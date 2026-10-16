"""Two-axis short LOB model + medium/long confidence filter.

Direction path (per design routes 2 & 3):
    event window (T x F)
      -> dilated causal conv5 stem (time local / widened receptive field)
      -> time-axis self-attention  ("which past moment matters")
      -> feature-axis attention over the feature dimension at the pooled step
         ("which level/factor matters")
      -> 3-class direction head
The feature-axis attention is implemented as a lightweight gated mixer over
feature tokens (TLOB-lite), not a full heavy dual transformer.

Confidence filter (route 4):
    a separate small scorer consumes the medium/long reversion/continuation
    signals and returns a tradable probability. It never changes the direction;
    it only gates whether a directional prediction is acted upon (selective).
"""
from __future__ import annotations

import torch
from torch import nn

TOKENS = 120


class DilatedCausalConv(nn.Module):
    def __init__(self, d_model, kernel=5, dilation=2, dropout=0.2):
        super().__init__()
        self.kernel = kernel
        self.dilation = dilation
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel, padding=pad,
                      dilation=dilation, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.net(h)
        return h[:, :, :x.size(1)].transpose(1, 2)


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.net(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(1), w


class FeatureAxisAttention(nn.Module):
    """Lightweight self-attention across a set of feature tokens.

    Reshape the pooled d_model vector into G groups (feature tokens), run one
    self-attention layer across groups, then flatten. This lets the model
    reweight which feature group is informative.
    """

    def __init__(self, d_model, groups=8, heads=4, dropout=0.2):
        super().__init__()
        assert d_model % groups == 0
        self.groups = groups
        self.group_dim = d_model // groups
        layer = nn.TransformerEncoderLayer(
            self.group_dim,
            1 if self.group_dim == 1 else min(heads, self.group_dim),
            max(16, self.group_dim * 2),
            dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q):
        b = q.size(0)
        tokens = q.view(b, self.groups, self.group_dim)
        out = self.encoder(tokens).reshape(b, -1)
        return self.norm(out + q)


class TwoAxisLOB(nn.Module):
    """Two-axis model with optional causal masking and dual readout.

    Variants for the E020 ablation (defaults reproduce the original model):
      * original: causal_mask=False, dual_readout=False
      * B:        causal_mask=True,  dual_readout=False  (mask + attention pool)
      * C:        causal_mask=True,  dual_readout=True   (mask + last/pool fuse)

    `causal_mask` upper-triangular mask blocks each frame from attending to
    later frames, so the time-axis attention (bidirectional by default in
    nn.TransformerEncoder) cannot see the future. `dual_readout` concatenates
    the attention-pooled summary with the LAST frame's vector and projects
    back to d_model, giving both global context and the latest instant.
    """

    def __init__(self, event_dim, d_model=64, layers=2, heads=4,
                 d_ff=128, dropout=0.2, feature_groups=8,
                 causal_mask=False, dual_readout=False):
        super().__init__()
        self.causal_mask = causal_mask
        self.dual_readout = dual_readout
        self.proj = nn.Sequential(
            nn.Linear(event_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = DilatedCausalConv(d_model)
        self.pos = nn.Parameter(torch.zeros(1, TOKENS, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(layer, layers)
        # Boolean additive mask: True = "may not attend". Row t may see
        # frames 0..t only (diagonal is False), so no row is fully masked.
        self.register_buffer(
            "causal_attn_mask",
            torch.triu(torch.ones(TOKENS, TOKENS, dtype=torch.bool),
                       diagonal=1),
            persistent=False,
        )
        self.pool = AttentionPool(d_model)
        if dual_readout:
            self.fuse = nn.Linear(2 * d_model, d_model)
        self.pre_feature_norm = nn.LayerNorm(d_model)
        self.feature_attention = FeatureAxisAttention(
            d_model, groups=feature_groups, heads=heads, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, events):
        h = self.proj(events)
        h = self.conv(h) + self.pos
        z = self.time_encoder(
            h, mask=self.causal_attn_mask if self.causal_mask else None
        )
        q_attn, time_weights = self.pool(z)
        if self.dual_readout:
            q = self.fuse(torch.cat([q_attn, z[:, -1, :]], dim=-1))
        else:
            q = q_attn
        q = self.pre_feature_norm(q)
        q = self.feature_attention(q)
        return self.head(q), q, time_weights


class LevelStructuredLOB(nn.Module):
    """Two-axis model with REAL level semantics (TLOB-style dual axis).

    Inputs keep the L depth levels explicit:
      levels:  (B, T, L, C) per-level channels (dist/qty/imbalance per side)
      globals: (B, T, G) book-wide channels (spread, top5 imbalance, ofi, ...)

    Time axis: a shared dilated causal conv + temporal self-attention is run
    independently for each level (weights shared across levels), then
    attention-pooled over T -> exactly one token per depth level.
    Feature axis: projected global tokens are appended and self-attention runs
    ACROSS the L level tokens, so token l literally represents depth level l.
    An attention pool over the level+global tokens gives the orderflow
    embedding q, which feeds the 3-class direction head.
    """

    def __init__(self, level_dim, global_dim, levels=10, d_model=64,
                 time_layers=1, heads=4, d_ff=128, dropout=0.2,
                 global_tokens=2, tokens=TOKENS):
        super().__init__()
        self.levels = levels
        self.global_tokens = global_tokens
        self.level_proj = nn.Sequential(
            nn.Linear(level_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = DilatedCausalConv(d_model)
        self.pos = nn.Parameter(torch.zeros(1, tokens, d_model))
        nn.init.normal_(self.pos, std=0.02)
        tlayer = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(tlayer, time_layers)
        self.time_pool = AttentionPool(d_model)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model), nn.GELU(),
            nn.Linear(d_model, global_tokens * d_model),
        )
        llayer = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.level_encoder = nn.TransformerEncoder(llayer, 1)
        self.level_pool = AttentionPool(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, levels, globals_):
        b, t, L, _ = levels.shape
        h = self.level_proj(levels)                  # b,T,L,d
        h = h.permute(0, 2, 1, 3).reshape(b * L, t, -1)
        h = self.conv(h) + self.pos
        h = self.time_encoder(h)
        per_level, _ = self.time_pool(h)            # b*L,d
        per_level = per_level.reshape(b, L, -1)
        g = self.global_proj(globals_[:, -1, :]).reshape(
            b, self.global_tokens, -1
        )
        tok = torch.cat([per_level, g], dim=1)      # b,L+K,d
        tok = self.level_encoder(tok)
        q, level_weights = self.level_pool(tok)
        return self.head(q), q, level_weights


class ConfidenceFilter(nn.Module):
    """Route 4: tradability scorer from slow reversion/continuation signals.

    Outputs P(tradable) in (0,1). It is direction-agnostic; downstream we only
    act on the direction when this confidence is high (selective prediction).
    """

    def __init__(self, slow_dim, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim, d_model), nn.GELU(),
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, slow):
        return torch.sigmoid(self.net(slow)).squeeze(-1)
