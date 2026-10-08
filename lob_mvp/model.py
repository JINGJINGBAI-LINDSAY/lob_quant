from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn

from .core import MVPConfig, build_orderflow_bars


def _debug_post(hypothesis_id, location, msg, data):
    env_path = Path(".dbg/transformer-baseline-gap.env")
    run_id = os.environ.get("DEBUG_RUN_ID")
    if not run_id or not env_path.exists():
        return
    try:
        server_url = next(
            line.split("=", 1)[1]
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("DEBUG_SERVER_URL=")
        )
        request = urllib.request.Request(
            server_url,
            data=json.dumps({
                "sessionId": "transformer-baseline-gap",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "msg": msg,
                "data": data,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=0.5).read()
    except (OSError, StopIteration, ValueError):
        pass


class SwiGLU(nn.Module):
    def __init__(self, d_ff):
        super().__init__()
        self.up = nn.Linear(d_ff, d_ff * 2)
        self.down = nn.Linear(d_ff, d_ff)

    def forward(self, x):
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * value)


class RegimeAwareLOB(nn.Module):
    def __init__(self, event_dim: int, regime_dim: int, d_model: int = 128, layers: int = 2,
                 heads: int = 4, d_ff: int = 256, dropout: float = 0.1, max_window: int = 100,
                 use_medium_context: bool = True, use_regime_context: bool = True,
                 use_event_gate: bool = True, fusion_type: str = "gated_add"):
        super().__init__()
        if fusion_type not in {"gated_add", "concat_mlp", "none"}:
            raise ValueError(f"Unsupported fusion_type={fusion_type}")
        self.use_medium_context = use_medium_context
        self.use_regime_context = use_regime_context
        self.use_event_gate = use_event_gate
        self.fusion_type = fusion_type
        self.event = nn.Sequential(nn.Linear(event_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.regime = nn.Sequential(nn.Linear(regime_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(d_model * 3 + 5, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.medium_pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.context_projection = nn.Linear(d_model, d_model)
        self.regime_projection = nn.Linear(d_model, d_model)
        self.context_gate = nn.Linear(d_model, 1)
        self.regime_gate = nn.Linear(d_model, 1)
        self.concat_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.time = nn.Linear(1, d_model)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        self.position = nn.Parameter(torch.zeros(1, max_window, d_model))
        nn.init.normal_(self.position, std=0.02)
        # The built-in layer owns the residual/norm/attention path; SwiGLU
        # supplies the requested 256-wide gated FFN through its activation hook.
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation=SwiGLU(d_ff)
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor, x_medium: torch.Tensor, regime: torch.Tensor, priors: torch.Tensor,
                delta_t: torch.Tensor):
        e = self.event(x)
        r = self.regime(regime).unsqueeze(1) if self.use_regime_context else torch.zeros_like(e[:, :1])
        if self.use_medium_context:
            medium_events = self.event(x_medium)
            c = self.medium_pool(medium_events.mean(dim=1)).unsqueeze(1)
        else:
            medium_events = torch.zeros_like(e[:, :1])
            c = torch.zeros_like(e[:, :1])
        if self.use_event_gate:
            score = torch.sigmoid(self.gate(
                torch.cat([e, c.expand_as(e), r.expand_as(e), priors], -1)
            )).squeeze(-1)
        else:
            score = torch.ones_like(e[..., 0])
        event_part = e * score.unsqueeze(-1)
        context_part = (
            self.context_projection(c) * torch.sigmoid(self.context_gate(c))
            if self.use_medium_context else torch.zeros_like(c)
        )
        regime_part = (
            self.regime_projection(r) * torch.sigmoid(self.regime_gate(r))
            if self.use_regime_context else torch.zeros_like(r)
        )
        if self.fusion_type == "concat_mlp":
            fused = self.concat_projection(torch.cat([
                event_part,
                c.expand_as(e),
                r.expand_as(e),
            ], dim=-1))
        else:
            fused = event_part + context_part + regime_part
        z = (
            fused
            + self.time(torch.log1p(delta_t / 1000.0).unsqueeze(-1)) * self.time_scale
            + self.position[:, :x.size(1)]
        )
        if not hasattr(self, "_debug_pre_fusion_logged") and os.environ.get("DEBUG_RUN_ID"):
            parts = {
                "event_gated": event_part,
                "time": self.time(torch.log1p(delta_t / 1000.0).unsqueeze(-1)) * self.time_scale,
                "position": self.position[:, :x.size(1)],
                "context": context_part,
                "regime": regime_part,
                "fused": z,
            }
            _debug_post("C", "model.py:forward/fusion", "[DEBUG] fusion component norms", {
                "fusion_type": self.fusion_type,
                "components": {
                    key: {
                        "rms": float(value.detach().pow(2).mean().sqrt()),
                        "abs_max": float(value.detach().abs().max()),
                    }
                    for key, value in parts.items()
                },
            })
            self._debug_pre_fusion_logged = True
        if not hasattr(self, "_debug_pre_gate_logged") and os.environ.get("DEBUG_RUN_ID"):
            _debug_post("B", "model.py:forward/gate", "[DEBUG] encoded branch and gate stats", {
                "event_rms": float(e.detach().pow(2).mean().sqrt()),
                "medium_event_rms": float(medium_events.detach().pow(2).mean().sqrt()),
                "context_rms": float(c.detach().pow(2).mean().sqrt()),
                "regime_rms": float(r.detach().pow(2).mean().sqrt()),
                "gate_mean": float(score.detach().mean()),
                "gate_std": float(score.detach().std()),
                "gate_min": float(score.detach().min()),
                "gate_max": float(score.detach().max()),
                "medium_enabled": self.use_medium_context,
                "regime_enabled": self.use_regime_context,
                "gate_enabled": self.use_event_gate,
            })
            self._debug_pre_gate_logged = True
        z = self.encoder(z)
        alpha = torch.softmax(self.pool(z).squeeze(-1), dim=1)
        q = (z * alpha.unsqueeze(-1)).sum(1)
        return self.head(q), q, score


def make_windows(frame, feature_columns, regime_columns, cfg: MVPConfig, normalization=None):
    xs, xm, rs, dts, ys = [], [], [], [], []
    priors = []
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        values = symbol_frame.reset_index(drop=True)
        event_times = values["event_time_ms"].to_numpy(np.int64)
        normalized = values.copy()
        if normalization:
            for column, stats in normalization.items():
                normalized[column] = (normalized[column] - stats["mean"]) / stats["std"]
        bars = build_orderflow_bars(values, cfg)
        if bars.empty:
            continue
        bar_times = bars["event_time_ms"].to_numpy(np.int64)
        normalized_bars = bars.copy()
        if normalization:
            for column, stats in normalization.items():
                if column in normalized_bars:
                    normalized_bars[column] = (
                        normalized_bars[column] - stats["mean"]
                    ) / stats["std"]
        event_matrix = normalized[feature_columns].to_numpy(np.float32)
        event_priors = normalized[cfg.prior_columns].to_numpy(np.float32)
        event_delta = values["delta_event_time_ms"].to_numpy(np.float32)
        regime_matrix = normalized[regime_columns].to_numpy(np.float32)
        labels = values["label_h50"].to_numpy(np.int64)
        bar_matrix = normalized_bars[feature_columns].to_numpy(np.float32)
        for end in range(0, len(values), cfg.sample_stride):
            end_time = int(event_times[end])
            short_start = end_time - cfg.short_window_hours * 60 * 60 * 1000
            short_start_idx = np.searchsorted(event_times, short_start, side="left")
            short_start_idx = max(short_start_idx, end + 1 - cfg.short_events)
            short_part = event_matrix[short_start_idx:end + 1]
            short_prior = event_priors[short_start_idx:end + 1]
            short_delta = event_delta[short_start_idx:end + 1]
            if len(short_part) == 0:
                continue
            medium_start = end_time - cfg.medium_window_days * 24 * 60 * 60 * 1000
            medium_start_idx = np.searchsorted(bar_times, medium_start, side="left")
            medium_end_idx = np.searchsorted(bar_times, end_time, side="right")
            medium_start_idx = max(medium_start_idx, medium_end_idx - cfg.medium_bars)
            medium_part = bar_matrix[medium_start_idx:medium_end_idx]
            if len(short_part) < cfg.short_events or len(medium_part) < cfg.medium_bars:
                continue
            xs.append(short_part)
            xm.append(medium_part)
            rs.append(regime_matrix[end])
            priors.append(short_prior)
            dts.append(short_delta)
            ys.append(int(labels[end]))
    return [torch.tensor(v) for v in (np.array(xs), np.array(xm), np.array(rs), np.array(priors),
                                      np.array(dts), np.array(ys))]


def window_sample_metadata(frame, cfg: MVPConfig):
    """Rows aligned with the samples emitted by make_windows."""
    rows = []
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        values = symbol_frame.reset_index(drop=True)
        if values.empty:
            continue
        event_times = values["event_time_ms"].to_numpy(np.int64)
        bars = build_orderflow_bars(values, cfg)
        bar_times = bars["event_time_ms"].to_numpy(np.int64)
        for end in range(0, len(values), cfg.sample_stride):
            end_time = int(event_times[end])
            short_start = end_time - cfg.short_window_hours * 60 * 60 * 1000
            short_start_idx = max(
                np.searchsorted(event_times, short_start, side="left"),
                end + 1 - cfg.short_events,
            )
            medium_start = end_time - cfg.medium_window_days * 24 * 60 * 60 * 1000
            medium_start_idx = max(
                np.searchsorted(bar_times, medium_start, side="left"),
                np.searchsorted(bar_times, end_time, side="right") - cfg.medium_bars,
            )
            medium_end_idx = np.searchsorted(bar_times, end_time, side="right")
            if end + 1 - short_start_idx < cfg.short_events:
                continue
            if medium_end_idx - medium_start_idx < cfg.medium_bars:
                continue
            rows.append(values.iloc[[end]])
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()
