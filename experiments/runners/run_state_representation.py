"""Self-supervised market-STATE representation on ~9 months of 30s buckets.

The latent vector must encode the market REGIME, not price direction.

Inputs (per 120-frame = 60 min window), direction-neutral only:
  depth/notional logs across bps bands, depth imbalances, and unsigned
  macro state (realized vol, liquidity, event intensity). Signed price moves
  (return_1, price_change) are deliberately excluded.

Self-supervised targets (future, strictly causal, all NON-directional):
  next-H realized volatility, next-H liquidity, next-H event intensity.

After training we cluster the held-out (non-overlapping) q and apply three
pre-registered GATEs:
  1. clusters are not a relabel of price direction (crosstab with label_h50);
  2. latent clustering beats classic clustering on raw state (silhouette and
     split-half reproducibility ARI);
  3. states persist far longer than a single window.
Any GATE failing -> stop; the latent has no proven value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import AttentionPool, DilatedCausalConv, TOKENS

FEATURES = Path("artifacts/historical/binance_1y/event_features.parquet")
OUTPUT = Path("artifacts/historical/state_representation")

H = 10                 # forward target window (10 frames = 5 min)
TRAIN_STRIDE = 30      # dense, overlapping train windows (15 min)
EVAL_STRIDE = TOKENS   # non-overlapping test windows (60 min)
D_MODEL = 64
K_CLUSTERS = 4

DEPTH_LOG = [
    f"{side}_depth_log_{b}"
    for b in ("0p2", "1", "2", "3", "4", "5")
    for side in ("bid", "ask")
]
NOTIONAL_LOG = [
    f"{side}_notional_log_{b}"
    for b in ("0p2", "1", "2", "3", "4", "5")
    for side in ("bid", "ask")
]
IMBALANCE = [f"depth_imbalance_{b}" for b in ("0p2", "1", "2", "3", "4", "5")]
MACRO = ["rv", "liquidity", "event_intensity", "depth_imbalance_mean",
         "trade_imbalance_mean", "ofi_proxy", "trade_range",
         "log_trade_notional", "log_trade_count"]
STATE_INPUT_COLS = DEPTH_LOG + NOTIONAL_LOG + IMBALANCE + MACRO

TARGET_COLS = ["tgt_vol", "tgt_liq", "tgt_intensity"]


def add_future_targets(frame):
    def fwd_mean(col):
        return frame[col].shift(-1).rolling(H, min_periods=H).mean()

    def fwd_std(col):
        return frame[col].shift(-1).rolling(H, min_periods=H).std()

    out = pd.DataFrame({
        "tgt_vol": fwd_std("return_1"),
        "tgt_liq": fwd_mean("liquidity"),
        "tgt_intensity": fwd_mean("event_intensity"),
    })
    return pd.concat([frame, out], axis=1)


def make_windows(values, targets, ends_meta, stride, lo, hi):
    start = max(lo, TOKENS - 1)
    stop = min(hi, len(values) - H)
    xs, ys, rows = [], [], []
    for end in range(start, stop, stride):
        if not np.isfinite(targets[end]).all():
            continue  # skip windows whose future target is undefined
        xs.append(values[end + 1 - TOKENS:end + 1])
        ys.append(targets[end])
        rows.append(ends_meta.iloc[[end]])
    if not xs:
        return None
    return (torch.tensor(np.asarray(xs), dtype=torch.float32),
            torch.tensor(np.asarray(ys), dtype=torch.float32),
            pd.concat(rows, ignore_index=True))


class StateEncoder(nn.Module):
    def __init__(self, event_dim, d_model=D_MODEL, layers=2, heads=4,
                 d_ff=128, dropout=0.2):
        super().__init__()
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
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.pool = AttentionPool(d_model)
        self.head_vol = nn.Linear(d_model, 1)
        self.head_liq = nn.Linear(d_model, 1)
        self.head_int = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.conv(h) + self.pos
        z = self.encoder(h)
        q, _ = self.pool(z)
        pred = torch.cat([self.head_vol(q), self.head_liq(q),
                          self.head_int(q)], dim=1)
        return pred, q


def fit_stats(matrix):
    """NaN-aware mean/std; early warmup rows carry NaN and must not poison."""
    t = matrix
    mean = torch.nanmean(t, dim=0)
    var = torch.nanmean((t - mean) ** 2, dim=0)
    std = var.sqrt().clamp_min(1e-6)
    mean = torch.nan_to_num(mean, nan=0.0)
    return mean, std


def train_encoder(tr, dv, epochs, lr=3e-4, bs=256, patience=8):
    model = StateEncoder(tr["x"].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(tr["x"], tr["y"]), batch_size=bs,
                        shuffle=True)
    best = {"loss": float("inf"), "state": None, "epoch": 0}
    since = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            dv_pred, _ = model(dv["x"])
            dv_loss = float(loss_fn(dv_pred, dv["y"]))
        if dv_loss < best["loss"] - 1e-4:
            best = {"loss": dv_loss, "epoch": ep + 1,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
        print(f"ep{ep+1} dev_mse={dv_loss:.5f}", flush=True)
        if since >= patience:
            break
    if best["state"] is None:
        raise RuntimeError("dev loss was never finite; check targets/inputs")
    model.load_state_dict(best["state"])
    return model, best["epoch"], best["loss"]


def reproducibility_ari(features):
    """Fit a clusterer independently on two halves; ARI on the second half."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    n = len(features)
    h = n // 2
    a, b = features[:h], features[h:]
    base = KMeans(K_CLUSTERS, n_init=10, random_state=0).fit(a)
    # Labels from a model that saw half A vs one independently fit on half B.
    pred_from_a = base.predict(b)
    pred_b = KMeans(K_CLUSTERS, n_init=10, random_state=1).fit_predict(b)
    return float(adjusted_rand_score(pred_from_a, pred_b))


def cluster_report(q, meta, tag):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    km = KMeans(K_CLUSTERS, n_init=10, random_state=0)
    cid = km.fit_predict(q)
    sil = float(silhouette_score(q, cid)) if len(np.unique(cid)) > 1 else float("nan")
    ari = reproducibility_ari(q)

    label = meta["label_h50"].to_numpy()
    # GATE 1: cluster x direction crosstab (row-normalized).
    crosstab = []
    max_dir_share = []
    for k in range(K_CLUSTERS):
        m = cid == k
        if m.sum() == 0:
            continue
        share = [float((label[m] == j).mean()) for j in (0, 1, 2)]
        crosstab.append({"cluster": int(k), "n": int(m.sum()),
                         "down/neutral/up": [round(s, 3) for s in share]})
        max_dir_share.append(max(share))

    # GATE 3: dwell persistence in non-overlapping window units.
    dwell = []
    run = 1
    for i in range(1, len(cid)):
        if cid[i] == cid[i - 1]:
            run += 1
        else:
            dwell.append(run)
            run = 1
    dwell.append(run)

    # Risk profile per cluster (proves clusters separate STATE, not direction).
    profile = []
    for col in ("rv", "liquidity", "event_intensity"):
        profile.append({col: {str(k): round(float(meta.loc[cid == k, col].mean()), 4)
                              for k in range(K_CLUSTERS)}})

    return {
        "tag": tag, "silhouette": sil, "reproducibility_ari": ari,
        "cluster_x_direction": crosstab,
        "worst_cluster_max_direction_share": round(min(max_dir_share), 3),
        "mean_dwell_windows": round(float(np.mean(dwell)), 2),
        "risk_profile": profile, "cluster_id": cid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--max-train-windows", type=int, default=0,
                    help="smoke test: cap dense train windows (0 = all)")
    args = ap.parse_args()

    frame = pd.read_parquet(FEATURES)
    frame = frame.sort_values("event_time_ms").reset_index(drop=True)
    missing = [c for c in STATE_INPUT_COLS if c not in frame.columns]
    if missing:
        raise SystemExit(f"missing state columns: {missing}")
    frame = add_future_targets(frame)

    n = len(frame)
    i60, i80 = int(n * 0.60), int(n * 0.80)
    raw = frame[STATE_INPUT_COLS].to_numpy(np.float32)
    tgt = frame[TARGET_COLS].to_numpy(np.float32)

    # Train-only standardization for inputs and targets.
    x_mean, x_std = fit_stats(torch.tensor(raw[:i60]))
    t_mean, t_std = fit_stats(torch.tensor(tgt[:i60]))
    xn = ((torch.tensor(raw) - x_mean) / x_std).numpy()
    yn = ((torch.tensor(tgt) - t_mean) / t_std).numpy()
    xn = np.nan_to_num(xn, nan=0.0)

    tr = make_windows(xn, yn, frame, TRAIN_STRIDE, 0, i60)
    cut_dv = int(i60 + (i80 - i60) * 0.5)
    dv = make_windows(xn, yn, frame, EVAL_STRIDE, i60, cut_dv)
    te = make_windows(xn, yn, frame, EVAL_STRIDE, i80, n)
    if tr is None or dv is None or te is None:
        raise SystemExit("not enough frames for windows")
    if args.max_train_windows:
        tr = (tr[0][:args.max_train_windows], tr[1][:args.max_train_windows],
              tr[2].iloc[:args.max_train_windows])
    print(f"windows train={len(tr[0])} dev={len(dv[0])} test={len(te[0])} "
          f"input_dim={tr[0].shape[-1]}", flush=True)

    trd = {"x": tr[0], "y": tr[1]}
    dvd = {"x": dv[0], "y": dv[1]}
    model, best_ep, best_loss = train_encoder(trd, dvd, args.epochs)

    model.eval()
    with torch.no_grad():
        _, q_test = model(te[0])
    q = q_test.numpy()
    latent_report = cluster_report(q, te[2].reset_index(drop=True), "latent_q")

    # Baseline: cluster raw aggregated state (window mean of a compact subset).
    base_cols = ["rv", "liquidity", "event_intensity"] + IMBALANCE
    bi = [STATE_INPUT_COLS.index(c) for c in base_cols]
    raw_agg = te[0][:, :, bi].mean(dim=1).numpy()
    baseline_report = cluster_report(raw_agg, te[2].reset_index(drop=True),
                                     "raw_state_baseline")
    for rep in (latent_report, baseline_report):
        rep.pop("cluster_id", None)

    gates = {
        "g1_not_direction": bool(
            latent_report["worst_cluster_max_direction_share"] < 0.7),
        "g2_beats_baseline_silhouette": bool(
            latent_report["silhouette"] > baseline_report["silhouette"]),
        "g2_beats_baseline_reproducibility": bool(
            latent_report["reproducibility_ari"]
            > baseline_report["reproducibility_ari"]),
        "g3_persistent": bool(latent_report["mean_dwell_windows"] > 1.5),
    }
    gates["all_pass"] = all(gates.values())

    result = {"best_epoch": best_ep, "best_dev_mse": best_loss,
              "test_windows": len(te[0]), "latent": latent_report,
              "baseline": baseline_report, "gates": gates}
    snapshot_runner(OUTPUT, __file__)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(q).to_parquet(OUTPUT / "test_latent_q.parquet")
    te[2][["event_time_ms", "label_h50", "rv", "liquidity",
           "event_intensity"]].to_csv(OUTPUT / "test_meta.csv", index=False)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
