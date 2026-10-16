"""Slow-state encoder: learn a continuous latent z from causal slow features.

Purpose (this run): produce the latent ONLY. No direction head, no clustering.

Input  : past WINDOW=120 30s frames (1h) of 15 causal slow features.
Encoder: causal conv stack + attention pool -> z (Z_DIM=12).
Heads  : 3 regression heads predicting the state 1h in the future:
           future log(rv_1h), liquidity_1h, log(event_intensity_1h).
The multi-task objective keeps z describing market STATE (volatility,
liquidity, activity) instead of collapsing onto direction.

Timeline split (single pass, by row position): 60% train / 10% valid / 30% OOS.
All scalers fit on train only. Windows that touch a data gap (a frame interval
!= 30s) are dropped. Export writes z for every frame whose past-window is
contiguous, tagged train/valid/oos.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from lob_mvp.two_axis_model import AttentionPool, DilatedCausalConv

INPUT_COLS = [
    "return_1", "ofi_proxy",
    "depth_imbalance_mean", "trade_imbalance_mean",
    "rv_1h", "rv_1d", "rv_7d",
    "liquidity_1h", "liquidity_1d", "liquidity_7d",
    "event_intensity_1h", "event_intensity_1d", "event_intensity_7d",
]
RAW = Path(
    "artifacts/historical/binance_1y/event_features.parquet"
)
OUT = Path("artifacts/historical/slow_state")

WINDOW = 120          # past frames fed to encoder
FUTURE_H = 120        # predict the state this many frames ahead
Z_DIM = 12
D_MODEL = 64
TRAIN_FRAC, VALID_FRAC = 0.60, 0.70
TRAIN_STRIDE = 10     # thin dense train windows (slow features are slow-moving)
BATCH = 512
SEED = 20261009


# ---------------------------------------------------------------- model
class SlowStateEncoder(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.inproj = nn.Linear(n_feat, D_MODEL)
        self.conv1 = DilatedCausalConv(D_MODEL, kernel=5, dilation=2)
        self.conv2 = DilatedCausalConv(D_MODEL, kernel=5, dilation=4)
        self.pool = AttentionPool(D_MODEL)
        self.to_z = nn.Linear(D_MODEL, Z_DIM)
        self.head_rv = nn.Linear(Z_DIM, 1)
        self.head_liq = nn.Linear(Z_DIM, 1)
        self.head_ei = nn.Linear(Z_DIM, 1)

    def forward(self, x):
        h = self.inproj(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h, _w = self.pool(h)
        z = self.to_z(h)
        heads = torch.cat([
            self.head_rv(z), self.head_liq(z), self.head_ei(z)
        ], dim=-1)
        return heads, z


# ---------------------------------------------------------------- data
def load_matrix():
    cols = ["symbol", "event_time_ms"] + INPUT_COLS
    df = pd.read_parquet(RAW, columns=cols).sort_values(
        "event_time_ms"
    ).reset_index(drop=True)
    return df


def standardize(x, fit_idx):
    mu = x[fit_idx].mean(0)
    sd = x[fit_idx].std(0).clip(1e-6)
    return (x - mu) / sd


def build_tensors(df):
    n = len(df)
    x_raw = df[INPUT_COLS].to_numpy(np.float32)

    # Targets in raw (untransformed) space, then log/compress.
    rv = np.log(df["rv_1h"].to_numpy(np.float64))
    liq = df["liquidity_1h"].to_numpy(np.float64)
    ei = np.log(df["event_intensity_1h"].to_numpy(np.float64))
    tgt = np.stack([rv, liq, ei], axis=1).astype(np.float32)

    i_train = int(n * TRAIN_FRAC)
    x = standardize(x_raw, slice(0, i_train)).astype(np.float32)
    y = standardize(tgt, slice(0, i_train)).astype(np.float32)

    # Targets are the state at t+FUTURE_H.
    y_future = np.full_like(y, np.nan)
    y_future[: n - FUTURE_H] = y[FUTURE_H:]

    seg = np.full(n, "oos", dtype=object)
    seg[: i_train] = "train"
    seg[i_train: int(n * VALID_FRAC)] = "valid"
    seg[int(n * VALID_FRAC):] = "oos"
    return x, y_future, seg


def window_batch(x, ends):
    return np.stack(
        [x[e + 1 - WINDOW: e + 1] for e in ends], axis=0
    )


# ---------------------------------------------------------------- train
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum())
                                  * np.sqrt((b * b).sum()) + 1e-12))


def evaluate(model, x, ends, yfut, device, bs=1024):
    preds, tgts = [], []
    with torch.no_grad():
        for i in range(0, len(ends), bs):
            e = ends[i:i + bs]
            xb = torch.tensor(window_batch(x, e)).to(device)
            p, _z = model(xb)
            preds.append(p.cpu().numpy()); tgts.append(yfut[e])
    P, T = np.concatenate(preds), np.concatenate(tgts)
    return {name: corr(P[:, j], T[:, j])
            for j, name in enumerate(("rv", "liq", "ei"))}


def train(model, x, yfut, device, epochs):
    n = len(x)
    i_tr0, i_tr1 = 0, int(n * TRAIN_FRAC)
    i_va0, i_va1 = int(n * TRAIN_FRAC), int(n * VALID_FRAC)
    allidx = np.arange(n)
    tr_pool = allidx[i_tr0 + WINDOW: i_tr1 - FUTURE_H]
    va_ends = allidx[i_va0: i_va1]
    # last FUTURE_H rows have no future target (NaN); finite selects the rest.
    tr_pool = tr_pool[np.isfinite(yfut[tr_pool, 0])]
    va_ends = va_ends[np.isfinite(yfut[va_ends, 0])]
    tr_ends = tr_pool[::TRAIN_STRIDE]

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(tr_ends)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            e = perm[i:i + BATCH]
            xb = torch.tensor(window_batch(x, e)).to(device)
            yb = torch.tensor(yfut[e]).to(device)
            opt.zero_grad()
            p, _z = model(xb)
            loss = loss_fn(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(e)
        sched.step()
        t1 = time.time()
        vc = evaluate(model, x, va_ends, yfut, device)
        history.append({"epoch": ep + 1, "train_loss": tot / len(perm),
                        "valid_corr": vc})
        print(ep + 1, f"loss={tot/len(perm):.4f} train={t1-t0:.1f}s "
              f"eval={time.time()-t1:.1f}s valid_corr rv={vc['rv']:+.3f} "
              f"liq={vc['liq']:+.3f} ei={vc['ei']:+.3f}", flush=True)
    return history, va_ends


# ---------------------------------------------------------------- export
def export_z(model, x, seg, ts, end_ok, device):
    ends = np.where(end_ok)[0]
    zs = []
    with torch.no_grad():
        for i in range(0, len(ends), 1024):
            e = ends[i:i + 1024]
            xb = torch.tensor(window_batch(x, e)).to(device)
            _p, z = model(xb)
            zs.append(z.cpu().numpy())
    z = np.concatenate(zs)
    out = pd.DataFrame(z, columns=[f"z_{j}" for j in range(Z_DIM)])
    out.insert(0, "event_time_ms", ts[ends])
    out.insert(1, "segment", seg[ends])
    out.insert(0, "symbol", "BTCUSDT")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    df = load_matrix()
    n = len(df)
    x, yfut, seg = build_tensors(df)
    # frames with enough past history get exported; targets use row-wise shift
    end_ok = np.zeros(n, dtype=bool)
    end_ok[WINDOW - 1:] = True

    model = SlowStateEncoder(len(INPUT_COLS)).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={nparam} frames={n}", flush=True)

    history, va_ends = train(model, x, yfut, device, args.epochs)

    OUT.mkdir(parents=True, exist_ok=True)
    zdf = export_z(model, x, seg, df["event_time_ms"].to_numpy(), end_ok, device)
    zdf.to_parquet(OUT / "slow_z.parquet", index=False)
    torch.save(model.state_dict(), OUT / "slow_encoder.pt")

    zstd = zdf[[f"z_{j}" for j in range(Z_DIM)]].std().to_dict()
    metrics = {"params": nparam, "device": device,
               "total_frames": n,
               "z_rows": len(zdf), "z_std": {k: float(v) for k, v in zstd.items()},
               "final_valid_corr": history[-1]["valid_corr"], "history": history}
    (OUT / "slow_state_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"exported z rows={len(zdf)} dropped={n-len(zdf)} "
          f"z_std range=({min(zstd.values()):.3f},{max(zstd.values()):.3f})",
          flush=True)


if __name__ == "__main__":
    main()
