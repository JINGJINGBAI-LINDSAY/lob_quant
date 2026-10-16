"""Experiment on one week of stateful BTCUSDT L2 features.

Core question: does making the feature-axis tokens correspond to REAL depth
levels (token l == level l, TLOB-style dual axis) beat the Logistic baseline
under the same strict chronological protocol used for the 30s bucket data?

Protocol (single symbol, continuous frames):
  - frame = 30s; target = future 5m average-price return (10 frames)
  - labels down/neutral/up, threshold = 70% quantile of |train forward|
  - 70% train / remaining 30% split evenly into dev/test by time
  - train-only normalization; train tail is horizon-purged; windows are full
    (T=120) and built within each segment, so no future info leaks
  - checkpoint chosen on dev; test scored exactly once

Artifacts include the encoder embedding q (the orderflow latent) for the test
set, not just classifications.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn

from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import LevelStructuredLOB

SEED = 20261002
LEVELS = 10
TOKENS = 120
HORIZON = 10
STRIDE = 10
TRAIN_FRACTION = 0.70
LEVEL_CHANNELS = ["bid_dist", "bid_qty", "ask_dist", "ask_qty", "imb"]
GLOBAL_COLS = ["spread_bps", "imb_top5", "ofi_1", "return_1"]
# Indices of heavy-tailed channels that need a (signed) log1p compression.
LV_LOG_IDX = (1, 3)          # bid_qty, ask_qty
G_SIGNED_LOG_IDX = (2,)      # ofi_1
MAX_DEEP_DIST_BPS = 5.0      # sane cap: normal level-9 p99 is ~0.39 bps
WINSOR_Q = (0.005, 0.995)
CLASS_WEIGHT_POWER = 0.7     # inverse-freq class weight exponent (single source)
FEATURES = Path(
    "artifacts/historical/l2_btc_week_features/l2_30s_features.parquet"
)
OUTPUT = Path("artifacts/historical/l2_btc_week_model")


def drop_warmup_frames(df):
    """Remove leading/any frames where the deepest visible level is absurd.

    Those are empty-book warmup snapshots (e.g. level9 at thousands of bps);
    a single such frame can blow up train-only standardization.
    """
    ok = ((df[f"bid_dist_{LEVELS - 1}"] < MAX_DEEP_DIST_BPS)
          & (df[f"ask_dist_{LEVELS - 1}"] < MAX_DEEP_DIST_BPS)
          & (df["spread_bps"] < MAX_DEEP_DIST_BPS))
    dropped = int((~ok).sum())
    return df[ok].reset_index(drop=True), dropped


def compress_tails(lv, gl):
    """Label-free elementwise compression of heavy-tailed channels."""
    for j in LV_LOG_IDX:
        lv[:, :, j] = np.log1p(np.clip(lv[:, :, j], 0, None))
    for j in G_SIGNED_LOG_IDX:
        x = gl[:, j]
        gl[:, j] = np.sign(x) * np.log1p(np.abs(x))
    return lv, gl


def build_arrays(df):
    n = len(df)
    lv = np.empty((n, LEVELS, len(LEVEL_CHANNELS)), np.float32)
    for l in range(LEVELS):
        lv[:, l, :] = df[
            [f"{ch}_{l}" for ch in LEVEL_CHANNELS]
        ].to_numpy(np.float32)
    gl = df[GLOBAL_COLS].to_numpy(np.float32)
    mid = df["mid"].to_numpy(np.float64)
    ma = np.convolve(mid, np.ones(HORIZON) / HORIZON, mode="valid")
    fwd = np.full(n, np.nan)
    fwd[: len(ma)] = ma / mid[: len(ma)] - 1.0
    return lv, gl, fwd


def make_windows(lv, gl, fwd):
    n = len(lv)
    wl, wg, ends = [], [], []
    for end in range(TOKENS - 1, n - HORIZON, STRIDE):
        wl.append(lv[end + 1 - TOKENS: end + 1])
        wg.append(gl[end + 1 - TOKENS: end + 1])
        ends.append(end)
    ends = np.asarray(ends)
    return (
        np.asarray(wl, np.float32),
        np.asarray(wg, np.float32),
        fwd[ends],
        ends,
    )


def labels_from(fwd_at_ends, threshold):
    y = np.ones(len(fwd_at_ends), dtype=np.int64)
    y[fwd_at_ends < -threshold] = 0
    y[fwd_at_ends > threshold] = 2
    return y


def warmup_cosine(step, total):
    warmup = max(1, int(total * 0.1))
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def summarize(y, prob, fwd):
    direction = prob[:, 2] - prob[:, 0]
    return {
        "macro_f1": float(f1_score(y, prob.argmax(1), average="macro",
                                  zero_division=0)),
        "auc": float(roc_auc_score(y == 2, direction)),
        "ic": float(np.corrcoef(direction, fwd)[0, 1]),
        "counts": np.bincount(prob.argmax(1), minlength=3).tolist(),
    }


def selective(y, prob, fwd, confidence, threshold):
    keep = confidence >= threshold
    if keep.sum() < 30:
        return None
    direction = prob[keep, 2] - prob[keep, 0]
    return {
        "macro_f1": float(f1_score(y[keep], prob[keep].argmax(1),
                                  average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y[keep] == 2, direction)),
        "ic": float(np.corrcoef(direction, fwd[keep])[0, 1]),
        "coverage": float(keep.mean()),
        "kept": int(keep.sum()),
    }


def prepare_l2_dataset():
    """Deterministic data prep reused by all L2 experiments.

    Returns normalized train/dev/test dicts ({L,G,y,f}) plus the label
    threshold, frame count and the underlying frame frame. Normalization is
    fitted on train only (robust: winsorized, heavy tails log-compressed).
    """
    frame = pd.read_parquet(FEATURES).sort_values("bin").reset_index(drop=True)
    frame, dropped = drop_warmup_frames(frame)
    print("dropped warmup/bad frames:", dropped,
          "remaining:", len(frame), flush=True)
    lv, gl, fwd = build_arrays(frame)
    lv, gl = compress_tails(lv, gl)
    n = len(frame)

    cut = int(n * TRAIN_FRACTION)
    ext_end = cut + (n - cut) // 2
    segments = {"train": (0, cut), "dev": (cut, ext_end),
                "test": (ext_end, n)}
    windows = {name: make_windows(lv[lo:hi], gl[lo:hi], fwd[lo:hi])
               for name, (lo, hi) in segments.items()}
    train_fwd = windows["train"][2]
    train_fwd = train_fwd[np.isfinite(train_fwd)]
    threshold = float(np.quantile(np.abs(train_fwd), 0.7))

    data = {}
    for name, (wl, wg, wf, _) in windows.items():
        defined = np.isfinite(wf)
        data[name] = {
            "L": torch.tensor(wl[defined]), "G": torch.tensor(wg[defined]),
            "y": torch.tensor(labels_from(wf[defined], threshold)),
            "f": wf[defined],
        }
    tr, dv, te = data["train"], data["dev"], data["test"]
    print("sizes", len(tr["L"]), len(dv["L"]), len(te["L"]), flush=True)

    def fit_robust(x):
        flat = x.reshape(-1, x.shape[-1])
        lo = torch.quantile(flat, WINSOR_Q[0], dim=0)
        hi = torch.quantile(flat, WINSOR_Q[1], dim=0)
        clipped = torch.clamp(flat, lo, hi)
        return lo, hi, clipped.mean(0), clipped.std(0).clamp_min(1e-6)

    l_lo, l_hi, l_mean, l_std = fit_robust(tr["L"])
    g_lo, g_hi, g_mean, g_std = fit_robust(tr["G"])

    def normalize(d):
        d["L"] = (torch.clamp(d["L"], l_lo, l_hi) - l_mean) / l_std
        d["G"] = (torch.clamp(d["G"], g_lo, g_hi) - g_mean) / g_std

    for d in (tr, dv, te):
        normalize(d)
    print("train level-channel std after scale:",
          np.round(tr["L"].reshape(-1, len(LEVEL_CHANNELS)).std(0).numpy(), 3),
          flush=True)
    return {"tr": tr, "dv": dv, "te": te, "threshold": threshold,
            "n": n, "frame": frame, "test_ends": windows["test"][3],
            "test_defined": np.isfinite(windows["test"][2]),
            "bounds": {"cut": cut, "ext_end": ext_end}}


def train_direction_model(model, tr, dv, lr=3e-4, epochs=24, verbose=True,
                          weight_power=CLASS_WEIGHT_POWER, patience=10,
                          batch_size=128):
    """Train a direction model, select checkpoint on dev Macro-F1.

    Loads the best checkpoint back into `model` and returns best_epoch.
    Class weights use the inverse-frequency exponent `weight_power` (sourced from
    one named constant/config, never an inline magic number).
    """
    counts = torch.bincount(tr["y"], minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).pow(weight_power)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    bs = batch_size
    total = epochs * math.ceil(len(tr["L"]) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: warmup_cosine(st, total)
    )
    best = {"f1": -1, "state": None, "epoch": 0}
    since = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr["L"]))
        for i in range(0, len(tr["L"]), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, _, _ = model(tr["L"][idx], tr["G"][idx])
            loss = loss_fn(logits, tr["y"][idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            dev_logits, _, _ = model(dv["L"], dv["G"])
        dev_prob = dev_logits.softmax(-1).numpy()
        dev_f1 = f1_score(dv["y"].numpy(), dev_prob.argmax(1),
                          average="macro", zero_division=0)
        if dev_f1 > best["f1"] + 1e-5:
            best = {
                "f1": dev_f1, "epoch": epoch + 1,
                "state": {k: v.clone() for k, v in
                          model.state_dict().items()},
            }
            since = 0
        else:
            since += 1
        if verbose:
            print(epoch + 1, f"dev_f1={dev_f1:.4f}", flush=True)
        if since >= patience:
            break
    model.load_state_dict(best["state"])
    return best["epoch"]


def eval_probs(model, d):
    model.eval()
    with torch.no_grad():
        logits, q, _ = model(d["L"], d["G"])
    return logits.softmax(-1).numpy(), q


def fit_logistic(tr, te):
    def flat_last(d):
        return torch.cat([d["L"][:, -1].reshape(len(d["L"]), -1),
                          d["G"][:, -1]], dim=1).numpy()

    logistic = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=SEED)
    logistic.fit(flat_last(tr), tr["y"].numpy())
    return float(f1_score(te["y"].numpy(), logistic.predict(flat_last(te)),
                          average="macro", zero_division=0))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = prepare_l2_dataset()
    tr, dv, te = ds["tr"], ds["dv"], ds["te"]

    model = LevelStructuredLOB(len(LEVEL_CHANNELS), len(GLOBAL_COLS),
                               levels=LEVELS)
    best_epoch = train_direction_model(model, tr, dv, lr=3e-4)

    test_prob, q_test = eval_probs(model, te)
    test_summary = summarize(te["y"].numpy(), test_prob, te["f"])

    dev_prob, _ = eval_probs(model, dv)
    dev_margin = np.abs(dev_prob[:, 2] - dev_prob[:, 0])
    test_margin = np.abs(test_prob[:, 2] - test_prob[:, 0])
    dev_gate = float(np.quantile(dev_margin, 0.5))
    selective_summary = selective(
        te["y"].numpy(), test_prob, te["f"], test_margin, dev_gate
    )
    logistic_test_f1 = fit_logistic(tr, te)

    result = {
        "data": {"frames": ds["n"],
                 "span": [int(ds["frame"]["bin"].iloc[0]),
                          int(ds["frame"]["bin"].iloc[-1])]},
        "samples": {"train": len(tr["L"]), "dev": len(dv["L"]),
                    "test": len(te["L"])},
        "label_threshold": ds["threshold"],
        "best_epoch": best_epoch,
        "level_model_all": test_summary,
        "level_model_selective": selective_summary,
        "logistic_test_macro_f1": logistic_test_f1,
        "vs_logistic": test_summary["macro_f1"] - logistic_test_f1,
        "exceeds_logistic": bool(test_summary["macro_f1"] > logistic_test_f1),
    }
    snapshot_runner(OUTPUT, __file__)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), OUTPUT / "level_model.pt")
    np.save(OUTPUT / "test_embedding.npy", q_test.numpy())
    ext_end = ds["bounds"]["ext_end"]
    test_ends = ds["test_ends"][ds["test_defined"]]
    test_bins = ds["frame"]["bin"].to_numpy()[ext_end + test_ends]
    pd.DataFrame({
        "bin": test_bins,
        "label": te["y"].numpy(), "return_h50": te["f"],
        "p_down": test_prob[:, 0], "p_neutral": test_prob[:, 1],
        "p_up": test_prob[:, 2], "margin": test_margin,
    }).to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
