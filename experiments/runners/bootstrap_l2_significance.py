"""Paired bootstrap: is the L2 level model really better than Logistic?

The headline gap (+0.054 Macro-F1) comes from only 285 test samples, so a
point estimate cannot separate signal from luck. This script rebuilds the
exact deterministic data prep, refits the Logistic baseline identically, and
uses the already-evaluated level-model probabilities, then resamples the 285
test indices WITH REPLACEMENT and recomputes Macro-F1 for both on the SAME
resample (paired).

Two schemes:
  - iid bootstrap (assumes exchangeable rows)
  - moving-block bootstrap (preserves time dependence; test windows are 5 min
    apart and likely serially correlated, so this is the conservative read)

Outputs the distribution of (level_f1 - logistic_f1): mean, 95% interval and
P(gap > 0). The comparison conditions on the single trained model, i.e. it
quantifies evaluation uncertainty, not training-seed uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from experiments.runners.run_l2_level_experiment import (
    FEATURES, SEED, WINSOR_Q,
    build_arrays, compress_tails, drop_warmup_frames, labels_from,
    make_windows,
)
from lob_mvp.run_io import snapshot_runner

B = 20_000
BLOCK = 10          # ~50 min blocks (test windows are 5 min apart)
OUTPUT = Path("artifacts/historical/l2_btc_week_model")


def prepare():
    df = pd.read_parquet(FEATURES).sort_values("bin").reset_index(drop=True)
    df, dropped = drop_warmup_frames(df)
    lv, gl, fwd = build_arrays(df)
    lv, gl = compress_tails(lv, gl)
    n = len(df)
    cut = int(n * 0.70)
    ext_end = cut + (n - cut) // 2
    segments = {"train": (0, cut), "dev": (cut, ext_end),
                "test": (ext_end, n)}
    windows = {name: make_windows(lv[lo:hi], gl[lo:hi], fwd[lo:hi])
               for name, (lo, hi) in segments.items()}
    t_fwd = windows["train"][2]
    t_fwd = t_fwd[np.isfinite(t_fwd)]
    threshold = float(np.quantile(np.abs(t_fwd), 0.7))

    data = {}
    for name, (wl, wg, wf, _) in windows.items():
        defined = np.isfinite(wf)
        data[name] = {
            "L": torch.tensor(wl[defined]), "G": torch.tensor(wg[defined]),
            "y": torch.tensor(labels_from(wf[defined], threshold)),
        }
    tr, te = data["train"], data["test"]

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

    normalize(tr); normalize(te)

    def flat_last(d):
        return torch.cat([d["L"][:, -1].reshape(len(d["L"]), -1),
                          d["G"][:, -1]], dim=1).numpy()

    logistic = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=SEED)
    logistic.fit(flat_last(tr), tr["y"].numpy())
    log_pred = logistic.predict(flat_last(te))
    return te["y"].numpy(), log_pred


def macro_f1(y, p):
    """Vectorized macro-F1 from integer true/pred labels (3 classes)."""
    key = y * 3 + p
    cm = np.bincount(key, minlength=9).reshape(3, 3).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


def block_index(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def main():
    rng = np.random.default_rng(SEED)
    y, log_pred = prepare()
    n = len(y)

    saved = pd.read_csv(OUTPUT / "test_predictions.csv")
    assert len(saved) == n, "model prediction count mismatch"
    assert (saved["label"].to_numpy() == y).all(), "label order mismatch"
    model_pred = saved[["p_down", "p_neutral", "p_up"]].to_numpy().argmax(1)

    point_model = macro_f1(y, model_pred)
    point_log = macro_f1(y, log_pred)
    print(f"point estimates: model={point_model:.4f} logistic={point_log:.4f} "
          f"gap={point_model - point_log:+.4f}  n={n}", flush=True)

    results = {}
    for scheme in ("iid", "block"):
        gaps = np.empty(B)
        for b in range(B):
            if scheme == "iid":
                idx = rng.integers(0, n, n)
            else:
                idx = block_index(n, BLOCK, rng)
            gaps[b] = macro_f1(y[idx], model_pred[idx]) - \
                macro_f1(y[idx], log_pred[idx])
        lo, hi = np.quantile(gaps, [0.025, 0.975])
        results[scheme] = {
            "gap_mean": float(gaps.mean()),
            "ci95": [float(lo), float(hi)],
            "p_gap_gt_0": float((gaps > 0).mean()),
        }
        print(f"[{scheme:5s}] gap mean={gaps.mean():+.4f} "
              f"95%CI=({lo:+.4f},{hi:+.4f}) P(gap>0)={(gaps > 0).mean():.3f}",
              flush=True)

    out = {
        "point": {"model_macro_f1": point_model,
                  "logistic_macro_f1": point_log,
                  "gap": point_model - point_log},
        "resamples": B, "block_len": BLOCK, "test_n": n,
        "iid": results["iid"], "block": results["block"],
        "note": "conditions on the single trained model; block bootstrap is "
                "the conservative read under serial correlation.",
    }
    snapshot_runner(OUTPUT, __file__)
    (OUTPUT / "bootstrap_significance.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("saved bootstrap_significance.json")


if __name__ == "__main__":
    main()
