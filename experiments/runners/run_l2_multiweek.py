"""Multi-week expanding-window experiment on dispersed L2 weeks (E019).

Four sampled weeks (each 7 days, ~120-frame windows built WITHIN the week so
they never span the multi-week gaps):
    2026-05-07, 2026-06-04, 2026-07-02, 2026-08-27

For each target week k (expanding window, strictly causal):
    train = ALL windows from earlier weeks + first 56% of week k
    dev   = next 14% of week k (contiguous, checkpoint selection)
    test  = last 30% of week k
so every week contributes an OOS test block; pooled OOS ~ 4 x ~290 windows.

Per fold, the label threshold (70% q of |train forward|) and robust
normalization are fitted on THAT fold's train only. A Logistic baseline is
refit identically per fold. Pooling folds gives test blocks spanning several
market regimes.

Paired bootstrap resamples WITHIN each week (moving blocks to preserve serial
dependence) and recomputes pooled Macro-F1 for both models on the same
resample. Decision rule: lower bound of the 95% CI of (level - logistic).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from experiments.runners.bootstrap_l2_significance import macro_f1
from experiments.runners.run_l2_level_experiment import (
    GLOBAL_COLS, LEVEL_CHANNELS, LEVELS, WINSOR_Q, build_arrays,
    compress_tails, drop_warmup_frames, eval_probs, labels_from,
    make_windows, train_direction_model,
)
from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import LevelStructuredLOB

B = 20_000
BLOCK = 10
WEEKS = [
    ("2026-04-23", Path("artifacts/historical/l2_weeks_features/2026-04-23/l2_30s_features.parquet")),
    ("2026-05-07", Path("artifacts/historical/l2_weeks_features/2026-05-07/l2_30s_features.parquet")),
    ("2026-05-21", Path("artifacts/historical/l2_weeks_features/2026-05-21/l2_30s_features.parquet")),
    ("2026-06-04", Path("artifacts/historical/l2_weeks_features/2026-06-04/l2_30s_features.parquet")),
    ("2026-06-18", Path("artifacts/historical/l2_weeks_features/2026-06-18/l2_30s_features.parquet")),
    ("2026-07-02", Path("artifacts/historical/l2_weeks_features/2026-07-02/l2_30s_features.parquet")),
    ("2026-07-16", Path("artifacts/historical/l2_weeks_features/2026-07-16/l2_30s_features.parquet")),
    ("2026-07-30", Path("artifacts/historical/l2_weeks_features/2026-07-30/l2_30s_features.parquet")),
    ("2026-08-13", Path("artifacts/historical/l2_weeks_features/2026-08-13/l2_30s_features.parquet")),
    ("2026-08-27", Path("artifacts/historical/l2_btc_week_features/l2_30s_features.parquet")),
    ("2026-09-10", Path("artifacts/historical/l2_weeks_features/2026-09-10/l2_30s_features.parquet")),
    ("2026-09-24", Path("artifacts/historical/l2_weeks_features/2026-09-24/l2_30s_features.parquet")),
]
OUTPUT = Path("artifacts/historical/l2_multiweek_expanding")


def week_arrays(path):
    frame = pd.read_parquet(path).sort_values("bin").reset_index(drop=True)
    frame, dropped = drop_warmup_frames(frame)
    lv, gl, fwd = build_arrays(frame)
    lv, gl = compress_tails(lv, gl)
    L, G, f, ends = make_windows(lv, gl, fwd)
    defined = np.isfinite(f)
    return {"L": torch.tensor(L[defined]), "G": torch.tensor(G[defined]),
            "f": f[defined], "dropped": dropped}


def fit_robust_scaler():
    def per(t, dim_last):
        flat = t.reshape(-1, dim_last)
        # torch.quantile sorts and needs large workspaces; on the biggest
        # expanding folds the flattened tensor is too large. Windows overlap
        # heavily, so uniformly subsampling rows gives essentially identical
        # quantiles while bounding memory.
        rows = flat.shape[0]
        sample = flat
        if rows > 200_000:
            step = int(np.ceil(rows / 200_000))
            sample = flat[::step]
        lo = torch.quantile(sample, WINSOR_Q[0], dim=0)
        hi = torch.quantile(sample, WINSOR_Q[1], dim=0)
        clipped = torch.clamp(flat, lo, hi)
        return lo, hi, clipped.mean(0), clipped.std(0).clamp_min(1e-6)
    return per


def scale(t, scaler):
    lo, hi, mean, std = scaler
    return (torch.clamp(t, lo, hi) - mean) / std


def flat_last(L, G):
    return torch.cat([L[:, -1].reshape(len(L), -1), G[:, -1]], dim=1).numpy()


def build_fold(weeks, k, train_stride=1):
    cur = weeks[k]
    n = len(cur["L"])
    i70, i56 = int(n * 0.70), int(n * 0.56)
    tr_L = [w["L"] for w in weeks[:k]] + [cur["L"][:i56]]
    tr_G = [w["G"] for w in weeks[:k]] + [cur["G"][:i56]]
    tr_f = [w["f"] for w in weeks[:k]] + [cur["f"][:i56]]
    trL, trG = torch.cat(tr_L), torch.cat(tr_G)
    trf = np.concatenate(tr_f)
    # Optional TRAIN-only density reduction: windows overlap ~92%, so stride on
    # the training set removes little independent information. Dev/test frames
    # below are never touched, and the Logistic baseline uses the same sample.
    if train_stride > 1:
        trL, trG, trf = trL[::train_stride], trG[::train_stride], trf[::train_stride]
    dvL, dvG = cur["L"][i56:i70], cur["G"][i56:i70]
    teL, teG = cur["L"][i70:], cur["G"][i70:]
    tef = cur["f"][i70:]

    threshold = float(np.quantile(np.abs(trf), 0.7))
    trY = torch.tensor(labels_from(trf, threshold))
    dvY = torch.tensor(labels_from(cur["f"][i56:i70], threshold))
    teY = torch.tensor(labels_from(tef, threshold))

    scaler_fn = fit_robust_scaler()
    lsc = scaler_fn(trL, len(LEVEL_CHANNELS))
    gsc = scaler_fn(trG, len(GLOBAL_COLS))
    tr = {"L": scale(trL, lsc), "G": scale(trG, gsc), "y": trY}
    dv = {"L": scale(dvL, lsc), "G": scale(dvG, gsc), "y": dvY}
    te = {"L": scale(teL, lsc), "G": scale(teG, gsc), "y": teY, "f": tef}

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=20261002)
    clf.fit(flat_last(tr["L"], tr["G"]), trY.numpy())
    log_pred = clf.predict(flat_last(te["L"], te["G"]))
    return tr, dv, te, threshold, log_pred


def week_block_indices(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, nb)
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]) % n
    return idx.reshape(-1)[:n]


def paired_bootstrap(grouped):
    rng = np.random.default_rng(20261002)
    gaps = np.empty(B)
    for b in range(B):
        my, mp, lp = [], [], []
        for g in grouped:
            idx = week_block_indices(len(g["y"]), rng)
            my.append(g["y"][idx]); mp.append(g["model"][idx])
            lp.append(g["log"][idx])
        yy = np.concatenate(my)
        gaps[b] = macro_f1(yy, np.concatenate(mp)) - macro_f1(yy, np.concatenate(lp))
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {"gap_mean": float(gaps.mean()), "ci95": [float(lo), float(hi)],
            "p_gap_gt_0": float((gaps > 0).mean())}


def main():
    torch.manual_seed(20261002)
    np.random.seed(20261002)
    weeks = []
    for name, path in WEEKS:
        if not path.exists():
            raise SystemExit(f"missing week features: {path}")
        w = week_arrays(path); w["name"] = name
        weeks.append(w)
        print(f"week {name}: windows={len(w['L'])} dropped={w['dropped']}",
              flush=True)

    grouped, fold_info = [], []
    for k in range(len(weeks)):
        tr, dv, te, threshold, log_pred = build_fold(weeks, k)
        model = LevelStructuredLOB(len(LEVEL_CHANNELS), len(GLOBAL_COLS),
                                   levels=LEVELS)
        best_epoch = train_direction_model(model, tr, dv, lr=3e-4)
        prob, q = eval_probs(model, te)
        model_pred = prob.argmax(1)
        y = te["y"].numpy()
        mf1, lf1 = macro_f1(y, model_pred), macro_f1(y, log_pred)
        print(f"[fold {weeks[k]['name']}] n={len(y)} model={mf1:.4f} "
              f"logistic={lf1:.4f} gap={mf1-lf1:+.4f} best_ep={best_epoch}",
              flush=True)
        grouped.append({"y": y, "model": model_pred, "log": log_pred})
        fold_info.append({"week": weeks[k]["name"], "test_n": len(y),
                          "model_macro_f1": mf1, "logistic_macro_f1": lf1,
                          "gap": mf1 - lf1, "best_epoch": best_epoch,
                          "threshold": threshold,
                          "counts": np.bincount(model_pred, minlength=3).tolist(),
                          "auc_up": _auc(y, prob), "ic": _ic(te["f"], prob)})

    all_y = np.concatenate([g["y"] for g in grouped])
    all_m = np.concatenate([g["model"] for g in grouped])
    all_l = np.concatenate([g["log"] for g in grouped])
    point = {"model_macro_f1": macro_f1(all_y, all_m),
             "logistic_macro_f1": macro_f1(all_y, all_l)}
    point["gap"] = point["model_macro_f1"] - point["logistic_macro_f1"]
    print(f"[POOLED] n={len(all_y)} model={point['model_macro_f1']:.4f} "
          f"logistic={point['logistic_macro_f1']:.4f} gap={point['gap']:+.4f}",
          flush=True)

    boot = paired_bootstrap(grouped)
    print(f"[BOOTSTRAP by-week blocks] mean={boot['gap_mean']:+.4f} "
          f"95%CI=({boot['ci95'][0]:+.4f},{boot['ci95'][1]:+.4f}) "
          f"P>0={boot['p_gap_gt_0']:.3f}", flush=True)

    result = {"point": point, "bootstrap": boot, "folds": fold_info,
              "resamples": B, "block_len": BLOCK,
              "decision_ci_lb_gt_0": bool(boot["ci95"][0] > 0)}
    snapshot_runner(OUTPUT, __file__)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    pd.DataFrame({
        "week": np.concatenate([[f["week"]] * f["test_n"] for f in fold_info]),
        "label": all_y, "model_pred": all_m, "logistic_pred": all_l,
    }).to_csv(OUTPUT / "pooled_oos_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


def _auc(y, prob):
    from sklearn.metrics import roc_auc_score
    d = prob[:, 2] - prob[:, 0]
    return float(roc_auc_score(y == 2, d))


def _ic(f, prob):
    d = prob[:, 2] - prob[:, 0]
    return float(np.corrcoef(d, f)[0, 1])


if __name__ == "__main__":
    main()
