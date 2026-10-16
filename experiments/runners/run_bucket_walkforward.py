"""One-year bucket walk-forward (E020): TLOB-lite vs Logistic.

Data: one year of Binance Vision percentage-depth buckets (30s), built by
`lob_mvp.binance_archive`. Unlike the earlier single-split E010, this uses a
rolling origin with non-overlapping TEST windows (independent OOS) and dense,
overlapping TRAIN windows (data augmentation; see --train-stride).

  * a window is TOKENS=120 frames = 60 minutes. TEST windows are sampled every
    120 frames (non-overlapping: consecutive windows share no frames and their
    labels, the mean over the next HORIZON=10 frames, do not overlap either),
    so OOS samples are independent. TRAIN/dev windows use --train-stride
    (default 30 frames = 15 min, dense/overlapping, acting as data
    augmentation for the neural model); dev windows are therefore correlated,
    which only makes best-epoch selection noisier;
  * rolling folds: TRAIN_MONTHS -> TEST_MONTHS, stepping STEP_MONTHS, with a
    HORIZON-frame purge/embargo between train and test to prevent label leak;
  * per fold the label threshold (70% q of |train forward|) and normalization
    are fitted on THAT fold's train; a Logistic baseline is refit identically.

Paired bootstrap resamples within each fold (same indices for both models)
and recomputes pooled Macro-F1. Decision: lower bound of the 95% CI of
(level - logistic) > 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from experiments.runners.bootstrap_l2_significance import macro_f1
from experiments.runners.run_two_axis_experiment import (
    HORIZON, TOKENS, prepare_symbol, score, warmup_cosine,
)
from lob_mvp.core import MVPConfig
from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import TwoAxisLOB

SEED = 20261008
SYMBOL = "BTCUSDT"
FEATURES = Path("artifacts/historical/binance_1y/event_features.parquet")
OUTPUT = Path("artifacts/historical/binance_1y_walkforward")
CONFIG = "config/binance_202609.json"
B = 20_000
TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS = 3, 3, 3


def month_segments(event_time):
    """Return (min_frame, max_frame, month_key) for each calendar month."""
    months = pd.Series(pd.to_datetime(
        np.asarray(event_time), unit="ms", utc=True)).dt.strftime("%Y-%m")
    dfm = pd.DataFrame({"m": months}).reset_index()
    grp = dfm.groupby("m", sort=True)["index"].agg(["min", "max"])
    return [(int(r["min"]), int(r["max"]), m) for m, r in grp.iterrows()]


def windows_over(group, event_cols, lo, hi, stride):
    """Non-overlapping (stride >= TOKENS) windows over frames [lo, hi)."""
    values = group[event_cols].to_numpy(np.float32)
    mid = group["mid_price"].to_numpy()
    conv = np.convolve(mid, np.ones(HORIZON) / HORIZON, mode="valid")
    forward = np.full(len(group), np.nan)
    forward[:len(conv)] = conv / mid[:len(conv)] - 1.0
    events, ends = [], []
    start = max(lo, TOKENS - 1)
    stop = min(hi, len(group) - HORIZON)
    for end in range(start, stop, stride):
        events.append(values[end + 1 - TOKENS:end + 1])
        ends.append(end)
    ends = np.asarray(ends)
    return (torch.tensor(np.asarray(events)), ends, forward[ends])


def build_folds(group):
    segs = month_segments(group["event_time_ms"])
    folds = []
    n = len(segs)
    for a in range(0, n - TRAIN_MONTHS - TEST_MONTHS + 1, STEP_MONTHS):
        tr_seg = segs[a:a + TRAIN_MONTHS]
        te_seg = segs[a + TRAIN_MONTHS:a + TRAIN_MONTHS + TEST_MONTHS]
        tr_lo = tr_seg[0][0]
        tr_hi = tr_seg[-1][1] + 1
        te_lo = te_seg[0][0]
        te_hi = te_seg[-1][1] + 1
        folds.append({
            "train_months": [s[2] for s in tr_seg],
            "test_months": [s[2] for s in te_seg],
            "tr_lo": tr_lo,
            # Purge: train must not end within HORIZON frames of the test start.
            "tr_hi": min(tr_hi, max(tr_lo, te_lo - HORIZON)),
            "te_lo": te_lo + HORIZON,  # embargo after train boundary
            "te_hi": te_hi,
        })
    return folds


def labels(forward, threshold):
    y = np.ones(len(forward), dtype=np.int64)
    y[forward < -threshold] = 0
    y[forward > threshold] = 2
    return torch.tensor(y)


def train_model(model, Etr, ytr, Edv, ydv, fdev, epochs=24, bs=256):
    import math
    from torch import nn
    counts = torch.bincount(ytr, minlength=3).float()
    cw = (counts.sum() / (3 * counts.clamp_min(1))).pow(0.5)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.02)
    total = epochs * math.ceil(len(Etr) / bs)
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda st: warmup_cosine(st, total))
    best = {"f1": -1.0, "state": None, "epoch": 0}
    since = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Etr))
        for i in range(0, len(Etr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, _, _ = model(Etr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            dev_logits, _, _ = model(Edv)
        summ, _ = score(ydv.numpy(), dev_logits, fdev)
        if summ["macro_f1"] > best["f1"] + 1e-5:
            best = {"f1": summ["macro_f1"], "epoch": ep + 1,
                    "state": {k: v.clone() for k, v in
                              model.state_dict().items()}}
            since = 0
        else:
            since += 1
        print(ep + 1, f"dev_f1={summ['macro_f1']:.4f}", flush=True)
        if since >= 10:
            break
    model.load_state_dict(best["state"])
    return best["epoch"]


def fit_norm(t):
    flat = t.reshape(-1, t.shape[-1])
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def paired_bootstrap(grouped):
    rng = np.random.default_rng(SEED)
    gaps = np.empty(B)
    for b in range(B):
        ys, ms, ls = [], [], []
        for g in grouped:
            n = len(g["y"])
            idx = rng.integers(0, n, n)
            ys.append(g["y"][idx]); ms.append(g["model"][idx])
            ls.append(g["log"][idx])
        y = np.concatenate(ys)
        gaps[b] = macro_f1(y, np.concatenate(ms)) - macro_f1(y, np.concatenate(ls))
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {"gap_mean": float(gaps.mean()), "ci95": [float(lo), float(hi)],
            "p_gap_gt_0": float((gaps > 0).mean())}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stride", type=int, default=30,
                        help="frame stride for train/dev windows (30=15min)")
    parser.add_argument("--eval-stride", type=int, default=TOKENS,
                        help="frame stride for test windows (120=non-overlap 1h)")
    parser.add_argument("--max-folds", type=int, default=100)
    parser.add_argument("--causal-mask", action="store_true",
                        help="causal mask on time-axis attention (variant B/C)")
    parser.add_argument("--dual-readout", action="store_true",
                        help="fuse attention-pool summary with last frame (C)")
    parser.add_argument("--tag", default="",
                        help="output dir suffix, e.g. B or C")
    args = parser.parse_args()
    out_dir = OUTPUT if not args.tag else OUTPUT.parent / f"{OUTPUT.name}_{args.tag}"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = MVPConfig.from_json(CONFIG)
    frame = pd.read_parquet(FEATURES)
    group = prepare_symbol(frame, SYMBOL)
    event_cols = list(cfg.feature_columns)
    folds = build_folds(group)[:args.max_folds]
    print(f"folds={len(folds)} symbol={SYMBOL} train_stride={args.train_stride} "
          f"causal_mask={args.causal_mask} dual_readout={args.dual_readout} "
          f"tag={args.tag or 'original'}", flush=True)
    for i, f in enumerate(folds):
        print(f"[fold {i}] train {f['train_months'][0]}.."
              f"{f['train_months'][-1]} test {f['test_months'][0]}.."
              f"{f['test_months'][-1]}", flush=True)

    grouped, fold_info = [], []
    for i, f in enumerate(folds):
        Etr, etr_idx, ftr = windows_over(
            group, event_cols, f["tr_lo"], f["tr_hi"], args.train_stride)
        Ete, ete_idx, fte = windows_over(
            group, event_cols, f["te_lo"], f["te_hi"], args.eval_stride)
        # Dev = last 20% of the train window (contiguous, never touches test).
        cut = int(len(Etr) * 0.8)
        threshold = float(np.quantile(np.abs(ftr[:cut]), 0.7))
        y_all = labels(ftr, threshold)
        yte = labels(fte, threshold)

        e_mean, e_std = fit_norm(Etr[:cut])
        norm = lambda t: (t - e_mean) / e_std
        Etr_n, Edv_n, Ete_n = norm(Etr[:cut]), norm(Etr[cut:]), norm(Ete)
        ytr, ydv = y_all[:cut], y_all[cut:]

        model = TwoAxisLOB(Etr_n.shape[-1], causal_mask=args.causal_mask,
                           dual_readout=args.dual_readout)
        best_ep = train_model(model, Etr_n, ytr, Edv_n, ydv, ftr[cut:])
        model.eval()
        with torch.no_grad():
            test_logits, _, _ = model(Ete_n)
        tsumm, tprob = score(yte.numpy(), test_logits, fte)
        model_pred = tprob.argmax(1)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Etr_n[:, -1, :].numpy(), ytr.numpy())
        log_pred = clf.predict(Ete_n[:, -1, :].numpy())
        lf1 = macro_f1(yte.numpy(), log_pred)
        print(f"[fold {i} TEST] n={len(yte)} model={tsumm['macro_f1']:.4f} "
              f"logistic={lf1:.4f} gap={tsumm['macro_f1']-lf1:+.4f} "
              f"best_ep={best_ep}", flush=True)

        win_month = pd.to_datetime(
            group["event_time_ms"].to_numpy()[ete_idx], unit="ms", utc=True
        ).strftime("%Y-%m").to_numpy()
        grouped.append({"y": yte.numpy(), "model": model_pred,
                        "log": log_pred, "month": win_month, "fold": i})
        fold_info.append({"test_months": f["test_months"], "test_n": len(yte),
                          "model_macro_f1": tsumm["macro_f1"],
                          "logistic_macro_f1": float(lf1),
                          "gap": tsumm["macro_f1"] - float(lf1),
                          "best_epoch": best_ep, "threshold": threshold,
                          "counts": tsumm["counts"], "auc": tsumm["auc"],
                          "ic": tsumm["ic"]})

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
    print(f"[BOOTSTRAP] mean={boot['gap_mean']:+.4f} "
          f"95%CI=({boot['ci95'][0]:+.4f},{boot['ci95'][1]:+.4f}) "
          f"P>0={boot['p_gap_gt_0']:.3f}", flush=True)

    result = {"point": point, "bootstrap": boot, "folds": fold_info,
              "resamples": B,
              "variant": {"causal_mask": args.causal_mask,
                          "dual_readout": args.dual_readout,
                          "tag": args.tag or "original"},
              "decision_ci_lb_gt_0": bool(boot["ci95"][0] > 0)}
    snapshot_runner(out_dir, __file__, extra_files=[CONFIG])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    all_fold = np.concatenate([g["fold"] * np.ones(len(g["y"]), int)
                               for g in grouped])
    all_month = np.concatenate([g["month"] for g in grouped])
    pd.DataFrame({
        "fold": all_fold, "test_month": all_month,
        "label": all_y, "model_pred": all_m, "logistic_pred": all_l,
    }).to_csv(out_dir / "pooled_oos_predictions.csv", index=False)
    print("DECISION ci_lb_gt_0 =", result["decision_ci_lb_gt_0"])


if __name__ == "__main__":
    main()
