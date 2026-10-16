"""Ablate last-frame global-token parameterizations for LevelStructuredLOB.

The first three variants keep the L2 backbone at d_model=64 and differ only
in how the last global frame is projected:

  split2x64       4 -> 64 -> 128 -> two 64-dimensional tokens (current model)
  single64        4 -> 64 -> 64  -> one 64-dimensional token
  single64_via128 4 -> 128 -> 64 -> one 64-dimensional token

The single128 variant uses one 128-dimensional global token and therefore also
widens the entire L2 backbone to d_model=128. Its result is a capacity test,
not a clean test of global-token count.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from experiments.runners.bootstrap_l2_significance import macro_f1
from experiments.runners.run_l2_level_experiment import (
    CLASS_WEIGHT_POWER,
    GLOBAL_COLS,
    LEVEL_CHANNELS,
    LEVELS,
    warmup_cosine,
)
from experiments.runners.run_l2_multiweek import (
    B,
    BLOCK,
    WEEKS,
    build_fold,
    week_arrays,
    week_block_indices,
)
from lob_mvp.two_axis_model import LevelStructuredLOB


OUTPUT = Path("artifacts/historical/l2_global_token_ablation")
SEED = 20261011
VARIANTS = {
    "split2x64": {
        "d_model": 64,
        "d_ff": 128,
        "global_tokens": 2,
        "global_hidden": 64,
        "interpretation": "current: last G -> two 64d tokens",
        "clean_64d_comparison": True,
    },
    "single64": {
        "d_model": 64,
        "d_ff": 128,
        "global_tokens": 1,
        "global_hidden": 64,
        "interpretation": "last G -> one 64d token",
        "clean_64d_comparison": True,
    },
    "single64_via128": {
        "d_model": 64,
        "d_ff": 128,
        "global_tokens": 1,
        "global_hidden": 128,
        "interpretation": "last G -> 128d hidden -> one 64d token",
        "clean_64d_comparison": True,
    },
    "single128": {
        "d_model": 128,
        "d_ff": 256,
        "global_tokens": 1,
        "global_hidden": 128,
        "interpretation": "one 128d token; entire L2 backbone widened to 128d",
        "clean_64d_comparison": False,
    },
}


def make_model(name: str) -> LevelStructuredLOB:
    cfg = VARIANTS[name]
    model = LevelStructuredLOB(
        len(LEVEL_CHANNELS),
        len(GLOBAL_COLS),
        levels=LEVELS,
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        global_tokens=cfg["global_tokens"],
    )
    out_dim = cfg["global_tokens"] * cfg["d_model"]
    model.global_proj = nn.Sequential(
        nn.Linear(len(GLOBAL_COLS), cfg["global_hidden"]),
        nn.GELU(),
        nn.Linear(cfg["global_hidden"], out_dim),
    )
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(model, data, device, batch_size):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(data["L"]), batch_size):
            logits, _, _ = model(
                data["L"][i:i + batch_size].to(device),
                data["G"][i:i + batch_size].to(device),
            )
            probs.append(logits.softmax(-1).cpu())
    return torch.cat(probs).numpy()


def train_model(model, tr, dv, device, lr, epochs, patience, batch_size):
    model.to(device)
    counts = torch.bincount(tr["y"], minlength=3).float()
    class_weights = (
        counts.sum() / (3 * counts.clamp_min(1))
    ).pow(CLASS_WEIGHT_POWER).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    total = epochs * math.ceil(len(tr["L"]) / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: warmup_cosine(step, total)
    )
    best = {"f1": -1.0, "state": None, "epoch": 0}
    since = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr["L"]))
        for i in range(0, len(tr["L"]), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits, _, _ = model(
                tr["L"][idx].to(device),
                tr["G"][idx].to(device),
            )
            loss = loss_fn(logits, tr["y"][idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        dev_prob = predict(model, dv, device, batch_size)
        dev_f1 = f1_score(
            dv["y"].numpy(),
            dev_prob.argmax(1),
            average="macro",
            zero_division=0,
        )
        if dev_f1 > best["f1"] + 1e-5:
            best = {
                "f1": dev_f1,
                "epoch": epoch + 1,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            since = 0
        else:
            since += 1
        if since >= patience:
            break
    model.load_state_dict(best["state"])
    return best["epoch"]


def block_bootstrap_gap(grouped, left: str, right: str, resamples: int):
    rng = np.random.default_rng(SEED)
    gaps = np.empty(resamples)
    for b in range(resamples):
        ys, lp, rp = [], [], []
        for fold in grouped:
            idx = week_block_indices(len(fold["y"]), rng)
            ys.append(fold["y"][idx])
            lp.append(fold[left][idx])
            rp.append(fold[right][idx])
        y = np.concatenate(ys)
        gaps[b] = (
            macro_f1(y, np.concatenate(lp))
            - macro_f1(y, np.concatenate(rp))
        )
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {
        "left": left,
        "right": right,
        "gap_mean": float(gaps.mean()),
        "ci95": [float(lo), float(hi)],
        "p_gap_gt_0": float((gaps > 0).mean()),
    }


def parse_indices(value: str, count: int):
    if value == "all":
        return list(range(count))
    out = sorted({int(x) for x in value.split(",")})
    if not out or out[0] < 0 or out[-1] >= count:
        raise ValueError(f"folds must be in [0,{count - 1}]")
    return out


def write_partial(payload):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics_partial.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def cap_training_windows(tr, max_samples):
    if max_samples <= 0 or len(tr["L"]) <= max_samples:
        return tr
    idx = torch.linspace(
        0, len(tr["L"]) - 1, steps=max_samples
    ).round().long()
    return {key: value[idx] for key, value in tr.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated subset of: {','.join(VARIANTS)}",
    )
    parser.add_argument("--folds", default="all", help="'all' or indices, e.g. 0,5,11")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="uniform deterministic cap after fold-local scaling; 0 disables",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=B)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    fold_indices = parse_indices(args.folds, len(WEEKS))
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    weeks = []
    for name, path in WEEKS:
        if not path.exists():
            raise SystemExit(f"missing week features: {path}")
        week = week_arrays(path)
        week["name"] = name
        weeks.append(week)
        print(
            f"week {name}: windows={len(week['L'])} dropped={week['dropped']}",
            flush=True,
        )

    grouped = []
    fold_results = []
    prediction_rows = []
    param_counts = {}

    for k in fold_indices:
        tr, dv, te, threshold, logistic_pred = build_fold(
            weeks, k, train_stride=args.train_stride
        )
        train_n_full = len(tr["L"])
        tr = cap_training_windows(tr, args.max_train_samples)
        y = te["y"].numpy()
        fold_predictions = {"y": y, "logistic": logistic_pred}
        fold_metrics = {
            "fold": k,
            "week": weeks[k]["name"],
            "test_n": len(y),
            "train_n_full": train_n_full,
            "train_n_used": len(tr["L"]),
            "threshold": threshold,
            "logistic_macro_f1": macro_f1(y, logistic_pred),
            "variants": {},
        }

        for name in variants:
            fold_seed = SEED + k
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)
            model = make_model(name)
            param_counts[name] = parameter_count(model)
            best_epoch = train_model(
                model,
                tr,
                dv,
                device,
                lr=3e-4,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )
            prob = predict(model, te, device, args.batch_size)
            pred = prob.argmax(1)
            score = macro_f1(y, pred)
            fold_predictions[name] = pred
            fold_metrics["variants"][name] = {
                "macro_f1": score,
                "gap_vs_logistic": score - fold_metrics["logistic_macro_f1"],
                "best_epoch": best_epoch,
                "prediction_counts": np.bincount(pred, minlength=3).tolist(),
            }
            print(
                f"[{weeks[k]['name']}][{name}] n={len(y)} "
                f"f1={score:.4f} log={fold_metrics['logistic_macro_f1']:.4f} "
                f"gap={score-fold_metrics['logistic_macro_f1']:+.4f} "
                f"best_ep={best_epoch} params={param_counts[name]:,}",
                flush=True,
            )
            del model, prob
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

        grouped.append(fold_predictions)
        fold_results.append(fold_metrics)
        for i in range(len(y)):
            row = {
                "fold": k,
                "week": weeks[k]["name"],
                "index_in_fold": i,
                "label": int(y[i]),
                "logistic": int(logistic_pred[i]),
            }
            row.update({name: int(fold_predictions[name][i]) for name in variants})
            prediction_rows.append(row)
        write_partial(
            {
                "config": vars(args),
                "variants": {name: VARIANTS[name] for name in variants},
                "parameter_counts": param_counts,
                "folds": fold_results,
            }
        )

    pooled_y = np.concatenate([x["y"] for x in grouped])
    pooled = {
        "logistic": macro_f1(
            pooled_y, np.concatenate([x["logistic"] for x in grouped])
        )
    }
    for name in variants:
        pooled[name] = macro_f1(
            pooled_y, np.concatenate([x[name] for x in grouped])
        )

    comparisons = []
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            comparisons.append(
                block_bootstrap_gap(
                    grouped, left, right, args.bootstrap_resamples
                )
            )

    result = {
        "config": vars(args),
        "variants": {name: VARIANTS[name] for name in variants},
        "parameter_counts": param_counts,
        "pooled_macro_f1": pooled,
        "pairwise_block_bootstrap": comparisons,
        "folds": fold_results,
        "bootstrap_block_len": BLOCK,
        "seed": SEED,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT / "pooled_oos_predictions.csv", index=False
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
