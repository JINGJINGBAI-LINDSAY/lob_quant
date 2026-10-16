"""Calibrate the E014 slow gate and direction thresholds on dev only.

The direction model and slow confidence filter reproduce the original E014
training sequence. Test is evaluated once after all threshold choices are
frozen on dev.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from experiments.runners.run_two_axis_experiment import (
    HORIZON,
    OUTPUT as ORIGINAL_OUTPUT,
    SEED,
    make_windows,
    prepare_symbol,
    warmup_cosine,
)
from lob_mvp.cli import _purge_train_horizon, _threshold
from lob_mvp.core import MVPConfig
from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import ConfidenceFilter, TwoAxisLOB

OUTPUT = Path(
    "artifacts/historical/binance_extended/"
    "two_axis_threshold_calibration_20261012"
)
COSTS = {"2bp": 0.0002, "7bp": 0.0007}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps"),
                        default="cpu")
    parser.add_argument("--direction-epochs", type=int, default=24)
    parser.add_argument("--confidence-epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(name)


def build_splits(cfg, frame):
    splits = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        group = prepare_symbol(frame, symbol)
        cut = int(len(group) * cfg.train_fraction)
        train_g, valid_g = group.iloc[:cut], group.iloc[cut:]
        sym_threshold = _threshold(
            _purge_train_horizon(train_g, HORIZON)
        )

        def assemble(g):
            events, slow, ends, fwd = make_windows(
                g, cfg.feature_columns
            )
            fwd_ends = fwd[ends]
            defined = np.isfinite(fwd_ends)
            labels = np.ones(len(ends), dtype=np.int64)
            labels[fwd_ends < -sym_threshold] = 0
            labels[fwd_ends > sym_threshold] = 2
            return (
                events[defined], slow[defined],
                torch.tensor(labels[defined]), fwd_ends[defined],
            )

        etr, str_, ytr, ftr = assemble(train_g)
        eva, sva, yva, fva = assemble(valid_g)
        half = len(eva) // 2
        splits[symbol] = {
            "train": (etr, str_, ytr, ftr),
            "dev": (eva[:half], sva[:half], yva[:half], fva[:half]),
            "test": (eva[half:], sva[half:], yva[half:], fva[half:]),
        }
    return splits


def gather(splits, part):
    e, s, y, f = [], [], [], []
    for symbol in splits:
        ev, sl, yy, ff = splits[symbol][part]
        e.append(ev)
        s.append(sl)
        y.append(yy)
        f.append(ff)
    return torch.cat(e), torch.cat(s), torch.cat(y), np.concatenate(f)


def fit_norm(tensor):
    flat = tensor.reshape(-1, tensor.shape[-1])
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def infer_direction(model, events, device, batch_size=512):
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            logits, _, _ = model(events[start:start + batch_size].to(device))
            parts.append(logits.cpu())
    return torch.cat(parts).softmax(-1).numpy()


def train_direction(events_train, labels_train, events_dev, labels_dev,
                    device, epochs):
    model = TwoAxisLOB(events_train.shape[-1]).to(device)
    counts = torch.bincount(labels_train, minlength=3).float()
    weights = (counts.sum() / (3 * counts.clamp_min(1))).pow(0.5).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=0.03
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, weight_decay=0.02
    )
    batch_size = 256
    total = epochs * math.ceil(len(events_train) / batch_size)
    scheduler = LambdaLR(
        optimizer, lambda step: warmup_cosine(step, total)
    )
    best = {"f1": -1.0, "epoch": 0, "state": None}
    since = 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(events_train))
        for start in range(0, len(events_train), batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits, _, _ = model(events_train[idx].to(device))
            loss = loss_fn(logits, labels_train[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        dev_prob = infer_direction(model, events_dev, device)
        dev_f1 = f1_score(
            labels_dev.numpy(), dev_prob.argmax(1),
            average="macro", zero_division=0,
        )
        if dev_f1 > best["f1"] + 1e-5:
            best = {
                "f1": float(dev_f1),
                "epoch": epoch + 1,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            since = 0
        else:
            since += 1
        print(f"direction epoch={epoch + 1} dev_f1={dev_f1:.4f}",
              flush=True)
        if since >= 8:
            break

    model.load_state_dict(best["state"])
    model.to(device)
    return model, best


def train_confidence(slow_train, returns_train, slow_dev, slow_test,
                     device, epochs):
    model = ConfidenceFilter(slow_train.shape[-1]).to(device)
    target = (
        np.abs(returns_train) > np.quantile(np.abs(returns_train), 0.6)
    ).astype(np.float32)
    target = torch.from_numpy(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        permutation = torch.randperm(len(slow_train))
        for start in range(0, len(slow_train), 256):
            idx = permutation[start:start + 256]
            optimizer.zero_grad()
            logits = model.net(slow_train[idx].to(device)).squeeze(-1)
            loss = loss_fn(logits, target[idx].to(device))
            loss.backward()
            optimizer.step()

    def infer(values):
        parts = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(values), 1024):
                parts.append(model(values[start:start + 1024].to(device)).cpu())
        return torch.cat(parts).numpy()

    return model, infer(slow_dev), infer(slow_test)


def safe_auc(y, score):
    if len(np.unique(y == 2)) < 2:
        return None
    return float(roc_auc_score(y == 2, score))


def safe_ic(score, returns):
    if len(score) < 2 or np.std(score) == 0 or np.std(returns) == 0:
        return None
    return float(np.corrcoef(score, returns)[0, 1])


def trading_metrics(pred, returns):
    position = np.where(pred == 2, 1, np.where(pred == 0, -1, 0))
    active = position != 0
    gross = position[active] * returns[active]
    result = {
        "trades": int(active.sum()),
        "gross_return_mean": float(gross.mean()) if len(gross) else None,
        "gross_return_sum": float(gross.sum()),
        "gross_mean_bp": float(gross.mean() * 1e4) if len(gross) else None,
        "win_rate": float((gross > 0).mean()) if len(gross) else None,
    }
    for name, cost in COSTS.items():
        net = gross - cost
        result[f"net_{name}_mean"] = (
            float(net.mean()) if len(net) else None
        )
        result[f"net_{name}_sum"] = float(net.sum())
        result[f"net_{name}_mean_bp"] = (
            float(net.mean() * 1e4) if len(net) else None
        )
    return result


def evaluate(y, prob, returns, confidence, gate, up_threshold,
             down_threshold):
    direction_score = prob[:, 2] - prob[:, 0]
    log_prob = np.log(np.clip(prob, 1e-12, 1.0))
    up_margin = log_prob[:, 2] - log_prob[:, 1]
    down_margin = log_prob[:, 0] - log_prob[:, 1]
    keep = confidence >= gate
    pred = np.ones(len(y), dtype=np.int64)
    pred[
        keep & (up_margin > up_threshold) & (prob[:, 2] >= prob[:, 0])
    ] = 2
    pred[
        keep & (down_margin > down_threshold) & (prob[:, 0] > prob[:, 2])
    ] = 0
    retained = keep.sum()
    return {
        "parameters": {
            "gate": float(gate),
            "up_threshold": float(up_threshold),
            "down_threshold": float(down_threshold),
        },
        "coverage": float(keep.mean()),
        "kept": int(retained),
        "all_timestamp_macro_f1": float(f1_score(
            y, pred, average="macro", zero_division=0
        )),
        "retained_macro_f1": float(f1_score(
            y[keep], pred[keep], average="macro", zero_division=0
        )) if retained else None,
        "retained_auc": safe_auc(y[keep], direction_score[keep]),
        "retained_ic": safe_ic(direction_score[keep], returns[keep]),
        "prediction_counts": np.bincount(pred, minlength=3).tolist(),
        "retained_prediction_counts": np.bincount(
            pred[keep], minlength=3
        ).tolist(),
        "trading": trading_metrics(pred, returns),
        "_pred": pred,
    }


def evaluate_argmax_gate(y, prob, returns, confidence, gate):
    keep = confidence >= gate
    pred = np.ones(len(y), dtype=np.int64)
    pred[keep] = prob[keep].argmax(1)
    direction_score = prob[:, 2] - prob[:, 0]
    result = {
        "parameters": {"gate": float(gate), "rule": "softmax_argmax"},
        "coverage": float(keep.mean()),
        "kept": int(keep.sum()),
        "all_timestamp_macro_f1": float(f1_score(
            y, pred, average="macro", zero_division=0
        )),
        "retained_macro_f1": float(f1_score(
            y[keep], pred[keep], average="macro", zero_division=0
        )),
        "retained_auc": safe_auc(y[keep], direction_score[keep]),
        "retained_ic": safe_ic(direction_score[keep], returns[keep]),
        "prediction_counts": np.bincount(pred, minlength=3).tolist(),
        "retained_prediction_counts": np.bincount(
            pred[keep], minlength=3
        ).tolist(),
        "trading": trading_metrics(pred, returns),
        "_pred": pred,
    }
    return result


def threshold_candidates(margin):
    quantiles = np.quantile(margin, np.linspace(0.02, 0.98, 25))
    return np.unique(np.r_[0.0, quantiles])


def search_thresholds(y, prob, returns, confidence, gate_quantiles,
                      objective):
    log_prob = np.log(np.clip(prob, 1e-12, 1.0))
    up_candidates = threshold_candidates(log_prob[:, 2] - log_prob[:, 1])
    down_candidates = threshold_candidates(log_prob[:, 0] - log_prob[:, 1])
    min_kept = max(200, int(math.ceil(0.20 * len(y))))
    best = None
    candidates_evaluated = 0
    for gate_q in gate_quantiles:
        gate = float(np.quantile(confidence, gate_q))
        for up_threshold in up_candidates:
            for down_threshold in down_candidates:
                result = evaluate(
                    y, prob, returns, confidence, gate,
                    up_threshold, down_threshold,
                )
                counts = result["retained_prediction_counts"]
                trades = result["trading"]["trades"]
                if result["kept"] < min_kept:
                    continue
                if counts[0] < 30 or counts[2] < 30 or trades < 100:
                    continue
                candidates_evaluated += 1
                if objective == "classification":
                    rank = (
                        result["retained_macro_f1"],
                        result["all_timestamp_macro_f1"],
                        result["coverage"],
                    )
                else:
                    rank = (
                        result["trading"]["net_7bp_sum"],
                        result["trading"]["net_7bp_mean"],
                        result["retained_macro_f1"],
                    )
                if best is None or rank > best[0]:
                    result["parameters"]["gate_quantile"] = float(gate_q)
                    best = rank, result
    if best is None:
        raise RuntimeError("No threshold candidate satisfies constraints")
    best[1]["search"] = {
        "objective": objective,
        "decision_rule": "class_log_probability_margin_vs_neutral",
        "candidates_evaluated": candidates_evaluated,
        "minimum_dev_coverage": 0.20,
        "minimum_each_direction_predictions": 30,
        "minimum_total_trades": 100,
    }
    return best[1]


def without_private(result):
    return {key: value for key, value in result.items()
            if not key.startswith("_")}


def raw_metrics(y, prob, returns):
    pred = prob.argmax(1)
    direction_score = prob[:, 2] - prob[:, 0]
    return {
        "macro_f1": float(f1_score(
            y, pred, average="macro", zero_division=0
        )),
        "auc": safe_auc(y, direction_score),
        "ic": safe_ic(direction_score, returns),
        "prediction_counts": np.bincount(pred, minlength=3).tolist(),
        "trading": trading_metrics(pred, returns),
        "_pred": pred,
    }


def main():
    args = parse_args()
    if args.smoke:
        args.direction_epochs = min(args.direction_epochs, 1)
        args.confidence_epochs = min(args.confidence_epochs, 2)
        args.output = args.output.with_name(args.output.name + "_smoke")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    cfg = MVPConfig.from_json("config/binance_202609.json")
    frame = pd.read_parquet(
        "artifacts/historical/binance_extended/event_features.parquet"
    )
    splits = build_splits(cfg, frame)
    event_train, slow_train, y_train, fwd_train = gather(splits, "train")
    event_dev, slow_dev, y_dev, fwd_dev = gather(splits, "dev")
    event_test, slow_test, y_test, fwd_test = gather(splits, "test")
    print(
        f"sizes train={len(event_train)} dev={len(event_dev)} "
        f"test={len(event_test)}",
        flush=True,
    )

    event_mean, event_std = fit_norm(event_train)
    slow_mean, slow_std = fit_norm(slow_train)
    event_train, event_dev, event_test = (
        (event_train - event_mean) / event_std,
        (event_dev - event_mean) / event_std,
        (event_test - event_mean) / event_std,
    )
    slow_train, slow_dev, slow_test = (
        (slow_train - slow_mean) / slow_std,
        (slow_dev - slow_mean) / slow_std,
        (slow_test - slow_mean) / slow_std,
    )

    direction_model, best = train_direction(
        event_train, y_train, event_dev, y_dev, device,
        args.direction_epochs,
    )
    dev_prob = infer_direction(direction_model, event_dev, device)
    test_prob = infer_direction(direction_model, event_test, device)
    confidence_model, dev_conf, test_conf = train_confidence(
        slow_train, fwd_train, slow_dev, slow_test,
        device, args.confidence_epochs,
    )

    y_dev_np = y_dev.numpy()
    y_test_np = y_test.numpy()
    original_gate = float(np.quantile(dev_conf, 0.5))
    dev_original = evaluate_argmax_gate(
        y_dev_np, dev_prob, fwd_dev, dev_conf, original_gate
    )
    test_original = evaluate_argmax_gate(
        y_test_np, test_prob, fwd_test, test_conf, original_gate
    )

    fixed_gate = search_thresholds(
        y_dev_np, dev_prob, fwd_dev, dev_conf,
        gate_quantiles=[0.5], objective="classification",
    )
    joint_classification = search_thresholds(
        y_dev_np, dev_prob, fwd_dev, dev_conf,
        gate_quantiles=np.linspace(0.0, 0.8, 17),
        objective="classification",
    )
    joint_trading = search_thresholds(
        y_dev_np, dev_prob, fwd_dev, dev_conf,
        gate_quantiles=np.linspace(0.0, 0.8, 17),
        objective="trading",
    )

    selected = {
        "fixed_gate_direction_calibration": fixed_gate,
        "joint_classification_calibration": joint_classification,
        "joint_trading_calibration_exploratory": joint_trading,
    }
    test_selected = {}
    for name, dev_result in selected.items():
        params = dev_result["parameters"]
        test_selected[name] = evaluate(
            y_test_np, test_prob, fwd_test, test_conf,
            params["gate"], params["up_threshold"],
            params["down_threshold"],
        )

    raw_dev = raw_metrics(y_dev_np, dev_prob, fwd_dev)
    raw_test = raw_metrics(y_test_np, test_prob, fwd_test)
    result = {
        "protocol": {
            "threshold_selection": "dev_only",
            "test_evaluations_after_selection": 1,
            "decision_rule": (
                "class log-probability margin versus neutral; "
                "zero thresholds exactly reproduce argmax"
            ),
            "ranking_score": "p_up_minus_p_down",
            "outside_gate_action": "neutral_no_trade",
            "cost_per_directional_prediction": COSTS,
            "return_definition": (
                "mean next 10 frames divided by current mid minus one"
            ),
            "device": str(device),
            "smoke": args.smoke,
        },
        "samples": {
            "train": len(event_train),
            "dev": len(event_dev),
            "test": len(event_test),
        },
        "best_direction_epoch": best["epoch"],
        "raw_argmax": {
            "dev": without_private(raw_dev),
            "test": without_private(raw_test),
        },
        "original_median_gate_argmax": {
            "dev": without_private(dev_original),
            "test": without_private(test_original),
        },
        "dev_selected": {
            name: without_private(value) for name, value in selected.items()
        },
        "test_once": {
            name: without_private(value)
            for name, value in test_selected.items()
        },
        "reference_e014_metrics_path": str(
            ORIGINAL_OUTPUT / "metrics.json"
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_runner(
        args.output, __file__,
        extra_files=[
            "config/binance_202609.json",
            "experiments/runners/run_two_axis_experiment.py",
        ],
    )
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(direction_model.state_dict(),
               args.output / "two_axis_model.pt")
    torch.save(confidence_model.state_dict(),
               args.output / "confidence_filter.pt")

    dev_table = pd.DataFrame({
        "label": y_dev_np,
        "return_h10": fwd_dev,
        "p_down": dev_prob[:, 0],
        "p_neutral": dev_prob[:, 1],
        "p_up": dev_prob[:, 2],
        "confidence": dev_conf,
        "raw_pred": raw_dev["_pred"],
        "original_pred": dev_original["_pred"],
    })
    test_table = pd.DataFrame({
        "label": y_test_np,
        "return_h10": fwd_test,
        "p_down": test_prob[:, 0],
        "p_neutral": test_prob[:, 1],
        "p_up": test_prob[:, 2],
        "confidence": test_conf,
        "raw_pred": raw_test["_pred"],
        "original_pred": test_original["_pred"],
    })
    for name, dev_result in selected.items():
        dev_table[name] = dev_result["_pred"]
        test_table[name] = test_selected[name]["_pred"]
    dev_table.to_csv(args.output / "dev_predictions.csv", index=False)
    test_table.to_csv(args.output / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
