from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .core import (
    MVPConfig, assign_labels, build_features, read_frame,
    write_dataset_artifacts, write_frame,
)
from .model import RegimeAwareLOB, make_windows, window_sample_metadata


def _debug_post(run_id, hypothesis_id, location, msg, data):
    """Best-effort diagnostics must never make a training run fail."""
    env_path = Path(".dbg/transformer-baseline-gap.env")
    if not os.environ.get("DEBUG_RUN_ID") or not env_path.exists():
        return
    try:
        server_url = next(
            line.split("=", 1)[1]
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("DEBUG_SERVER_URL=")
        )
        payload = {
            "sessionId": "transformer-baseline-gap",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        request = urllib.request.Request(
            server_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=0.5).read()
    except (OSError, StopIteration, ValueError):
        pass


def _threshold(frame, fraction=0.7):
    return float(frame["return_h50"].abs().quantile(fraction))


def _metrics(y, proba, returns):
    score = proba[:, 2] - proba[:, 0]
    auc = roc_auc_score((y == 2).astype(int), score) if len(np.unique(y == 2)) > 1 else float("nan")
    return {"macro_f1": float(f1_score(y, proba.argmax(1), average="macro", zero_division=0)),
            "auc_up_vs_rest": float(auc), "ic": float(np.corrcoef(score, returns)[0, 1])}


def _classification_summary(y, proba, returns):
    result = _metrics(y, proba, returns)
    predicted = proba.argmax(1)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1, 2], zero_division=0
    )
    result["per_class"] = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(("down", "neutral", "up"))
    }
    result["confusion_matrix"] = confusion_matrix(
        y, predicted, labels=[0, 1, 2]
    ).tolist()
    result["prediction_counts"] = np.bincount(predicted, minlength=3).tolist()
    return result


def _blend_probabilities(logistic_proba, transformer_proba, logistic_weight):
    if not 0.0 <= logistic_weight <= 1.0:
        raise ValueError("logistic_blend_weight must be between 0 and 1")
    return (
        logistic_weight * logistic_proba
        + (1.0 - logistic_weight) * transformer_proba
    )


def _time_split_by_symbol(frame, fraction):
    train_parts, valid_parts = [], []
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        cut = int(len(symbol_frame) * fraction)
        train_parts.append(symbol_frame.iloc[:cut])
        valid_parts.append(symbol_frame.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def _purge_train_horizon(frame, horizon):
    parts = []
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        if len(symbol_frame) <= horizon:
            continue
        parts.append(symbol_frame.iloc[:-horizon])
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _fit_normalization(frame, columns):
    stats = {}
    for column in columns:
        mean = float(frame[column].mean())
        std = float(frame[column].std())
        stats[column] = {"mean": mean, "std": std if np.isfinite(std) and std > 1e-8 else 1.0}
    return stats


def ingest(args):
    cfg = MVPConfig.from_json(args.config) if args.config else MVPConfig()
    events = [json.loads(line) for line in Path(args.events).read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = []
    for symbol in sorted({event["symbol"] for event in events}):
        frame = build_features([event for event in events if event["symbol"] == symbol], cfg)
        if not frame.empty:
            frames.append(assign_labels(frame, _threshold(frame.iloc[:int(len(frame) * 0.8)])))
    if not frames:
        raise RuntimeError("No usable book events found in the input JSONL")
    frame = __import__("pandas").concat(frames, ignore_index=True)
    paths = write_dataset_artifacts(events, frame, args.output, cfg)
    print(json.dumps({"dataset": paths, "rows": len(frame),
                      "symbols": frame.symbol.unique().tolist()}, indent=2))


def train(args):
    cfg = MVPConfig.from_json(args.config) if args.config else MVPConfig()
    overrides = {
        key: value for key, value in {
            "seed": args.seed,
            "epochs": args.epochs,
            "class_weight_power": args.class_weight_power,
            "use_medium_context": None if args.medium_context is None else args.medium_context == "on",
            "use_regime_context": None if args.regime_context is None else args.regime_context == "on",
            "use_event_gate": None if args.event_gate is None else args.event_gate == "on",
            "fusion_type": args.fusion_type,
        }.items() if value is not None
    }
    cfg = replace(cfg, **overrides)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    frame = read_frame(args.features)
    train_frame, valid_frame = _time_split_by_symbol(frame, cfg.train_fraction)
    label_threshold = _threshold(_purge_train_horizon(train_frame, cfg.horizon))
    train_frame = assign_labels(
        _purge_train_horizon(train_frame, cfg.horizon), label_threshold
    )
    valid_frame = assign_labels(valid_frame, label_threshold)
    feature_columns, regime_columns = cfg.feature_columns, cfg.regime_columns
    prior_columns = cfg.prior_columns
    categorical = ["event_add", "event_cancel", "event_trade", "event_depth_update", "side_signed"]
    normalization_columns = [c for c in feature_columns + regime_columns + prior_columns
                            if c not in categorical]
    normalization = _fit_normalization(train_frame, normalization_columns)
    xtr, xmtr, rtr, ptr, ttr, ytr = make_windows(
        train_frame, feature_columns, regime_columns, cfg, normalization)
    xva, xmva, rva, pva, tva, yva = make_windows(
        valid_frame, feature_columns, regime_columns, cfg, normalization)
    _debug_post(args.run_id, "B", "cli.py:train/windows", "[DEBUG] branch tensor scales", {
        "short": {"mean": float(xtr.mean()), "std": float(xtr.std()), "abs_max": float(xtr.abs().max())},
        "medium": {"mean": float(xmtr.mean()), "std": float(xmtr.std()), "abs_max": float(xmtr.abs().max())},
        "regime": {"mean": float(rtr.mean()), "std": float(rtr.std()), "abs_max": float(rtr.abs().max())},
        "short_shape": list(xtr.shape),
        "medium_shape": list(xmtr.shape),
    })
    _debug_post(args.run_id, "D", "cli.py:train/padding", "[DEBUG] repeated prefix ratios", {
        "short_repeated_prefix": float((xtr[:, 1:] == xtr[:, :-1]).all(dim=2).float().mean()),
        "medium_repeated_prefix": float((xmtr[:, 1:] == xmtr[:, :-1]).all(dim=2).float().mean()),
    })
    if len(ytr) == 0 or len(yva) == 0:
        raise RuntimeError(
            f"Not enough sequential data for window={cfg.window}, "
            f"medium_bar_minutes={cfg.medium_bar_minutes}; collect more real history per symbol."
        )
    if args.baseline_only:
        valid_samples = window_sample_metadata(valid_frame, cfg).reset_index(drop=True)
        valid_returns = valid_samples["return_h50"].to_numpy()
        baseline = LogisticRegression(
            max_iter=200, class_weight="balanced", random_state=cfg.seed
        )
        baseline.fit(xtr[:, -1, :].numpy(), ytr.numpy())
        train_proba = baseline.predict_proba(xtr[:, -1, :].numpy())
        valid_proba = baseline.predict_proba(xva[:, -1, :].numpy())
        result = _classification_summary(yva.numpy(), valid_proba, valid_returns)
        result.update({
            "best_epoch": None,
            "train_loss": float(log_loss(ytr.numpy(), train_proba, labels=[0, 1, 2])),
            "valid_loss": float(log_loss(yva.numpy(), valid_proba, labels=[0, 1, 2])),
            "loss_type": "logistic_cross_entropy",
            "config": {
                "seed": cfg.seed,
                "epochs": 0,
                "class_weight_power": 0.0,
                "use_medium_context": False,
                "use_regime_context": False,
                "use_event_gate": False,
                "fusion_type": "none",
            },
        })
        majority = int(ytr.bincount(minlength=3).argmax())
        majority_proba = np.zeros((len(yva), 3), dtype=float)
        majority_proba[:, majority] = 1.0
        result["majority"] = _classification_summary(
            yva.numpy(), majority_proba, valid_returns
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (out / "normalization.json").write_text(
            json.dumps(normalization, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return
    model = RegimeAwareLOB(
        len(feature_columns), len(regime_columns), d_model=cfg.d_model,
        layers=cfg.n_layers, heads=cfg.n_heads, d_ff=cfg.d_ff,
        dropout=cfg.dropout, max_window=cfg.window,
        use_medium_context=cfg.use_medium_context,
        use_regime_context=cfg.use_regime_context,
        use_event_gate=cfg.use_event_gate,
        fusion_type=cfg.fusion_type,
    )
    loader = DataLoader(TensorDataset(xtr, xmtr, rtr, ptr, ttr, ytr), batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    counts = torch.bincount(ytr, minlength=3).float()
    class_weights = (counts.sum() / (3.0 * counts.clamp_min(1.0))).pow(cfg.class_weight_power)
    _debug_post(args.run_id, "A", "cli.py:train/loss", "[DEBUG] class counts and CE weights", {
        "counts": counts.tolist(),
        "weights": class_weights.tolist(),
        "weight_ratio": float(class_weights.max() / class_weights.min()),
        "class_weight_power": cfg.class_weight_power,
    })
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    valid_samples = window_sample_metadata(
        valid_frame, cfg
    ).reset_index(drop=True)
    valid_returns = valid_samples["return_h50"].to_numpy()
    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    best_valid_loss = float("inf")
    best_train_loss = float("nan")
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x, xm, r, p, dt, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x, xm, r, p, dt)[0], y)
            loss.backward()
            if epoch == 0 and len(losses) == 0:
                gate_gradient = model.gate[0].weight.grad
                _debug_post(args.run_id, "A", "cli.py:train/backward", "[DEBUG] first batch gradient norms", {
                    "event": float(model.event[0].weight.grad.norm()),
                    "gate": None if gate_gradient is None else float(gate_gradient.norm()),
                    "head": float(model.head.weight.grad.norm()),
                })
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            valid_logits = model(xva, xmva, rva, pva, tva)[0]
            valid_loss = loss_fn(valid_logits, yva).item()
            valid_proba = valid_logits.softmax(-1).numpy()
        valid_macro_f1 = _metrics(yva.numpy(), valid_proba, valid_returns)["macro_f1"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_macro_f1": valid_macro_f1,
        })
        if (
            valid_macro_f1 > best_macro_f1
            or (valid_macro_f1 == best_macro_f1 and valid_loss < best_valid_loss)
        ):
            best_epoch = epoch + 1
            best_macro_f1 = valid_macro_f1
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} valid_macro_f1={valid_macro_f1:.5f}"
        )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, embedding, gate = model(xva, xmva, rva, pva, tva)
        transformer_proba = logits.softmax(-1).numpy()
    baseline = LogisticRegression(
        max_iter=200, class_weight="balanced", random_state=cfg.seed
    )
    baseline.fit(xtr[:, -1, :].numpy(), ytr.numpy())
    logistic_proba = baseline.predict_proba(xva[:, -1, :].numpy())
    blend_weight = args.logistic_blend_weight
    proba = _blend_probabilities(logistic_proba, transformer_proba, blend_weight)
    predicted = proba.argmax(1)
    _debug_post(args.run_id, "E", "cli.py:train/eval", "[DEBUG] prediction and logit distribution", {
        "true_counts": np.bincount(yva.numpy(), minlength=3).tolist(),
        "pred_counts": np.bincount(predicted, minlength=3).tolist(),
        "logit_mean": logits.mean(0).tolist(),
        "logit_std": logits.std(0).tolist(),
        "direction_mean": float((proba[:, 2] - proba[:, 0]).mean()),
        "direction_std": float((proba[:, 2] - proba[:, 0]).std()),
        "best_epoch": best_epoch,
        "logistic_blend_weight": blend_weight,
    })
    result = _classification_summary(yva.numpy(), proba, valid_returns)
    result["best_epoch"] = best_epoch
    result["train_loss"] = best_train_loss
    result["valid_loss"] = best_valid_loss
    result["loss_type"] = "cross_entropy"
    result["config"] = {
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "class_weight_power": cfg.class_weight_power,
        "use_medium_context": cfg.use_medium_context,
        "use_regime_context": cfg.use_regime_context,
        "use_event_gate": cfg.use_event_gate,
        "fusion_type": cfg.fusion_type,
        "logistic_blend_weight": blend_weight,
    }
    majority = int(ytr.bincount(minlength=3).argmax())
    majority_proba = np.zeros((len(yva), 3), dtype=float)
    majority_proba[:, majority] = 1.0
    majority_result = _classification_summary(yva.numpy(), majority_proba, valid_returns)
    result["majority_macro_f1"] = majority_result["macro_f1"]
    logistic_result = _classification_summary(yva.numpy(), logistic_proba, valid_returns)
    result["logistic_macro_f1"] = logistic_result["macro_f1"]
    result["baselines"] = {
        "majority": majority_result,
        "logistic": logistic_result,
    }
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "symbol": valid_samples["symbol"], "event_time_ms": valid_samples["event_time_ms"],
        "label_h50": yva.numpy(), "p_down": proba[:, 0], "p_neutral": proba[:, 1],
        "p_up": proba[:, 2], "direction_score": proba[:, 2] - proba[:, 0],
        "transformer_p_down": transformer_proba[:, 0],
        "transformer_p_neutral": transformer_proba[:, 1],
        "transformer_p_up": transformer_proba[:, 2],
        "logistic_p_down": logistic_proba[:, 0],
        "logistic_p_neutral": logistic_proba[:, 1],
        "logistic_p_up": logistic_proba[:, 2],
        "event_gate_mean": gate.numpy().mean(axis=1),
        "regime_volatility": valid_samples["rv"],
        "regime_liquidity": valid_samples["liquidity"],
        "regime_pressure": valid_samples["depth_imbalance_mean"],
        "regime_intensity": valid_samples["event_intensity"],
    })
    embedding_np = embedding.numpy()
    embeddings = pd.DataFrame(
        embedding_np, columns=[f"embedding_{i:03d}" for i in range(embedding_np.shape[1])]
    )
    predictions = pd.concat([predictions, embeddings], axis=1)
    prediction_path = write_frame(predictions, out / "predictions.parquet")
    for symbol in predictions["symbol"].unique():
        mask = predictions["symbol"].to_numpy() == symbol
        result[f"{symbol}_macro_f1"] = _metrics(yva.numpy()[mask], proba[mask],
                                                  valid_samples["return_h50"].to_numpy()[mask])["macro_f1"]
    torch.save(model.state_dict(), out / "model.pt")
    if blend_weight > 0:
        import joblib
        joblib.dump(baseline, out / "logistic_model.joblib")
    result["predictions"] = str(prediction_path)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--config"); p.add_argument("--events", required=True); p.add_argument("--output", default="artifacts/live/dataset"); p.set_defaults(func=ingest)
    p = sub.add_parser("train"); p.add_argument("--config"); p.add_argument("--features", required=True); p.add_argument("--output-dir", default="artifacts/run")
    p.add_argument("--run-id", default="post-fix")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--class-weight-power", type=float)
    p.add_argument("--medium-context", choices=("on", "off"))
    p.add_argument("--regime-context", choices=("on", "off"))
    p.add_argument("--event-gate", choices=("on", "off"))
    p.add_argument("--fusion-type", choices=("gated_add", "concat_mlp"))
    p.add_argument("--logistic-blend-weight", type=float, default=0.0)
    p.set_defaults(func=train)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
