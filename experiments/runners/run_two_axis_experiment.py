"""Experiment: two-axis direction model + slow confidence filter.

Reports on the held-out chronological test set:
  - two-axis model Macro-F1/AUC/IC (all predictions)
  - selective predictions kept only when confidence >= threshold
  - comparison vs pure short (conv5) reproduction and Logistic
The confidence threshold is selected on dev; test is scored once.
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

from lob_mvp.cli import (
    _purge_train_horizon, _threshold,
)
from lob_mvp.core import MVPConfig
from lob_mvp.medium_long_features import (
    MEDIUM_LONG_COLUMNS, build_medium_long,
)
from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import (
    ConfidenceFilter, TOKENS, TwoAxisLOB,
)

SEED = 20240929
STRIDE = 10
HORIZON = 10
OUTPUT = Path(
    "artifacts/historical/binance_extended/two_axis_20240929"
)


def prepare_symbol(frame, symbol):
    group = frame[frame.symbol == symbol].sort_values(
        "event_time_ms"
    ).reset_index(drop=True)
    ml = build_medium_long(group)
    group = pd.concat(
        [group.drop(columns=[c for c in MEDIUM_LONG_COLUMNS
                             if c in group.columns]), ml],
        axis=1,
    ).dropna(subset=MEDIUM_LONG_COLUMNS).reset_index(drop=True)
    return group


def make_windows(group, event_cols):
    event_values = group[event_cols].to_numpy(np.float32)
    slow_values = group[MEDIUM_LONG_COLUMNS].to_numpy(np.float32)
    mid = group["mid_price"].to_numpy()
    conv = np.convolve(mid, np.ones(HORIZON) / HORIZON, mode="valid")
    forward = np.full(len(group), np.nan)
    forward[:len(conv)] = conv / mid[:len(conv)] - 1.0
    events, slow_last, ends = [], [], []
    for end in range(TOKENS - 1, len(group) - HORIZON, STRIDE):
        events.append(event_values[end + 1 - TOKENS:end + 1])
        slow_last.append(slow_values[end])
        ends.append(end)
    return (
        torch.tensor(np.asarray(events)),
        torch.tensor(np.asarray(slow_last)),
        np.asarray(ends),
        forward,
    )


def warmup_cosine(step, total):
    warmup = max(1, int(total * 0.1))
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def score(y, logits, forward):
    prob = logits.detach().softmax(-1).numpy()
    direction = prob[:, 2] - prob[:, 0]
    return {
        "macro_f1": float(f1_score(y, prob.argmax(1), average="macro",
                                 zero_division=0)),
        "auc": float(roc_auc_score(y == 2, direction)),
        "ic": float(np.corrcoef(direction, forward)[0, 1]),
        "counts": np.bincount(prob.argmax(1), minlength=3).tolist(),
    }, prob


def selective_metrics(y, prob, forward, confidence, threshold):
    keep = confidence >= threshold
    if keep.sum() < 50:
        return None
    yy, pp, ff = y[keep], prob[keep], forward[keep]
    direction = pp[:, 2] - pp[:, 0]
    return {
        "macro_f1": float(f1_score(yy, pp.argmax(1), average="macro",
                                 zero_division=0)),
        "auc": float(roc_auc_score(yy == 2, direction)),
        "ic": float(np.corrcoef(direction, ff)[0, 1]),
        "coverage": float(keep.mean()),
        "kept": int(keep.sum()),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = MVPConfig.from_json("config/binance_202609.json")
    frame = pd.read_parquet(
        "artifacts/historical/binance_extended/event_features.parquet"
    )
    event_cols = cfg.feature_columns

    splits = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        group = prepare_symbol(frame, symbol)
        cut = int(len(group) * cfg.train_fraction)
        train_g, valid_g = group.iloc[:cut], group.iloc[cut:]
        sym_threshold = _threshold(
            _purge_train_horizon(train_g, HORIZON)
        )

        def assemble(g):
            events, slow, ends, fwd = make_windows(g, event_cols)
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

    def gather(part):
        e, s, y, f = [], [], [], []
        for symbol in splits:
            ev, sl, yy, ff = splits[symbol][part]
            e.append(ev)
            s.append(sl)
            y.append(yy)
            f.append(ff)
        return torch.cat(e), torch.cat(s), torch.cat(y), np.concatenate(f)

    Etr, Str, ytr, ftr = gather("train")
    Edev, Sdev, ydev, fdev = gather("dev")
    Etest, Stest, ytest, ftest = gather("test")
    print("sizes", len(Etr), len(Edev), len(Etest), flush=True)

    # Train-only normalization (events and slow separately).
    def fit_norm(t):
        flat = t.reshape(-1, t.shape[-1])
        return flat.mean(0), flat.std(0).clamp_min(1e-6)

    e_mean, e_std = fit_norm(Etr)
    s_mean, s_std = fit_norm(Str)
    Etr, Edev, Etest = ((Etr - e_mean) / e_std,
                         (Edev - e_mean) / e_std,
                         (Etest - e_mean) / e_std)
    Str, Sdev, Stest = ((Str - s_mean) / s_std,
                        (Sdev - s_mean) / s_std,
                        (Stest - s_mean) / s_std)

    # ---- Direction model ----
    model = TwoAxisLOB(Etr.shape[-1])
    counts = torch.bincount(ytr, minlength=3).float()
    class_weights = (counts.sum() / (3 * counts.clamp_min(1))).pow(0.5)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.03
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.02)
    bs = 256
    total = 24 * math.ceil(len(Etr) / bs)
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda st: warmup_cosine(st, total))
    best = {"f1": -1, "state": None, "epoch": 0}
    since = 0
    for epoch in range(24):
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
            dev_logits, _, _ = model(Edev)
        dev_summary, _ = score(ydev.numpy(), dev_logits, fdev)
        if dev_summary["macro_f1"] > best["f1"] + 1e-5:
            best = {
                "f1": dev_summary["macro_f1"], "epoch": epoch + 1,
                "state": {k: v.clone() for k, v in
                          model.state_dict().items()},
            }
            since = 0
        else:
            since += 1
        print(epoch + 1, f"dev_f1={dev_summary['macro_f1']:.4f}", flush=True)
        if since >= 8:
            break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        test_logits, _, _ = model(Etest)
    test_summary, test_prob = score(ytest.numpy(), test_logits, ftest)

    # ---- Confidence filter (trained to recognize high-margin/tradable) ----
    confidence_net = ConfidenceFilter(Str.shape[-1])
    # Target: samples where the realized return exceeds the label threshold
    # (proxy for "tradable"); learned only on train.
    train_target = (
        np.abs(ftr) > np.quantile(np.abs(ftr), 0.6)
    ).astype(np.float32)
    c_opt = torch.optim.AdamW(confidence_net.parameters(), lr=1e-3,
                             weight_decay=1e-4)
    c_loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(60):
        perm = torch.randperm(len(Str))
        for i in range(0, len(Str), 256):
            idx = perm[i:i + 256]
            c_opt.zero_grad()
            logit = confidence_net.net(Str[idx]).squeeze(-1)
            loss = c_loss_fn(logit, torch.tensor(train_target[idx]))
            loss.backward()
            c_opt.step()
    confidence_net.eval()
    with torch.no_grad():
        dev_conf = confidence_net(Sdev).numpy()
        test_conf = confidence_net(Stest).numpy()

    # Pick confidence threshold on dev (target coverage 50%).
    dev_threshold = float(np.quantile(dev_conf, 0.5))
    selective = selective_metrics(
        ytest.numpy(), test_prob, ftest, test_conf, dev_threshold
    )

    # ---- Baselines ----
    logistic = LogisticRegression(
        max_iter=200, class_weight="balanced", random_state=SEED
    )
    logistic.fit(Etr[:, -1, :].numpy(), ytr.numpy())
    logistic_test_f1 = f1_score(
        ytest.numpy(), logistic.predict(Etest[:, -1, :].numpy()),
        average="macro", zero_division=0,
    )

    result = {
        "samples": {"train": len(Etr), "dev": len(Edev),
                    "test": len(Etest)},
        "best_epoch": best["epoch"],
        "two_axis_all_predictions": test_summary,
        "two_axis_selective": selective,
        "logistic_test_macro_f1": float(logistic_test_f1),
        "two_axis_vs_logistic": test_summary["macro_f1"] - logistic_test_f1,
        "exceeds_logistic": bool(
            test_summary["macro_f1"] > logistic_test_f1
        ),
    }
    snapshot_runner(OUTPUT, __file__,
                    extra_files=["config/binance_202609.json"])
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), OUTPUT / "two_axis_model.pt")
    pd.DataFrame({
        "label_h50": ytest.numpy(), "return_h50": ftest,
        "p_down": test_prob[:, 0], "p_neutral": test_prob[:, 1],
        "p_up": test_prob[:, 2], "confidence": test_conf,
    }).to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
