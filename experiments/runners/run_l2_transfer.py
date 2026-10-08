"""Transfer learning for the small one-week L2 model (zero extra download).

The level model is data-starved (1376 train windows; dev->test dropped ~0.10;
the +0.054 gap was bootstrap-insignificant). This script pretrains only the
TEMPORAL backbone on the local 2-month 30s bucket dataset, then fine-tunes on
the one-week L2 data.

Leakage control: pretraining uses only bucket frames strictly BEFORE the L2
evaluation week (< 2026-08-27 UTC), so the test period never participates in
pretraining. The level/global input projections, the across-level encoder and
the head cannot be transferred (different structure) and stay random-init.

Transferred submodules (d_model=64, T=120, 1 temporal layer):
    conv, pos, time_encoder  and  pool -> time_pool.
Then the same strict L2 protocol and paired iid/block bootstrap are applied.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn

from experiments.runners.bootstrap_l2_significance import block_index, macro_f1
from experiments.runners.run_l2_level_experiment import (
    CLASS_WEIGHT_POWER, GLOBAL_COLS, LEVEL_CHANNELS, LEVELS, SEED, eval_probs,
    labels_from, prepare_l2_dataset, train_direction_model,
)
from experiments.runners.run_two_axis_experiment import (
    make_windows as bucket_windows, prepare_symbol,
)
from lob_mvp.core import MVPConfig
from lob_mvp.run_io import snapshot_runner
from lob_mvp.two_axis_model import LevelStructuredLOB, TwoAxisLOB

PRETRAIN_BEFORE = pd.Timestamp("2026-08-27", tz="UTC")
PRETRAIN_EPOCHS = 20
FINETUNE_LR = 2e-4
B = 20_000
BLOCK = 10
OUTPUT = Path("artifacts/historical/l2_btc_week_model_transfer")


def pretrain_temporal():
    cfg = MVPConfig.from_json("config/binance_202609.json")
    event_cols = cfg.feature_columns
    frame = pd.read_parquet(
        "artifacts/historical/binance_extended/event_features.parquet"
    )
    cutoff_ms = int(PRETRAIN_BEFORE.timestamp() * 1000)

    all_events, all_fwd = [], []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        group = prepare_symbol(frame, symbol)
        group = group[group["event_time_ms"] < cutoff_ms]
        events, _, ends, fwd = bucket_windows(group, event_cols)
        defined = np.isfinite(fwd[ends])
        all_events.append(events[defined]); all_fwd.append(fwd[ends][defined])
    E = torch.cat(all_events)
    fwd = np.concatenate(all_fwd)
    thr = float(np.quantile(np.abs(fwd), 0.7))
    y = torch.tensor(labels_from(fwd, thr))
    print("pretrain windows:", len(E), "features:", E.shape[-1], flush=True)

    model = TwoAxisLOB(E.shape[-1], layers=1)
    counts = torch.bincount(y, minlength=3).float()
    cw = (counts.sum() / (3 * counts.clamp_min(1))).pow(CLASS_WEIGHT_POWER)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    bs = 256

    def cos(step, total):
        wu = max(1, int(total * 0.1))
        if step < wu:
            return step / wu
        p = (step - wu) / max(1, total - wu)
        return 0.5 * (1 + math.cos(math.pi * p))

    total = PRETRAIN_EPOCHS * math.ceil(len(E) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cos(s, total))
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(len(E))
        for i in range(0, len(E), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits, _, _ = model(E[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if (epoch + 1) % 5 == 0:
            print(f"pretrain epoch {epoch+1}/{PRETRAIN_EPOCHS} "
                  f"loss={loss.item():.4f}", flush=True)
    return model


def transfer_weights(level_model, pretrained):
    src = pretrained.state_dict()
    mapped = {}
    for k, v in src.items():
        if k == "pos" or k.startswith("conv.") or k.startswith("time_encoder."):
            mapped[k] = v
        elif k.startswith("pool."):
            mapped["time_pool." + k[len("pool."):]] = v
    report = level_model.load_state_dict(mapped, strict=False)
    print("transferred tensors:", len(mapped))
    print("missing (fresh init):", report.missing_keys)
    return level_model


def logistic_test_predictions(tr, te):
    def flat_last(d):
        return torch.cat([d["L"][:, -1].reshape(len(d["L"]), -1),
                          d["G"][:, -1]], dim=1).numpy()

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(flat_last(tr), tr["y"].numpy())
    return clf.predict(flat_last(te))


def paired_bootstrap(y, model_pred, log_pred):
    rng = np.random.default_rng(SEED)
    n = len(y)
    out = {}
    for scheme in ("iid", "block"):
        gaps = np.empty(B)
        for b in range(B):
            idx = (rng.integers(0, n, n) if scheme == "iid"
                   else block_index(n, BLOCK, rng))
            gaps[b] = macro_f1(y[idx], model_pred[idx]) - \
                macro_f1(y[idx], log_pred[idx])
        lo, hi = np.quantile(gaps, [0.025, 0.975])
        out[scheme] = {"gap_mean": float(gaps.mean()),
                       "ci95": [float(lo), float(hi)],
                       "p_gap_gt_0": float((gaps > 0).mean())}
        print(f"[{scheme:5s}] mean={gaps.mean():+.4f} "
              f"95%CI=({lo:+.4f},{hi:+.4f}) P>0={(gaps>0).mean():.3f}",
              flush=True)
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = prepare_l2_dataset()
    tr, dv, te = ds["tr"], ds["dv"], ds["te"]

    pretrained = pretrain_temporal()
    model = LevelStructuredLOB(len(LEVEL_CHANNELS), len(GLOBAL_COLS),
                               levels=LEVELS)
    transfer_weights(model, pretrained)

    best_epoch = train_direction_model(model, tr, dv, lr=FINETUNE_LR)
    test_prob, q_test = eval_probs(model, te)
    model_pred = test_prob.argmax(1)
    y = te["y"].numpy()

    log_pred = logistic_test_predictions(tr, te)
    model_f1 = macro_f1(y, model_pred)
    log_f1 = macro_f1(y, log_pred)
    print(f"point: transfer model={model_f1:.4f} logistic={log_f1:.4f} "
          f"gap={model_f1-log_f1:+.4f}", flush=True)
    boot = paired_bootstrap(y, model_pred, log_pred)

    result = {
        "pretrain_before_utc": str(PRETRAIN_BEFORE),
        "finetune_lr": FINETUNE_LR, "best_epoch": best_epoch,
        "point": {"transfer_model_macro_f1": model_f1,
                  "logistic_macro_f1": log_f1, "gap": model_f1 - log_f1},
        "counts": np.bincount(model_pred, minlength=3).tolist(),
        "bootstrap": boot,
    }
    snapshot_runner(OUTPUT, __file__)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), OUTPUT / "transfer_model.pt")
    np.save(OUTPUT / "test_embedding.npy", q_test.numpy())
    ext_end = ds["bounds"]["ext_end"]
    test_ends = ds["test_ends"][ds["test_defined"]]
    test_bins = ds["frame"]["bin"].to_numpy()[ext_end + test_ends]
    pd.DataFrame({
        "bin": test_bins, "label": y, "return_h50": te["f"],
        "p_down": test_prob[:, 0], "p_neutral": test_prob[:, 1],
        "p_up": test_prob[:, 2],
    }).to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
