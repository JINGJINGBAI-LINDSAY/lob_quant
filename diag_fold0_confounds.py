"""Fold0 diagnostic: quantify two confounds in the E020 bucket comparison.

Reproduces fold0 exactly (train_stride=30, eval_stride=120, cut=0.8), then
varies ONLY (a) class-weight scheme and (b) input scope, all with logistic
models (seconds to fit), to measure how much of the reported -0.101 gap is
caused by design asymmetry rather than model capacity.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.runners.bootstrap_l2_significance import macro_f1
from experiments.runners.run_bucket_walkforward import (
    build_folds, labels, windows_over,
)
from experiments.runners.run_two_axis_experiment import prepare_symbol
from lob_mvp.core import MVPConfig
import pandas as pd

CONFIG = "config/binance_202609.json"
FEATURES = "artifacts/historical/binance_1y/event_features.parquet"

cfg = MVPConfig.from_json(CONFIG)
frame = pd.read_parquet(FEATURES)
group = prepare_symbol(frame, "BTCUSDT")
event_cols = list(cfg.feature_columns)
fold = build_folds(group)[0]

Etr, _, ftr = windows_over(group, event_cols, fold["tr_lo"], fold["tr_hi"], 30)
Ete, _, fte = windows_over(group, event_cols, fold["te_lo"], fold["te_hi"], 120)
cut = int(len(Etr) * 0.8)
threshold = float(np.quantile(np.abs(ftr[:cut]), 0.7))
y_all = labels(ftr, threshold).numpy()
ytr = y_all[:cut]
yte = labels(fte, threshold).numpy()

flat = Etr[:cut].reshape(-1, Etr.shape[-1])
e_mean, e_std = flat.mean(0), flat.std(0).clip(1e-6)
norm = lambda t: (t - e_mean) / e_std
Etr_n, Ete_n = norm(Etr[:cut]), norm(Ete)

counts = np.bincount(ytr, minlength=3).astype(float)
N = counts.sum()
sqrt_w = {c: float((N / (3 * max(counts[c], 1))) ** 0.5) for c in range(3)}
bal_w = {c: float(N / (3 * max(counts[c], 1))) for c in range(3)}

def fit_report(name, Xtr, Xte, cw):
    clf = LogisticRegression(max_iter=2000, class_weight=cw,
                             random_state=20261008)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = macro_f1(yte, pred)
    dist = np.bincount(pred, minlength=3).tolist()
    print(f"{name:34s} f1={f1:.4f}  pred_dist={dist}")
    return f1

print(f"label counts train={np.bincount(ytr).tolist()} "
      f"test={np.bincount(yte).tolist()} threshold={threshold:.6f}")
print("--- single frame (what the current baseline uses) ---")
f_bal = fit_report("single-frame  balanced (as run)", Etr_n[:, -1, :],
                   Ete_n[:, -1, :], "balanced")
f_sqrt = fit_report("single-frame  sqrt (=nn scheme)", Etr_n[:, -1, :],
                    Ete_n[:, -1, :], sqrt_w)
f_none = fit_report("single-frame  none", Etr_n[:, -1, :],
                    Ete_n[:, -1, :], None)
print("--- sequence flatten 120x49 (same info as nn) ---")
Xtr_seq = Etr_n.reshape(len(Etr_n), -1)
Xte_seq = Ete_n.reshape(len(Ete_n), -1)
s_bal = fit_report("sequence-flatten balanced", Xtr_seq, Xte_seq, "balanced")
s_sqrt = fit_report("sequence-flatten sqrt", Xtr_seq, Xte_seq, sqrt_w)

print("\n--- gap decomposition (nn reported 0.3348) ---")
print(f"effect of class-weight alone on logistic: balanced->sqrt = "
      f"{f_sqrt - f_bal:+.4f}")
print(f"effect of input scope (balanced): single->sequence = {s_bal - f_bal:+.4f}")
print(f"fair gap (nn 0.3348 vs same-scheme sqrt single logistic {f_sqrt:.4f}) = "
      f"{0.3348 - f_sqrt:+.4f}")
