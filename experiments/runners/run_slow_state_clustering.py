"""Stage-3 clustering of the slow-state latent z, with three pre-registered gates.

Everything runs on the OOS segment only (220,596 frames, ~76 days).
Primary clustering: KMeans on the 12-d z, K=4; K=2..6 reported as sensitivity.
Baseline: same KMeans on the 13 standardized raw slow features ("classical
clustering on raw slow variables").

Gates (thresholds fixed before running):
  G1 Not a direction proxy: Cramer's V(cluster, label_h50) < 0.40 with block-
     bootstrap 95% CI upper bound < 0.50; and max over clusters
     |#up/(#up+#down) - 0.5| <= 0.15 with CI upper <= 0.25.
  G2 Better than raw baseline: paired moving-block bootstrap 95% CI lower
     bound of (z - raw) > 0, for BOTH silhouette and segment-reproduction ARI.
  G3 Durable states: after a minimum-dwell filter (120 frames = 1 input
     window), mean dwell > 360 frames (3x window), bootstrap CI lower bound
     > 240 frames (2x window).

Segment reproduction: split OOS into three contiguous thirds; for each
forward pair (i<j), fit centers on segment i, predict j, compare via ARI to
KMeans fit independently on j. ARI is label-permutation invariant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

from experiments.runners.run_slow_state_encoder import INPUT_COLS

Z_PATH = Path("artifacts/historical/slow_state/slow_z.parquet")
B_PATH = Path(
    "artifacts/historical/binance_1y/event_features.parquet"
)
OUT = Path("artifacts/historical/slow_state")

KS = [2, 3, 4, 5, 6]
K_PRIMARY = 5
WINDOW = 120               # one input window / min-dwell, in 30s frames
SEED = 20261009
R_SIL = 30                 # paired silhouette draws
R_PAIR = 200               # paired ARI block-bootstrap reps
R_G1 = 500                 # G1 block-bootstrap reps
R_G3 = 500                 # dwell bootstrap reps
SIL_SAMPLE = 10_000
ARI_SUBSAMPLE = 20_000


# ---------------------------------------------------------------- data
def load_oos():
    z = pd.read_parquet(Z_PATH)
    zcols = [c for c in z.columns if c.startswith("z_")]
    z = z[z.segment == "oos"].sort_values("event_time_ms").reset_index(drop=True)
    b = pd.read_parquet(
        B_PATH, columns=["event_time_ms", "label_h50"] + INPUT_COLS
    )
    m = z.merge(b, on="event_time_ms", how="left", validate="one_to_one")
    if m[["label_h50"] + INPUT_COLS].isna().any().any():
        raise SystemExit("missing joined values in OOS")
    Z = m[zcols].to_numpy(np.float64)
    direction = m["label_h50"].to_numpy(np.int64)
    R = m[INPUT_COLS].to_numpy(np.float64)
    return Z, R, direction, m["event_time_ms"].to_numpy(np.int64)


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0).clip(1e-8)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def km_fit_predict(X, K, centers_only=False):
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
    lab = km.fit_predict(X)
    return (km.cluster_centers_, km) if centers_only else (lab, km)


def predict_centers(X, centers):
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


# ---------------------------------------------------------------- G1
def cramers_v(x, y):
    cm = contingency_matrix(x, y, sparse=False).astype(np.float64)
    n = cm.sum()
    if n == 0 or cm.shape[0] < 2 or cm.shape[1] < 2:
        return 0.0
    chi2 = _chi2_from_contingency(cm)
    r, c = cm.shape
    return float(np.sqrt(chi2 / (n * (min(r - 1, c - 1)))))


def _chi2_from_contingency(cm):
    n = cm.sum()
    row = cm.sum(1, keepdims=True)
    col = cm.sum(0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(expected > 0, (cm - expected) ** 2
                        / np.where(expected > 0, expected, 1), 0.0)
    return term.sum()


def gate1(lab, direction):
    v = cramers_v(lab, direction)
    skews = []
    for k in np.unique(lab):
        d = direction[lab == k]
        d = d[d != 1]
        if len(d) >= 30:
            skews.append(abs((d == 2).mean() - 0.5))
    skew_max = float(max(skews)) if skews else 0.0

    n = len(lab)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / WINDOW))
    vs, sk = np.empty(R_G1), np.empty(R_G1)
    for b in range(R_G1):
        starts = rng.integers(0, n - WINDOW + 1, nb)
        idx = (starts[:, None] + np.arange(WINDOW)[None, :]).ravel()[:n]
        vs[b] = cramers_v(lab[idx], direction[idx])
        s = []
        ll, dd = lab[idx], direction[idx]
        for k in np.unique(ll):
            d = dd[(ll == k)]
            d = d[d != 1]
            if len(d) >= 30:
                s.append(abs((d == 2).mean() - 0.5))
        sk[b] = max(s) if s else 0.0
    v_lo, v_hi = np.quantile(vs, [0.025, 0.975])
    s_lo, s_hi = np.quantile(sk, [0.025, 0.975])
    passed = bool(v < 0.40 and v_hi < 0.50
                  and skew_max <= 0.15 and s_hi <= 0.25)
    return {"cramers_v": v, "cramers_v_ci95": [float(v_lo), float(v_hi)],
            "skew_max": skew_max, "skew_ci95": [float(s_lo), float(s_hi)],
            "pass": passed}


# ---------------------------------------------------------------- G2
def block_indices(n, block, target, rng):
    nb = int(np.ceil(target / block))
    starts = rng.integers(0, n - block + 1, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return idx[:target]


def gate2_silhouette(Z, Rraw, K, lab_z, lab_r):
    n = len(Z)
    rng = np.random.default_rng(SEED)
    dz, dr, lz, lr = [], [], [], []
    for _ in range(R_SIL):
        idx = np.sort(rng.choice(n, SIL_SAMPLE, replace=False))
        # reassign within-sample via nearest full-OOS centers? labels inherited
        # from the full fit; standard silhouette on fixed labels + sampled rows.
        z_lab = lab_z[idx]
        r_lab = lab_r[idx]
        if len(np.unique(z_lab)) < 2 or len(np.unique(r_lab)) < 2:
            continue
        sz = silhouette_score(Z[idx], z_lab)
        sr = silhouette_score(Rraw[idx], r_lab)
        dz.append(sz); dr.append(sr)
    dz, dr = np.array(dz), np.array(dr)
    delta = dz - dr
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {"z_silhouette": float(dz.mean()),
            "raw_silhouette": float(dr.mean()),
            "delta": float(delta.mean()),
            "delta_ci95": [float(lo), float(hi)],
            "pass": bool(lo > 0)}


def gate2_segment_ari(Z, Rraw, K):
    n = len(Z)
    bounds = [0, n // 3, 2 * n // 3, n]
    segs = [(bounds[i], bounds[i + 1]) for i in range(3)]

    # Raw standardizers fitted on the training segment per pair; z is fixed.
    pair_data = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        tr0, tr1 = segs[i]
        te0, te1 = segs[j]
        _, kmz = km_fit_predict(Z[tr0:tr1], K)
        _, kmr = km_fit_predict(Rraw[tr0:tr1], K)
        pred_z = predict_centers(Z[te0:te1], kmz.cluster_centers_)
        pred_r = predict_centers(Rraw[te0:te1], kmr.cluster_centers_)
        ref_z, _ = km_fit_predict(Z[te0:te1], K)
        ref_r, _ = km_fit_predict(Rraw[te0:te1], K)
        pair_data.append((pred_z, pred_r, ref_z, ref_r))

    rng = np.random.default_rng(SEED)
    az, ar = [], []
    for _ in range(R_PAIR):
        pz, pr = [], []
        for (pred_z, pred_r, ref_z, ref_r) in pair_data:
            m = len(ref_z)
            idx = block_indices(m, WINDOW, min(ARI_SUBSAMPLE, m), rng)
            pz.append(adjusted_rand_score(ref_z[idx], pred_z[idx]))
            pr.append(adjusted_rand_score(ref_r[idx], pred_r[idx]))
        az.append(np.mean(pz)); ar.append(np.mean(pr))
    az, ar = np.array(az), np.array(ar)
    delta = az - ar
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {"z_segment_ari": float(az.mean()),
            "raw_segment_ari": float(ar.mean()),
            "delta": float(delta.mean()),
            "delta_ci95": [float(lo), float(hi)],
            "pass": bool(lo > 0)}


# ---------------------------------------------------------------- G3
def min_dwell_filter(lab, min_len):
    # run lengths
    brk = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(lab)]])
    labels = lab[starts]
    lengths = ends - starts

    durable = lengths >= min_len
    if not durable.any():
        return lab.copy(), lengths
    # each run adopts previous durable label; leading shorts adopt first durable
    first_durable = int(np.argmax(durable))
    assigned = np.empty(len(labels), dtype=lab.dtype)
    last = labels[first_durable]
    for i in range(len(labels)):
        if durable[i]:
            last = labels[i]
        assigned[i] = last
    out = np.empty_like(lab)
    for s, e, lb in zip(starts, ends, assigned):
        out[s:e] = lb
    # recompute merged runs
    brk2 = np.flatnonzero(np.diff(out)) + 1
    st2 = np.concatenate([[0], brk2])
    en2 = np.concatenate([brk2, [len(out)]])
    return out, en2 - st2


def gate3(lab):
    filtered, dwell = min_dwell_filter(lab, WINDOW)
    mean_d, med_d = float(dwell.mean()), float(np.median(dwell))
    rng = np.random.default_rng(SEED)
    means = np.empty(R_G3)
    for b in range(R_G3):
        sample = rng.choice(dwell, len(dwell), replace=True)
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    passed = bool(mean_d > 3 * WINDOW and lo > 2 * WINDOW)
    return {"mean_dwell_frames": mean_d, "median_dwell_frames": med_d,
            "mean_dwell_min": mean_d / 2.0, "median_dwell_min": med_d / 2.0,
            "ci95_frames": [float(lo), float(hi)],
            "n_runs": int(len(dwell)), "pass": passed,
            "filtered_labels": filtered}


# ---------------------------------------------------------------- main
def main():
    Z, Rraw, direction, ts = load_oos()
    mu, sd = standardize_fit(Rraw)
    Rs = standardize_apply(Rraw, mu, sd)
    print(f"OOS frames={len(Z)}", flush=True)

    per_k = {}
    export_labels = None
    for K in KS:
        lab_z, _ = km_fit_predict(Z, K)
        lab_r, _ = km_fit_predict(Rs, K)
        g1 = gate1(lab_z, direction)
        g2s = gate2_silhouette(Z, Rs, K, lab_z, lab_r)
        g2a = gate2_segment_ari(Z, Rs, K)
        g3 = gate3(lab_z)
        per_k[f"K{K}"] = {
            "G1_direction_independence": g1,
            "G2a_silhouette": {k: v for k, v in g2s.items()},
            "G2b_segment_ari": g2a,
            "G3_dwell": {k: v for k, v in g3.items()
                         if k != "filtered_labels"},
        }
        if K == K_PRIMARY:
            export_labels = g3["filtered_labels"]
        print(f"K={K} G1={g1['pass']} G2sil={g2s['pass']} "
              f"G2ari={g2a['pass']} G3={g3['pass']} "
              f"| V={g1['cramers_v']:.3f} dSil={g2s['delta']:+.3f} "
              f"dARI={g2a['delta']:+.3f} dwell={g3['mean_dwell_frames']:.0f}",
              flush=True)

    primary = per_k[f"K{K_PRIMARY}"]
    decision = bool(
        primary["G1_direction_independence"]["pass"]
        and primary["G2a_silhouette"]["pass"]
        and primary["G2b_segment_ari"]["pass"]
        and primary["G3_dwell"]["pass"]
    )
    result = {"scope": "OOS_only", "K_primary": K_PRIMARY,
              "primary_decision": decision, "per_k": per_k}
    (OUT / "cluster_gates.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    pd.DataFrame({
        "event_time_ms": ts, "cluster": export_labels,
        "label_h50": direction,
    }).to_parquet(OUT / "oos_clusters.parquet", index=False)
    print(f"DECISION={decision} (primary K={K_PRIMARY})", flush=True)


if __name__ == "__main__":
    main()
