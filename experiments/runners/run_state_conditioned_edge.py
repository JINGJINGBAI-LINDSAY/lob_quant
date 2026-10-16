"""State-conditioned direction edge: stratify the L2 direction bets by regime.

This is the only remaining test that can overturn the "gross edge ~ 0" verdict
without rebuilding any direction model: take the already-trained 12-fold L2
direction bets, assign each to the K=5 slow-state cluster at its timestamp, and
ask whether the gross edge concentrates in one state strongly enough to cover
costs.

Two label sets:
  strict   : dwell-filtered K=5 labels on the OOS segment only
             (artifacts/historical/slow_state/oos_clusters.parquet); this covers
             folds 6-11 and is the leakage-free analysis.
  extended : the SAME KMeans centers (fitted on OOS z, deterministic refit here)
             assigned to every frame and dwell-filtered over the full series;
             covers all 12 folds. Caveat: for folds 0-5 the encoder params were
             trained partly on those windows -> supportive evidence only.

Pre-registered primary hypothesis (fixed before looking):
  strict / 5min / cluster 0 (high-volatility + high-activity state) has gross
  edge whose block-bootstrap 95% CI lower bound exceeds the 7bp round-trip cost.
Pass rule for any cell: gross mean CI low > 7bp (equiv. net7 CI low > 0),
  edge present in >= n_folds-1 represented folds, and >= 2 independent folds.
Concentration is also reported directly: bootstrap CI of (cluster gross mean
  minus gross mean of all other bets in the same tier).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from experiments.runners.run_horizon_scan import (
    CLOCK_FRAMES, CLOCK_LABELS, COSTS, build_bets,
)
from experiments.runners.run_l2_multiweek import WEEKS
from experiments.runners.run_slow_state_clustering import (
    SEED, WINDOW, min_dwell_filter,
)

OUT = Path("artifacts/historical/state_conditioned_edge")
SLOW = Path("artifacts/historical/slow_state")
STRICT_PATH = SLOW / "oos_clusters.parquet"
Z_PATH = SLOW / "slow_z.parquet"
EXT_PATH = SLOW / "clusters_extended.parquet"
R_BOOT = 5_000
STRIDE_BETS = 10          # bets are 10 frames (5min) apart
PRIMARY_TIER = "5min"
PRIMARY_CLUSTER = 0


# -------------------------------------------------------------- labels
def build_extended_labels():
    """Frozen OOS K=5 centers assigned to every frame; dwell-filtered."""
    z = pd.read_parquet(Z_PATH).sort_values("event_time_ms").reset_index(drop=True)
    zcols = [c for c in z.columns if c.startswith("z_")]
    Z = z[zcols].to_numpy()
    centers = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit(
        Z[z.segment == "oos"]).cluster_centers_
    lab = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    filt, _ = min_dwell_filter(lab, WINDOW)
    ext = pd.DataFrame({"event_time_ms": z.event_time_ms, "cluster": filt})
    ext.to_parquet(EXT_PATH, index=False)
    return ext


# -------------------------------------------------------------- per-bet PnL
def gross_table(bets, week_mid):
    """Long table of gross directional returns: fold/week/ts/gross per tier."""
    d = bets[bets.dir != 0]
    parts = []
    for h, label in zip(CLOCK_FRAMES, CLOCK_LABELS):
        rows = []
        for name, _p in WEEKS:
            mid, n_frames = week_mid[name]
            b = d[d.week == name]
            end = b.end.to_numpy()
            entry = b.entry_mid.to_numpy()
            direc = b.dir.to_numpy()
            ok = end + h < n_frames
            e = end[ok]
            g = direc[ok] * (mid[e + h] / entry[ok] - 1.0)
            rows.append(pd.DataFrame({
                "fold": b.fold.to_numpy()[ok], "week": name,
                "ts": b.ts.to_numpy()[ok], "gross": g,
            }))
        t = pd.concat(rows, ignore_index=True)
        t["tier"] = label
        t["h"] = h
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# -------------------------------------------------------------- bootstrap
def _draw_mean(by_fold, folds, blen, rng):
    vals = []
    for f in rng.choice(folds, len(folds)):
        a = by_fold[int(f)]
        if len(a) >= blen:
            s = rng.integers(0, len(a) - blen + 1)
            vals.append(a[s:s + blen])
        else:
            vals.append(a)
    return float(np.concatenate(vals).mean())


def cluster_cell(x, rest, blen, rng):
    folds = sorted(x.fold.unique())
    by_fold = {int(f): x[x.fold == f].gross.to_numpy() for f in folds}
    r_folds = sorted(rest.fold.unique())
    r_by = {int(f): rest[rest.fold == f].gross.to_numpy() for f in r_folds}

    gm = np.empty(R_BOOT)
    rm = np.empty(R_BOOT)
    for b in range(R_BOOT):
        gm[b] = _draw_mean(by_fold, folds, blen, rng)
        rm[b] = _draw_mean(r_by, r_folds, blen, rng)

    cost7, cost2 = COSTS["7bp"], COSTS["2bp"]
    g = x.gross.to_numpy()
    lo, hi = np.quantile(gm, [0.025, 0.975])
    d = gm - rm
    dlo, dhi = np.quantile(d, [0.025, 0.975])
    fold_means = x.groupby("fold").gross.mean()
    folds_pos = int((fold_means > cost7).sum())
    return {
        "n": int(len(x)), "n_folds": len(folds), "folds": [int(f) for f in folds],
        "gross_mean": float(g.mean()), "gross_median": float(np.median(g)),
        "win_rate": float((g > 0).mean()),
        "net7_mean": float(g.mean() - cost7),
        "net2_mean": float(g.mean() - cost2),
        "gross_ci95": [float(lo), float(hi)],
        "p_gross_gt_7bp": float((gm > cost7).mean()),
        "folds_positive_net7": folds_pos,
        "edge_vs_rest_mean": float(d.mean()),
        "edge_vs_rest_ci95": [float(dlo), float(dhi)],
        "pass": bool(lo > cost7 and folds_pos >= len(folds) - 1
                     and len(folds) >= 2),
    }


def analyze(gt, labels, tag):
    """gt: gross long table; labels: event_time_ms -> cluster."""
    m = gt.merge(labels.rename(columns={labels.columns[0]: "ts"}),
                 on="ts", how="left")
    result = {}
    for label in CLOCK_LABELS:
        tier = m[m.tier == label]
        h = int(tier.h.iloc[0])
        blen = max(1, int(np.ceil(h / STRIDE_BETS)))
        rng = np.random.default_rng(SEED)
        cells = {}
        present = sorted(tier.cluster.dropna().unique())
        for c in present:
            cells[str(int(c))] = cluster_cell(
                tier[tier.cluster == c],
                tier[tier.cluster != c], blen, rng)
        result[label] = {"h_frames": h, "block_bets": blen,
                         "n_dir": int(tier.cluster.notna().sum()),
                         "clusters": cells}
    return result


# -------------------------------------------------------------- profiles
def cluster_profiles():
    c = pd.read_parquet(STRICT_PATH)
    b = pd.read_parquet(
        "artifacts/historical/binance_1y/event_features.parquet",
        columns=["event_time_ms", "rv_1h", "liquidity_1h",
                 "event_intensity_1h", "depth_imbalance_mean"])
    m = c.merge(b, on="event_time_ms", how="left", validate="one_to_one")
    g = m.groupby("cluster")
    return {
        str(int(k)): {
            "frame_share": float(len(v) / len(m)),
            "rv_1h": float(v.rv_1h.mean()),
            "liquidity": float(v.liquidity_1h.mean()),
            "event_intensity": float(v.event_intensity_1h.mean()),
            "depth_imbalance": float(v.depth_imbalance_mean.mean()),
        }
        for k, v in g
    }


def main():
    bets, week_mid = build_bets()
    print(f"bets={len(bets)} directional={int((bets.dir != 0).sum())}",
          flush=True)

    gt = gross_table(bets, week_mid)
    strict = pd.read_parquet(STRICT_PATH)[["event_time_ms", "cluster"]]
    ext = build_extended_labels()

    res_strict = analyze(gt, strict, "strict")
    res_ext = analyze(gt, ext, "extended")

    primary = res_strict[PRIMARY_TIER]["clusters"][str(PRIMARY_CLUSTER)]
    result = {
        "scope": "L2 12-fold direction bets stratified by K=5 slow states",
        "preregistered_primary": {
            "label_set": "strict_oos", "tier": PRIMARY_TIER,
            "cluster": PRIMARY_CLUSTER, **primary,
        },
        "cluster_profiles_oos": cluster_profiles(),
        "strict_oos": res_strict,
        "extended_12weeks": res_ext,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_conditioned_edge.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    # compact console tables
    for tag, res in [("STRICT OOS", res_strict), ("EXTENDED 12w", res_ext)]:
        print(f"\n== {tag} ==")
        for tier in CLOCK_LABELS:
            cell = res[tier]["clusters"]
            line = " | ".join(
                f"c{c}: n={v['n']} gross={1e4*v['gross_mean']:+.1f}bp "
                f"CI[{1e4*v['gross_ci95'][0]:+.1f},{1e4*v['gross_ci95'][1]:+.1f}]"
                f"{' PASS' if v['pass'] else ''}"
                for c, v in cell.items())
            print(f"{tier:>5}: {line}")
    p = result["preregistered_primary"]
    print(f"\nPRIMARY strict/5min/cluster0: gross={1e4*p['gross_mean']:+.1f}bp "
          f"CI[{1e4*p['gross_ci95'][0]:+.1f},{1e4*p['gross_ci95'][1]:+.1f}] "
          f"PASS={p['pass']}")


if __name__ == "__main__":
    main()
