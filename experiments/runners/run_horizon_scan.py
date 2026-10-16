"""Multi-scale horizon scan: PnL of the 12-fold L2 direction bets.

Two time conventions share the SAME bets, costs and statistical engine:
  A fixed-clock : exit after h 30s frames (mid at t+h).
  B event-time  : exit after N aggTrades (mean of N trade prices vs entry mid).

Bets come from the already-trained 12-fold OOS predictions
(pooled_oos_predictions.csv). Signal timestamps are rebuilt deterministically
by replaying the identical week/window pipeline, and the rebuilt test counts
are asserted against the pooled CSV before any PnL is computed.

Only directional bets (model_pred up/down) are taken; neutral -> no position.
Statistics: per-week moving blocks (block = return span in bet units) nested in
a week-level (regime) resample. Pass rule fixed up front:
  mean net > 0, 95% CI lower bound > 0, >= 9/12 weeks positive, and gross edge
  median above cost.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runners.run_l2_level_experiment import (
    STRIDE, build_arrays, compress_tails, drop_warmup_frames,
    make_windows,
)
from experiments.runners.run_l2_multiweek import WEEKS

POOLED = Path(
    "artifacts/historical/l2_multiweek_expanding/pooled_oos_predictions.csv"
)
AGG_DIR = Path("artifacts/historical/binance_1y/raw/aggTrades/BTCUSDT")
OUT = Path("artifacts/historical/horizon_scan")

CLOCK_FRAMES = [10, 60, 240, 480, 960]
CLOCK_LABELS = ["5min", "30min", "2h", "4h", "8h"]
CLOCK_SECONDS = [300, 1800, 7200, 14400, 28800]
COSTS = {"7bp": 0.0007, "2bp": 0.0002}
R_BOOT = 10_000
SEED = 20261009

# Event-time calibration: 12 evenly spaced strictly pre-test days.
CALIB_START = date(2025, 10, 8)
CALIB_END = date(2026, 4, 22)
CALIB_N = 12
# Load each week plus this many trailing days so exits after week-end resolve.
WEEK_BUFFER_DAYS = 8


# ---------------------------------------------------------------- timestamp
def week_test_bets(path):
    """Return per-test-bet (end_idx, ts_ms, entry_mid) in pooled row order."""
    frame = pd.read_parquet(path).sort_values("bin").reset_index(drop=True)
    frame, _dropped = drop_warmup_frames(frame)
    lv, gl, fwd = build_arrays(frame)
    lv, gl = compress_tails(lv, gl)
    _L, _G, f, ends = make_windows(lv, gl, fwd)
    defined = np.isfinite(f)
    ends = ends[defined]
    i70 = int(len(ends) * 0.70)
    te_ends = ends[i70:]
    bin_ = frame["bin"].to_numpy(np.int64)
    mid_ = frame["mid"].to_numpy(np.float64)
    return {
        "end": te_ends,
        "ts": bin_[te_ends],
        "entry_mid": mid_[te_ends],
        "mid": mid_,
        "n_frames": len(frame),
    }


def build_bets():
    pooled = pd.read_csv(POOLED)
    parts = []
    week_mid = {}
    for k, (name, path) in enumerate(WEEKS):
        if not path.exists():
            raise SystemExit(f"missing week features: {path}")
        w = week_test_bets(path)
        rows = pooled[pooled.week == name].reset_index(drop=True)
        if len(rows) != len(w["end"]):
            raise SystemExit(
                f"week {name}: pooled {len(rows)} != rebuilt {len(w['end'])}"
            )
        direction = np.where(rows.model_pred == 2, 1,
                             np.where(rows.model_pred == 0, -1, 0))
        parts.append(pd.DataFrame({
            "week": name, "fold": k,
            "end": w["end"], "ts": w["ts"],
            "entry_mid": w["entry_mid"], "dir": direction,
        }))
        week_mid[name] = (w["mid"], w["n_frames"])
    bets = pd.concat(parts, ignore_index=True)
    return bets, week_mid


# ---------------------------------------------------------------- stats
def block_resample(n, blen, rng):
    nb = int(np.ceil(n / blen))
    starts = rng.integers(0, max(1, n - blen + 1), nb)
    idx = (starts[:, None] + np.arange(blen)[None, :]).ravel()
    return idx[idx < n]


def summarize(gross_by_week, blen):
    """gross_by_week: list of gross arrays (directional bets only)."""
    rng = np.random.default_rng(SEED)
    n_weeks = len(gross_by_week)
    out = {}
    for cname, cost in COSTS.items():
        net = [g - cost for g in gross_by_week]
        means = np.empty(R_BOOT)
        for b in range(R_BOOT):
            wk = rng.integers(0, n_weeks, n_weeks)
            vals = []
            for w in wk:
                x = net[w]
                if len(x) == 0:
                    continue
                vals.append(x[block_resample(len(x), blen, rng)])
            flat = np.concatenate(vals) if vals else np.array([0.0])
            means[b] = flat.mean()
        lo, hi = np.quantile(means, [0.025, 0.975])
        point_net = np.concatenate(net)
        point_gross = np.concatenate(gross_by_week)
        weeks_pos = int(sum(int(x.mean() - cost > 0)
                            for x in gross_by_week if len(x)))
        out[cname] = {
            "n": int(len(point_net)),
            "gross_mean": float(point_gross.mean()),
            "gross_median": float(np.median(point_gross)),
            "net_mean": float(point_net.mean()),
            "ci95": [float(lo), float(hi)],
            "p_net_gt_0": float((means > 0).mean()),
            "weeks_positive": weeks_pos,
            "pass": bool(point_net.mean() > 0 and lo > 0
                         and weeks_pos >= 9
                         and np.median(point_gross) > cost),
        }
    return out


# ---------------------------------------------------------------- A clock
def run_clock(bets, week_mid, frames):
    result = {}
    bets_d = bets[bets.dir != 0]
    for h, label in zip(frames, CLOCK_LABELS):
        gross_week = []
        for name, _path in WEEKS:
            mid, n_frames = week_mid[name]
            b = bets_d[bets_d.week == name]
            end = b.end.to_numpy()
            valid = end + h < n_frames
            end = end[valid]
            entry = b.entry_mid.to_numpy()[valid]
            exit_ = mid[end + h]
            d = b.dir.to_numpy()[valid]
            gross_week.append(d * (exit_ / entry - 1.0))
        blen = max(1, int(np.ceil(h / STRIDE)))
        result[label] = {"h_frames": h, "block_bets": blen,
                         **summarize(gross_week, blen)}
    return result


# ---------------------------------------------------------------- B events
def _read_day_trades(day: date):
    path = AGG_DIR / f"BTCUSDT-aggTrades-{day.isoformat()}.zip"
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    return pd.read_csv(io.BytesIO(raw),
                       usecols=["agg_trade_id", "price", "transact_time"])


def calibrate_event_n():
    """Map each clock tier to N events using the median pre-test event rate."""
    span = (CALIB_END - CALIB_START).days
    days = [CALIB_START + timedelta(days=round(i * span / (CALIB_N - 1)))
            for i in range(CALIB_N)]
    counts = np.array([len(_read_day_trades(d)) for d in days])
    rate = float(np.median(counts) / 86400.0)
    n_events = {
        label: int(round(rate * sec))
        for label, sec in zip(CLOCK_LABELS, CLOCK_SECONDS)
    }
    return n_events, {"days": [d.isoformat() for d in days],
                      "day_counts": counts.tolist(),
                      "events_per_sec_median": rate}


def week_event_series(week_name: str):
    start = date.fromisoformat(week_name)
    days = [start + timedelta(days=i)
            for i in range(7 + WEEK_BUFFER_DAYS)]
    frames = []
    for d in days:
        path = AGG_DIR / f"BTCUSDT-aggTrades-{d.isoformat()}.zip"
        if path.exists():  # trailing buffer beyond local archive: skip
            frames.append(_read_day_trades(d))
    if not frames:
        raise SystemExit(f"no trade files for week {week_name}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["transact_time", "agg_trade_id"],
                        kind="mergesort").reset_index(drop=True)
    t = df["transact_time"].to_numpy(np.int64)
    p = df["price"].to_numpy(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(p)])
    return t, p, csum


def run_event(bets):
    n_events, calib = calibrate_event_n()
    print(f"[calib] rate={calib['events_per_sec_median']:.2f}/s "
          f"N={n_events}", flush=True)

    bets_d = bets[bets.dir != 0]
    tier_gross = {l: [] for l in CLOCK_LABELS}
    tier_dur = {l: [] for l in CLOCK_LABELS}
    for name, _path in WEEKS:
        t, _p, csum = week_event_series(name)
        b = bets_d[bets_d.week == name]
        ts = b.ts.to_numpy()
        entry = b.entry_mid.to_numpy()
        d = b.dir.to_numpy()
        e0 = np.searchsorted(t, ts, side="left")
        for label in CLOCK_LABELS:
            N = n_events[label]
            ok = e0 + N <= len(t)
            e = e0[ok]
            mean_exit = (csum[e + N] - csum[e]) / N
            gross = d[ok] * (mean_exit / entry[ok] - 1.0)
            tier_gross[label].append(gross)
            tier_dur[label].append(t[np.minimum(e + N - 1, len(t) - 1)]
                                   - ts[ok])

    result = {}
    for label in CLOCK_LABELS:
        dur_ms = np.concatenate(tier_dur[label])
        p95 = float(np.quantile(dur_ms, 0.95))
        blen = max(1, int(np.ceil(p95 / 300_000.0)))  # bets are 5min apart
        result[label] = {
            "N_events": n_events[label], "block_bets": blen,
            "dur_median_min": float(np.median(dur_ms) / 60_000),
            "dur_p95_min": float(p95 / 60_000),
            **summarize(tier_gross[label], blen),
        }
    return result, calib


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--only", default=None, help="single tier label (pipeline check)")
    args = ap.parse_args()

    bets, week_mid = build_bets()
    ndir = int((bets.dir != 0).sum())
    print(f"bets={len(bets)} directional={ndir} weeks={bets.week.nunique()}",
          flush=True)

    if args.mode == "A":
        frames = CLOCK_FRAMES
        if args.only:
            j = CLOCK_LABELS.index(args.only)
            frames = [frames[j]]
        res = run_clock(bets, week_mid, frames)
        result = {"mode": "fixed_clock", "tiers": res}
    else:
        res, calib = run_event(bets)
        result = {"mode": "event_time", "calibration": calib,
                  "tiers": res}

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.mode if not args.only else f"{args.mode}_{args.only}"
    (OUT / f"scan_{tag}.json").write_text(json.dumps(result, indent=2),
                                          encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
