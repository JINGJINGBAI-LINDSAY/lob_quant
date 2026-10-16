"""Multi-scale MEDIUM / LONG window slow features (research-validated).

Key empirical result (Aug-Sep 2026, both symbols):
  At 1h-7d scales the order book is dominated by short-horizon MEAN REVERSION;
  naive momentum / relative position carry negative forward IC. Genuine
  continuation appears only in order-flow ACCELERATION and in persistent
  long-window trade pressure.

Every time scale is expressed in 30s frames and is configurable below (see the
MOM_SCALES / RELPOS_SCALES / PRESSURE_SCALES tables). Each raw slow signal is
z-scored against its own strictly-causal baseline window. Adding or removing a
scale does not change the composite magnitude, because reversion/continuation
average the per-scale z's (scale-count invariant).

This module returns, per 30s row and strictly causal:
  medium_long = [reversion, continuation, ...one z column per scale]
It feeds the direction-agnostic confidence gate only (it never touches the
direction model). Standardization is causal rolling mean/std; refit statistics
externally if a strict train/valid split of the normalizers is required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRAME_SECONDS = 30
CONT_WEIGHT = 1.0

# Scale table: name -> (lookback, lookback_minp, z_baseline, z_baseline_minp).
# All window lengths are in 30s frames. The z_baseline must be >= lookback and
# its minp is kept at <= 2880 (1d) so warmup never discards more than ~1 day.
MOM_SCALES = {
    "1h": (120, 30, 2880, 240),     # original
    "4h": (480, 120, 2880, 240),    # original
    "12h": (1440, 360, 14400, 2880),  # added slower reversion scale (5d base)
}
RELPOS_SCALES = {
    "2h": (240, 60, 2880, 240),     # original
    "8h": (960, 240, 14400, 2880),  # added (5d base)
}
PRESSURE_SCALES = {
    "1d": (2880, 720, 20160, 2880),  # added
    "7d": (20160, 2880, 20160, 2880),  # original
}

# Order-flow acceleration is a single diff; its z baseline is configurable.
DEPTH_ACC_BASELINE = 2880
DEPTH_ACC_MINP = 240


def _mom_col(name: str) -> str:
    return f"ml_z_mom_{name}"


def _relpos_col(name: str) -> str:
    return f"ml_z_relpos_{name}"


def _pressure_col(name: str) -> str:
    return f"ml_z_tradepress_{name}"


_MOM_COLS = [_mom_col(n) for n in MOM_SCALES]
_RELPOS_COLS = [_relpos_col(n) for n in RELPOS_SCALES]
_PRESSURE_COLS = [_pressure_col(n) for n in PRESSURE_SCALES]

MEDIUM_LONG_COLUMNS = (
    ["ml_reversion", "ml_continuation", "ml_composite"]
    + _MOM_COLS + _RELPOS_COLS + ["ml_z_depth_acc"] + _PRESSURE_COLS
)


def _rolling_z(series: pd.Series, baseline: int, min_periods: int) -> pd.Series:
    mean = series.rolling(baseline, min_periods=min_periods).mean()
    std = series.rolling(baseline, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _mean_z(cols: dict[str, pd.Series]) -> pd.Series:
    """Scale-count invariant mean of per-scale z columns."""
    return sum(cols[c] for c in cols) / len(cols)


def build_medium_long(group: pd.DataFrame) -> pd.DataFrame:
    """group: one symbol's 30s frame sorted by event_time_ms."""
    r = group["return_1"]
    p = group["mid_price"]
    high = group["trade_high"]
    low = group["trade_low"]

    # --- Reversion: momentum at multiple scales ---
    mom = {
        _mom_col(name): _rolling_z(
            r.rolling(lookback, min_periods=lmin).sum(), base, bmin
        )
        for name, (lookback, lmin, base, bmin) in MOM_SCALES.items()
    }

    # --- Reversion: relative position in the high-low range ---
    relpos = {}
    for name, (lookback, lmin, base, bmin) in RELPOS_SCALES.items():
        hi = high.rolling(lookback, min_periods=lmin).max()
        lo = low.rolling(lookback, min_periods=lmin).min()
        position = (p - lo) / (hi - lo).replace(0, np.nan) - 0.5
        relpos[_relpos_col(name)] = _rolling_z(position, base, bmin)

    # --- Continuation: persistent trade pressure ---
    pressure = {
        _pressure_col(name): _rolling_z(
            group["trade_imbalance"].rolling(
                lookback, min_periods=lmin
            ).mean(),
            base, bmin,
        )
        for name, (lookback, lmin, base, bmin) in PRESSURE_SCALES.items()
    }

    # --- Continuation: order-flow acceleration (single scale) ---
    depth_acc = group["depth_imbalance_mean"].diff()
    z_depth_acc = _rolling_z(
        depth_acc, DEPTH_ACC_BASELINE, DEPTH_ACC_MINP
    )

    # --- Composites (families averaged over their scales) ---
    reversion = -(_mean_z(mom) + _mean_z(relpos)) / 2.0
    continuation = 0.7 * z_depth_acc + 0.5 * _mean_z(pressure)

    out = {
        "ml_reversion": reversion,
        "ml_continuation": continuation,
        "ml_composite": reversion + CONT_WEIGHT * continuation,
        **mom, **relpos, "ml_z_depth_acc": z_depth_acc, **pressure,
    }
    return pd.DataFrame(out)[MEDIUM_LONG_COLUMNS]
