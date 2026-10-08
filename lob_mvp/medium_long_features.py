"""Reusable MEDIUM / LONG window feature construction (research-validated).

Key empirical result (Aug-Sep 2026, both symbols):
  At 1h-7d scales the order book is dominated by short-horizon MEAN REVERSION;
  naive momentum / relative position carry negative forward IC. Genuine
  continuation appears only in order-flow ACCELERATION and in persistent
  long-window trade pressure.

This module returns, per 30s row and strictly causal:
  medium_long = [reversion, continuation, ...subcomponents]
It can be concatenated to event features or used as context to modulate the
short encoder. Standardization uses rolling means/std (train-only behaviour is
preserved because the transforms are causal; refit statistics externally if a
strict train/valid split of the normalizers is required).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONT_WEIGHT = 1.0


def _rolling_z(series: pd.Series, baseline: int, min_periods: int) -> pd.Series:
    mean = series.rolling(baseline, min_periods=min_periods).mean()
    std = series.rolling(baseline, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def build_medium_long(group: pd.DataFrame) -> pd.DataFrame:
    """group: one symbol's 30s frame sorted by event_time_ms."""
    r = group["return_1"]
    p = group["mid_price"]
    high = group["trade_high"]
    low = group["trade_low"]

    # --- Reversion inputs (1-4h) ---
    mom_1h = r.rolling(120, min_periods=30).sum()
    mom_4h = r.rolling(480, min_periods=120).sum()
    high_2h = high.rolling(240, min_periods=60).max()
    low_2h = low.rolling(240, min_periods=60).min()
    relpos_2h = (
        (p - low_2h) / (high_2h - low_2h).replace(0, np.nan) - 0.5
    )

    z_mom1 = _rolling_z(mom_1h, 2880, 240)
    z_mom4 = _rolling_z(mom_4h, 2880, 240)
    z_pos = _rolling_z(relpos_2h, 2880, 240)
    reversion = -(z_mom1 + 0.5 * z_mom4 + z_pos) / 2.5

    # --- Continuation inputs ---
    depth_acc = group["depth_imbalance_mean"].diff()
    z_depth_acc = _rolling_z(depth_acc, 2880, 240)
    trade_pressure_7d = group["trade_imbalance"].rolling(
        20160, min_periods=2880
    ).mean()
    z_trade_persist = _rolling_z(trade_pressure_7d, 20160, 2880)
    continuation = 0.7 * z_depth_acc + 0.5 * z_trade_persist

    return pd.DataFrame({
        "ml_reversion": reversion,
        "ml_continuation": continuation,
        "ml_composite": reversion + CONT_WEIGHT * continuation,
        "ml_z_mom1h": z_mom1,
        "ml_z_mom4h": z_mom4,
        "ml_z_relpos2h": z_pos,
        "ml_z_depth_acc": z_depth_acc,
        "ml_z_trade_persist7d": z_trade_persist,
    })


MEDIUM_LONG_COLUMNS = [
    "ml_reversion", "ml_continuation", "ml_composite",
    "ml_z_mom1h", "ml_z_mom4h", "ml_z_relpos2h",
    "ml_z_depth_acc", "ml_z_trade_persist7d",
]
