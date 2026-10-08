"""Stateful replay of raw L2 updates into 30s book snapshots.

Correctly maintains the resting book: apply every incremental update in event
order to a price->quantity map per side, and at each 30s boundary snapshot the
top LEVELS. State is carried across hour files so the book stays continuous.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = 10
BIN_MS = 30_000
SRC = Path("artifacts/historical/l2_btc_week")
OUT = Path("artifacts/historical/l2_btc_week_features")


HEAL_REMOVED = []   # number of stale prices purged at each healed boundary


def heal_crossed(bids_book, asks_book):
    """Self-heal a crossed reconstruction.

    The incremental stream can miss the removal of a resting order that was
    matched in reality, leaving a stale price that crosses the other side.
    Such prices would have been filled, so purge them: delete every bid >= best
    ask, then every ask <= the resulting best bid. Mutates the books and
    returns how many stale prices were removed.
    """
    removed = 0
    if not bids_book or not asks_book:
        return 0
    pa = min(p for p, q in asks_book.items() if q > 0)
    for p in [p for p, q in bids_book.items() if q > 0 and p >= pa]:
        bids_book.pop(p, None)
        removed += 1
    live_bids = [p for p, q in bids_book.items() if q > 0]
    if not live_bids:
        return removed
    pb = max(live_bids)
    for p in [p for p, q in asks_book.items() if q > 0 and p <= pb]:
        asks_book.pop(p, None)
        removed += 1
    return removed


def snapshot(bids_book, asks_book, bin_start):
    if any(q > 0 for q in bids_book.values()) and \
            any(q > 0 for q in asks_book.values()):
        pb_now = max(p for p, q in bids_book.items() if q > 0)
        pa_now = min(p for p, q in asks_book.items() if q > 0)
        if pb_now >= pa_now:
            HEAL_REMOVED.append(heal_crossed(bids_book, asks_book))

    bid_prices = sorted(
        (p for p, q in bids_book.items() if q > 0), reverse=True
    )[:LEVELS]
    ask_prices = sorted(
        (p for p, q in asks_book.items() if q > 0)
    )[:LEVELS]
    if not bid_prices or not ask_prices:
        return None
    pb, pa = bid_prices[0], ask_prices[0]
    if pb >= pa:
        return None  # still crossed after healing: give up on this frame
    mid = (pb + pa) / 2
    row = {"bin": bin_start, "mid": mid,
           "spread_bps": (pa - pb) / mid * 1e4}
    for i in range(LEVELS):
        if i < len(bid_prices):
            p = bid_prices[i]
            row[f"bid_dist_{i}"] = (mid - p) / mid * 1e4
            row[f"bid_qty_{i}"] = bids_book[p]
        if i < len(ask_prices):
            p = ask_prices[i]
            row[f"ask_dist_{i}"] = (p - mid) / mid * 1e4
            row[f"ask_qty_{i}"] = asks_book[p]
    return row


def process_file(path, bids_book, asks_book, rows, first_bin_aligned):
    df = pd.read_parquet(path, columns=["event_time", "side", "price", "quantity"])
    df = df.sort_values("event_time", kind="mergesort")
    times = df["event_time"].to_numpy()
    sides = df["side"].to_numpy()
    prices = df["price"].to_numpy()
    qtys = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).to_numpy()

    next_bin = None
    n = len(df)
    for i in range(n):
        t = int(times[i])
        current_bin = (t // BIN_MS) * BIN_MS
        if next_bin is None:
            next_bin = current_bin + BIN_MS
        # Snapshot at every boundary crossed before applying this update.
        while t >= next_bin:
            row = snapshot(bids_book, asks_book, next_bin - BIN_MS)
            if row is not None:
                rows.append(row)
            next_bin += BIN_MS
        book = bids_book if sides[i] == "bid" else asks_book
        q = float(qtys[i])
        price = float(prices[i])
        if q > 0:
            book[price] = q
        else:
            book.pop(price, None)


def add_imbalance_and_ofi(frames):
    for i in range(LEVELS):
        b, a = frames[f"bid_qty_{i}"], frames[f"ask_qty_{i}"]
        frames[f"imb_{i}"] = (b - a) / (b + a).replace(0, np.nan)
    frames["imb_top5"] = sum(
        frames[f"imb_{i}"] for i in range(min(5, LEVELS))
    ) / min(5, LEVELS)
    frames["imb_1"] = frames["imb_0"]
    frames["ofi_1"] = (
        frames["bid_qty_0"].diff().fillna(0)
        - frames["ask_qty_0"].diff().fillna(0)
    )
    frames["return_1"] = frames["mid"].pct_change().fillna(0)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = sorted(src.glob("*/*/BTCUSDT_orderbook.parquet"))
    print("files", len(files), flush=True)
    bids_book, asks_book, rows = {}, {}, []
    for i, path in enumerate(files):
        process_file(path, bids_book, asks_book, rows, i == 0)
        if (i + 1) % 12 == 0:
            print(f"processed {i+1}/{len(files)} files, snapshots={len(rows)}",
                  flush=True)
    frames = pd.DataFrame(rows).sort_values("bin")
    frames = frames.drop_duplicates("bin", keep="last").reset_index(drop=True)
    # Ensure level columns exist and fill.
    for i in range(LEVELS):
        for side in ("bid", "ask"):
            qc, dc = f"{side}_qty_{i}", f"{side}_dist_{i}"
            if qc not in frames:
                frames[qc] = 0.0
            if dc not in frames:
                frames[dc] = np.nan
            frames[qc] = frames[qc].fillna(0.0)
            frames[dc] = frames[dc].ffill().bfill()
    frames = add_imbalance_and_ofi(frames)
    out.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out / "l2_30s_features.parquet", index=False)
    healed = [n for n in HEAL_REMOVED if n > 0]
    print(f"healed boundaries={len(healed)} stale-prices-removed={sum(healed)}")
    print("frames", len(frames))
    print(frames[["mid", "spread_bps", "imb_1", "imb_top5"]].describe()
          .loc[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
