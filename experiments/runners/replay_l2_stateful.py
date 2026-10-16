"""Stateful replay of raw L2 updates into 30s book snapshots.

Correctly maintains the resting book: apply every incremental update in event
order to a price->quantity map per side, and at each 30s boundary snapshot the
top LEVELS. State is carried across hour files so the book stays continuous.
"""
from __future__ import annotations

import argparse
import heapq
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
    if not bids_book or not asks_book:
        return None
    # Single scan per side: take top LEVELS, detect crossed from the tops and
    # only then heal (rare) and recompute. All dict entries have qty > 0.
    bid_prices = heapq.nlargest(LEVELS, bids_book)
    ask_prices = heapq.nsmallest(LEVELS, asks_book)
    if bid_prices and ask_prices and bid_prices[0] >= ask_prices[0]:
        HEAL_REMOVED.append(heal_crossed(bids_book, asks_book))
        bid_prices = heapq.nlargest(LEVELS, bids_book)
        ask_prices = heapq.nsmallest(LEVELS, asks_book)
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


def process_file(path, bids_book, asks_book, rows):
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


def _apply_bin_groups(book, groups, b):
    gp = groups.get(b)
    if gp is None:
        return
    pos = gp.loc[gp["qty"] > 0]
    if len(pos):
        book.update(zip(pos["price"].tolist(), pos["qty"].tolist()))
    for p in gp.loc[gp["qty"] == 0, "price"].tolist():
        book.pop(p, None)


def process_file_vec(path, bids_book, asks_book, rows, state=None):
    """Vectorized replay producing identical state to the per-event loop.

    Two facts let us avoid the per-event Python loop:
      * an update at a price REPLACES the quantity (never accumulates), so
        within one 30s bin only the LAST update per (bin, side, price) can
        matter -> reduce the hour in C and bulk-merge each bin;
      * a frame for bin b is emitted in the event loop only when the first
        later event (bin > b) crosses the boundary. We therefore hold the
        latest frame pending (`state`) and flush it only when a strictly
        later bin appears. Events from a new file that land on an earlier
        bin refresh the pending frame, and the final bin with no follower
        is never emitted -- exactly the loop's behaviour.

    `state` carries the pending frame across hour files.
    """
    if state is None:
        state = {"bin": None, "row": None}
    df = pd.read_parquet(path, columns=["event_time", "side", "price", "quantity"])
    df = df.sort_values("event_time", kind="mergesort")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["qty"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["bin"] = df["event_time"].astype("int64") // BIN_MS

    last = df.drop_duplicates(subset=["bin", "side", "price"], keep="last")
    bid_groups = {b: gp[["price", "qty"]] for b, gp in
                  last.loc[last["side"] == "bid"].groupby("bin", sort=True)}
    ask_groups = {b: gp[["price", "qty"]] for b, gp in
                  last.loc[last["side"] == "ask"].groupby("bin", sort=True)}

    first_bin = int(df["bin"].iloc[0])
    last_bin = int(df["bin"].iloc[-1])
    for b in range(first_bin, last_bin + 1):
        # A strictly later bin triggers the pending boundary frame, which is
        # emitted before applying the new bin's updates.
        if state["bin"] is not None and b > state["bin"]:
            if state["row"] is not None:
                rows.append(state["row"])
            state["bin"], state["row"] = None, None
        _apply_bin_groups(bids_book, bid_groups, b)
        _apply_bin_groups(asks_book, ask_groups, b)
        if state["bin"] is None:
            # Most advanced bin seen: this frame is now pending.
            state["bin"], state["row"] = b, snapshot(bids_book, asks_book, b * BIN_MS)
        else:
            # Late event on an earlier, not-yet-flushed bin: the pending
            # boundary frame must reflect it.
            state["row"] = snapshot(bids_book, asks_book, state["bin"] * BIN_MS)


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
    parser.add_argument("--engine", choices=("loop", "vec"), default="vec",
                        help="per-event loop or exact vectorized replay")
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    files = sorted(src.glob("*/*/BTCUSDT_orderbook.parquet"))
    print("files", len(files), "engine", args.engine, flush=True)
    process = process_file if args.engine == "loop" else process_file_vec
    bids_book, asks_book, rows = {}, {}, []
    pending = {"bin": None, "row": None}
    for i, path in enumerate(files):
        if args.engine == "vec":
            process(path, bids_book, asks_book, rows, pending)
        else:
            process(path, bids_book, asks_book, rows)
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
