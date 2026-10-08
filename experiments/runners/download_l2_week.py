"""Download one week of raw multi-level L2 data for BTCUSDT.

Source: CryptoHFTData free per-hour parquet (no API key for the download
endpoint). Interval: 2026-08-27 .. 2026-09-02 inclusive (168 hours).
Files are saved as
  {out}/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet
Already-present, non-trivial files are skipped; failures are retried.
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

from urllib.request import Request, urlopen

OUT = Path("artifacts/historical/l2_btc_week")
START = date(2026, 8, 27)
DAYS = 7
MIN_BYTES = 1_000_000  # a real hour is ~16-26MB


def hour_url(day: date, hour: int) -> str:
    key = f"binance_futures/{day.isoformat()}/{hour:02d}/BTCUSDT_orderbook.parquet"
    return "https://api.cryptohftdata.com/download?file=" + quote(key, safe="")


def _valid_parquet(dest: Path) -> bool:
    try:
        import pyarrow.parquet as _pq
        _pq.ParquetFile(dest).metadata
        return True
    except Exception:  # noqa: BLE001
        return False


def _maybe_decode(data: bytes) -> bytes:
    """The endpoint may serve a zstd-compressed body; transparently inflate.

    A zstd frame begins with magic 0xFD2FB528 (bytes 28 B5 2F FD). The
    inflated body is the real parquet starting with PAR1.
    """
    if data[:4] == b"(\xb5/\xfd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data)
    return data


def _fetch_once(url: str, read_timeout: int, wall_timeout: float) -> bytes:
    """Download one URL in a worker thread with a hard wall-clock watchdog.

    `urlopen(timeout=...)` only bounds individual socket operations; on a
    stalled-but-open connection it can hang indefinitely. Running the blocking
    read in a daemon thread and joining with `wall_timeout` guarantees the
    attempt is abandoned after that many real seconds even if the socket never
    times out.
    """
    box: dict = {}

    def work() -> None:
        try:
            req = Request(url, headers={"User-Agent": "l2-downloader"})
            with urlopen(req, timeout=read_timeout) as response:
                box["data"] = response.read()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(wall_timeout)
    if thread.is_alive():
        raise TimeoutError(f"hard watchdog tripped after {wall_timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["data"]


def download_one(dest: Path, url: str, attempts: int = 4,
                 read_timeout: int = 120, wall_timeout: float = 120):
    """Download one hour. Returns bytes, or None if all attempts failed.

    Never raises: a single bad hour must not abort the whole week. The caller
    retries failed hours after finishing the rest. Accepts a download only if
    it is a real, readable parquet (guards against truncated/error bodies and
    transparently inflates a zstd-compressed response). A hard watchdog stops
    an indefinitely stalled socket regardless of the socket-level timeout.
    """
    if dest.exists() and dest.stat().st_size > MIN_BYTES and _valid_parquet(dest):
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        tmp = dest.with_suffix(".parquet.part")
        try:
            data = _fetch_once(url, read_timeout, wall_timeout)
            data = _maybe_decode(data)
            if len(data) < MIN_BYTES:
                raise IOError(f"file too small: {len(data)}")
            tmp.write_bytes(data)
            if not _valid_parquet(tmp):
                raise IOError("downloaded body is not a valid parquet")
            tmp.replace(dest)
            return len(data)
        except Exception:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 15))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START.isoformat(),
                        help="first day YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--passes", type=int, default=20,
                        help="max full retries of any failed hours")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent download threads")
    parser.add_argument("--attempts", type=int, default=4,
                        help="per-hour attempts inside a pass")
    parser.add_argument("--read-timeout", type=int, default=120)
    parser.add_argument("--wall-timeout", type=float, default=120,
                        help="hard per-attempt wall-clock watchdog (seconds)")
    args = parser.parse_args()
    out = Path(args.out)
    start = date.fromisoformat(args.start)

    total = args.days * 24
    all_hours = [(start + timedelta(days=d), h)
                 for d in range(args.days) for h in range(24)]

    def dest_of(day: date, hour: int) -> Path:
        return out / day.isoformat() / f"{hour:02d}" / "BTCUSDT_orderbook.parquet"

    # Skip hours already on disk as valid parquet (resume, never re-download).
    pending, completed = [], 0
    for day, hour in all_hours:
        dest = dest_of(day, hour)
        if dest.exists() and dest.stat().st_size > MIN_BYTES and _valid_parquet(dest):
            completed += 1
        else:
            pending.append((day, hour))
    print(f"start: {completed}/{total} already present, "
          f"{len(pending)} to fetch, {args.workers} workers", flush=True)

    for pass_no in range(1, args.passes + 1):
        still_failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for day, hour in pending:
                futures[pool.submit(
                    download_one, dest_of(day, hour), hour_url(day, hour),
                    args.attempts, args.read_timeout, args.wall_timeout
                )] = (day, hour)
            for fut in as_completed(futures):
                day, hour = futures[fut]
                size = fut.result()
                if size is None:
                    still_failed.append((day, hour))
                else:
                    completed += 1
                    print(f"[{completed}/{total}] {day} {hour:02d}h "
                          f"{size/1e6:.1f}MB", flush=True)
        pending = still_failed
        if not pending:
            print("done")
            return
        print(f"pass {pass_no}: {len(pending)} hours still failing, retrying",
              flush=True)
    raise SystemExit(f"giving up; {len(pending)} hours failed: "
                     f"{[(d.isoformat(), h) for d, h in pending]}")


if __name__ == "__main__":
    main()
