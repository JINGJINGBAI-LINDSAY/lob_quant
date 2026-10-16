from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .core import MVPConfig, assign_labels, write_frame


BASE_URL = "https://data.binance.vision/data/futures/um/daily"
BUCKETS = ((-5.0, "5"), (-4.0, "4"), (-3.0, "3"), (-2.0, "2"),
           (-1.0, "1"), (-0.2, "0p2"), (0.2, "0p2"), (1.0, "1"),
           (2.0, "2"), (3.0, "3"), (4.0, "4"), (5.0, "5"))


def _dates(start: str, end: str) -> list[date]:
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    result = []
    while current < stop:
        result.append(current)
        current += timedelta(days=1)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, destination: Path, attempts: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            partial.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError):
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)


def _download_one(raw_dir: Path, symbol: str, dataset: str, day: date) -> dict:
    filename = f"{symbol}-{dataset}-{day.isoformat()}.zip"
    destination = raw_dir / dataset / symbol / filename
    url = f"{BASE_URL}/{dataset}/{symbol}/{filename}"
    checksum_url = f"{url}.CHECKSUM"
    with urlopen(checksum_url, timeout=60) as response:
        expected = response.read().decode("ascii").split()[0]
    if not destination.exists() or _sha256(destination) != expected:
        _fetch(url, destination)
    actual = _sha256(destination)
    if actual != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch: {destination}")
    return {
        "symbol": symbol,
        "dataset": dataset,
        "date": day.isoformat(),
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": actual,
        "url": url,
    }


def download_archive(raw_dir: str | Path, symbols: list[str],
                     start: str, end: str, workers: int = 8) -> list[dict]:
    raw_dir = Path(raw_dir)
    tasks = [
        (symbol, dataset, day)
        for symbol in symbols
        for dataset in ("bookDepth", "aggTrades")
        for day in _dates(start, end)
    ]
    manifest = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, raw_dir, symbol, dataset, day):
            (symbol, dataset, day)
            for symbol, dataset, day in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            item = future.result()
            manifest.append(item)
            print(
                f"[{completed}/{len(tasks)}] {item['symbol']} "
                f"{item['dataset']} {item['date']}",
                flush=True,
            )
    manifest.sort(key=lambda item: (item["symbol"], item["dataset"], item["date"]))
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _read_depth(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, compression="zip")
    timestamp = pd.to_datetime(source["timestamp"], utc=True)
    source["event_time_ms"] = (
        timestamp.astype("int64") // 10**6 // 30_000 * 30_000
    ).astype("int64")
    pivot = source.pivot_table(
        index="event_time_ms",
        columns="percentage",
        values=["depth", "notional"],
        aggfunc="last",
    )
    # Binance only added the +/-0.2 buckets on 2026-01-15; earlier days lack
    # those columns. Reindex to the full schema so missing levels become NaN
    # instead of raising KeyError (callers may simply exclude 0p2 features).
    percentages = [percentage for percentage, _ in BUCKETS]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product(
            [("depth", "notional"), percentages]
        )
    )
    rows = pd.DataFrame(index=pivot.index)
    for percentage, label in BUCKETS:
        side = "bid" if percentage < 0 else "ask"
        rows[f"{side}_depth_{label}"] = pivot[("depth", percentage)]
        rows[f"{side}_notional_{label}"] = pivot[("notional", percentage)]
    return rows.reset_index()


def _read_trades(path: Path) -> pd.DataFrame:
    source = pd.read_csv(
        path,
        compression="zip",
        usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
    )
    source["event_time_ms"] = (
        source["transact_time"].astype("int64") // 30_000 * 30_000
    )
    source["notional"] = source["price"] * source["quantity"]
    source["buy_quantity"] = np.where(
        ~source["is_buyer_maker"].astype(bool), source["quantity"], 0.0
    )
    source["sell_quantity"] = np.where(
        source["is_buyer_maker"].astype(bool), source["quantity"], 0.0
    )
    source["buy_count"] = (~source["is_buyer_maker"].astype(bool)).astype("int32")
    source["sell_count"] = source["is_buyer_maker"].astype(bool).astype("int32")
    grouped = source.groupby("event_time_ms", sort=True)
    result = grouped.agg(
        trade_last=("price", "last"),
        trade_high=("price", "max"),
        trade_low=("price", "min"),
        trade_quantity=("quantity", "sum"),
        trade_notional=("notional", "sum"),
        buy_quantity=("buy_quantity", "sum"),
        sell_quantity=("sell_quantity", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
    ).reset_index()
    result["trade_vwap"] = result["trade_notional"] / result["trade_quantity"]
    return result


def _base_symbol_frame(raw_dir: Path, symbol: str,
                       start: str, end: str) -> pd.DataFrame:
    parts = []
    for day in _dates(start, end):
        depth_path = (
            raw_dir / "bookDepth" / symbol
            / f"{symbol}-bookDepth-{day.isoformat()}.zip"
        )
        trade_path = (
            raw_dir / "aggTrades" / symbol
            / f"{symbol}-aggTrades-{day.isoformat()}.zip"
        )
        depth = _read_depth(depth_path)
        trades = _read_trades(trade_path)
        merged = depth.merge(trades, on="event_time_ms", how="inner", validate="one_to_one")
        parts.append(merged)
        print(f"converted {symbol} {day.isoformat()}: {len(merged)} rows", flush=True)
    frame = pd.concat(parts, ignore_index=True).sort_values("event_time_ms")
    return frame.drop_duplicates("event_time_ms", keep="last").reset_index(drop=True)


def _build_features(base: pd.DataFrame, symbol: str, cfg: MVPConfig,
                    start: str, end: str) -> pd.DataFrame:
    frame = base.copy()
    frame.insert(0, "symbol", symbol)
    frame["event_type"] = "depth_trade_30s"
    frame["mid_price"] = frame["trade_last"]
    epsilon = cfg.epsilon
    labels = ("0p2", "1", "2", "3", "4", "5")
    for label in labels:
        bid_depth = frame[f"bid_depth_{label}"]
        ask_depth = frame[f"ask_depth_{label}"]
        total_depth = bid_depth + ask_depth
        frame[f"bid_depth_log_{label}"] = np.log1p(bid_depth)
        frame[f"ask_depth_log_{label}"] = np.log1p(ask_depth)
        frame[f"bid_notional_log_{label}"] = np.log1p(frame[f"bid_notional_{label}"])
        frame[f"ask_notional_log_{label}"] = np.log1p(frame[f"ask_notional_{label}"])
        frame[f"depth_imbalance_{label}"] = (
            bid_depth - ask_depth
        ) / (total_depth + epsilon)
        frame[f"depth_change_{label}"] = (
            bid_depth.diff().fillna(0) - ask_depth.diff().fillna(0)
        ) / (total_depth.shift(1).fillna(total_depth) + epsilon)
    total_quantity = frame["buy_quantity"] + frame["sell_quantity"]
    total_count = frame["buy_count"] + frame["sell_count"]
    frame["trade_imbalance"] = (
        frame["buy_quantity"] - frame["sell_quantity"]
    ) / (total_quantity + epsilon)
    frame["trade_count_imbalance"] = (
        frame["buy_count"] - frame["sell_count"]
    ) / (total_count + epsilon)
    frame["trade_vwap_return"] = (
        frame["trade_vwap"] / frame["mid_price"].shift(1) - 1
    ).fillna(0)
    frame["trade_range"] = (
        (frame["trade_high"] - frame["trade_low"]) / frame["trade_vwap"]
    ).fillna(0)
    frame["log_trade_notional"] = np.log1p(frame["trade_notional"])
    frame["log_trade_count"] = np.log1p(total_count)
    frame["return_1"] = frame["mid_price"].pct_change().fillna(0)
    inner_lag = (
        frame["bid_depth_0p2"] + frame["ask_depth_0p2"]
    ).shift(1).bfill()
    signed_trade = frame["buy_quantity"] - frame["sell_quantity"]
    frame["ofi_proxy"] = (
        frame["depth_change_0p2"] + signed_trade / (inner_lag + epsilon)
    )
    rolling_mean = frame["trade_notional"].rolling(120, min_periods=2).mean()
    rolling_std = frame["trade_notional"].rolling(120, min_periods=2).std()
    frame["volume_zscore"] = (
        (frame["trade_notional"] - rolling_mean)
        / rolling_std.replace(0, np.nan)
    ).fillna(0)
    frame["ofi_jump"] = frame["ofi_proxy"].diff().fillna(0)
    frame["price_change"] = frame["return_1"]
    rolling_count_median = total_count.rolling(120, min_periods=1).median()
    frame["trade_burst"] = (total_count > rolling_count_median).astype(float)
    frame["delta_event_time_ms"] = (
        frame["event_time_ms"].diff().fillna(0).clip(lower=0)
    )

    frame["rv"] = frame["return_1"].pow(2).rolling(120, min_periods=1).sum().pow(.5)
    frame["liquidity"] = np.log1p(
        frame["bid_notional_0p2"] + frame["ask_notional_0p2"]
    )
    frame["depth_imbalance_mean"] = (
        frame["depth_imbalance_0p2"].rolling(120, min_periods=1).mean()
    )
    frame["event_intensity"] = total_count / 30.0
    frame["trade_imbalance_mean"] = (
        frame["trade_imbalance"].rolling(120, min_periods=1).mean()
    )
    index = pd.to_datetime(frame["event_time_ms"], unit="ms", utc=True)
    indexed = frame.set_index(index)
    for label, duration in (("1h", "1h"), ("1d", "1d"), ("7d", "7d")):
        frame[f"rv_{label}"] = (
            indexed["return_1"].pow(2).rolling(duration, min_periods=1).sum().pow(.5)
            .to_numpy()
        )
        frame[f"liquidity_{label}"] = (
            indexed["liquidity"].rolling(duration, min_periods=1).mean().to_numpy()
        )
        frame[f"event_intensity_{label}"] = (
            indexed["event_intensity"].rolling(duration, min_periods=1).mean().to_numpy()
        )

    future = (
        frame["mid_price"].shift(-1)
        .rolling(cfg.horizon, min_periods=cfg.horizon).mean()
        .shift(-(cfg.horizon - 1))
    )
    frame["return_h50"] = (future - frame["mid_price"]) / frame["mid_price"]
    warmup = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=cfg.long_window_days)
    stop = pd.Timestamp(end, tz="UTC")
    timestamp = pd.to_datetime(frame["event_time_ms"], unit="ms", utc=True)
    frame = frame.loc[(timestamp >= warmup) & (timestamp < stop)]
    required = cfg.feature_columns + cfg.regime_columns + ["return_h50"]
    return frame.dropna(subset=required).reset_index(drop=True)


def build_dataset(raw_dir: str | Path, output_dir: str | Path,
                  symbols: list[str], start: str, end: str,
                  cfg: MVPConfig) -> dict:
    if cfg.feature_schema != "binance_depth_buckets":
        raise ValueError(
            "Binance archive conversion requires "
            "feature_schema=binance_depth_buckets")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    frames = [
        _build_features(
            _base_symbol_frame(raw_dir, symbol, start, end),
            symbol, cfg, start, end,
        )
        for symbol in symbols
    ]
    combined = pd.concat(frames, ignore_index=True)
    train_parts = [
        group.iloc[:max(0, int(len(group) * cfg.train_fraction) - cfg.horizon)]
        for _, group in combined.groupby("symbol", sort=False)
    ]
    threshold = float(pd.concat(train_parts)["return_h50"].abs().quantile(.7))
    combined = assign_labels(combined, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = write_frame(combined, output_dir / "event_features.parquet")
    metadata = {
        "source": "Binance Vision official USD-M Futures daily archive",
        "source_url": BASE_URL,
        "venue": "Binance USD-M Futures",
        "symbols": symbols,
        "archive_interval": {"start_inclusive": start, "end_exclusive": end},
        "usable_interval": {
            "start_inclusive": (
                pd.Timestamp(start) + pd.Timedelta(days=cfg.long_window_days)
            ).date().isoformat(),
            "end_exclusive": end,
        },
        "base_resolution": "30 seconds",
        "depth_semantics": "real cumulative depth/notional within +/-0.2,1,2,3,4,5 percent",
        "price_semantics": "last aggregate-trade price in each 30-second bin; not quote mid",
        "label": {
            "column": "return_h50",
            "definition": f"mean price over next {cfg.horizon} 30-second rows versus current price",
            "horizon_minutes": cfg.horizon * .5,
            "threshold_train_abs_return_q70": threshold,
        },
        "context": {
            "short": f"{cfg.short_events} x 30-second rows",
            "medium": f"{cfg.medium_bars} x {cfg.medium_bar_minutes}-minute bars",
            "long": ["rolling 1h", "rolling 1d", "rolling 7d"],
            "warmup_days_excluded": cfg.long_window_days,
        },
        "rows": {
            symbol: int((combined["symbol"] == symbol).sum()) for symbol in symbols
        },
        "features": str(feature_path),
        "config": asdict(cfg),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "build", "all"))
    parser.add_argument("--config", default="config/binance_202609.json")
    parser.add_argument("--raw-dir", default="artifacts/historical/binance_futures_202609/raw")
    parser.add_argument("--output-dir", default="artifacts/historical/binance_futures_202609")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-09-29")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = MVPConfig.from_json(args.config)
    if args.command in {"download", "all"}:
        download_archive(args.raw_dir, args.symbols, args.start, args.end, args.workers)
    if args.command in {"build", "all"}:
        build_dataset(
            args.raw_dir, args.output_dir, args.symbols, args.start, args.end, cfg
        )


if __name__ == "__main__":
    main()
