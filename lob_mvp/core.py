from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MVPConfig:
    feature_schema: str = "lob_levels"
    levels: int = 20
    # Percentage buckets used by the depth-BUCKET schema (symmetric, both
    # sides). This is NOT a number of price levels: the real per-level schema
    # uses `levels`; bucket schema aggregates depth at these % distances.
    depth_buckets_pct: Tuple[float, ...] = (0.2, 1, 2, 3, 4, 5)
    short_events: int = 100
    medium_bar_minutes: int = 5
    medium_bars: int = 288
    sample_stride: int = 5
    short_window_hours: int = 1
    medium_window_days: int = 1
    long_window_days: int = 7
    horizon: int = 50
    epsilon: float = 1e-8
    seed: int = 7
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 3
    train_fraction: float = 0.7
    class_weight_power: float = 0.5
    use_medium_context: bool = True
    use_regime_context: bool = True
    use_event_gate: bool = True
    fusion_type: str = "gated_add"

    @classmethod
    def from_json(cls, path: str | Path) -> "MVPConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        # Backward compatibility: old name for the bucket schema.
        if values.get("feature_schema") == "binance_depth":
            values["feature_schema"] = "binance_depth_buckets"
        return cls(**values)

    @property
    def window(self) -> int:
        return self.short_events

    @property
    def depth_bucket_labels(self) -> List[str]:
        """Column label for each percentage bucket, e.g. 0.2 -> '0p2', 1 -> '1'."""
        return [("%g" % p).replace(".", "p") for p in self.depth_buckets_pct]

    @property
    def feature_columns(self) -> List[str]:
        if self.feature_schema == "binance_depth_buckets":
            buckets = self.depth_bucket_labels
            columns = []
            for kind in ("depth_log", "notional_log"):
                for side in ("bid", "ask"):
                    columns.extend(f"{side}_{kind}_{bucket}" for bucket in buckets)
            columns.extend(f"depth_imbalance_{bucket}" for bucket in buckets)
            columns.extend(f"depth_change_{bucket}" for bucket in buckets)
            return columns + [
                "trade_imbalance", "trade_count_imbalance",
                "trade_vwap_return", "trade_range", "log_trade_notional",
                "log_trade_count", "return_1", "ofi_proxy",
                "volume_zscore", "ofi_jump", "price_change", "trade_burst",
                "delta_event_time_ms",
            ]
        cols = []
        for side in ("bid_price_dist", "ask_price_dist", "bid_size_log", "ask_size_log"):
            cols.extend(f"{side}_{i}" for i in range(1, self.levels + 1))
        return cols + [
            "spread_norm", "imbalance_1", "ofi_1", "trade_imbalance",
            "delta_event_time_ms",
            "event_add", "event_cancel", "event_trade", "event_depth_update",
            "side_signed", "cancel", "volume_zscore", "ofi_jump", "spread_change",
            "event_burst",
        ]

    @property
    def regime_columns(self) -> List[str]:
        if self.feature_schema == "binance_depth_buckets":
            return [
                "rv", "liquidity", "depth_imbalance_mean",
                "event_intensity", "trade_imbalance_mean",
                "rv_1h", "rv_1d", "rv_7d",
                "liquidity_1h", "liquidity_1d", "liquidity_7d",
                "event_intensity_1h", "event_intensity_1d",
                "event_intensity_7d",
            ]
        return ["rv", "liquidity", "spread_mean", "depth_imbalance_mean",
                "event_intensity", "cancel_rate", "rv_1h", "rv_1d", "rv_7d",
                "liquidity_1h", "liquidity_1d", "liquidity_7d",
                "event_intensity_1h", "event_intensity_1d", "event_intensity_7d"]

    @property
    def prior_columns(self) -> List[str]:
        if self.feature_schema == "binance_depth_buckets":
            return [
                "volume_zscore", "ofi_jump", "price_change",
                "trade_imbalance", "trade_burst",
            ]
        return ["volume_zscore", "ofi_jump", "spread_change", "cancel", "event_burst"]

    @property
    def book_state_columns(self) -> List[str]:
        columns = ["symbol", "event_time_ms", "update_id", "best_bid", "best_ask",
                   "mid_price", "spread"]
        for side in ("bid_price", "bid_size", "ask_price", "ask_size"):
            columns.extend(f"{side}_{i}" for i in range(1, self.levels + 1))
        return columns


def _book_frame(book: Dict[str, float], levels: int) -> Tuple[np.ndarray, np.ndarray]:
    bids = sorted(((-p, s) for (p, s) in book.items() if p < 0), reverse=True)
    asks = sorted(((p, s) for (p, s) in book.items() if p > 0))
    return np.array(bids[:levels], dtype=float), np.array(asks[:levels], dtype=float)


def _apply_delta(book: Dict[float, float], updates: Iterable[Iterable[str]], side: int) -> None:
    for price, size in updates:
        price, size = float(price), float(size)
        key = side * price
        if size <= 0:
            book.pop(key, None)
        else:
            book[key] = size


def _event_features(events: List[dict], cfg: MVPConfig) -> pd.DataFrame:
    book: Dict[float, float] = {}
    rows: List[dict] = []
    previous_ofi = previous_time = None
    previous_trade = 0.0
    for event in events:
        if event["type"] == "depth":
            if event.get("book_snapshot"):
                book.clear()
            _apply_delta(book, event.get("bids_delta", []), -1)
            _apply_delta(book, event.get("asks_delta", []), 1)
        bids, asks = _book_frame(book, cfg.levels)
        if len(bids) == 0 or len(asks) == 0:
            continue
        bid_p, ask_p = bids[0, 0], asks[0, 0]
        mid, spread = (ask_p - bid_p) / 2 + bid_p, ask_p - bid_p
        bid_sizes = np.pad(bids[:, 1], (0, cfg.levels - len(bids)))
        ask_sizes = np.pad(asks[:, 1], (0, cfg.levels - len(asks)))
        bid_prices = np.pad(bids[:, 0], (0, cfg.levels - len(bids)), constant_values=bid_p)
        ask_prices = np.pad(asks[:, 0], (0, cfg.levels - len(asks)), constant_values=ask_p)
        if event["type"] == "trade":
            signed = float(event.get("size", 0.0)) * (1 if event.get("side") == "buy" else -1)
            previous_trade = signed
            event_kind, side_signed = "trade", np.sign(signed)
        else:
            signed = 0.0
            event_kind = event.get("event_type", "depth_update")
            side_signed = 0.0
        ofi = float(bid_sizes[0] - ask_sizes[0]) + signed
        updates = event.get("bids_delta", []) + event.get("asks_delta", [])
        event_volume = abs(float(event.get("size", 0.0))) + sum(abs(float(size)) for _, size in updates)
        row = {
            "symbol": event["symbol"], "event_time_ms": int(event["event_time_ms"]),
            "update_id": event.get("final_update_id"),
            "event_type": event_kind,
            "best_bid": bid_p, "best_ask": ask_p,
            "mid_price": mid, "spread": spread, "ofi_raw": ofi,
            "trade_signed": previous_trade, "cancel": int(event.get("event_type") == "cancel"),
            "event_volume": event_volume, "side_signed": side_signed,
        }
        for kind in ("add", "cancel", "trade", "depth_update"):
            row[f"event_{kind}"] = int(event_kind == kind)
        for i in range(cfg.levels):
            row[f"bid_price_{i+1}"] = bid_prices[i]
            row[f"ask_price_{i+1}"] = ask_prices[i]
            row[f"bid_size_{i+1}"] = bid_sizes[i]
            row[f"ask_size_{i+1}"] = ask_sizes[i]
            row[f"bid_price_dist_{i+1}"] = (bid_prices[i] - mid) / mid
            row[f"ask_price_dist_{i+1}"] = (ask_prices[i] - mid) / mid
            row[f"bid_size_log_{i+1}"] = np.log1p(bid_sizes[i])
            row[f"ask_size_log_{i+1}"] = np.log1p(ask_sizes[i])
        k_bid, k_ask = bid_sizes[:5].sum(), ask_sizes[:5].sum()
        row["spread_norm"] = spread / mid
        row["imbalance_1"] = (bid_sizes[0] - ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0] + cfg.epsilon)
        row["ofi_1"] = ofi - (previous_ofi or 0.0)
        row["trade_imbalance"] = signed / (abs(signed) + cfg.epsilon)
        row["delta_event_time_ms"] = 0 if previous_time is None else max(0, int(event["event_time_ms"]) - previous_time)
        row["_depth_imbalance"] = (k_bid - k_ask) / (k_bid + k_ask + cfg.epsilon)
        rows.append(row)
        previous_ofi, previous_time = ofi, int(event["event_time_ms"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("event_time_ms").reset_index(drop=True)
    frame["return_1"] = frame["mid_price"].pct_change().fillna(0.0)
    frame["rv"] = frame["return_1"].rolling(cfg.window, min_periods=1).apply(
        lambda x: float(np.sqrt(np.square(x).sum())), raw=True)
    frame["liquidity"] = frame[[f"bid_size_log_{i}" for i in range(1, 6)] +
                               [f"ask_size_log_{i}" for i in range(1, 6)]].sum(axis=1)
    frame["spread_mean"] = frame["spread_norm"].rolling(cfg.window, min_periods=1).mean()
    frame["depth_imbalance_mean"] = frame["_depth_imbalance"].rolling(cfg.window, min_periods=1).mean()
    elapsed = frame["event_time_ms"].rolling(cfg.window, min_periods=2).apply(lambda x: max(1.0, x[-1] - x[0]), raw=True)
    frame["event_intensity"] = cfg.window / elapsed.replace(0, np.nan)
    frame["event_intensity"] = frame["event_intensity"].fillna(0.0)
    frame["cancel_rate"] = frame["cancel"].rolling(cfg.window, min_periods=1).mean()
    vol_mean = frame["event_volume"].rolling(cfg.window, min_periods=1).mean()
    vol_std = frame["event_volume"].rolling(cfg.window, min_periods=1).std().replace(0, np.nan)
    frame["volume_zscore"] = ((frame["event_volume"] - vol_mean) / vol_std).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    frame["ofi_jump"] = frame["ofi_1"].diff().fillna(0.0)
    frame["spread_change"] = frame["spread_norm"].diff().fillna(0.0)
    intensity_median = frame["event_intensity"].rolling(cfg.window, min_periods=1).median()
    frame["event_burst"] = (frame["event_intensity"] > intensity_median).astype(float)
    time_index = pd.to_datetime(frame["event_time_ms"], unit="ms")
    timed = frame.set_index(time_index)
    for label, duration in (("1h", "1h"), ("1d", "1d"), ("7d", "7d")):
        frame[f"rv_{label}"] = timed["return_1"].pow(2).rolling(duration, min_periods=1).sum().pow(0.5).to_numpy()
        frame[f"liquidity_{label}"] = timed["liquidity"].rolling(duration, min_periods=1).mean().to_numpy()
        counts = timed["event_time_ms"].rolling(duration, min_periods=1).count()
        frame[f"event_intensity_{label}"] = (counts / (pd.Timedelta(duration).total_seconds())).to_numpy()
    return frame


def build_features(events: Iterable[dict], cfg: MVPConfig = MVPConfig()) -> pd.DataFrame:
    """Build event features and h=50 labels using future mean mid-price."""
    frame = _event_features(list(events), cfg)
    if frame.empty:
        return frame
    future = frame["mid_price"].shift(-1).rolling(cfg.horizon, min_periods=cfg.horizon).mean().shift(-(cfg.horizon - 1))
    frame["return_h50"] = (future - frame["mid_price"]) / frame["mid_price"]
    frame = frame.dropna(subset=["return_h50"]).reset_index(drop=True)
    # Threshold is deliberately computed by the caller on the training split.
    frame["label_h50"] = 1
    frame.attrs["config"] = asdict(cfg)
    return frame


def assign_labels(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    r = frame["return_h50"]
    frame["label_h50"] = np.select([r > threshold, r < -threshold], [2, 0], default=1).astype(int)
    return frame


def build_orderflow_bars(frame: pd.DataFrame, cfg: MVPConfig) -> pd.DataFrame:
    """Compress event features into fixed-time order-flow bars."""
    bar_columns = cfg.feature_columns
    parts = []
    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        values = symbol_frame.sort_values("event_time_ms").copy()
        values.index = pd.to_datetime(values["event_time_ms"], unit="ms")
        aggregations = {}
        for column in bar_columns:
            if column not in values:
                continue
            aggregations[column] = "sum" if column in {
                "ofi_1", "ofi_jump", "event_add", "event_cancel", "event_trade",
                "event_depth_update", "cancel", "event_volume", "ofi_proxy",
                "log_trade_notional", "log_trade_count",
            } else "mean"
        bars = values.resample(f"{cfg.medium_bar_minutes}min").agg(aggregations)
        bars["symbol"] = symbol
        bars["event_time_ms"] = (bars.index.view("int64") // 10**6).astype("int64")
        anchor = "trade_imbalance" if cfg.feature_schema == "binance_depth_buckets" else "spread_norm"
        bars = bars.dropna(subset=[anchor]).reset_index(drop=True)
        parts.append(bars)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def generate_synthetic_events(symbol: str = "BTCUSDT", count: int = 900, seed: int = 7) -> List[dict]:
    """Deterministic depth/trade stream for an offline end-to-end smoke test."""
    rng = np.random.default_rng(seed)
    mid = 63800.0 if symbol == "BTCUSDT" else 3200.0
    events = []
    for i in range(count):
        mid += float(rng.normal(0, mid * 0.00002))
        spread = max(0.01, mid * 0.00003)
        bid, ask = mid - spread / 2, mid + spread / 2
        depth = {"type": "depth", "symbol": symbol, "event_time_ms": i * 25,
                 "event_type": "cancel" if i % 11 == 0 else "depth_update",
                 "bids_delta": [[str(bid - j * spread), str(abs(rng.normal(0.8, 0.25)))] for j in range(20)],
                 "asks_delta": [[str(ask + j * spread), str(abs(rng.normal(0.8, 0.25)))] for j in range(20)]}
        events.append(depth)
        if i % 3 == 0:
            events.append({"type": "trade", "symbol": symbol, "event_time_ms": i * 25 + 1,
                           "side": "buy" if rng.random() > 0.48 else "sell",
                           "size": float(abs(rng.normal(0.35, 0.15)))})
    return events


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ModuleNotFoundError):
        path = path.with_suffix(".csv")
        frame.to_csv(path, index=False)
    return path


def write_dataset_artifacts(events: List[dict], frame: pd.DataFrame,
                            output_dir: str | Path, cfg: MVPConfig) -> dict:
    """Persist the documented raw/book_state/features/metadata layers."""
    output_dir = Path(output_dir)
    raw_dir, state_dir, feature_dir, metadata_dir = (
        output_dir / "raw", output_dir / "book_state",
        output_dir / "features", output_dir / "metadata"
    )
    for directory in (raw_dir, state_dir, feature_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_events(events, raw_dir / "events.jsonl")
    state_columns = [c for c in cfg.book_state_columns if c in frame.columns]
    state = frame[state_columns].copy()
    state_path = write_frame(state, state_dir / "book_state.parquet")
    feature_columns = ["symbol", "event_time_ms", "event_type", "return_h50",
                       "label_h50"] + cfg.feature_columns + cfg.regime_columns
    feature_columns = [c for c in feature_columns if c in frame.columns]
    features = frame[feature_columns].copy()
    feature_path = write_frame(features, feature_dir / "event_features.parquet")
    metadata = {
        "config": asdict(cfg),
        "schemas": {"raw": "raw/events.jsonl", "book_state": state_columns,
                    "event_features": feature_columns},
        "rows": {"raw_events": len(events), "book_state": len(state),
                 "event_features": len(features)},
        "symbols": sorted(frame["symbol"].dropna().unique().tolist()),
        "label_stats": frame["label_h50"].value_counts(dropna=False).to_dict()
        if "label_h50" in frame else {},
    }
    metadata_path = metadata_dir / "dataset.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"raw": str(raw_dir / "events.jsonl"), "book_state": str(state_path),
            "features": str(feature_path), "metadata": str(metadata_path)}


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def save_events(events: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
