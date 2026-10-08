# LOB Order-Flow MVP

面向 BTCUSDT / ETHUSDT 的可回放订单流建模 MVP。默认闭环：

1. depth update + trade JSONL -> top20 order book
2. event features + 100-event window -> h=50 label
3. majority / logistic baseline
4. Regime-aware Event Gate + Transformer encoder
5. Macro-F1、AUC、IC 和可解释的 event gate 输出

## Data rule

正式采集、训练和评估只使用 Binance 真实行情。代码不再提供 demo 运行入口。
原始输入是 `type=depth` 或 `type=trade` 的 JSONL，字段定义见
`lob_mvp/core.py`。

## Collect real Binance data

The collector uses Binance's public WebSocket and does not require API keys:

```bash
pip install -r requirements.txt
python3 -m lob_mvp.collector --symbols BTCUSDT ETHUSDT --seconds 3600 \
  --output artifacts/live/events.jsonl
python3 -m lob_mvp.cli ingest --events artifacts/live/events.jsonl \
  --output artifacts/live/dataset
python3 -m lob_mvp.cli train \
  --config config/default.json \
  --features artifacts/live/dataset/features/event_features.csv \
  --output-dir artifacts/live/run
```

The stream uses `depth20@100ms` and `aggTrade`, records exchange and local
receive timestamps, and writes replayable JSONL. Then convert each symbol's
events with `build_features()` and train using the resulting feature file.
`depth20` is a top-20 partial-book feed, not L3 order-level data.

当前时间尺度配置为：短期 1 小时、中期 1 天、长期 1 周。中期分支最多保留
500 个事件进入模型，长期分支使用 1h/1d/7d rolling statistics；不把整天或整周
的原始事件直接塞入 Transformer。

## Configuration and Bash

模型和数据参数集中在 [config/default.json](config/default.json)：

```json
{
  "window": 100,
  "short_events": 100,
  "medium_bar_minutes": 5,
  "medium_bars": 288,
  "short_window_hours": 1,
  "medium_window_days": 1,
  "long_window_days": 7,
  "horizon": 50,
  "d_model": 128,
  "n_layers": 2,
  "n_heads": 4,
  "d_ff": 256,
  "dropout": 0.1,
  "batch_size": 256,
  "learning_rate": 0.0002,
  "epochs": 3
}
```

完整真实数据流程可以直接用：

```bash
chmod +x run_live.sh
SECONDS_TO_COLLECT=3600 ./run_live.sh config/default.json
```

也可以分步运行：

```bash
python3 -m lob_mvp.collector --symbols BTCUSDT ETHUSDT --seconds 3600 \
  --output artifacts/live/events.jsonl
python3 -m lob_mvp.cli ingest --config config/default.json \
  --events artifacts/live/events.jsonl --output artifacts/live/dataset
python3 -m lob_mvp.cli train --config config/default.json \
  --features artifacts/live/dataset/features/event_features.csv \
  --output-dir artifacts/live/run
```

Parquet is preferred when `pyarrow` is installed. CSV is used as a portable fallback
for local smoke tests. Install production dependencies with:

```bash
pip install -r requirements.txt
```
