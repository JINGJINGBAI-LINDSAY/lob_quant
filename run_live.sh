#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/default.json}"
SECONDS_TO_COLLECT="${SECONDS_TO_COLLECT:-3600}"
ROOT="artifacts/live/$(date +%Y%m%d_%H%M%S)"
EVENTS="$ROOT/events.jsonl"
DATASET="$ROOT/dataset"
RUN="$ROOT/run"

mkdir -p "$ROOT"
python3 -m lob_mvp.collector \
  --symbols BTCUSDT ETHUSDT \
  --seconds "$SECONDS_TO_COLLECT" \
  --output "$EVENTS"

python3 -m lob_mvp.cli ingest \
  --config "$CONFIG" \
  --events "$EVENTS" \
  --output "$DATASET"

FEATURES="$DATASET/features/event_features.csv"

python3 -m lob_mvp.cli train \
  --config "$CONFIG" \
  --features "$FEATURES" \
  --output-dir "$RUN"

echo "completed: $RUN"
