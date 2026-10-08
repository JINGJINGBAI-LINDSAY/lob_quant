"""Binance public market-data collector.

Uses partial-book snapshots rather than L3 order events. Each depth message is
stored as a top-20 book update and can be replayed by the existing pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets


def _streams(symbols):
    return "/".join(
        f"{symbol.lower()}@depth20@100ms/{symbol.lower()}@aggTrade"
        for symbol in symbols
    )


async def collect(symbols, seconds, output):
    url = "wss://stream.binance.com:9443/stream?streams=" + _streams(symbols)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + seconds
    count = 0
    with output.open("a", encoding="utf-8") as file:
        while time.monotonic() < deadline:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_size=8 * 1024 * 1024) as socket:
                    while time.monotonic() < deadline:
                        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
                        payload = message.get("data", message)
                        stream_symbol = message.get("stream", "").split("@", 1)[0].upper()
                        symbol = payload.get("s", stream_symbol)
                        if not symbol:
                            continue
                        receive_time_ms = int(time.time() * 1000)
                        if "bids" in payload and "asks" in payload:
                            event = {
                                "type": "depth", "symbol": symbol,
                                "book_snapshot": True,
                                "event_time_ms": int(payload.get("E", receive_time_ms)),
                                "receive_time_ms": receive_time_ms,
                                "final_update_id": payload.get("u", payload.get("lastUpdateId")),
                                "bids_delta": payload["bids"],
                                "asks_delta": payload["asks"],
                            }
                        elif payload.get("e") == "aggTrade":
                            event = {
                                "type": "trade", "symbol": symbol,
                                "event_time_ms": int(payload["T"]),
                                "receive_time_ms": receive_time_ms,
                                "side": "sell" if payload["m"] else "buy",
                                "size": payload["q"],
                                "price": payload["p"],
                            }
                        else:
                            continue
                        file.write(json.dumps(event) + "\n")
                        count += 1
                        if count % 1000 == 0:
                            file.flush()
            except Exception:
                if time.monotonic() >= deadline:
                    break
                file.flush()
                await asyncio.sleep(min(10, max(1, deadline - time.monotonic())))
    return count


def main():
    parser = argparse.ArgumentParser(description="Collect Binance public LOB data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--output", default="artifacts/live/events.jsonl")
    args = parser.parse_args()
    count = asyncio.run(collect(args.symbols, args.seconds, args.output))
    print(json.dumps({"events": count, "output": args.output, "symbols": args.symbols}))


if __name__ == "__main__":
    main()
