"""Binance trade stream -> ClickHouse: batched inserts, automatic reconnect, gap detection.

Run with:  python -m tiq.live.ingest     (needs `pip install -e ".[live]"` and a running ClickHouse)
"""
import asyncio
import json
import logging
import time
from datetime import UTC, datetime

import websockets

from tiq.live.clickhouse import get_client
from tiq.live.schema import init_db

log = logging.getLogger("ingest")

SYMBOLS = ["btcusdt", "ethusdt"]
WS_URL = "wss://stream.binance.com:9443/stream?streams=" + "/".join(f"{s}@trade" for s in SYMBOLS)
COLUMNS = ["ts", "symbol", "trade_id", "price", "qty", "buyer_is_maker"]
BATCH_MAX = 50_000     # rows per INSERT (upper bound)
FLUSH_SECONDS = 5.0    # longest a row waits in memory (= most you can lose on a hard crash)
_STOP = object()


def parse_trade(msg: dict) -> tuple:
    """Binance trade event -> row matching COLUMNS. T = trade time (ms), m = buyer is the maker."""
    return (
        datetime.fromtimestamp(msg["T"] / 1000, tz=UTC),
        msg["s"],
        msg["t"],
        float(msg["p"]),
        float(msg["q"]),
        bool(msg["m"]),
    )


async def reader(queue: asyncio.Queue) -> None:
    """Receive trades forever; reconnect with exponential backoff; warn about missing trade ids."""
    last_id: dict[str, int] = {}
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL, close_timeout=2) as ws:
                log.info("connected: %s", ", ".join(SYMBOLS))
                backoff = 1
                async for raw in ws:
                    row = parse_trade(json.loads(raw)["data"])
                    symbol, trade_id = row[1], row[2]
                    prev = last_id.get(symbol)
                    if prev is not None and trade_id != prev + 1:
                        log.warning("gap in %s: trade_id %d -> %d", symbol, prev, trade_id)
                    last_id[symbol] = trade_id
                    await queue.put(row)
        except (websockets.ConnectionClosed, OSError) as exc:
            log.warning("connection lost (%s); reconnecting in %ds", exc, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


async def writer(queue: asyncio.Queue, sink) -> None:
    """Collect rows into batches (size- or time-triggered) and hand them to the blocking sink in a thread."""
    while True:
        first = await queue.get()
        if first is _STOP:
            return
        batch, stop = [first], False
        deadline = time.monotonic() + FLUSH_SECONDS
        while len(batch) < BATCH_MAX:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                row = await asyncio.wait_for(queue.get(), left)
            except TimeoutError:
                break
            if row is _STOP:
                stop = True
                break
            batch.append(row)
        await asyncio.to_thread(sink, batch)
        if stop:
            return


def make_sink():
    """A blocking function that inserts one batch into ClickHouse, retrying with backoff."""
    client = get_client()
    init_db(client)

    def sink(rows: list) -> None:
        delay = 1
        for attempt in range(1, 6):
            try:
                client.insert("ticks", rows, column_names=COLUMNS)
                log.debug("inserted %d rows", len(rows))
                return
            except Exception as exc:  # log every failure; re-raise after the last attempt
                log.error("insert failed (attempt %d/5): %s", attempt, exc)
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2

    return sink


async def run(sink) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=100_000)
    tasks = [asyncio.create_task(reader(queue)), asyncio.create_task(writer(queue, sink))]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        tasks[0].cancel()                  # stop reading
        if not tasks[1].done():
            await queue.put(_STOP)         # let the writer flush whatever is still in memory
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:                 # reached only if a task failed on its own
        if isinstance(result, Exception):
            raise result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run(make_sink()))
    except KeyboardInterrupt:
        pass
