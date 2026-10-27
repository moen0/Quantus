"""Live ingest batching (no network, no database) and CLI helpers."""
import asyncio

import pytest

from tiq.cli import build_parser, parse_cpus

ingest = pytest.importorskip("tiq.live.ingest", reason="needs the [live] extra")


def _row(i):
    return (None, "BTCUSDT", i, 100.0, 1.0, False)


async def _run_writer(rows, monkeypatch, batch_max=1000, flush=0.05, stop=True):
    monkeypatch.setattr(ingest, "BATCH_MAX", batch_max)
    monkeypatch.setattr(ingest, "FLUSH_SECONDS", flush)
    batches = []
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(ingest.writer(queue, lambda b: batches.append(list(b))))
    for r in rows:
        await queue.put(r)
    await asyncio.sleep(flush * 3)
    if stop:
        await queue.put(ingest._STOP)
    await asyncio.wait_for(task, 2)
    return batches


def test_writer_batches_by_size(monkeypatch):
    batches = asyncio.run(_run_writer([_row(i) for i in range(25)], monkeypatch, batch_max=10))
    assert [len(b) for b in batches] == [10, 10, 5]


def test_writer_flushes_on_time_and_on_stop(monkeypatch):
    batches = asyncio.run(_run_writer([_row(i) for i in range(3)], monkeypatch))
    assert sum(len(b) for b in batches) == 3 and [r[2] for b in batches for r in b] == [0, 1, 2]


def test_parse_trade_uses_trade_time_in_utc():
    row = ingest.parse_trade({"T": 1700000000123, "s": "BTCUSDT", "t": 42, "p": "37000.5", "q": "0.01", "m": True})
    assert row[0].tzinfo is not None and row[0].isoformat().startswith("2023-11-14T22:13:20.123")
    assert row[1:] == ("BTCUSDT", 42, 37000.5, 0.01, True)


def test_cli_helpers():
    assert parse_cpus("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    args = build_parser().parse_args(["run", "experiments/01_donchian_btc.toml", "--quick"])
    assert args.quick and args.func.__name__ == "cmd_run"
