"""Shared test helpers: synthetic panels and kline frames. No network, no lake on disk unless tmp_path."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from tiq.data.binance_vision import INTERVAL_SECONDS, KLINE_SCHEMA
from tiq.data.panel import Panel

START = datetime(2020, 1, 1, tzinfo=UTC)


def make_panel(close, interval: str = "1d", symbols=None, open_=None, volume=None, tradable=None,
               start: datetime = START) -> Panel:
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    T, N = close.shape
    if open_ is None:
        open_ = np.vstack([close[:1], close[:-1]])  # open = previous close (24/7 market, no gaps)
        open_ = np.where(np.isfinite(open_), open_, close)
    open_ = np.asarray(open_, dtype=float).reshape(T, N)
    volume = np.where(np.isfinite(close), 1e6, np.nan) if volume is None else np.asarray(volume, float).reshape(T, N)
    if tradable is None:
        tradable = np.isfinite(close) & (np.nan_to_num(volume) > 0)
    step = INTERVAL_SECONDS[interval] * 1000
    index = int(start.timestamp() * 1000) + step * np.arange(T, dtype=np.int64)
    return Panel(index=index, symbols=tuple(symbols or [f"C{j}USDT" for j in range(N)]), interval=interval,
                 open=open_, high=np.fmax(open_, close), low=np.fmin(open_, close), close=close,
                 volume=volume, quote_volume=volume * close, tradable=np.asarray(tradable, bool))


def random_walk(T: int, N: int = 1, seed: int = 0, drift: float = 0.0, vol: float = 0.03,
                start_price: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start_price * np.exp(np.cumsum(rng.normal(drift, vol, (T, N)), axis=0))


def kline_frame(closes, start: datetime = START, interval: str = "1d", volume: float = 1000.0) -> pl.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    step = INTERVAL_SECONDS[interval] * 1000
    ts = int(start.timestamp() * 1000) + step * np.arange(n, dtype=np.int64)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pl.DataFrame({
        "open_time": pl.Series(ts).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "open": opens, "high": np.maximum(opens, closes) * 1.01, "low": np.minimum(opens, closes) * 0.99,
        "close": closes, "volume": np.full(n, volume), "quote_volume": volume * closes,
        "trades": np.full(n, 100, dtype=np.int64), "taker_buy_volume": np.full(n, volume / 2),
        "taker_buy_quote_volume": volume * closes / 2,
    }).select([pl.col(c).cast(t) for c, t in KLINE_SCHEMA.items()])


@pytest.fixture
def rw_panel() -> Panel:
    return make_panel(random_walk(1500, 3, seed=1, drift=0.0005))
