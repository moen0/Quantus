"""ClickHouse -> Polars -> Panel. Aggregate live ticks into OHLCV bars inside the database.

`load_live_panel` turns collected ticks into the same Panel the research engine uses, so a
strategy can be backtested on your own recent data with exactly the code used on the archive.
"""
import polars as pl

from tiq.data.binance_vision import INTERVAL_SECONDS
from tiq.data.panel import Panel, build_panel
from tiq.live.clickhouse import get_client

BARS_SQL = """
SELECT
    toDateTime64(toStartOfInterval(ts, INTERVAL {seconds} SECOND), 3, 'UTC') AS bar,
    argMin(price, (ts, trade_id))                    AS open,
    max(price)                                       AS high,
    min(price)                                       AS low,
    argMax(price, (ts, trade_id))                    AS close,
    sum(qty)                                         AS volume,
    sum(price * qty)                                 AS quote_volume,
    count()                                          AS n_trades
FROM ticks
WHERE symbol = %(symbol)s
  AND ts >= toDateTime64(%(start)s, 3, 'UTC')
  AND ts <  toDateTime64(%(end)s,   3, 'UTC')
GROUP BY bar
ORDER BY bar
"""


def load_bars(symbol: str, seconds: int, start: str, end: str) -> pl.DataFrame:
    """OHLCV bars for one symbol. start/end are UTC strings such as '2026-10-14 00:00:00'."""
    sql = BARS_SQL.format(seconds=int(seconds))  # int() makes the interpolation injection-safe
    # query_arrow: columnar transfer, no row-by-row Python objects
    table = get_client().query_arrow(sql, parameters={"symbol": symbol, "start": start, "end": end})
    return pl.from_arrow(table)


def load_live_panel(symbols: list[str], interval: str, start: str, end: str) -> Panel:
    """Bars from collected ticks as a Panel (interval must be a Binance interval such as '1m' or '1h')."""
    frames = {}
    for sym in symbols:
        bars = load_bars(sym, INTERVAL_SECONDS[interval], start, end)
        frames[sym] = bars.rename({"bar": "open_time"}).select(
            "open_time", "open", "high", "low", "close", "volume", "quote_volume")
    return build_panel(frames, interval, start.replace(" ", "T"), end.replace(" ", "T"))
