"""ClickHouse table for live trades. Safe to run repeatedly (IF NOT EXISTS)."""

TICKS_DDL = """
CREATE TABLE IF NOT EXISTS ticks
(
    ts              DateTime64(3, 'UTC')  CODEC(DoubleDelta, ZSTD(1)),
    symbol          LowCardinality(String),
    trade_id        UInt64                CODEC(Delta, ZSTD(1)),
    price           Float64               CODEC(Gorilla, ZSTD(1)),
    qty             Float64               CODEC(Gorilla, ZSTD(1)),
    buyer_is_maker  Bool
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (symbol, ts, trade_id)
"""


def init_db(client) -> None:
    client.command(TICKS_DDL)
