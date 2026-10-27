"""Live data stack (optional): Binance trade stream -> ClickHouse -> bars -> Panel.

Install the extra first: pip install -e ".[live]". Modules import clickhouse_connect and
websockets, so they are not imported by the research code.
"""
