"""ClickHouse connection for the live stack. Credentials come from the environment (.env), never from code."""
import os

import clickhouse_connect
from dotenv import load_dotenv

from tiq.paths import PROJECT_ROOT

_ENV = PROJECT_ROOT / ".env"
load_dotenv(_ENV if _ENV.exists() else None)  # project-root .env first, else search upwards


def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.environ["CLICKHOUSE_DB"],
    )
