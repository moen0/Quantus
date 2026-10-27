"""Backfill against a fake archive served by httpx.MockTransport: planning, resuming, verification."""
import hashlib
import io
import zipfile
from datetime import UTC, date

import httpx

from tiq.data import binance_vision as bv
from tiq.data.backfill import backfill_klines
from tiq.data.lake import Lake


def _zip(name: str, csv: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv)
    return buf.getvalue()


def _day_rows(month: str, days: range) -> bytes:
    from datetime import datetime
    rows = []
    for d in days:
        ts = int(datetime(int(month[:4]), int(month[5:7]), d, tzinfo=UTC).timestamp() * 1000)
        rows.append(f"{ts},100,101,99,100.5,10,{ts + 86_399_999},1005,7,5,502,0")
    return ("\n".join(rows) + "\n").encode()


class FakeArchive:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.corrupt: set[str] = set()

    def add(self, key: str, csv: bytes) -> None:
        z = _zip(key.rsplit("/", 1)[-1].replace(".zip", ".csv"), csv)
        self.files[key] = z
        self.files[key + ".CHECKSUM"] = f"{hashlib.sha256(z).hexdigest()}  x.zip".encode()

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.startswith(bv.LIST_URL):
            prefix = request.url.params["prefix"]
            keys = sorted(k for k in self.files if k.startswith(prefix))
            body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
            xml = (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false'
                   f"</IsTruncated>{body}</ListBucketResult>")
            return httpx.Response(200, content=xml.encode())
        key = url[len(bv.DATA_URL) + 1:]
        if key not in self.files:
            return httpx.Response(404)
        data = self.files[key]
        if key in self.corrupt:
            data = data[:-1] + b"X"
        return httpx.Response(200, content=data)


def _setup():
    fake = FakeArchive()
    for m in ("2024-08", "2024-09"):
        fake.add(bv.kline_key("spot", "AAAUSDT", "1d", m), _day_rows(m, range(1, bv.days_in_month(m) + 1)))
    for d in (1, 2, 3):  # current month only has daily files so far
        fake.add(bv.kline_key("spot", "AAAUSDT", "1d", f"2024-10-0{d}"), _day_rows("2024-10", range(d, d + 1)))
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return fake, client


def test_backfill_monthly_and_daily_then_resume(tmp_path):
    fake, client = _setup()
    lake = Lake(tmp_path)
    today = date(2024, 10, 4)
    rep = backfill_klines(["AAAUSDT"], "1d", lake=lake, client=client, today=today, workers=2)
    assert rep.written == 3 and not rep.failures
    df = lake.read_klines("spot", "1d", "AAAUSDT")
    assert df.height == 31 + 30 + 3
    manifest = lake.read_manifest(lake.kline_dir("spot", "1d", "AAAUSDT"))
    assert manifest["2024-10"]["source"] == "daily" and manifest["2024-10"]["days"] == [
        "2024-10-01", "2024-10-02", "2024-10-03"]

    # nothing new: second run downloads nothing
    n = len(fake.requests)
    rep = backfill_klines(["AAAUSDT"], "1d", lake=lake, client=client, today=today, workers=2)
    assert rep.written == 0 and rep.skipped == 3
    assert not any(u.endswith(".zip") for u in fake.requests[n:])

    # a new daily file appears: only the current month is rebuilt
    fake.add(bv.kline_key("spot", "AAAUSDT", "1d", "2024-10-04"), _day_rows("2024-10", range(4, 5)))
    rep = backfill_klines(["AAAUSDT"], "1d", lake=lake, client=client, today=date(2024, 10, 5), workers=2)
    assert rep.written == 1
    assert lake.read_klines("spot", "1d", "AAAUSDT").height == 31 + 30 + 4

    # the monthly file is published: it replaces the daily build
    fake.add(bv.kline_key("spot", "AAAUSDT", "1d", "2024-10"), _day_rows("2024-10", range(1, 32)))
    rep = backfill_klines(["AAAUSDT"], "1d", lake=lake, client=client, today=date(2024, 11, 2), workers=2)
    assert rep.written == 1
    manifest = lake.read_manifest(lake.kline_dir("spot", "1d", "AAAUSDT"))
    assert manifest["2024-10"]["source"] == "monthly"
    assert lake.read_klines("spot", "1d", "AAAUSDT").height == 31 + 30 + 31


def test_checksum_mismatch_is_reported_not_written(tmp_path):
    fake, client = _setup()
    bad = bv.kline_key("spot", "AAAUSDT", "1d", "2024-09")
    fake.corrupt.add(bad)
    lake = Lake(tmp_path)
    rep = backfill_klines(["AAAUSDT"], "1d", lake=lake, client=client, today=date(2024, 10, 4), workers=2)
    assert [f[1] for f in rep.failures] == ["2024-09"]
    assert "sha256" in rep.failures[0][2]
    assert not (lake.kline_dir("spot", "1d", "AAAUSDT") / "2024-09.parquet").exists()
    assert "2024-09" not in lake.read_manifest(lake.kline_dir("spot", "1d", "AAAUSDT"))


def test_start_month_filter(tmp_path):
    fake, client = _setup()
    lake = Lake(tmp_path)
    rep = backfill_klines(["AAAUSDT"], "1d", start_month="2024-09", lake=lake, client=client,
                          today=date(2024, 10, 4), workers=2)
    assert rep.written == 2
    assert not (lake.kline_dir("spot", "1d", "AAAUSDT") / "2024-08.parquet").exists()
