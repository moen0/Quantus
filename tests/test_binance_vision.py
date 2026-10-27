import hashlib
import io
import zipfile

import polars as pl
import pytest

from tiq.data import binance_vision as bv

# First rows of real archive files (fetched 2026-10-27).
SPOT_MS = (b"1502942400000,4261.48000000,4313.62000000,4261.32000000,4308.83000000,47.18100900,1502945999999,"
           b"202366.13839304,171,35.16050300,150952.47794304,7887.63551305\n")
SPOT_US = (b"1735689600000000,93576.00000000,94509.42000000,93489.03000000,94401.14000000,755.99010000,"
           b"1735693199999999,71068810.55946380,93525,421.08319000,39596777.77992590,0\n")
FUTURES_HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                  b"taker_buy_quote_volume,ignore\n1788220800000,78549.60,79196.00,76368.00,77400.10,151357.832,"
                  b"1788307199999,11776717838.40160,3622828,72544.811,5646145648.39730,0\n")
FUNDING = (b"calc_time,funding_interval_hours,last_funding_rate\n"
           b"1788220800005,8,0.00008482\n1788249600000,8,0.00010000\n")


def test_parse_spot_milliseconds():
    df = bv.parse_klines(SPOT_MS)
    assert df.schema == pl.Schema(bv.KLINE_SCHEMA)
    assert str(df["open_time"][0]) == "2017-08-17 04:00:00+00:00"
    assert df["close"][0] == pytest.approx(4308.83)
    assert df["trades"][0] == 171


def test_parse_spot_microseconds_since_2025():
    df = bv.parse_klines(SPOT_US)
    assert str(df["open_time"][0]) == "2025-01-01 00:00:00+00:00"
    assert df["quote_volume"][0] == pytest.approx(71068810.5594638)


def test_parse_mixed_units_in_one_file():
    df = bv.parse_klines(SPOT_MS + SPOT_US)
    assert [str(x)[:10] for x in df["open_time"]] == ["2017-08-17", "2025-01-01"]


def test_parse_futures_with_header():
    df = bv.parse_klines(FUTURES_HEADER)
    assert df.height == 1
    assert df["trades"][0] == 3622828
    assert str(df["open_time"][0]).startswith("2026-09-01")


def test_parse_empty_file():
    assert bv.parse_klines(b"").height == 0


def test_parse_funding_rounds_jitter():
    df = bv.parse_funding(FUNDING)
    assert [str(x) for x in df["funding_time"]] == ["2026-09-01 00:00:00+00:00", "2026-09-01 08:00:00+00:00"]
    assert df["funding_rate"][0] == pytest.approx(0.00008482)


def test_checksum_verification():
    data = b"hello"
    good = hashlib.sha256(data).hexdigest() + "  file.zip"
    assert bv.verify_sha256(data, good) == hashlib.sha256(data).hexdigest()
    with pytest.raises(bv.ChecksumError):
        bv.verify_sha256(b"tampered", good)


def test_unzip_single_csv():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1d-2024-01.csv", SPOT_MS)
    assert bv.unzip_single_csv(buf.getvalue()) == SPOT_MS


LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>data.binance.vision</Name>
<Prefix>data/spot/monthly/klines/</Prefix><Marker></Marker><MaxKeys>1000</MaxKeys><Delimiter>/</Delimiter>
<IsTruncated>true</IsTruncated><NextMarker>data/spot/monthly/klines/ETHUSDT/</NextMarker>
<Contents><Key>data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2017-08.zip</Key></Contents>
<Contents><Key>data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2017-08.zip.CHECKSUM</Key></Contents>
<CommonPrefixes><Prefix>data/spot/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>data/spot/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>
</ListBucketResult>"""


def test_parse_listing():
    keys, prefixes, truncated, marker = bv.parse_listing(LISTING)
    assert len(keys) == 2 and prefixes[-1].endswith("ETHUSDT/")
    assert truncated and marker == "data/spot/monthly/klines/ETHUSDT/"
    assert bv.periods_from_keys(keys, "BTCUSDT-1d-") == ["2017-08"]


def test_keys_and_months():
    assert bv.kline_key("spot", "BTCUSDT", "1h", "2024-01") == \
        "data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    assert bv.kline_key("spot", "BTCUSDT", "1h", "2024-01-05").startswith("data/spot/daily/")
    assert bv.funding_key("BTCUSDT", "2024-01").endswith("fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip")
    assert bv.months_between("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]
    assert bv.shift_month("2024-01", -1) == "2023-12"
    assert bv.days_in_month("2024-02") == 29


@pytest.mark.network
def test_live_archive_roundtrip():
    with bv.make_client() as client:
        months = bv.list_kline_periods(client, "spot", "BTCUSDT", "1d")
        assert months[0] == "2017-08"
        csv_bytes, sha = bv.fetch_verified(client, bv.kline_key("spot", "BTCUSDT", "1d", "2024-01"))
    df = bv.parse_klines(csv_bytes)
    assert df.height == 31 and len(sha) == 64
