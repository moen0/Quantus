"""Lake, quality checks, resampling, panel alignment and gap handling."""
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tests.conftest import START, kline_frame
from tiq.data.lake import Lake
from tiq.data.panel import build_panel, load_panel, resample_klines
from tiq.data.quality import find_gaps, kline_qc


def test_lake_roundtrip_dedupes_and_filters(tmp_path):
    lake = Lake(tmp_path)
    d = lake.kline_dir("spot", "1d", "AAAUSDT")
    df = kline_frame(np.arange(1, 41, dtype=float))
    lake.write_month(d, "2020-01", df.head(31), "open_time")
    lake.write_month(d, "2020-02", pl.concat([df.slice(30, 10), df.slice(30, 1)]), "open_time")  # overlap + dup
    out = lake.read_klines("spot", "1d", "AAAUSDT")
    assert out.height == 40 and out["open_time"].is_sorted() and out["open_time"].n_unique() == 40
    window = lake.read_klines("spot", "1d", "AAAUSDT", "2020-01-10", "2020-01-20")
    assert window.height == 10 and str(window["open_time"][0])[:10] == "2020-01-10"
    assert lake.read_klines("spot", "1d", "NOPE").height == 0
    assert lake.kline_symbols("spot", "1d") == ["AAAUSDT"]
    Lake.write_manifest(d, {"2020-02": {"x": 1}, "2020-01": {"x": 2}})
    assert list(Lake.read_manifest(d)) == ["2020-01", "2020-02"]


def test_find_gaps_and_qc():
    df = kline_frame(np.linspace(100, 110, 20))
    holed = pl.concat([df.head(5), df.slice(9, 11)])  # bars 5..8 missing
    gaps = find_gaps(holed["open_time"], 86400)
    assert gaps.height == 1 and gaps["missing"][0] == 4
    bad = holed.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(50.0).otherwise(pl.col("high")).alias("high"))
    bad = pl.concat([bad, bad.head(1)])  # duplicate row
    qc = kline_qc(bad, "1d")
    assert qc["gaps"] == 1 and qc["missing_bars"] == 4 and qc["duplicates"] == 1 and qc["ohlc_violations"] == 1


def test_qc_flags_extreme_moves():
    closes = np.full(30, 100.0)
    closes[10] = 300.0
    assert kline_qc(kline_frame(closes), "1d")["extreme_moves"] == 2


def test_resample_hourly_to_daily_and_drop_incomplete_tail():
    hourly = kline_frame(np.arange(1, 24 * 3 + 7, dtype=float), interval="1h")  # 3 full days + 6 hours
    daily = resample_klines(hourly, "1h", "1d")
    assert daily.height == 3  # the partial 4th day is dropped
    first_day = hourly.head(24)
    assert daily["open"][0] == first_day["open"][0]
    assert daily["close"][0] == first_day["close"][-1]
    assert daily["high"][0] == pytest.approx(first_day["high"].max())
    assert daily["volume"][0] == pytest.approx(first_day["volume"].sum())


def test_resample_4h_is_utc_aligned():
    hourly = kline_frame(np.arange(1, 49, dtype=float), start=START + timedelta(hours=2), interval="1h")
    four = resample_klines(hourly, "1h", "4h")
    hours = [t.hour for t in four["open_time"]]
    assert set(hours) <= {0, 4, 8, 12, 16, 20}
    assert four["open_time"][0] == START  # first bucket 00:00-04:00 holds only 2 hourly bars (kept: not last)


def test_panel_alignment_listing_and_gaps():
    # A: full history. B: listed later. C: short gap (2 bars, carried forward) then long gap (relisting)
    a = kline_frame(np.linspace(100, 130, 30))
    b = kline_frame(np.linspace(10, 12, 10), start=START + timedelta(days=20))
    c_full = kline_frame(np.linspace(1, 2, 30))
    keep = [i for i in range(30) if i not in (5, 6) and not (12 <= i <= 21)]
    c = c_full[keep]
    relisted = c.filter(pl.col("open_time") >= START + timedelta(days=22)).with_columns(pl.col("close") * 1000)
    c = pl.concat([c.filter(pl.col("open_time") < START + timedelta(days=22)), relisted])
    panel = build_panel({"AUSDT": a, "BUSDT": b, "CUSDT": c}, "1d", max_fill_bars=3)
    assert panel.shape == (30, 3)
    jb, jc = panel.col("BUSDT"), panel.col("CUSDT")
    assert np.isnan(panel.close[:20, jb]).all() and np.isfinite(panel.close[20:, jb]).all()
    # short gap: carried forward, zero volume, not tradable
    assert panel.close[5, jc] == panel.close[4, jc] == panel.close[6, jc]
    assert panel.volume[5, jc] == 0 and not panel.tradable[5, jc] and panel.tradable[7, jc]
    # long gap: stays NaN, so the 1000x "return" across it can never be booked
    assert np.isnan(panel.close[12:22, jc]).all()
    assert panel.close[22, jc] > 900 * panel.close[11, jc]
    assert panel.position(datetime(2020, 1, 21, tzinfo=UTC)) == 20
    sub = panel.select(["CUSDT"]).rows(10, 25)
    assert sub.shape == (15, 1) and sub.symbols == ("CUSDT",)


def test_panel_drops_misaligned_rows_and_respects_bounds():
    df = kline_frame(np.arange(1, 11, dtype=float))
    shifted = df.with_columns(pl.col("open_time") + pl.duration(hours=3)).head(2)
    panel = build_panel({"AUSDT": pl.concat([df, shifted])}, "1d", start="2020-01-03", end="2020-01-08")
    assert panel.shape == (5, 1)
    assert panel.close[0, 0] == 3.0 and panel.close[-1, 0] == 7.0


def test_load_panel_resamples_from_lake(tmp_path):
    lake = Lake(tmp_path)
    hourly = kline_frame(np.arange(1, 24 * 10 + 1, dtype=float), interval="1h")
    lake.write_month(lake.kline_dir("spot", "1h", "AUSDT"), "2020-01", hourly, "open_time")
    panel = load_panel(lake, "spot", "1d", ["AUSDT"], source_interval="1h")
    assert panel.shape == (10, 1) and panel.close[0, 0] == 24.0 and panel.periods_per_year == 365


def test_gap_threshold_is_in_days_at_any_bar_size(tmp_path):
    """A 32-hour outage is carried forward at 4h bars too; a 10-day hole still splits the series."""
    lake = Lake(tmp_path)
    hourly = kline_frame(np.linspace(100, 200, 24 * 40), interval="1h")
    ts = hourly["open_time"]
    outage = (ts >= START + timedelta(days=10)) & (ts < START + timedelta(days=10, hours=32))
    delisting = (ts >= START + timedelta(days=25)) & (ts < START + timedelta(days=35))
    lake.write_month(lake.kline_dir("spot", "1h", "AUSDT"), "2020-01", hourly.filter(~outage & ~delisting), "open_time")
    panel = load_panel(lake, "spot", "4h", ["AUSDT"], source_interval="1h", max_fill_days=5)
    c = panel.close[:, 0]
    rows_outage = slice(60, 68)                       # day 10 00:00 .. day 11 08:00 in 4h bars
    assert np.isfinite(c[rows_outage]).all() and not panel.tradable[rows_outage, 0].any()
    assert np.isnan(c[25 * 6:35 * 6]).all()
