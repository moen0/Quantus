"""Rolling kernels against naive NumPy references, NaN handling, and causality."""
import numpy as np
import pytest

from tiq import indicators as ind


def naive(fn, x, n):
    out = np.full_like(x, np.nan)
    for t in range(n - 1, x.shape[0]):
        w = x[t - n + 1:t + 1]
        ok = np.all(np.isfinite(w), axis=0)
        if ok.any():
            out[t, ok] = fn(w[:, ok])
    return out


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    x = 50_000 + np.cumsum(rng.normal(0, 100, (800, 4)), axis=0)
    x[100:103, 1] = np.nan       # gap inside a series
    x[:50, 2] = np.nan           # late listing
    x[700:, 3] = np.nan          # delisting
    return x


@pytest.mark.parametrize("n", [1, 2, 7, 60, 65, 200])
def test_rolling_kernels_match_naive(data, n):
    assert np.allclose(ind.rolling_mean(data, n), naive(lambda w: w.mean(0), data, n), equal_nan=True, rtol=1e-12)
    assert np.array_equal(ind.rolling_max(data, n), naive(lambda w: w.max(0), data, n), equal_nan=True)
    assert np.array_equal(ind.rolling_min(data, n), naive(lambda w: w.min(0), data, n), equal_nan=True)
    assert np.array_equal(ind.rolling_median(data, n), naive(lambda w: np.median(w, 0), data, n), equal_nan=True)
    if n > 1:
        ref = naive(lambda w: w.std(0, ddof=1), data, n)
        assert np.allclose(ind.rolling_std(data, n, 1), ref, equal_nan=True, rtol=1e-9)


@pytest.mark.parametrize("n", [60, 100, 365])
def test_rolling_std_is_stable_at_price_level_over_long_series(n):
    rng = np.random.default_rng(0)
    x = 95_000 + np.cumsum(rng.normal(0, 50, (60_000, 1)), axis=0)
    got = ind.rolling_std(x, n, 1)
    ref = naive(lambda w: w.std(0, ddof=1), x, n)
    assert np.nanmax(np.abs(got - ref) / ref) < 1e-9


def test_window_never_spans_a_gap():
    x = np.arange(20, dtype=float)[:, None]
    x[10] = np.nan
    m = ind.rolling_mean(x, 3)
    assert np.isnan(m[10:13]).all() and m[13, 0] == pytest.approx(12.0)


def test_pct_change_and_returns():
    x = np.array([[100.0], [110.0], [np.nan], [121.0], [133.1]])
    r = ind.returns(x)
    assert np.isnan(r[0, 0]) and r[1, 0] == pytest.approx(0.1)
    assert np.isnan(r[3, 0])                    # no return across the gap
    assert r[4, 0] == pytest.approx(0.1)
    assert np.isnan(ind.pct_change(x, 2)[4, 0])  # window would span the gap


def test_consecutive_valid_and_shift():
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    assert ind.consecutive_valid(x)[:, 0].tolist() == [1, 2, 0, 1, 2, 3]
    s = ind.shift(x, 2)
    assert np.isnan(s[:2]).all() and s[3, 0] == 2.0


def test_causality_prefix_property(data):
    """Truncating the input must not change any earlier output: nothing looks ahead."""
    for f in (lambda x: ind.rolling_mean(x, 20), lambda x: ind.rolling_std(x, 20, 1),
              lambda x: ind.rolling_max(x, 20), lambda x: ind.rolling_median(x, 9),
              lambda x: ind.pct_change(x, 5), lambda x: ind.annualized_vol(x, 30, 365)):
        full = f(data)
        for cut in (60, 333, 701):
            assert np.array_equal(f(data[:cut]), full[:cut], equal_nan=True)
