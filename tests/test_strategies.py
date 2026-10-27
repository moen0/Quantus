import math

import numpy as np
import pytest

from tests.conftest import make_panel, random_walk
from tiq import indicators as ind
from tiq.strategies import STRATEGIES, Context, get_strategy, size_positions
from tiq.strategies.meanrev import zscore_state
from tiq.strategies.trend import donchian_long_state

GRID_SAMPLES = {
    "donchian_ensemble": {"lookbacks": [5, 20, 60]},
    "ma_crossover": {"fast_days": 10, "slow_days": 50},
    "tsmom": {"lookback_days": 30},
    "zscore_reversion": {"window_days": 15, "entry": 1.5, "exit": 0.3},
    "buy_and_hold": {},
    "vol_target_buy_and_hold": {},
}


def _ctx(panel, top_n=None):
    universe = np.isfinite(panel.close)
    return Context(panel, universe, top_n or panel.shape[1])


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_is_causal(name):
    """Weights computed on a truncated history equal the first rows of the full computation."""
    close = random_walk(700, 3, seed=11, drift=0.0003)
    close[:120, 2] = np.nan
    panel = make_panel(close)
    s = get_strategy(name)
    p = s.params(GRID_SAMPLES[name])
    full = s.weights(_ctx(panel), p)
    assert np.isfinite(full).all()
    for cut in (130, 377, 699):
        assert np.allclose(s.weights(_ctx(panel.rows(0, cut)), p), full[:cut], atol=1e-15)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_long_only_and_no_leverage_by_default(name):
    panel = make_panel(random_walk(500, 4, seed=12, vol=0.01))  # low vol: vol targeting wants leverage
    s = get_strategy(name)
    w = s.weights(_ctx(panel), s.params(GRID_SAMPLES[name]))
    assert (w >= 0).all()
    assert (np.abs(w).sum(axis=1) <= 1.0 + 1e-12).all()


def test_donchian_state_machine():
    close = np.array([10, 11, 12, 11, 13, 14, 15, 14.6, 12.4, 13, 16], dtype=float)[:, None]
    n = 3
    hi, lo = ind.rolling_max(close, n), ind.rolling_min(close, n)
    s = donchian_long_state(close, ind.shift(hi, 1), (hi + lo) / 2)[:, 0]
    # t=3: upper_prev = max(10,11,12)=12, close 11 -> flat; t=4: 13 > max(11,12,11) -> long
    assert s[:4].tolist() == [0, 0, 0, 0]
    assert s[4] == 1
    # stop ratchets: mid t=4 12, t=5 12.5, t=6 13.5, t=7 14.5 (14.6 >= 14.5 stays), t=8 13.7 < stop 14.5 -> exit
    assert s[5:8].tolist() == [1, 1, 1]
    assert s[8] == 0
    # t=10: 16 > max(14.6, 12.4, 13) -> long again
    assert s[10] == 1


def test_zscore_state_machine():
    z = np.array([0, -2.5, -1.0, -0.4, 0.1, 2.5, 1.0, 0.2, np.nan], dtype=float)[:, None]
    long_only = zscore_state(z, 2.0, 0.5, False)[:, 0]
    assert long_only.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0]
    both = zscore_state(z, 2.0, 0.5, True)[:, 0]
    assert both.tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]


def test_vol_targeting_sizing_rules():
    panel = make_panel(random_walk(400, 2, seed=13, vol=0.04))
    universe = np.ones(panel.shape, dtype=bool)
    universe[:, 1] = False                       # second asset never in the universe
    ctx = Context(panel, universe, slots=2)
    signal = np.ones(panel.shape)
    p = get_strategy("vol_target_buy_and_hold").params({"target_vol": 0.3, "vol_lookback_days": 30})
    w = size_positions(signal, ctx, p)
    vol = ind.annualized_vol(panel.close, 30, 365)[:, 0]
    t = 200
    assert w[t, 0] == pytest.approx(min(2.0, 0.3 / vol[t]) / 2)  # one of two sleeves
    assert (w[:, 1] == 0).all()
    assert (w[:30, 0] == 0).all()                # no volatility estimate yet -> no position


def test_gross_cap_scales_proportionally():
    panel = make_panel(random_walk(300, 3, seed=14, vol=0.005))
    ctx = _ctx(panel)
    p = get_strategy("vol_target_buy_and_hold").params({"target_vol": 0.5, "max_gross": 1.0})
    w = size_positions(np.ones(panel.shape), ctx, p)
    assert np.abs(w[100:]).sum(axis=1) == pytest.approx(np.ones(200))


def test_shorts_only_when_allowed():
    close = np.concatenate([np.linspace(100, 50, 200), np.linspace(50, 80, 100)])
    panel = make_panel(close)
    s = get_strategy("tsmom")
    w_long = s.weights(_ctx(panel), s.params({"lookback_days": 20, "target_vol": None}))
    w_short = s.weights(_ctx(panel), s.params({"lookback_days": 20, "target_vol": None, "allow_short": True}))
    assert (w_long[50:190] == 0).all() and (w_short[50:190] == -1).all()


def test_params_reject_unknown_keys():
    with pytest.raises(KeyError, match="unknown parameter"):
        get_strategy("donchian_ensemble").params({"lookback": [5]})
    with pytest.raises(KeyError):
        get_strategy("does_not_exist")


def test_ma_crossover_degenerate_grid_point_never_trades():
    panel = make_panel(random_walk(300, 1, seed=15))
    s = get_strategy("ma_crossover")
    assert (s.weights(_ctx(panel), s.params({"fast_days": 50, "slow_days": 20})) == 0).all()


def test_days_convert_to_bars_for_intraday_panels():
    hourly = make_panel(random_walk(24 * 60, 1, seed=16), interval="1h")
    s = get_strategy("tsmom")
    w = s.weights(_ctx(hourly), s.params({"lookback_days": 5, "target_vol": None}))
    assert (w[: 5 * 24] == 0).all() and math.isclose(hourly.bars_per_day, 24)
