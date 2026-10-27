"""Backtester accounting, verified against hand calculations, plus look-ahead canaries."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_panel, random_walk
from tiq.backtest.costs import CostModel
from tiq.backtest.engine import run_backtest
from tiq.backtest.metrics import sharpe_ratio

FREE = CostModel(fee_bps=0, half_spread_bps=0, impact_coef=0)
CLOSE = np.array([100.0, 110.0, 121.0, 133.1, 120.0])
OPEN = np.array([100.0, 105.0, 115.0, 130.0, 125.0])


def _bh(costs, delay=1, band=0.0):
    panel = make_panel(CLOSE, open_=OPEN)
    return run_backtest(panel, np.ones((5, 1)), costs, delay=delay, band=band)


def test_buy_and_hold_without_costs_fills_at_next_open():
    res = _bh(FREE)
    assert res.nav[0] == pytest.approx(100_000)
    assert res.nav[-1] == pytest.approx(100_000 * CLOSE[-1] / OPEN[1])
    assert res.returns[1] == pytest.approx(CLOSE[1] / OPEN[1] - 1)
    assert res.returns[2] == pytest.approx(CLOSE[2] / CLOSE[1] - 1)
    assert res.entries.sum() == 1 and res.turnover[1] == pytest.approx(1.0)


def test_fee_is_paid_once_on_entry():
    res = _bh(CostModel(fee_bps=10, half_spread_bps=0, impact_coef=0))
    assert res.nav[-1] == pytest.approx(100_000 / 1.001 * CLOSE[-1] / OPEN[1])
    assert res.fees.sum() == pytest.approx(100_000 / 1.001 * 0.001)


def test_half_spread_moves_the_fill_price():
    res = _bh(CostModel(fee_bps=0, half_spread_bps=5, impact_coef=0))
    assert res.nav[-1] == pytest.approx(100_000 / 1.0005 * CLOSE[-1] / OPEN[1])
    assert res.slippage.sum() == pytest.approx(100_000 / 1.0005 * 0.0005)


def test_round_trip_costs():
    panel = make_panel(CLOSE, open_=OPEN)
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])[:, None]
    res = run_backtest(panel, target, CostModel(fee_bps=10, half_spread_bps=0, impact_coef=0))
    units = 100_000 / (OPEN[1] * 1.001)
    assert res.nav[-1] == pytest.approx(units * OPEN[2] * 0.999)
    assert res.entries.sum() == 1 and (res.gross_exposure[2:] == 0).all()


def test_delay_two_fills_one_bar_later():
    res = _bh(FREE, delay=2)
    assert res.nav[1] == pytest.approx(100_000)
    assert res.nav[-1] == pytest.approx(100_000 * CLOSE[-1] / OPEN[2])


def test_delay_zero_is_refused():
    with pytest.raises(ValueError, match="delay"):
        _bh(FREE, delay=0)


def test_rebalance_band_skips_small_changes_but_never_exits():
    close = np.full(6, 100.0)
    panel = make_panel(close)
    target = np.array([0.50, 0.52, 0.49, 0.51, 0.0, 0.0])[:, None]
    res = run_backtest(panel, target, FREE, band=0.05)
    assert res.turnover[2:5].sum() == 0              # changes under 5 points ignored
    assert res.turnover[5] == pytest.approx(0.5)      # exit to zero always executes
    assert res.gross_exposure[-1] == 0


def test_long_only_never_borrows_even_with_costs():
    close = random_walk(300, 2, seed=21, vol=0.05)
    panel = make_panel(close)
    rng = np.random.default_rng(0)
    target = np.zeros((300, 2))
    flip = rng.integers(0, 2, 300)
    target[np.arange(300), flip] = 1.0               # rotate the whole portfolio almost every bar
    res = run_backtest(panel, target, CostModel(fee_bps=10, half_spread_bps=5, impact_coef=1))
    assert (res.gross_exposure <= 1 + 1e-9).all()
    assert (res.weights >= -1e-12).all()
    assert np.isfinite(res.nav).all() and (res.nav > 0).all()


def test_impact_grows_with_order_size():
    close = random_walk(200, 1, seed=22)
    panel = make_panel(close, volume=np.full((200, 1), 10.0))  # thin market: ~1,000 USDT per bar
    target = (np.arange(200) % 20 < 10).astype(float)[:, None]
    small = run_backtest(panel, target, CostModel(fee_bps=0, half_spread_bps=0, impact_coef=1, capital=1e2))
    large = run_backtest(panel, target, CostModel(fee_bps=0, half_spread_bps=0, impact_coef=1, capital=1e6))
    cost_small = small.slippage.sum() / small.turnover.sum() / small.nav.mean()
    cost_large = large.slippage.sum() / large.turnover.sum() / large.nav.mean()
    assert cost_large > 5 * cost_small


def test_delisting_forces_exit_and_a_relisted_ticker_cannot_produce_profit():
    """LUNA case: the old asset collapses and stops trading; a new asset reuses the ticker at 1,000x."""
    close = np.full(30, 1.0)
    close[5:10] = [0.5, 0.1, 0.01, 0.001, 0.0001]
    close[10:20] = np.nan
    close[20:] = 100.0
    panel = make_panel(close)
    res = run_backtest(panel, np.ones((30, 1)), FREE, max_stale_bars=0)
    assert res.forced_exits.sum() == 1
    assert res.nav[12] == pytest.approx(100_000 * 0.0001)    # liquidated at the last real close
    assert res.nav[-1] == pytest.approx(res.nav[12])         # the 1,000,000x jump across the gap is never booked


def test_short_gap_is_held_through():
    close = np.array([100, 101, 102, 102, 102, 103, 104], dtype=float)
    tradable = np.array([1, 1, 1, 0, 0, 1, 1], dtype=bool)[:, None]
    vol = np.where(tradable, 1e6, 0.0)
    panel = make_panel(close, volume=vol, tradable=tradable)
    target = np.ones((7, 1))
    target[3:5] = 0.0                                      # wants out during the outage: impossible
    res = run_backtest(panel, target, FREE)
    assert res.turnover[3:5].sum() == 0 and res.forced_exits.sum() == 0
    assert res.turnover[5] == pytest.approx(1.0)           # exits at the first tradable open (decided at t=4)


def test_no_edge_on_a_random_walk():
    """A causal momentum rule on driftless random walks must average zero Sharpe: no hidden look-ahead."""
    sharpes = []
    for seed in range(120):
        close = random_walk(1000, 1, seed=100 + seed)
        panel = make_panel(close)
        r = np.diff(np.log(close[:, 0]), prepend=np.nan)
        target = (np.nan_to_num(r) > 0).astype(float)[:, None]   # uses data up to the close of t
        sharpes.append(sharpe_ratio(run_backtest(panel, target, FREE).returns[1:], 365))
    sharpes = np.array(sharpes)
    assert abs(sharpes.mean()) < 3 * sharpes.std(ddof=1) / np.sqrt(sharpes.size)


def test_look_ahead_canary_is_detected():
    """Peeking at the next bar must look spectacular; if it did not, the test above would prove nothing."""
    close = random_walk(1000, 1, seed=7)
    panel = make_panel(close)
    nxt = np.roll(np.diff(np.log(close[:, 0]), prepend=np.nan), -1)
    cheat = (np.nan_to_num(nxt) > 0).astype(float)[:, None]
    assert sharpe_ratio(run_backtest(panel, cheat, FREE).returns[1:], 365) > 10


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), fee=st.floats(0, 50), band=st.floats(0, 0.1), delay=st.integers(1, 3))
def test_engine_invariants_hold_for_random_portfolios(seed, fee, band, delay):
    """Property test: whatever the targets, costs and gaps, a long-only spot account never borrows,
    never goes short, never pays negative costs and keeps a finite, positive equity."""
    rng = np.random.default_rng(seed)
    close = random_walk(150, 3, seed=seed, vol=0.06)
    close[60:70, 1] = np.nan                                 # a delisting-style hole in one asset
    panel = make_panel(close)
    target = rng.dirichlet(np.ones(4), size=150)[:, :3]      # weights >= 0 summing to <= 1
    target[rng.random(150) < 0.2] = 0.0                      # some bars fully in cash
    res = run_backtest(panel, target, CostModel(fee_bps=fee, half_spread_bps=fee / 2, impact_coef=0.5),
                       delay=delay, band=band)
    assert np.isfinite(res.nav).all() and (res.nav > 0).all()
    assert (res.gross_exposure <= 1 + 1e-9).all() and (res.weights >= -1e-12).all()
    assert (res.fees >= 0).all() and (res.slippage >= 0).all() and (res.turnover >= 0).all()
    assert (res.weights[61:70, 1] == 0).all()                # the hole was liquidated, not held
