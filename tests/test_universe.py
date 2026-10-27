import numpy as np

from tests.conftest import make_panel, random_walk
from tiq.data.symbols import eligible_symbols, exclusion_reason, is_leveraged_token
from tiq.data.universe import UniverseConfig, build_universe, static_mask


def test_symbol_rules():
    syms = ["BTCUSDT", "ETHUSDT", "USDCUSDT", "BTCUPUSDT", "ETHDOWNUSDT", "JUPUSDT", "BULLUSDT", "PAXGUSDT",
            "WBTCUSDT", "EURUSDT", "BTCBUSD"]
    assert eligible_symbols(syms) == ["BTCUSDT", "ETHUSDT", "JUPUSDT"]
    bases = {"BTC", "ETH", "J"}
    assert is_leveraged_token("BTCUP", bases) and not is_leveraged_token("JUP", bases)
    assert exclusion_reason("BTCBUSD") == "not quoted in USDT"
    assert exclusion_reason("FDUSDUSDT") == "stablecoin or fiat"


def test_history_liquidity_and_top_n():
    T = 200
    close = random_walk(T, 4, seed=2)
    close[:100, 3] = np.nan                                   # listed at bar 100
    volume = np.tile([1e6, 3e6, 2e6, 9e6], (T, 1))
    volume[np.isnan(close)] = np.nan
    panel = make_panel(close, volume=volume, symbols=["AUSDT", "BUSDT", "CUSDT", "DUSDT"])
    cfg = UniverseConfig(top_n=2, min_history_days=30, liquidity_lookback_days=10, rebalance_days=1)
    u = build_universe(panel, cfg)
    assert not u[:29].any()                                   # nobody has 30 days of history yet
    # quote volume = volume * price; ranking uses the median over 10 days
    qv = volume * close
    t = 50
    expected = set(np.argsort(-np.median(qv[t - 9:t + 1, :3], axis=0))[:2])
    assert set(np.flatnonzero(u[t])) == expected
    assert not u[100:129, 3].any()                            # D needs its own 30 days
    assert u.sum(axis=1).max() == 2


def test_membership_changes_only_at_rebalance_but_exits_on_missing_data():
    T = 60
    close = random_walk(T, 2, seed=3)
    volume = np.ones((T, 2)) * 1e6
    volume[:, 1] = np.linspace(1e5, 1e8, T)                   # B overtakes A over time
    close[40:, 0] = np.nan                                    # A delisted at bar 40
    volume[40:, 0] = np.nan
    panel = make_panel(close, volume=volume, symbols=["AUSDT", "BUSDT"])
    u = build_universe(panel, UniverseConfig(top_n=1, liquidity_lookback_days=1, rebalance_days=7))
    changes = np.flatnonzero(np.any(u[1:] != u[:-1], axis=1)) + 1
    assert all(c % 7 == 0 or c == 40 for c in changes)
    assert not u[40:, 0].any()


def test_low_volatility_filter_catches_pegged_assets():
    T = 400
    close = random_walk(T, 2, seed=4)
    close[:, 1] = 1.0 + np.random.default_rng(0).normal(0, 0.0005, T)   # a stablecoin-like series
    panel = make_panel(close, symbols=["AUSDT", "NEWSTABLEUSDT"])
    u = build_universe(panel, UniverseConfig(min_annual_vol=0.10, vol_lookback_days=30, rebalance_days=1))
    assert u[100:, 0].all() and not u[:, 1].any()


def test_static_mask_and_explicit_symbols():
    syms = ("BTCUSDT", "ETHUSDT", "USDCUSDT")
    assert static_mask(syms, UniverseConfig()).tolist() == [True, True, False]
    assert static_mask(syms, UniverseConfig(symbols=("ETHUSDT",))).tolist() == [False, True, False]
    assert static_mask(syms, UniverseConfig(exclude=("BTCUSDT",))).tolist() == [False, True, False]


def test_universe_is_causal():
    T = 300
    close = random_walk(T, 5, seed=5)
    close[:80, 4] = np.nan
    volume = np.abs(random_walk(T, 5, seed=6)) * 1e4
    panel = make_panel(close, volume=volume, symbols=[f"X{j}USDT" for j in range(5)])
    cfg = UniverseConfig(top_n=3, min_history_days=20, liquidity_lookback_days=15, min_annual_vol=0.05,
                         rebalance_days=5)
    full = build_universe(panel, cfg)
    for cut in (50, 151, 299):
        assert np.array_equal(build_universe(panel.rows(0, cut), cfg), full[:cut])
