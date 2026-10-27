import math

import numpy as np
import pytest
from scipy import stats as sps

from tiq.backtest.metrics import calendar_returns, drawdown, performance, rolling_sharpe, sharpe_ratio
from tiq.stats import (
    bootstrap_ci,
    deflated_sharpe_ratio,
    effective_trials,
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe_std_error,
    spa_test,
    suggested_blocks,
)
from tiq.stats.sharpe import psr_from_stats


def test_performance_hand_example():
    r = np.array([0.10, -0.20, 0.05, 0.10])
    m = performance(r, periods_per_year=4)
    eq = np.cumprod(1 + r)
    assert m["total_return"] == pytest.approx(eq[-1] - 1)
    assert m["cagr"] == pytest.approx(eq[-1] - 1)          # exactly one year of quarterly bars
    assert m["max_drawdown"] == pytest.approx(0.20)        # 1.10 -> 0.88
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * 2)
    assert m["sortino"] == pytest.approx(r.mean() * 4 / (math.sqrt(np.mean(np.minimum(r, 0) ** 2)) * 2))
    assert drawdown(r)[1] == pytest.approx(0.2)


def test_rolling_sharpe_matches_direct():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, 400)
    rs = rolling_sharpe(r, 100, 365)
    assert rs[150] == pytest.approx(sharpe_ratio(r[51:151], 365), rel=1e-9)
    assert np.isnan(rs[:99]).all()


def test_calendar_returns():
    idx = (np.datetime64("2023-12-30") + np.arange(4)).astype("datetime64[ms]").astype(np.int64)
    out = calendar_returns(np.array([0.1, 0.1, 0.5, -0.5]), idx)
    assert out == {"2023": pytest.approx(0.21), "2024": pytest.approx(-0.25)}


def test_deflated_sharpe_reproduces_bailey_lopez_de_prado_example():
    """Numerical example of Bailey & Lopez de Prado (2014): N=100 trials, V[SR]=0.5 (annualised),
    best SR 2.5 (annualised), T=1250 daily observations, skew -3, kurtosis 10 -> DSR ~ 0.90."""
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    dsr = psr_from_stats(2.5 / math.sqrt(250), 1250, -3.0, 10.0, sr0)
    assert dsr == pytest.approx(0.9004, abs=2e-3)


def test_psr_matches_normal_closed_form_and_mintrl_is_consistent():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 2000)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = sps.skew(r), sps.kurtosis(r, fisher=False)
    expected = sps.norm.cdf(sr * math.sqrt(1999) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    assert probabilistic_sharpe_ratio(r) == pytest.approx(expected)
    n_star = min_track_record_length(r, 0.0, 0.95)
    assert psr_from_stats(sr, int(round(n_star)), skew, kurt) == pytest.approx(0.95, abs=0.003)
    assert sharpe_std_error(0.0, 101) == pytest.approx(0.1)


def test_deflation_raises_the_bar_with_more_trials():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0008, 0.01, 1500)
    d1, _ = deflated_sharpe_ratio(r, 1, 1e-4)
    d100, sr0 = deflated_sharpe_ratio(r, 100, 1e-4)
    assert d1 == pytest.approx(probabilistic_sharpe_ratio(r)) and d100 < d1 and sr0 > 0


def test_effective_trials():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(1000, 1))
    assert effective_trials(np.hstack([base] * 10)) == pytest.approx(1.0, abs=1e-6)
    assert effective_trials(rng.normal(size=(5000, 10))) > 9


def test_pbo_noise_is_about_half_and_skill_is_low():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.01, (2000, 20))
    assert 0.3 < pbo_cscv(noise, 10).pbo < 0.7
    skill = noise.copy()
    skill[:, 7] += 0.004                          # one variant is genuinely better in every period
    res = pbo_cscv(skill, 10)
    assert res.pbo < 0.05 and res.prob_oos_loss < 0.05
    assert suggested_blocks(2000) == 16 and suggested_blocks(100) == 4


def test_spa_rejects_only_a_real_edge():
    rng = np.random.default_rng(5)
    bench = np.zeros(1500)
    noise = rng.normal(0, 0.01, (1500, 15))
    assert spa_test(bench, noise, reps=500, seed=1)["p_consistent"] > 0.10
    edge = noise.copy()
    edge[:, 3] += 0.002
    out = spa_test(bench, edge, reps=500, seed=1)
    assert out["p_consistent"] < 0.05 and out["best_trial"] == 3


def test_bootstrap_ci_covers_the_truth():
    rng = np.random.default_rng(6)
    r = rng.normal(0.001, 0.01, 3000)
    lo, hi, block = bootstrap_ci(r, lambda x: x.mean() / x.std(ddof=1) * math.sqrt(365), reps=400, seed=0)
    assert lo < 0.1 * math.sqrt(365) < hi and block >= 1
