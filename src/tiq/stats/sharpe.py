"""Inference on Sharpe ratios: how sure can we be that a backtest's Sharpe ratio is real?

Everything here works with per-period (not annualised) Sharpe ratios; multiply by
sqrt(periods_per_year) only for display.

- Sharpe standard error with skewness and fat tails (Mertens 2002; Bailey & Lopez de Prado 2012):
      Var[SR] ~= (1 - skew * SR + (kurt - 1) / 4 * SR^2) / (n - 1)       kurt = non-excess kurtosis
  Negative skew and fat tails, typical of crypto and of trend strategies' exits, widen it.
- Probabilistic Sharpe Ratio, PSR(SR*): probability that the true Sharpe exceeds SR*.
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): PSR against the Sharpe you would
  expect from the best of N worthless strategies,
      SR0 = sqrt(V[SR_n]) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))),  g = Euler's constant.
  The more variants you try, the higher the bar.
- Minimum track record length: bars needed before PSR(SR*) reaches a given confidence.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def sample_moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    """(per-period Sharpe, skewness, non-excess kurtosis, n)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    sd = r.std(ddof=1) if n > 1 else 0.0
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    return sr, float(sps.skew(r)), float(sps.kurtosis(r, fisher=False)), n


def sharpe_std_error(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(n - 1, 1)
    return math.sqrt(max(var, 1e-300))


def psr_from_stats(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    return float(sps.norm.cdf((sr - sr_benchmark) / sharpe_std_error(sr, n, skew, kurt)))


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    sr, skew, kurt, n = sample_moments(returns)
    return psr_from_stats(sr, n, skew, kurt, sr_benchmark)


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials strategies whose true Sharpe is zero."""
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(var_sr, 0.0)) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: float, var_sr_trials: float) -> tuple[float, float]:
    """(DSR, SR0): probability that the selected strategy's true Sharpe beats the best-of-N noise level."""
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return probabilistic_sharpe_ratio(returns, sr0), sr0


def min_track_record_length(returns: np.ndarray, sr_benchmark: float = 0.0, prob: float = 0.95) -> float:
    """Bars of history needed for PSR(sr_benchmark) >= prob, given the observed moments."""
    sr, skew, kurt, _ = sample_moments(returns)
    if sr <= sr_benchmark:
        return float("inf")
    z = sps.norm.ppf(prob)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) * (z / (sr - sr_benchmark)) ** 2)


def effective_trials(trial_returns: np.ndarray) -> float:
    """Effective number of independent strategies: participation ratio (sum l)^2 / sum l^2 of the
    eigenvalues of the trials' correlation matrix. Equals N for uncorrelated trials and 1 for
    identical ones. A rough stand-in for the clustering approach of Lopez de Prado (2019)."""
    x = np.asarray(trial_returns, dtype=float)
    keep = x.std(axis=0) > 0
    x = x[:, keep]
    if x.shape[1] <= 1:
        return float(x.shape[1])
    eig = np.clip(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)), 0.0, None)
    return float(eig.sum() ** 2 / np.sum(eig**2))
