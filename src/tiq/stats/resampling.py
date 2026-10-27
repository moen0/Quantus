"""Resampling-based inference: bootstrap confidence intervals and Hansen's SPA test.

Returns are autocorrelated and volatility clusters, so an ordinary bootstrap (which shuffles
single bars) understates uncertainty. The stationary bootstrap (Politis & Romano 1994)
resamples blocks of random length instead; the mean block length comes from the automatic
selector of Politis & White (2004).

Hansen's Superior Predictive Ability test (2005) answers the multiple-testing question
directly: given every variant that was tried, is the best one really better than the
benchmark, or is its edge what you would expect from searching that many variants? A small
p-value means at least one variant beats the benchmark after accounting for the search.
Implementation: the `arch` package (Kevin Sheppard).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from arch.bootstrap import SPA, StationaryBootstrap, optimal_block_length


def block_length(returns: np.ndarray, minimum: int = 1, maximum: int | None = None) -> int:
    """Mean block length for the stationary bootstrap: the Politis-White automatic choice, taken
    as the larger of the estimates for the returns and for the squared returns. Daily returns
    are nearly uncorrelated (estimate ~1 bar) while their volatility clusters for weeks; the
    squared series captures that, so intervals are not too narrow."""
    r = np.asarray(returns, dtype=float)
    if r.size < 20 or r.std() == 0:
        return max(minimum, 1)
    est = optimal_block_length(np.column_stack([r, r * r]))["stationary"].to_numpy()
    est = est[np.isfinite(est)]
    b = int(math.ceil(est.max())) if est.size else minimum
    if maximum is not None:
        b = min(b, maximum)
    return max(minimum, b)


def bootstrap_ci(
    returns: np.ndarray, func: Callable[[np.ndarray], float], reps: int = 2000, alpha: float = 0.05,
    seed: int = 0, block: int | None = None,
) -> tuple[float, float, int]:
    """Percentile confidence interval of func(returns) under the stationary bootstrap.
    Returns (low, high, mean block length used)."""
    r = np.asarray(returns, dtype=float)
    block = block or block_length(r, maximum=max(1, r.size // 10))
    bs = StationaryBootstrap(block, r, seed=seed)
    ci = bs.conf_int(lambda x: np.array([func(x)]), reps=reps, method="percentile", size=1.0 - alpha)
    return float(ci[0, 0]), float(ci[1, 0]), block


def spa_test(
    benchmark_returns: np.ndarray, trial_returns: np.ndarray, reps: int = 2000, seed: int = 0,
    block: int | None = None,
) -> dict[str, float]:
    """Hansen's SPA test of 'no trial beats the benchmark'. Loss = minus the per-bar return."""
    b = np.asarray(benchmark_returns, dtype=float)
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    block = block or block_length(m.mean(axis=1) - b, maximum=max(1, b.size // 10))
    spa = SPA(-b, -m, block_size=block, reps=reps, bootstrap="stationary", studentize=True, seed=seed)
    spa.compute()
    p = spa.pvalues
    best = int(np.argmax(m.mean(axis=0) - b.mean()))
    return {"p_lower": float(p["lower"]), "p_consistent": float(p["consistent"]), "p_upper": float(p["upper"]),
            "best_trial": best, "block": block}
