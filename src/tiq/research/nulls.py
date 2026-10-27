"""Random-timing null: would the same exposure, placed at random times, have done as well?

Each asset's weight series is rotated (circularly shifted) by a random offset inside every
stretch where the asset was in the universe. This keeps the amount of exposure, the holding
periods, the turnover and the sizing pattern, and destroys only the alignment between
positions and subsequent price moves, which is precisely the timing skill being claimed.

The test statistic is the mean net return per bar, not the Sharpe ratio. Rotating assets
independently also de-synchronises their exposures, which diversifies the null portfolio over
time and lowers its volatility; a Sharpe comparison would then reward the null for that, not
for timing. A portfolio's mean return is the sum of each asset's exposure times its return, so
it is unaffected by how the assets' exposures line up with each other.

p-value = (1 + #null mean >= actual mean) / (1 + reps). A trend strategy whose p-value is large
made its money from being exposed to assets that went up, not from timing them.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as half-open (start, end) pairs."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def rotate_weights(weights: np.ndarray, valid: np.ndarray, rows: slice, rng: np.random.Generator) -> np.ndarray:
    out = weights.copy()
    block = out[rows]
    vblock = valid[rows]
    for j in range(block.shape[1]):
        col = block[:, j]
        if not np.any(col):
            continue
        for a, b in _segments(vblock[:, j]):
            if b - a > 1:
                col[a:b] = np.roll(col[a:b], int(rng.integers(1, b - a)))
    return out


def timing_null(
    weights: np.ndarray, valid: np.ndarray, rows: slice, run_returns: Callable[[np.ndarray], np.ndarray],
    reps: int, seed: int = 0, workers: int | None = None,
) -> np.ndarray:
    """Mean net return per bar of `reps` rotated copies, over `rows`."""
    rngs = [np.random.default_rng([seed, k]) for k in range(reps)]

    def one(rng: np.random.Generator) -> float:
        return float(run_returns(rotate_weights(weights, valid, rows, rng))[rows].mean())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.array(list(pool.map(one, rngs)))


def null_pvalue(actual: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= actual)) / (1 + null.size))
