"""Causal rolling indicators on (time x asset) arrays, compiled with Numba.

Conventions shared by every function:
- Input and output are float64 arrays of shape (T, N); column j is one asset.
- Row t uses only rows <= t. Nothing here looks ahead; tests check this by truncating inputs.
- A window must consist of `n` consecutive finite values. A NaN (no data, or a long gap that
  splits a series into separate listings) resets the window, so an indicator never spans a gap
  and a re-listed ticker starts its history from scratch.
- Output is NaN until a full window is available.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def consecutive_valid(x):
    """Number of consecutive finite values ending at each row (0 where x is not finite)."""
    T, N = x.shape
    out = np.zeros((T, N))
    for j in range(N):
        run = 0
        for t in range(T):
            if np.isfinite(x[t, j]):
                run += 1
            else:
                run = 0
            out[t, j] = run
    return out


@njit(cache=True, nogil=True)
def shift(x, k):
    """Value k rows earlier (k >= 0); NaN for the first k rows."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for t in range(k, T):
        for j in range(N):
            out[t, j] = x[t - k, j]
    return out


@njit(cache=True, nogil=True)
def pct_change(x, k):
    """x[t] / x[t-k] - 1, only when the k+1 values x[t-k..t] are all finite (no gap in between)."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        run = 0
        for t in range(T):
            if np.isfinite(x[t, j]):
                run += 1
            else:
                run = 0
                continue
            if run > k and x[t - k, j] != 0.0:
                out[t, j] = x[t, j] / x[t - k, j] - 1.0
    return out


@njit(cache=True, nogil=True)
def rolling_mean(x, n):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        run = 0
        s = 0.0
        for t in range(T):
            v = x[t, j]
            if not np.isfinite(v):
                run = 0
                s = 0.0
                continue
            run += 1
            s += v
            if run > n:
                s -= x[t - n, j]
            if run >= n:
                out[t, j] = s / n
    return out


@njit(cache=True, nogil=True)
def _window_std(x, t, j, n, ddof):
    mean = 0.0
    for k in range(t - n + 1, t + 1):
        mean += x[k, j]
    mean /= n
    m2 = 0.0
    for k in range(t - n + 1, t + 1):
        m2 += (x[k, j] - mean) ** 2
    return np.sqrt(m2 / (n - ddof)), mean, m2


@njit(cache=True, nogil=True)
def rolling_std(x, n, ddof=1):
    """Rolling standard deviation. Windows up to 64 bars are computed exactly (two passes); longer
    windows use Welford add/remove updates, re-anchored with an exact pass every 128 bars so that
    rounding cannot drift at price levels like 95,000."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if n <= ddof:
        return out
    exact = n <= 64
    for j in range(N):
        run = 0
        mean = 0.0
        m2 = 0.0
        for t in range(T):
            v = x[t, j]
            if not np.isfinite(v):
                run = 0
                mean = 0.0
                m2 = 0.0
                continue
            run += 1
            if exact:
                if run >= n:
                    out[t, j] = _window_std(x, t, j, n, ddof)[0]
                continue
            if run <= n:  # filling the first window
                d = v - mean
                mean += d / run
                m2 += d * (v - mean)
            elif run % 128 == 0:
                sd, mean, m2 = _window_std(x, t, j, n, ddof)
            else:  # slide: replace the value leaving the window by the new one
                y = x[t - n, j]
                old_mean = mean
                mean += (v - y) / n
                m2 += (v - y) * (v - mean + y - old_mean)
            if run >= n:
                out[t, j] = np.sqrt(max(m2, 0.0) / (n - ddof))
    return out


@njit(cache=True, nogil=True)
def rolling_max(x, n):
    """Maximum of the last n values (current row included), O(T) per column via a monotonic deque."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    dq = np.empty(T, dtype=np.int64)
    for j in range(N):
        head = 0
        tail = 0
        run = 0
        for t in range(T):
            v = x[t, j]
            if not np.isfinite(v):
                head = 0
                tail = 0
                run = 0
                continue
            run += 1
            while tail > head and x[dq[tail - 1], j] <= v:
                tail -= 1
            dq[tail] = t
            tail += 1
            while dq[head] <= t - n:
                head += 1
            if run >= n:
                out[t, j] = x[dq[head], j]
    return out


@njit(cache=True, nogil=True)
def rolling_min(x, n):
    return -rolling_max(-x, n)


@njit(cache=True, nogil=True)
def rolling_median(x, n):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        run = 0
        for t in range(T):
            if np.isfinite(x[t, j]):
                run += 1
            else:
                run = 0
                continue
            if run >= n:
                out[t, j] = np.median(x[t - n + 1:t + 1, j])
    return out


def returns(close: np.ndarray) -> np.ndarray:
    """Simple one-bar returns; NaN across gaps and on the first bar of every listing."""
    return pct_change(close, 1)


def annualized_vol(close: np.ndarray, lookback: int, periods_per_year: float) -> np.ndarray:
    """Rolling standard deviation of one-bar returns over `lookback` bars, annualised."""
    return rolling_std(returns(close), lookback, 1) * np.sqrt(periods_per_year)
