"""Numba kernels: rolling z-score, mean-reversion state machine, P&L, and a parallel parameter sweep."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _zscore_at(x, i, window):
    """z-score of x[i] against the `window` values *before* it (x[i] itself is excluded)."""
    w = x[i - window:i]
    m = w.mean()
    s = w.std()
    if s > 1e-9 * abs(m):          # relative epsilon: independent of the price level
        return (x[i] - m) / s
    return np.nan


@njit(cache=True)
def _zscore_series(x, window):
    z = np.full(x.shape[0], np.nan)
    for i in range(window, x.shape[0]):
        z[i] = _zscore_at(x, i, window)
    return z


@njit(parallel=True, cache=True)
def rolling_zscore(x, window):
    """Parallel over time steps. NaN until `window` values of history exist."""
    z = np.full(x.shape[0], np.nan)
    for i in prange(window, x.shape[0]):
        z[i] = _zscore_at(x, i, window)
    return z


@njit(cache=True)
def positions_from_zscore(z, entry, exit_):
    """Mean-reversion state machine (inherently sequential).
    Short above +entry, long below -entry; flat again once z has come back to exit_."""
    pos = np.zeros(z.shape[0], dtype=np.int8)
    state = 0
    for i in range(z.shape[0]):
        zi = z[i]
        if np.isnan(zi):
            state = 0
        elif state == 0:
            if zi > entry:
                state = -1
            elif zi < -entry:
                state = 1
        elif state == -1:
            if zi < exit_:
                state = 0
        else:
            if zi > -exit_:
                state = 0
        pos[i] = state
    return pos


@njit(cache=True)
def strategy_returns(close, pos, fee):
    """Net return per bar.
    pos[i] is decided at the close of bar i, so it earns the return of bar i+1 (no look-ahead).
    A fee (fraction per unit traded, e.g. 0.001 = 10 bps) is charged whenever the position changes."""
    r = np.zeros(close.shape[0])
    for i in range(1, close.shape[0]):
        before = float(pos[i - 2]) if i >= 2 else 0.0
        held = float(pos[i - 1])
        r[i] = held * (close[i] / close[i - 1] - 1.0) - fee * abs(held - before)
    return r


@njit(cache=True)
def summarize(r, bars_per_year):
    """(total return, annualised Sharpe, max drawdown as a positive fraction)."""
    equity, peak, max_dd = 1.0, 1.0, 0.0
    s, s2 = 0.0, 0.0
    for x in r:
        equity *= 1.0 + x
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
        s += x
        s2 += x * x
    n = r.shape[0]
    mean = s / n
    var = s2 / n - mean * mean
    sharpe = mean / np.sqrt(var) * np.sqrt(bars_per_year) if var > 0.0 else 0.0
    return equity - 1.0, sharpe, max_dd


@njit(cache=True)
def count_entries(pos):
    n = 0
    for i in range(1, pos.shape[0]):
        if pos[i] != 0 and pos[i - 1] == 0:
            n += 1
    return n


@njit(parallel=True, cache=True)
def sweep(close, windows, entries, exits, fee, bars_per_year):
    """One complete backtest per parameter combination, spread over all threads.
    Returns shape (n_combos, 4): total return, Sharpe, max drawdown, number of trades."""
    k = windows.shape[0]
    out = np.zeros((k, 4))
    for j in prange(k):
        z = _zscore_series(close, windows[j])
        pos = positions_from_zscore(z, entries[j], exits[j])
        r = strategy_returns(close, pos, fee)
        tot, sharpe, dd = summarize(r, bars_per_year)
        out[j, 0] = tot
        out[j, 1] = sharpe
        out[j, 2] = dd
        out[j, 3] = count_entries(pos)
    return out
