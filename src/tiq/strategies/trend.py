"""Trend-following strategies (long/flat by default, since spot accounts cannot short).

Evidence that crypto trends, with the caveat that crypto history is short and dominated by a
few large cycles:
- Moskowitz, Ooi, Pedersen (2012), "Time series momentum", JFE: the sign of an asset's own past
  return predicts its future return across 58 futures markets.
- Liu, Tsyvinski (2021), "Risks and returns of cryptocurrency", RFS: strong time-series momentum
  in Bitcoin at one- to eight-week horizons.
- Zarattini, Pagani, Barbon (2025), "Catching crypto trends", SSRN 5209907: an ensemble of
  Donchian breakouts on the most liquid coins, volatility-targeted, reports a net Sharpe above
  1.5. DonchianEnsemble follows its published design; results here are our own replication.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from tiq import indicators as ind
from tiq.strategies.base import Context, Strategy, days_to_bars


@njit(cache=True, nogil=True)
def donchian_long_state(close, upper_prev, mid):
    """Long/flat state for one breakout lookback.

    Enter when the close breaks above the highest close of the previous n bars (upper_prev).
    While long, a trailing stop ratchets up to the channel midpoint (max + min) / 2 of the last
    n closes and never moves down; exit when the close falls below it. Missing data -> flat.
    """
    T, N = close.shape
    out = np.zeros((T, N))
    for j in range(N):
        long = False
        stop = 0.0
        for t in range(T):
            c = close[t, j]
            if not (np.isfinite(c) and np.isfinite(upper_prev[t, j]) and np.isfinite(mid[t, j])):
                long = False
                continue
            if long:
                stop = max(stop, mid[t, j])
                if c < stop:
                    long = False
            elif c > upper_prev[t, j]:
                long = True
                stop = mid[t, j]
            if long:
                out[t, j] = 1.0
    return out


class DonchianEnsemble(Strategy):
    name = "donchian_ensemble"
    description = "Equal-weight ensemble of Donchian breakouts with midpoint trailing stops"
    defaults = {"lookbacks": [5, 10, 20, 30, 60, 90, 150, 250, 360], "target_vol": 0.25}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        close = ctx.panel.close
        total = np.zeros_like(close)
        for days in p["lookbacks"]:
            n = days_to_bars(days, ctx.panel, minimum=2)
            hi = ind.rolling_max(close, n)
            lo = ind.rolling_min(close, n)
            total += donchian_long_state(close, ind.shift(hi, 1), (hi + lo) / 2.0)
        return total / len(p["lookbacks"])


class MovingAverageCrossover(Strategy):
    name = "ma_crossover"
    description = "Long while the fast simple moving average is above the slow one"
    defaults = {"fast_days": 20, "slow_days": 100, "target_vol": 0.25}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        if p["fast_days"] >= p["slow_days"]:
            return np.zeros_like(ctx.panel.close)  # degenerate grid point: never trades
        close = ctx.panel.close
        fast = ind.rolling_mean(close, days_to_bars(p["fast_days"], ctx.panel))
        slow = ind.rolling_mean(close, days_to_bars(p["slow_days"], ctx.panel))
        valid = np.isfinite(fast) & np.isfinite(slow)
        up = valid & (fast > slow)
        down = valid & (fast < slow)
        return up.astype(float) - (down.astype(float) if p["allow_short"] else 0.0)


class TimeSeriesMomentum(Strategy):
    name = "tsmom"
    description = "Long while the asset's own return over the lookback is positive (Moskowitz et al. 2012)"
    defaults = {"lookback_days": 90, "target_vol": 0.25}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        r = ind.pct_change(ctx.panel.close, days_to_bars(p["lookback_days"], ctx.panel))
        up = np.nan_to_num(r, nan=0.0) > 0
        down = np.nan_to_num(r, nan=0.0) < 0
        return up.astype(float) - (down.astype(float) if p["allow_short"] else 0.0)
