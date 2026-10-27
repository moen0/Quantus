"""Single-asset z-score mean reversion: the original project's rule, ported to the shared engine.

It lost money on one-minute BTC bars after fees in the first review (GUIDE.md, live stack).
It stays as a reference and a negative control. It is not statistical arbitrage: that needs a
spread between related instruments (see the roadmap).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from tiq import indicators as ind
from tiq.strategies.base import Context, Strategy, days_to_bars


@njit(cache=True, nogil=True)
def zscore_state(z, entry, exit_, allow_short):
    """Hysteresis state machine: long below -entry until z rises above -exit_; short (if allowed)
    above +entry until z falls below +exit_. NaN forces flat."""
    T, N = z.shape
    out = np.zeros((T, N))
    for j in range(N):
        state = 0
        for t in range(T):
            zi = z[t, j]
            if not np.isfinite(zi):
                state = 0
            elif state == 0:
                if zi < -entry:
                    state = 1
                elif allow_short and zi > entry:
                    state = -1
            elif state == 1:
                if zi > -exit_:
                    state = 0
            elif zi < exit_:
                state = 0
            out[t, j] = state
    return out


class ZScoreReversion(Strategy):
    name = "zscore_reversion"
    description = "Buy when price is far below its rolling mean, exit when it returns"
    defaults = {"window_days": 20, "entry": 2.0, "exit": 0.5}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        close = ctx.panel.close
        n = days_to_bars(p["window_days"], ctx.panel, minimum=3)
        mean = ind.shift(ind.rolling_mean(close, n), 1)  # window excludes the current bar
        std = ind.shift(ind.rolling_std(close, n, 1), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 1e-9 * np.abs(mean), (close - mean) / std, np.nan)
        return zscore_state(z, float(p["entry"]), float(p["exit"]), bool(p["allow_short"]))
