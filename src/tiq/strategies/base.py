"""Strategy interface and the shared position-sizing step.

A strategy is split in two, on purpose:

    signal(ctx, params)  -> (T, N) array in [-1, 1]: direction and conviction, no sizing
    size_positions(...)  -> (T, N) target weights (fraction of portfolio value per asset)

Keeping sizing separate means every strategy gets the same risk treatment (universe filter,
volatility targeting, caps), and the validation code can build a "random timing" null by
scrambling the signal while keeping the sizing.

Timing convention: row t of the weights is decided at the close of bar t using data up to and
including bar t. The backtester executes it at the open of bar t + delay (delay >= 1).

All lookbacks are given in days and converted to bars, so one configuration works for daily
and intraday bars alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tiq import indicators as ind
from tiq.data.panel import Panel

SIZING_DEFAULTS: dict[str, Any] = {
    "target_vol": None,          # annualised volatility target per asset sleeve; None = no targeting
    "vol_lookback_days": 90,     # window for the realised-volatility estimate
    "max_asset_leverage": 2.0,   # cap on target_vol / realised_vol within a sleeve
    "max_gross": 1.0,            # cap on the sum of |weights|; 1.0 = no leverage (spot)
    "allow_short": False,        # spot accounts cannot short; perpetuals could
}


@dataclass(frozen=True)
class Context:
    """Everything computed once per experiment and shared by all parameter sets."""
    panel: Panel
    universe: np.ndarray  # bool (T, N)
    slots: int            # number of equal sleeves: top_n of the universe, or the number of symbols


def days_to_bars(days: float, panel: Panel, minimum: int = 1) -> int:
    return max(minimum, int(round(float(days) * panel.bars_per_day)))


def size_positions(signal: np.ndarray, ctx: Context, p: dict[str, Any]) -> np.ndarray:
    """Signal -> weights: universe filter, optional volatility targeting, equal sleeves, gross cap."""
    s = np.where(ctx.universe, np.nan_to_num(signal, nan=0.0), 0.0)
    if not p["allow_short"]:
        s = np.clip(s, 0.0, None)
    if p["target_vol"]:
        bars = days_to_bars(p["vol_lookback_days"], ctx.panel, minimum=2)
        vol = ind.annualized_vol(ctx.panel.close, bars, ctx.panel.periods_per_year)
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.minimum(p["max_asset_leverage"], p["target_vol"] / vol)
        mult = np.where(np.isfinite(mult) & (mult > 0), mult, 0.0)  # no volatility estimate yet -> no position
        s = s * mult
    w = s / max(1, ctx.slots)
    gross = np.abs(w).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(gross > p["max_gross"], p["max_gross"] / gross, 1.0)
    return w * scale[:, None]


class Strategy:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    defaults: ClassVar[dict[str, Any]] = {}

    def params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Defaults merged with overrides; unknown keys are an error (catches typos in configs)."""
        allowed = {**SIZING_DEFAULTS, **self.defaults}
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(allowed)
        if unknown:
            raise KeyError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
        return {**allowed, **overrides}

    def signal(self, ctx: Context, p: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def weights(self, ctx: Context, p: dict[str, Any]) -> np.ndarray:
        return size_positions(self.signal(ctx, p), ctx, p)
