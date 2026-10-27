"""Passive benchmarks. A timing strategy has to beat both of these, not just cash.

BuyAndHold is equal weight across the universe (100% of the asset for a one-asset universe).
VolTargetBuyAndHold holds the same assets, sized with the same volatility targeting as the
strategy. Volatility targeting alone tends to raise the Sharpe ratio of crypto (Harvey et al.
2018, "The impact of volatility targeting", JPM), so a trend rule only adds value if it beats
this second benchmark too: otherwise the gain came from the sizing, not from the timing.
"""
from __future__ import annotations

import numpy as np

from tiq.strategies.base import Context, Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    description = "Equal weight in every universe member, always invested"
    defaults = {}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        return np.isfinite(ctx.panel.close).astype(float)


class VolTargetBuyAndHold(Strategy):
    name = "vol_target_buy_and_hold"
    description = "Always long every universe member, sized to a volatility target"
    defaults = {"target_vol": 0.25}

    def signal(self, ctx: Context, p: dict) -> np.ndarray:
        return np.isfinite(ctx.panel.close).astype(float)
