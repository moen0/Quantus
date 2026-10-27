"""Trading-cost model: exchange fee + half the bid-ask spread + square-root market impact.

Per unit of notional traded, an order pays

    fee_bps / 1e4                                       exchange commission (taker by default)
  + half_spread_bps / 1e4                               crossing half the bid-ask spread
  + impact_coef * sigma_bar * sqrt(notional / adv_bar)  price pushed by the order itself

The impact term is the "square-root law" (Almgren et al. 2005; Bouchaud et al. 2018): cost grows
with volatility and with the square root of the order's share of typical volume. It is close to
zero for a small account in BTC and large for the same order in a thin altcoin, which is exactly
the distinction a flat fee misses. sigma_bar and adv_bar are trailing estimates known before the
order (no look-ahead).

Defaults are deliberately conservative for a retail Binance spot account: 10 bps taker fee
(VIP 0, no BNB discount), 5 bps half-spread, impact coefficient 1.0. Every experiment also
reports results at 0x, 2x and 3x these costs and the break-even multiple.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 10.0
    half_spread_bps: float = 5.0
    impact_coef: float = 1.0
    capital: float = 100_000.0   # starting equity in quote currency; matters only for impact
    adv_lookback_days: float = 30.0
    max_slippage: float = 0.10   # cap per order (10%), a guard against absurd impact estimates

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 1e4

    @property
    def half_spread(self) -> float:
        return self.half_spread_bps / 1e4

    def scaled(self, multiple: float) -> CostModel:
        """Same model with fee, spread and impact multiplied (capital unchanged)."""
        return CostModel(self.fee_bps * multiple, self.half_spread_bps * multiple, self.impact_coef * multiple,
                         self.capital, self.adv_lookback_days, self.max_slippage)
