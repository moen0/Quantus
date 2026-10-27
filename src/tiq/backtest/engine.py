"""Portfolio simulator: target weights in, a cost-aware equity curve out.

Event order within each bar t:

  1. Open of bar t: execute the target weights decided at the close of bar t - delay.
     a. Value the portfolio at the open (assets without a tradable open are marked at their
        last close).
     b. Liquidate any position whose asset has had no price for more than `max_stale_bars`
        bars (delisting, or a long gap that splits a re-used ticker), at the last close minus
        `delist_haircut`.
     c. Sell first, then buy. Without leverage, buys are scaled down if cash runs short.
     d. Every fill pays fee + half-spread + square-root impact (tiq.backtest.costs).
  2. Close of bar t: mark everything to the close and record equity, weights and exposure.

Rules that keep the simulation honest:
- Weights decided at a close are never filled at that same close: delay >= 1 bar.
- An asset that is not tradable at an open (no data, or a carried-forward bar) cannot be traded
  then; an existing position is simply held.
- Trades inside the rebalance band are skipped, except exits to zero, which always execute.

The loop is compiled with Numba and releases the GIL, so many parameter sets can run in
parallel threads.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from tiq import indicators as ind
from tiq.backtest.costs import CostModel
from tiq.data.panel import Panel


@njit(cache=True, nogil=True)
def _simulate(open_, close, tradable, target, delay, fee_rate, half_spread, impact_coef, adv, sigma,
              capital, band, max_stale, delist_haircut, allow_short, max_slippage):
    T, N = close.shape
    nav = np.zeros(T)
    turnover = np.zeros(T)
    fees = np.zeros(T)
    slippage = np.zeros(T)
    entries = np.zeros(T, dtype=np.int64)
    forced = np.zeros(T, dtype=np.int64)
    gross = np.zeros(T)
    net = np.zeros(T)
    weights = np.zeros((T, N))
    units = np.zeros(N)
    last_px = np.full(N, np.nan)
    stale = np.zeros(N, dtype=np.int64)
    delta = np.zeros(N)
    slip = np.zeros(N)
    cash = capital
    for t in range(T):
        if t >= delay and nav[t - 1] > 0.0:
            d = t - delay
            # 1a. value at the open
            nav_open = cash
            for i in range(N):
                if units[i] != 0.0:
                    if tradable[t, i] and np.isfinite(open_[t, i]):
                        nav_open += units[i] * open_[t, i]
                    else:
                        nav_open += units[i] * last_px[i]
            traded = 0.0
            # 1b. forced exits from assets that stopped trading
            for i in range(N):
                if units[i] != 0.0 and not (tradable[t, i] and np.isfinite(open_[t, i])) and stale[i] > max_stale:
                    if units[i] > 0.0:
                        px = last_px[i] * (1.0 - delist_haircut)
                    else:
                        px = last_px[i] * (1.0 + delist_haircut)
                    notional = abs(units[i]) * px
                    fee = notional * fee_rate
                    cash += units[i] * px - fee
                    fees[t] += fee
                    traded += notional
                    forced[t] += 1
                    units[i] = 0.0
            # 1c. desired trades and their slippage rates
            for i in range(N):
                delta[i] = 0.0
                slip[i] = 0.0
                if not (tradable[t, i] and np.isfinite(open_[t, i])):
                    continue
                w = target[d, i]
                if not np.isfinite(w):
                    w = 0.0
                if not allow_short and w < 0.0:
                    w = 0.0
                p = open_[t, i]
                current = units[i] * p / nav_open
                if w != 0.0 and abs(w - current) < band:
                    continue
                if w == 0.0 and units[i] == 0.0:
                    continue
                delta[i] = w * nav_open / p - units[i]
                s = half_spread
                notional = abs(delta[i]) * p
                if impact_coef > 0.0 and np.isfinite(sigma[d, i]):
                    a = adv[d, i]
                    if np.isfinite(a) and a > 0.0:
                        s += impact_coef * sigma[d, i] * np.sqrt(notional / a)
                    else:
                        s += impact_coef * sigma[d, i]
                slip[i] = min(s, max_slippage)
            # sells (and short sales) first, to raise cash
            for i in range(N):
                if delta[i] < 0.0:
                    p = open_[t, i]
                    fill = p * (1.0 - slip[i])
                    notional = -delta[i] * fill
                    fee = notional * fee_rate
                    was_flat = units[i] == 0.0
                    cash += notional - fee
                    units[i] += delta[i]
                    if abs(units[i] * p) < 1e-9 * nav_open:
                        units[i] = 0.0
                    if was_flat and units[i] != 0.0:
                        entries[t] += 1
                    fees[t] += fee
                    slippage[t] += -delta[i] * p * slip[i]
                    traded += notional
            # buys, scaled to the cash available when leverage is not allowed
            need = 0.0
            for i in range(N):
                if delta[i] > 0.0:
                    need += delta[i] * open_[t, i] * (1.0 + slip[i]) * (1.0 + fee_rate)
            scale = 1.0
            if not allow_short and need > 0.0 and need > cash:  # cash can be -1e-12 after a full buy
                scale = max(cash, 0.0) / need
            for i in range(N):
                if delta[i] > 0.0:
                    p = open_[t, i]
                    q = delta[i] * scale
                    fill = p * (1.0 + slip[i])
                    notional = q * fill
                    fee = notional * fee_rate
                    was_flat = units[i] == 0.0
                    cash -= notional + fee
                    units[i] += q
                    if abs(units[i] * p) < 1e-9 * nav_open:
                        units[i] = 0.0
                    if was_flat and units[i] != 0.0:
                        entries[t] += 1
                    fees[t] += fee
                    slippage[t] += q * p * slip[i]
                    traded += notional
            turnover[t] = traded / nav_open
        # 2. mark to the close
        for i in range(N):
            c = close[t, i]
            if np.isfinite(c):
                last_px[i] = c
                stale[i] = 0
            else:
                stale[i] += 1
        value = cash
        for i in range(N):
            if units[i] != 0.0:
                value += units[i] * last_px[i]
        nav[t] = value
        if value > 0.0:
            g = 0.0
            ne = 0.0
            for i in range(N):
                if units[i] != 0.0:
                    v = units[i] * last_px[i]
                    weights[t, i] = v / value
                    g += abs(v)
                    ne += v
            gross[t] = g / value
            net[t] = ne / value
    return nav, turnover, fees, slippage, entries, forced, gross, net, weights


@dataclass(frozen=True)
class BacktestResult:
    nav: np.ndarray          # equity at each close, in quote currency
    returns: np.ndarray      # close-to-close net return per bar (0 on the first bar)
    turnover: np.ndarray     # traded notional / equity at the open, per bar
    fees: np.ndarray         # commissions paid per bar (quote currency)
    slippage: np.ndarray     # spread + impact paid per bar (quote currency)
    entries: np.ndarray      # positions opened per bar
    forced_exits: np.ndarray  # liquidations caused by missing data
    gross_exposure: np.ndarray
    net_exposure: np.ndarray
    weights: np.ndarray      # realised weights at each close (T, N)

    @property
    def costs(self) -> np.ndarray:
        return self.fees + self.slippage

    def cost_returns(self) -> np.ndarray:
        """Costs per bar as a fraction of the previous close's equity."""
        prev = np.concatenate([[self.nav[0]], self.nav[:-1]])
        return np.divide(self.costs, prev, out=np.zeros_like(prev), where=prev > 0)


@dataclass(frozen=True)
class MarketInputs:
    """Trailing liquidity and volatility per bar, used by the impact model. Row t uses data <= t."""
    adv: np.ndarray    # mean quote volume per bar over the lookback
    sigma: np.ndarray  # standard deviation of one-bar returns over the lookback

    @classmethod
    def from_panel(cls, panel: Panel, lookback_days: float = 30.0) -> MarketInputs:
        bars = max(2, int(round(lookback_days * panel.bars_per_day)))
        return cls(adv=ind.rolling_mean(panel.quote_volume, bars),
                   sigma=ind.rolling_std(ind.returns(panel.close), bars, 1))


def run_backtest(
    panel: Panel, target: np.ndarray, costs: CostModel | None = None, market: MarketInputs | None = None,
    delay: int = 1, band: float = 0.0, max_stale_bars: int = 0, delist_haircut: float = 0.0,
    allow_short: bool = False,
) -> BacktestResult:
    if delay < 1:
        raise ValueError("delay must be >= 1: a decision at a close cannot fill at that same close")
    if target.shape != panel.close.shape:
        raise ValueError(f"target shape {target.shape} != panel shape {panel.close.shape}")
    costs = costs or CostModel()
    market = market or MarketInputs.from_panel(panel, costs.adv_lookback_days)
    nav, turnover, fees, slippage, entries, forced, gross, net, weights = _simulate(
        panel.open, panel.close, panel.tradable, np.ascontiguousarray(target, dtype=np.float64), int(delay),
        costs.fee_rate, costs.half_spread, costs.impact_coef, market.adv, market.sigma, float(costs.capital),
        float(band), int(max_stale_bars), float(delist_haircut), bool(allow_short), float(costs.max_slippage),
    )
    rets = np.zeros_like(nav)
    prev = nav[:-1]
    rets[1:] = np.divide(nav[1:], prev, out=np.full_like(prev, -1.0), where=prev > 0) - 1.0
    return BacktestResult(nav, rets, turnover, fees, slippage, entries, forced, gross, net, weights)
