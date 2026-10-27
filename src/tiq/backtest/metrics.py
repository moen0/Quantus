"""Performance statistics for a series of per-bar returns.

Conventions: simple (not log) returns; annualisation with `periods_per_year` (365 days for
crypto); risk-free rate 0 (state it if you use another); Sharpe uses the sample standard
deviation (ddof=1). Point estimates only: their uncertainty is in tiq.stats.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from tiq.backtest.engine import BacktestResult


def equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(returns, dtype=float))


def drawdown(returns: np.ndarray) -> np.ndarray:
    """Fraction below the running peak (0 at a new high, 0.3 = 30% below the peak)."""
    eq = np.concatenate([[1.0], equity_curve(returns)])
    return (1.0 - eq / np.maximum.accumulate(eq))[1:]


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 1.0) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def rolling_sharpe(returns: np.ndarray, window: int, periods_per_year: float) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    if r.size < window:
        return out
    c1 = np.concatenate([[0.0], np.cumsum(r)])
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    mean = s1 / window
    var = np.maximum(s2 / window - mean**2, 0.0) * window / (window - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window - 1:] = np.where(var > 0, mean / np.sqrt(var) * math.sqrt(periods_per_year), np.nan)
    return out


def _longest_underwater(dd: np.ndarray) -> int:
    longest = run = 0
    for x in dd:
        run = run + 1 if x > 1e-12 else 0
        longest = max(longest, run)
    return longest


def performance(
    returns: np.ndarray, periods_per_year: float, result: BacktestResult | None = None,
    benchmark: np.ndarray | None = None, rows: slice | None = None,
) -> dict[str, float]:
    """Standard statistics. `rows` restricts everything (including `result` arrays) to a window."""
    rows = rows or slice(None)
    r = np.asarray(returns, dtype=float)[rows]
    n = r.size
    out: dict[str, float] = {"bars": float(n)}
    if n < 2:
        return out
    years = n / periods_per_year
    eq = equity_curve(r)
    dd = drawdown(r)
    mean, sd = r.mean(), r.std(ddof=1)
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    q05 = np.quantile(r, 0.05)
    out.update(
        years=years,
        total_return=float(eq[-1] - 1.0),
        cagr=float(eq[-1] ** (1.0 / years) - 1.0) if eq[-1] > 0 else -1.0,
        ann_vol=float(sd * math.sqrt(periods_per_year)),
        sharpe=float(mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        sortino=(float(mean * periods_per_year / (downside * math.sqrt(periods_per_year)))
                 if downside > 0 else float("nan")),
        max_drawdown=float(dd.max()),
        longest_drawdown_days=float(_longest_underwater(dd) / periods_per_year * 365),
        skew=float(sps.skew(r)),
        excess_kurtosis=float(sps.kurtosis(r)),
        var_95=float(-q05),
        cvar_95=float(-r[r <= q05].mean()),
        best_bar=float(r.max()),
        worst_bar=float(r.min()),
        positive_bars=float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else float("nan"),
    )
    out["calmar"] = out["cagr"] / out["max_drawdown"] if out["max_drawdown"] > 0 else float("nan")
    if result is not None:
        cost_r = result.cost_returns()[rows]
        gross_exp = result.gross_exposure[rows]
        out.update(
            turnover_per_year=float(result.turnover[rows].sum() / years),
            cost_drag_per_year=float(cost_r.sum() / years),
            avg_gross_exposure=float(gross_exp.mean()),
            time_in_market=float(np.mean(gross_exp > 0.01)),
            trades=float(result.entries[rows].sum()),
            trades_per_year=float(result.entries[rows].sum() / years),
            forced_exits=float(result.forced_exits[rows].sum()),
        )
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)[rows]
        var_b = b.var(ddof=1)
        beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
        active = r - b
        te = active.std(ddof=1)
        out.update(
            beta=beta,
            alpha_ann=float((mean - beta * b.mean()) * periods_per_year),
            correlation=float(np.corrcoef(r, b)[0, 1]) if var_b > 0 and sd > 0 else float("nan"),
            tracking_error=float(te * math.sqrt(periods_per_year)),
            information_ratio=float(active.mean() / te * math.sqrt(periods_per_year)) if te > 0 else float("nan"),
        )
    return out


def calendar_returns(returns: np.ndarray, epoch_ms: np.ndarray, freq: str = "Y") -> dict[str, float]:
    """Compounded return per calendar year ('Y') or month ('M'), keyed '2024' or '2024-03'."""
    dates = np.asarray(epoch_ms).astype("datetime64[ms]")
    keys = dates.astype("datetime64[Y]" if freq == "Y" else "datetime64[M]").astype(str)
    out: dict[str, float] = {}
    for k in dict.fromkeys(keys):
        out[k] = float(np.prod(1.0 + np.asarray(returns)[keys == k]) - 1.0)
    return out
