"""Run one pre-registered experiment end to end.

    config -> panel -> point-in-time universe -> every parameter set backtested
           -> walk-forward selection -> stitched out-of-sample track record
           -> benchmarks, statistics, robustness checks -> kill criteria -> report + registry

The verdict is mechanical: every kill criterion in the config must pass. A FAIL is a valid,
useful result; the point of the bench is to make it hard to fool yourself.
"""
from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiq import indicators as ind
from tiq.backtest.costs import CostModel
from tiq.backtest.engine import BacktestResult, MarketInputs, run_backtest
from tiq.backtest.metrics import calendar_returns, performance, rolling_sharpe
from tiq.data.lake import Lake
from tiq.data.panel import Panel, load_panel
from tiq.data.universe import build_universe, static_mask
from tiq.research import registry
from tiq.research.config import ExperimentConfig
from tiq.research.nulls import null_pvalue, timing_null
from tiq.research.walkforward import Fold, make_folds, select_trials, stitch
from tiq.stats.overfitting import PBOResult, pbo_cscv, suggested_blocks
from tiq.stats.resampling import bootstrap_ci, spa_test
from tiq.stats.sharpe import (
    deflated_sharpe_ratio,
    effective_trials,
    min_track_record_length,
    psr_from_stats,
    sample_moments,
    sharpe_std_error,
)
from tiq.strategies import Context, get_strategy
from tiq.strategies.base import days_to_bars

log = logging.getLogger(__name__)
BENCHMARKS = ("buy_and_hold", "vol_target_buy_and_hold")


@dataclass
class ExperimentResult:
    cfg: ExperimentConfig
    panel: Panel
    universe: np.ndarray
    folds: list[Fold]
    chosen: np.ndarray
    fold_train_sharpe: np.ndarray
    trials: list[dict[str, Any]]
    trial_ids: list[str]
    trial_returns: np.ndarray
    trial_entries: np.ndarray
    eval_rows: slice
    oos_rows: slice
    strategy: BacktestResult
    benchmarks: dict[str, BacktestResult]
    metrics: dict[str, dict[str, float]]
    statistics: dict[str, Any]
    sensitivity: dict[str, Any]
    stability: dict[str, Any]
    criteria: list[dict[str, Any]]
    verdict: str
    pbo: PBOResult | None = None
    null_means: np.ndarray | None = None
    runtime_seconds: float = 0.0
    out_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _sharpe_pp(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if r.size > 1 and sd > 0 else 0.0


def _cagr(r: np.ndarray, ppy: float) -> float:
    g = float(np.prod(1.0 + r))
    return g ** (ppy / r.size) - 1.0 if g > 0 and r.size else -1.0


def _normalize(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def resolve_symbols(cfg: ExperimentConfig, lake: Lake) -> list[str]:
    interval = cfg.data.source_interval or cfg.data.interval
    stored = lake.kline_symbols(cfg.data.market, interval)
    hint = f"run: tiq backfill --interval {interval} --symbols ..."
    if cfg.universe.symbols:
        missing = [s for s in cfg.universe.symbols if s not in stored]
        if missing:
            raise FileNotFoundError(f"not in the lake at {interval}: {missing}; {hint}")
        return list(cfg.universe.symbols)
    symbols = [s for s in stored if s.endswith(cfg.universe.quote)]
    if not symbols:
        raise FileNotFoundError(f"no {cfg.universe.quote} symbols in the lake at {interval}; {hint}")
    return symbols


def _regimes(panel: Panel, rows: slice, strat_r: np.ndarray, bench_r: np.ndarray) -> dict[str, Any]:
    """Performance by BTC trend regime (close vs 200-day average) and BTC volatility tercile.
    Regimes are labelled with information available before each bar (no look-ahead)."""
    if "BTCUSDT" not in panel.symbols:
        return {}
    btc = panel.close[:, [panel.col("BTCUSDT")]]
    sma = ind.rolling_mean(btc, days_to_bars(200, panel))[:, 0]
    vol = ind.annualized_vol(btc, days_to_bars(30, panel, 2), panel.periods_per_year)[:, 0]
    prev_close, prev_sma, prev_vol = (np.concatenate([[np.nan], a[:-1]]) for a in (btc[:, 0], sma, vol))
    above = np.where(prev_close > prev_sma, "BTC above 200d avg", "BTC below 200d avg")
    trend = np.where(np.isfinite(prev_sma), above, "")
    tercile = np.full(panel.shape[0], "", dtype=object)
    hist: list[float] = []
    for t in range(panel.shape[0]):  # expanding quantiles: thresholds use only earlier observations
        v = prev_vol[t]
        if np.isfinite(v):
            if len(hist) >= 250:
                lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
                tercile[t] = "low vol" if v < lo else ("high vol" if v > hi else "mid vol")
            hist.append(v)
    ppy = panel.periods_per_year
    out: dict[str, Any] = {}
    for name, labels in (("trend", trend), ("volatility", tercile)):
        lab = np.asarray(labels, dtype=object)[rows]
        rows_out = {}
        for regime in sorted({x for x in lab if x}):
            m = lab == regime
            s, b = strat_r[rows][m], bench_r[rows][m]
            rows_out[regime] = {
                "share_of_time": float(m.mean()),
                "strategy_ann_return": float(s.mean() * ppy), "strategy_sharpe": _sharpe_pp(s) * math.sqrt(ppy),
                "benchmark_ann_return": float(b.mean() * ppy), "benchmark_sharpe": _sharpe_pp(b) * math.sqrt(ppy),
            }
        out[name] = rows_out
    return out


def run_experiment(
    cfg: ExperimentConfig, lake: Lake | None = None, out_dir: Path | None = None, quick: bool = False,
    workers: int | None = None, use_registry: bool = True, registry_file: Path | None = None,
) -> ExperimentResult:
    t_start = time.time()
    lake = lake or Lake()
    workers = workers or os.cpu_count() or 4
    v = cfg.validation
    if quick:
        v = replace(v, bootstrap_reps=min(v.bootstrap_reps, 200), null_reps=min(v.null_reps, 40),
                    spa_reps=min(v.spa_reps, 200))

    # ---- data and universe -------------------------------------------------------------------
    symbols = resolve_symbols(cfg, lake)
    log.info("loading %d symbols at %s", len(symbols), cfg.data.interval)
    panel = load_panel(lake, cfg.data.market, cfg.data.interval, symbols, cfg.data.history_start, cfg.data.end,
                       cfg.data.source_interval, cfg.data.max_fill_days)
    universe = build_universe(panel, cfg.universe)
    slots = cfg.universe.top_n or int(static_mask(panel.symbols, cfg.universe).sum())
    ctx = Context(panel, universe, slots)
    ppy = panel.periods_per_year
    T, N = panel.shape
    market = MarketInputs.from_panel(panel, cfg.costs.adv_lookback_days)
    ex = cfg.execution

    def backtest(weights: np.ndarray, costs: CostModel | None = None, delay: int | None = None,
                 allow_short: bool = False) -> BacktestResult:
        return run_backtest(panel, weights, costs or cfg.costs, market, delay or ex.delay, ex.rebalance_band,
                            ex.max_stale_bars, ex.delist_haircut, allow_short)

    eval_start = panel.position(cfg.data.start)
    if eval_start >= T - 2:
        raise ValueError(f"no data after {cfg.data.start}")
    if v.mode == "fixed":  # nothing is chosen, so everything from `start` is out-of-sample
        folds = [Fold(eval_start, eval_start, eval_start, T)]
    else:
        folds = make_folds(eval_start, T, days_to_bars(v.train_days, panel, 2), days_to_bars(v.test_days, panel),
                           v.anchored)
    if not folds:
        raise ValueError(f"not enough data after {cfg.data.start} for a {v.train_days}-day training window")
    eval_rows = slice(eval_start, T)
    oos_rows = slice(folds[0].test_start + 1, T)

    # ---- every parameter set ----------------------------------------------------------------
    strategy = get_strategy(cfg.strategy.name)
    trials = [strategy.params(p) for p in cfg.trial_grid()]
    universe_key = {k: getattr(cfg.universe, k) for k in cfg.universe.__dataclass_fields__}
    trial_ids = [registry.trial_id(strategy.name, p, universe_key, cfg.data.interval) for p in trials]
    log.info("%s: %d parameter set(s), %d bars x %d assets", strategy.name, len(trials), T, N)

    def run_trial(k: int) -> tuple[np.ndarray, np.ndarray]:
        res = backtest(strategy.weights(ctx, trials[k]), allow_short=trials[k]["allow_short"])
        return res.returns, res.entries

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(run_trial, range(len(trials))))
    R = np.column_stack([o[0] for o in outputs])
    E = np.column_stack([o[1] for o in outputs])

    # ---- walk-forward selection and the stitched out-of-sample record -------------------------
    if v.mode == "fixed":
        chosen, fold_scores = np.zeros(1, dtype=np.int64), np.full(1, np.nan)
    else:
        chosen, fold_scores = select_trials(R, E, folds, v.min_trades)
    stitched = stitch(lambda k: strategy.weights(ctx, trials[k]), folds, chosen, (T, N))
    any_short = any(p["allow_short"] for p in trials)
    strat = backtest(stitched, allow_short=any_short)

    base = strategy.params(cfg.strategy.params)
    bench_params = {
        "buy_and_hold": {},
        "vol_target_buy_and_hold": {k: base[k] for k in ("vol_lookback_days", "max_asset_leverage", "max_gross")}
        | {"target_vol": base["target_vol"] or 0.25},
    }
    benchmarks = {name: backtest(get_strategy(name).weights(ctx, get_strategy(name).params(bench_params[name])))
                  for name in BENCHMARKS}
    bh_r = benchmarks["buy_and_hold"].returns

    metrics = {"strategy": performance(strat.returns, ppy, strat, bh_r, oos_rows)}
    for name, res in benchmarks.items():
        metrics[name] = performance(res.returns, ppy, res, bh_r, oos_rows)
    post_publication: dict[str, dict[str, float]] = {}
    if cfg.experiment.published:
        pub = max(panel.position(cfg.experiment.published), oos_rows.start)
        if T - pub > 30:
            rows_pub = slice(pub, T)
            post_publication = {name: performance(res.returns, ppy, res, bh_r, rows_pub)
                                for name, res in (("strategy", strat), *benchmarks.items())}

    # ---- statistics ---------------------------------------------------------------------------
    r_oos = strat.returns[oos_rows]
    sr, skew, kurt, n = sample_moments(r_oos)
    sharpe_ci = bootstrap_ci(r_oos, lambda x: _sharpe_pp(x) * math.sqrt(ppy), v.bootstrap_reps, seed=v.seed)
    cagr_ci = bootstrap_ci(r_oos, lambda x: _cagr(x, ppy), v.bootstrap_reps, seed=v.seed)
    R_eval = R[eval_rows]
    trial_sr = np.array([_sharpe_pp(R_eval[:, k]) for k in range(R.shape[1])])
    best = int(np.argmax(trial_sr))
    n_eval = R_eval.shape[0]
    var_sr = float(np.var(trial_sr, ddof=1)) if len(trials) > 1 else 1.0 / max(n_eval - 1, 1)
    family_ids = (registry.family_trial_ids(cfg.experiment.family, registry_file) if use_registry else set())
    family_ids |= set(trial_ids)
    n_family = len(family_ids)
    dsr, sr0 = deflated_sharpe_ratio(R_eval[:, best], n_family, var_sr)
    n_eff = effective_trials(R_eval) if len(trials) > 1 else 1.0
    dsr_eff, sr0_eff = deflated_sharpe_ratio(R_eval[:, best], max(1.0, n_eff * n_family / len(trials)), var_sr)

    pbo = None
    if len(trials) >= 2 and n_eval >= 40:
        pbo = pbo_cscv(R_eval, suggested_blocks(n_eval, v.pbo_blocks))

    spa_bench = v.spa_benchmark
    if spa_bench == "cash":
        spa = spa_test(np.zeros(n_eval), R_eval, v.spa_reps, v.seed)
    else:
        b_eval = benchmarks[spa_bench].returns[eval_rows]
        spa = spa_test(_normalize(b_eval[:, None])[:, 0], _normalize(R_eval), v.spa_reps, v.seed)
    spa_cash = spa_test(np.zeros(n_eval), R_eval, v.spa_reps, v.seed)

    statistics = {
        "oos_sharpe_per_period": sr, "oos_skew": skew, "oos_kurtosis": kurt, "oos_bars": n,
        "oos_sharpe_se_ann": sharpe_std_error(sr, n, skew, kurt) * math.sqrt(ppy),
        "oos_sharpe_ci95": sharpe_ci[:2], "oos_cagr_ci95": cagr_ci[:2], "bootstrap_block": sharpe_ci[2],
        "psr": psr_from_stats(sr, n, skew, kurt, 0.0),
        "min_track_record_years": min_track_record_length(r_oos, 0.0, 0.95) / ppy,
        "trials_this_run": len(trials), "trials_in_family": n_family, "effective_trials": n_eff,
        "best_trial": best, "best_trial_sharpe_ann": float(trial_sr[best] * math.sqrt(ppy)),
        "trial_sharpe_std_ann": float(math.sqrt(var_sr) * math.sqrt(ppy)),
        "dsr": dsr, "dsr_sr0_ann": sr0 * math.sqrt(ppy), "dsr_effective": dsr_eff,
        "dsr_effective_sr0_ann": sr0_eff * math.sqrt(ppy),
        "pbo": pbo.pbo if pbo else None,
        "pbo_prob_oos_loss": pbo.prob_oos_loss if pbo else None,
        "pbo_degradation_slope": pbo.degradation_slope if pbo else None,
        "pbo_splits": pbo.n_splits if pbo else None,
        "spa_benchmark": spa_bench, "spa_p": spa["p_consistent"], "spa_p_lower": spa["p_lower"],
        "spa_p_upper": spa["p_upper"], "spa_cash_p": spa_cash["p_consistent"],
    }

    # ---- robustness: costs, latency, random timing ----------------------------------------------
    by_cost = {}
    for m in v.cost_multipliers:
        res = strat if m == 1.0 else backtest(stitched, cfg.costs.scaled(m), allow_short=any_short)
        rr = res.returns[oos_rows]
        by_cost[m] = {"sharpe": _sharpe_pp(rr) * math.sqrt(ppy), "cagr": _cagr(rr, ppy), "mean": float(rr.mean())}
    gross_mean = (by_cost[0.0]["mean"] if 0.0 in by_cost
                  else float(backtest(stitched, cfg.costs.scaled(0.0), allow_short=any_short).returns[oos_rows].mean()))
    net_mean = by_cost[1.0]["mean"]
    if gross_mean <= 0:
        breakeven = 0.0
    elif gross_mean <= net_mean:
        breakeven = float("inf")
    else:
        breakeven = gross_mean / (gross_mean - net_mean)
    late = backtest(stitched, delay=ex.delay + 1, allow_short=any_short).returns[oos_rows]
    null = timing_null(stitched, universe, oos_rows, lambda w: backtest(w, allow_short=any_short).returns,
                       v.null_reps, v.seed, workers)
    sensitivity = {
        "by_cost_multiple": by_cost, "breakeven_cost_multiple": breakeven,
        "breakeven_cost_bps_per_side": breakeven * (cfg.costs.fee_bps + cfg.costs.half_spread_bps),
        "extra_delay_sharpe": _sharpe_pp(late) * math.sqrt(ppy), "extra_delay_cagr": _cagr(late, ppy),
        "null_pvalue": null_pvalue(float(r_oos.mean()), null), "null_actual_return_ann": float(r_oos.mean() * ppy),
        "null_median_return_ann": float(np.median(null) * ppy),
        "null_p95_return_ann": float(np.quantile(null, 0.95) * ppy), "null_reps": int(null.size),
    }

    # ---- stability ----------------------------------------------------------------------------
    fold_rows = []
    for f, k, s in zip(folds, chosen, fold_scores, strict=True):
        rr = strat.returns[f.test_start + 1:f.test_end] if f.test_end - f.test_start > 2 else np.zeros(2)
        fold_rows.append({
            "test_start": str(panel.dates[f.test_start])[:10], "test_end": str(panel.dates[f.test_end - 1])[:10],
            "chosen": int(k), "params": {kk: trials[k][kk] for kk in cfg.strategy.grid} if k >= 0 else None,
            "train_sharpe": float(s * math.sqrt(ppy)) if np.isfinite(s) else None,
            "test_sharpe": _sharpe_pp(rr) * math.sqrt(ppy), "test_return": float(np.prod(1 + rr) - 1),
        })
    idx_oos = panel.index[oos_rows]
    window = days_to_bars(365, panel, 2)
    stability = {
        "folds": fold_rows,
        "calendar": {name: calendar_returns(res.returns[oos_rows], idx_oos)
                     for name, res in (("strategy", strat), *benchmarks.items())},
        "rolling_sharpe_window_days": 365,
        "rolling_sharpe": {name: rolling_sharpe(res.returns[oos_rows], window, ppy)
                           for name, res in (("strategy", strat), *benchmarks.items())},
        "regimes": _regimes(panel, oos_rows, strat.returns, bh_r),
        "avg_universe_size": float(universe[oos_rows].sum(axis=1).mean()),
        "post_publication": post_publication,
    }

    # ---- kill criteria ------------------------------------------------------------------------
    k = cfg.kill_criteria
    ms = metrics["strategy"]
    rows: list[tuple[str, float | None, str, float]] = [
        ("OOS Sharpe after costs", ms.get("sharpe"), ">=", k.min_oos_sharpe),
        ("PSR: P(true OOS Sharpe > 0)", statistics["psr"], ">=", k.min_psr),
        (f"Deflated Sharpe of best variant (N={n_family})", dsr, ">=", k.min_dsr),
        ("Probability of backtest overfitting", statistics["pbo"], "<=", k.max_pbo),
        (f"SPA p-value vs {spa_bench}", statistics["spa_p"], "<=", k.max_spa_pvalue),
        ("Random-timing null p-value", sensitivity["null_pvalue"], "<=", k.max_null_pvalue),
        ("Break-even cost multiple", breakeven, ">=", k.min_breakeven_cost_multiple),
        ("OOS max drawdown", ms.get("max_drawdown"), "<=", k.max_drawdown),
    ]
    for name in k.must_beat:
        if name in metrics:
            gap = ms.get("sharpe", float("nan")) - metrics[name]["sharpe"]
            rows.append((f"OOS Sharpe minus {name}", gap, ">=", 0.0))
    criteria = []
    for name, value, op, threshold in rows:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            criteria.append({"name": name, "value": None, "op": op, "threshold": threshold, "pass": None})
            continue
        ok = value >= threshold if op == ">=" else value <= threshold
        criteria.append({"name": name, "value": float(value), "op": op, "threshold": threshold, "pass": bool(ok)})
    verdict = "PASS" if all(c["pass"] is not False for c in criteria) else "FAIL"

    result = ExperimentResult(
        cfg=cfg, panel=panel, universe=universe, folds=folds, chosen=chosen, fold_train_sharpe=fold_scores,
        trials=trials, trial_ids=trial_ids, trial_returns=R, trial_entries=E, eval_rows=eval_rows,
        oos_rows=oos_rows, strategy=strat, benchmarks=benchmarks, metrics=metrics, statistics=statistics,
        sensitivity=sensitivity, stability=stability, criteria=criteria, verdict=verdict, pbo=pbo,
        null_means=null, runtime_seconds=time.time() - t_start,
    )
    result.extra["trial_sharpe_ann"] = trial_sr * math.sqrt(ppy)
    result.extra["quick"] = quick

    if out_dir is not None:
        from tiq.research.report import write_report
        write_report(result, out_dir)
        result.out_dir = out_dir
    if use_registry:
        registry.append_entry({
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "experiment": cfg.experiment.name, "family": cfg.experiment.family,
            "config": cfg.source_path, "config_sha256": cfg.source_sha256, "git": registry.git_state(),
            "quick": quick, "verdict": verdict, "trials": len(trials), "trial_ids": trial_ids,
            "oos_sharpe": ms.get("sharpe"), "psr": statistics["psr"], "dsr": dsr, "pbo": statistics["pbo"],
            "spa_p": statistics["spa_p"], "null_p": sensitivity["null_pvalue"],
            "report": str(out_dir) if out_dir else None,
        }, registry_file)
    log.info("verdict %s in %.1fs", verdict, result.runtime_seconds)
    return result
