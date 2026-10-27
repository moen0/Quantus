"""Experiment report: report.md with figures, plus summary.json for machines (or a web front end)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tiq.backtest.metrics import drawdown, equity_curve  # noqa: E402
from tiq.research.registry import git_state  # noqa: E402

if TYPE_CHECKING:
    from tiq.research.experiment import ExperimentResult

LABELS = {"strategy": "Strategy", "buy_and_hold": "Buy & hold", "vol_target_buy_and_hold": "Vol-target buy & hold"}
COLORS = {"strategy": "#1f5fa8", "buy_and_hold": "#9a9a9a", "vol_target_buy_and_hold": "#d08a2c"}


def _fmt(x: Any, kind: str = "num") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if kind == "pct":
        return f"{x * 100:.1f}%"
    if kind == "int":
        return f"{x:,.0f}"
    if kind == "p":
        return f"{x:.3f}"
    return f"{x:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [None if not np.isfinite(v) else float(v) for v in o.ravel()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def _plots(res: ExperimentResult, out: Path) -> None:
    rows = res.oos_rows
    dates = res.panel.dates[rows]
    series = {"strategy": res.strategy.returns[rows]}
    series |= {k: v.returns[rows] for k, v in res.benchmarks.items()}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, r in series.items():
        ax1.plot(dates, equity_curve(r), label=LABELS[name], color=COLORS[name], lw=1.6 if name == "strategy" else 1.1)
        ax2.plot(dates, -drawdown(r) * 100, color=COLORS[name], lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log scale)")
    how = "fixed parameters" if res.cfg.validation.mode == "fixed" else "walk-forward"
    ax1.set_title(f"{res.cfg.experiment.name}: out-of-sample ({how}), after costs")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, rs in res.stability["rolling_sharpe"].items():
        ax.plot(dates, rs, label=LABELS[name], color=COLORS[name], lw=1.4 if name == "strategy" else 1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Rolling 365-day Sharpe ratio (stability check)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "rolling_sharpe.png", dpi=130)
    plt.close(fig)

    ppy = res.panel.periods_per_year
    fig, axes = plt.subplots(1, 3 if res.pbo else 2, figsize=(13, 3.6))
    ax = axes[0]
    if res.null_means is not None and res.null_means.size:
        ax.hist(res.null_means * ppy * 100, bins=30, color="#bbbbbb")
        ax.axvline(res.sensitivity["null_actual_return_ann"] * 100, color=COLORS["strategy"], lw=2, label="actual")
        ax.set_title(f"Random-timing null (p = {res.sensitivity['null_pvalue']:.3f})")
        ax.set_xlabel("OOS mean return, % per year")
        ax.legend(frameon=False)
    ax = axes[1]
    ax.hist(res.extra["trial_sharpe_ann"], bins=min(30, max(5, len(res.trials))), color="#9db7d8")
    ax.set_title(f"Sharpe of all {len(res.trials)} variants (full period)")
    ax.set_xlabel("Sharpe")
    if res.pbo:
        ax = axes[2]
        ax.hist(res.pbo.logits, bins=30, color="#d8a29d")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"CSCV logits (PBO = {res.pbo.pbo:.2f})")
        ax.set_xlabel("logit of in-sample winner's OOS rank")
    fig.tight_layout()
    fig.savefig(out / "overfitting.png", dpi=130)
    plt.close(fig)


def render_markdown(res: ExperimentResult) -> str:
    cfg, m, st, se, stab = res.cfg, res.metrics, res.statistics, res.sensitivity, res.stability
    panel = res.panel
    d = panel.dates
    oos_start, oos_end = str(d[res.oos_rows.start])[:10], str(d[-1])[:10]
    git = git_state()
    out = [f"# {cfg.experiment.name}: {res.verdict}", ""]
    out.append(f"Family `{cfg.experiment.family}` | config `{cfg.source_path}` (sha256 {cfg.source_sha256[:12]}) | "
               f"git {git.get('commit')}{' (uncommitted changes)' if git.get('dirty') else ''} | "
               f"runtime {res.runtime_seconds:.0f}s{' | QUICK MODE' if res.extra.get('quick') else ''}")
    if cfg.experiment.hypothesis:
        out += ["", "> " + cfg.experiment.hypothesis.strip().replace("\n", "\n> ")]

    out += ["", "## Verdict", "",
            "Every criterion was fixed in the config before the run. One failure means FAIL.", ""]
    crit_rows = []
    for c in res.criteria:
        mark = "n/a" if c["pass"] is None else ("pass" if c["pass"] else "**FAIL**")
        crit_rows.append([c["name"], _fmt(c["value"], "p" if "p-value" in c["name"] or "PSR" in c["name"]
                                          or "Deflated" in c["name"] or "overfitting" in c["name"] else "num"),
                          f"{c['op']} {c['threshold']}", mark])
    out.append(_table(["Criterion", "Value", "Required", "Result"], crit_rows))

    grid_desc = ", ".join(f"{k} in {v}" for k, v in cfg.strategy.grid.items()) or "none (single variant)"
    u = cfg.universe
    uni_desc = (f"{', '.join(u.symbols)}" if u.symbols else
                f"all {u.quote} pairs in the lake, filtered point-in-time")
    if u.top_n:
        uni_desc += f"; top {u.top_n} by {u.liquidity_lookback_days:g}-day median quote volume"
    if u.min_history_days:
        uni_desc += f"; at least {u.min_history_days:g} days of history"
    if u.min_median_quote_volume:
        uni_desc += f"; median volume >= {u.min_median_quote_volume:,.0f} {u.quote}/day"
    out += ["", "## Setup", "",
            f"- **Data:** Binance {cfg.data.market} {cfg.data.interval} bars, {panel.shape[1]} symbols loaded, "
            f"{str(d[0])[:10]} to {str(d[-1])[:10]}. Out-of-sample: **{oos_start} to {oos_end}** "
            f"({m['strategy'].get('years', 0):.1f} years).",
            f"- **Universe:** {uni_desc}. Average size out-of-sample: {stab['avg_universe_size']:.1f}.",
            f"- **Strategy:** `{cfg.strategy.name}` with fixed parameters `{cfg.strategy.params}`; "
            f"grid: {grid_desc} ({len(res.trials)} variants).",
            f"- **Execution:** decide at the close, fill at the next open (delay {cfg.execution.delay}); "
            f"rebalance band {cfg.execution.rebalance_band}.",
            f"- **Costs:** fee {cfg.costs.fee_bps:g} bps + half-spread {cfg.costs.half_spread_bps:g} bps per side "
            f"+ square-root impact (coefficient {cfg.costs.impact_coef:g}, capital {cfg.costs.capital:,.0f}).",
            (f"- **Validation:** one pre-specified variant, no parameter selection, so everything from "
             f"{cfg.data.start} on is out-of-sample with respect to tuning."
             if cfg.validation.mode == "fixed" else
             f"- **Validation:** walk-forward, {'anchored' if cfg.validation.anchored else 'rolling'} "
             f"{cfg.validation.train_days:g}-day training windows, {cfg.validation.test_days:g}-day test windows "
             f"({len(res.folds)} folds); parameters chosen by training Sharpe (min {cfg.validation.min_trades} trades).")]

    out += ["", "## Out-of-sample performance (after costs)", ""]
    names = ["strategy", *res.benchmarks]
    spec = [("CAGR", "cagr", "pct"), ("Annual volatility", "ann_vol", "pct"), ("Sharpe", "sharpe", "num"),
            ("Sortino", "sortino", "num"), ("Max drawdown", "max_drawdown", "pct"), ("Calmar", "calmar", "num"),
            ("Longest drawdown (days)", "longest_drawdown_days", "int"), ("Worst bar", "worst_bar", "pct"),
            ("Skew", "skew", "num"), ("Beta to buy & hold", "beta", "num"),
            ("Time in market", "time_in_market", "pct"), ("Average gross exposure", "avg_gross_exposure", "pct"),
            ("Turnover per year", "turnover_per_year", "num"), ("Cost drag per year", "cost_drag_per_year", "pct"),
            ("Trades per year", "trades_per_year", "num"), ("Forced exits (delistings)", "forced_exits", "int")]
    out.append(_table(["", *(LABELS[n] for n in names)],
                      [[label, *(_fmt(m[n].get(key), kind) for n in names)] for label, key, kind in spec]))
    out += ["", "![equity](equity.png)"]

    lo, hi = st["oos_sharpe_ci95"]
    clo, chi = st["oos_cagr_ci95"]
    out += ["", "## How much of this is luck?", "",
            f"- **Sharpe uncertainty:** 95% bootstrap interval {_fmt(lo)} to {_fmt(hi)} "
            f"(stationary bootstrap, mean block {st['bootstrap_block']} bars); analytic standard error "
            f"{_fmt(st['oos_sharpe_se_ann'])} with skew {_fmt(st['oos_skew'])} and kurtosis {_fmt(st['oos_kurtosis'])}. "
            f"CAGR interval {_fmt(clo, 'pct')} to {_fmt(chi, 'pct')}.",
            f"- **Probabilistic Sharpe ratio:** P(true Sharpe > 0) = {_fmt(st['psr'], 'p')}. Track record needed "
            f"for 95% confidence at this Sharpe: {_fmt(st['min_track_record_years'])} years.",
            f"- **Deflated Sharpe ratio:** the best of the {st['trials_this_run']} variants had a full-period Sharpe of "
            f"{_fmt(st['best_trial_sharpe_ann'])}. With {st['trials_in_family']} distinct variants tried in this family, "
            f"pure noise would be expected to produce a best Sharpe of about {_fmt(st['dsr_sr0_ann'])}; "
            f"DSR = {_fmt(st['dsr'], 'p')} (counting only {st['effective_trials']:.1f} effective independent variants: "
            f"{_fmt(st['dsr_effective'], 'p')}).",
            ("- **Probability of backtest overfitting:** " + (
                f"{_fmt(st['pbo'], 'p')} over {st['pbo_splits']:,} CSCV splits; the in-sample winner lost money "
                f"out-of-sample in {_fmt(st['pbo_prob_oos_loss'], 'pct')} of splits; OOS-on-IS slope "
                f"{_fmt(st['pbo_degradation_slope'])}." if st["pbo"] is not None else "not applicable (one variant).")),
            f"- **SPA test (Hansen 2005):** p = {_fmt(st['spa_p'], 'p')} for 'no variant has a higher Sharpe than "
            f"{LABELS.get(st['spa_benchmark'], st['spa_benchmark'])}' (p = {_fmt(st['spa_cash_p'], 'p')} against cash).",
            f"- **Random timing:** {se['null_reps']} copies with the same exposure at random times earned a median "
            f"{_fmt(se['null_median_return_ann'], 'pct')} a year (95th percentile {_fmt(se['null_p95_return_ann'], 'pct')}), "
            f"against {_fmt(se['null_actual_return_ann'], 'pct')} for the strategy (arithmetic mean, after costs); "
            f"p = {_fmt(se['null_pvalue'], 'p')}.",
            "", "![overfitting](overfitting.png)"]

    out += ["", "## Robustness", ""]
    out.append(_table(["Cost multiple", "Sharpe", "CAGR"],
                      [[f"{k:g}x", _fmt(v["sharpe"]), _fmt(v["cagr"], "pct")] for k, v in se["by_cost_multiple"].items()]))
    out += ["", f"Break-even at {_fmt(se['breakeven_cost_multiple'])}x the assumed costs "
                f"(about {_fmt(se['breakeven_cost_bps_per_side'])} bps per side before impact). "
                f"One extra bar of latency: Sharpe {_fmt(se['extra_delay_sharpe'])}, CAGR {_fmt(se['extra_delay_cagr'], 'pct')}."]

    out += ["", "## Stability", "", "Walk-forward folds (parameters chosen on the preceding training window):", ""]
    out.append(_table(["Test window", "Chosen", "Train Sharpe", "Test Sharpe", "Test return"],
                      [[f"{f['test_start']} to {f['test_end']}", "flat" if f["params"] is None else str(f["params"] or "base"),
                        _fmt(f["train_sharpe"]), _fmt(f["test_sharpe"]), _fmt(f["test_return"], "pct")]
                       for f in stab["folds"]]))
    cal = stab["calendar"]
    years = list(cal["strategy"])
    out += ["", "Calendar-year returns (out-of-sample):", ""]
    out.append(_table(["Year", *(LABELS[n] for n in cal)], [[y, *(_fmt(cal[n].get(y), "pct") for n in cal)] for y in years]))
    if stab.get("post_publication"):
        pp = stab["post_publication"]
        out += ["", f"After publication ({cfg.experiment.published}, {pp['strategy'].get('years', 0):.1f} years): "
                    "the only period the rule's authors could not have seen.", ""]
        out.append(_table(["", *(LABELS[n] for n in pp)],
                          [[label, *(_fmt(pp[n].get(key), kind) for n in pp)]
                           for label, key, kind in (("CAGR", "cagr", "pct"), ("Sharpe", "sharpe", "num"),
                                                    ("Max drawdown", "max_drawdown", "pct"))]))
    for name, table in stab.get("regimes", {}).items():
        out += ["", f"By {name} regime (labelled with information available before each bar):", ""]
        out.append(_table(["Regime", "Share of time", "Strategy return/yr", "Strategy Sharpe", "B&H return/yr", "B&H Sharpe"],
                          [[r, _fmt(x["share_of_time"], "pct"), _fmt(x["strategy_ann_return"], "pct"),
                            _fmt(x["strategy_sharpe"]), _fmt(x["benchmark_ann_return"], "pct"), _fmt(x["benchmark_sharpe"])]
                           for r, x in table.items()]))
    out += ["", "![rolling](rolling_sharpe.png)"]

    sharpe = res.extra["trial_sharpe_ann"]
    order = np.argsort(-sharpe)[:10]
    times = {int(k): int((res.chosen == k).sum()) for k in set(res.chosen.tolist()) if k >= 0}
    keys = list(cfg.strategy.grid)
    out += ["", "## Variants (full evaluation period, top 10 by Sharpe)", ""]
    out.append(_table(["#", *(keys or ["params"]), "Sharpe", "Chosen in folds"],
                      [[str(i), *([str(res.trials[i][k]) for k in keys] or ["base"]), _fmt(float(sharpe[i])),
                        str(times.get(int(i), 0))] for i in order]))

    caveats = []
    if m["strategy"].get("years", 0) < 4:
        caveats.append(f"Short out-of-sample record ({m['strategy'].get('years', 0):.1f} years): a few market phases dominate.")
    if cfg.universe.top_n and stab["avg_universe_size"] < 0.9 * cfg.universe.top_n:
        caveats.append(f"The universe averaged {stab['avg_universe_size']:.1f} of {cfg.universe.top_n} slots; empty sleeves stayed in cash.")
    if res.extra.get("quick"):
        caveats.append("Quick mode: bootstrap, SPA and null use few replications; rerun without --quick before trusting p-values.")
    caveats.append("Costs are modelled, not measured: no order book, no partial fills, no funding or borrow costs. "
                   "Paper-trade before risking money.")
    caveats.append("Crypto has one short history with a few large cycles; out-of-sample here means unseen by the "
                   "parameter choice, not unseen by the researcher who designed the rule.")
    out += ["", "## Caveats", ""] + [f"- {c}" for c in caveats]
    return "\n".join(out) + "\n"


def write_report(res: ExperimentResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plots(res, out_dir)
    (out_dir / "report.md").write_text(render_markdown(res))
    summary = {
        "experiment": res.cfg.to_dict(), "verdict": res.verdict, "criteria": res.criteria,
        "metrics": res.metrics, "statistics": res.statistics, "sensitivity": res.sensitivity,
        "folds": res.stability["folds"], "calendar": res.stability["calendar"],
        "regimes": res.stability["regimes"], "post_publication": res.stability["post_publication"],
        "runtime_seconds": res.runtime_seconds,
        "oos_dates": [str(x)[:10] for x in res.panel.dates[res.oos_rows]],
        "oos_equity": {"strategy": equity_curve(res.strategy.returns[res.oos_rows])}
        | {k: equity_curve(v.returns[res.oos_rows]) for k, v in res.benchmarks.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(_sanitize(summary), indent=1, default=_json_default))
    return out_dir / "report.md"


def _sanitize(o: Any) -> Any:
    """Replace NaN/inf with None so the JSON is standard (browsers reject NaN)."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_sanitize(float(v)) for v in o.ravel()]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o
