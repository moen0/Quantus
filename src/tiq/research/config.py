"""Experiment configuration: one TOML file per experiment, validated strictly.

Unknown sections or keys are errors, so a typo cannot silently fall back to a default. The file
is the pre-registration: hypothesis, data, universe, parameter grid, costs, validation settings
and the kill criteria are all fixed before the run, and the report records the file's hash.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from tiq.backtest.costs import CostModel
from tiq.data.universe import UniverseConfig


@dataclass(frozen=True)
class ExperimentMeta:
    name: str
    family: str                  # trials are counted across every run in a family (deflated Sharpe)
    hypothesis: str = ""
    notes: str = ""
    published: str | None = None  # date the rule was published; the report adds a post-publication section


@dataclass(frozen=True)
class DataConfig:
    market: str = "spot"
    interval: str = "1d"
    source_interval: str | None = None  # stored interval to resample from (default: same as interval)
    start: str = "2019-01-01"           # first training window starts here; out-of-sample starts one
                                        # training window later
    end: str | None = None              # exclusive; None = all data in the lake
    history_start: str | None = None    # load from here for indicator warm-up; None = everything stored
    max_fill_days: float = 5.0          # gaps up to this long are carried forward (outages; Binance's
                                        # longest was 75 hours in Feb 2018); longer ones split the series
                                        # (delistings, re-used tickers: LUNA 17 days, FTT 310 days)


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConfig:
    delay: int = 1                 # bars between decision (close) and fill (open); >= 1
    rebalance_band: float = 0.0    # skip weight changes smaller than this (exits always execute)
    max_stale_bars: int = 0        # liquidate after this many bars without a price
    delist_haircut: float = 0.0    # extra loss on forced liquidation, as a fraction of the last price


@dataclass(frozen=True)
class ValidationConfig:
    mode: str = "walk_forward"     # "walk_forward": choose among the grid on past data, fold by fold;
                                   # "fixed": one pre-specified variant, out-of-sample from `start`
    train_days: float = 730.0
    test_days: float = 182.0
    anchored: bool = False         # True: training window grows from `start`; False: rolling window
    min_trades: int = 3            # a parameter set needs this many entries in a training window
    cost_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    bootstrap_reps: int = 2000
    null_reps: int = 300
    spa_reps: int = 2000
    pbo_blocks: int = 16
    spa_benchmark: str = "vol_target_buy_and_hold"
    seed: int = 7


@dataclass(frozen=True)
class KillCriteria:
    min_oos_sharpe: float = 0.5        # walk-forward out-of-sample, after costs
    min_psr: float = 0.95              # P(true OOS Sharpe > 0)
    min_dsr: float = 0.95              # best full-sample variant, deflated by the family's trial count
    max_pbo: float = 0.5
    max_spa_pvalue: float = 0.10       # Sharpe-based SPA test against `spa_benchmark`
    max_null_pvalue: float = 0.10      # random-timing null with identical exposure
    min_breakeven_cost_multiple: float = 2.0
    max_drawdown: float = 0.5
    must_beat: tuple[str, ...] = ("buy_and_hold", "vol_target_buy_and_hold")  # OOS Sharpe must exceed


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentMeta
    data: DataConfig
    universe: UniverseConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    costs: CostModel
    validation: ValidationConfig
    kill_criteria: KillCriteria
    source_path: str = ""
    source_sha256: str = ""

    def trial_grid(self) -> list[dict[str, Any]]:
        """Every parameter set to test: fixed params merged with each point of the grid."""
        keys = sorted(self.strategy.grid)
        combos = itertools.product(*(self.strategy.grid[k] for k in keys)) if keys else [()]
        return [{**self.strategy.params, **dict(zip(keys, c, strict=True))} for c in combos]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "experiment": ExperimentMeta, "data": DataConfig, "universe": UniverseConfig, "strategy": StrategyConfig,
    "execution": ExecutionConfig, "costs": CostModel, "validation": ValidationConfig,
    "kill_criteria": KillCriteria,
}
_REQUIRED = {"experiment", "strategy"}


def _build(cls, raw: dict[str, Any], section: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"[{section}] unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {}
    for k, v in raw.items():
        if isinstance(v, list) and k not in ("params", "grid"):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def parse_config(raw: dict[str, Any], source_path: str = "", source_sha256: str = "") -> ExperimentConfig:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}; allowed: {sorted(_SECTIONS)}")
    missing = _REQUIRED - set(raw)
    if missing:
        raise ValueError(f"missing required section(s) {sorted(missing)}")
    built = {name: _build(cls, raw.get(name, {}), name) for name, cls in _SECTIONS.items()}
    cfg = ExperimentConfig(**built, source_path=source_path, source_sha256=source_sha256)
    if cfg.execution.delay < 1:
        raise ValueError("[execution] delay must be >= 1")
    if not cfg.validation.cost_multipliers or 1.0 not in cfg.validation.cost_multipliers:
        raise ValueError("[validation] cost_multipliers must include 1.0 (the base case)")
    for k, v in cfg.strategy.grid.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"[strategy.grid] {k} must be a non-empty list")
    if cfg.validation.mode not in ("walk_forward", "fixed"):
        raise ValueError("[validation] mode must be 'walk_forward' or 'fixed'")
    if cfg.validation.mode == "fixed" and len(cfg.trial_grid()) != 1:
        raise ValueError("[validation] mode 'fixed' tests one pre-specified variant: leave [strategy.grid] empty")
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    data = path.read_bytes()
    return parse_config(tomllib.loads(data.decode()), str(path), hashlib.sha256(data).hexdigest())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
