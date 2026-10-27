"""Walk-forward mechanics, nulls, config validation, registry, and a full experiment on a synthetic lake."""
import json
from datetime import timedelta

import numpy as np
import pytest

from tests.conftest import START, kline_frame, random_walk
from tiq.data.lake import Lake
from tiq.research.config import parse_config
from tiq.research.experiment import run_experiment
from tiq.research.nulls import null_pvalue, rotate_weights
from tiq.research.registry import family_trial_ids, read_registry
from tiq.research.walkforward import make_folds, select_trials, stitch


def test_folds_rolling_and_anchored():
    folds = make_folds(10, 100, train_bars=30, test_bars=20)
    assert [(f.test_start, f.test_end) for f in folds] == [(40, 60), (60, 80), (80, 100)]
    assert all(f.train_end == f.test_start and f.train_end - f.train_start == 30 for f in folds)
    anchored = make_folds(10, 100, 30, 20, anchored=True)
    assert all(f.train_start == 10 for f in anchored)
    assert make_folds(10, 35, 30, 20) == []


def test_selection_uses_only_training_windows():
    T = 120
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (T, 2))
    folds = make_folds(0, T, 40, 20)
    for f in folds:
        R[f.train_start:f.train_end, 0] += 0.01    # trial 0 shines in every training window...
    for f in folds:
        R[f.test_start:f.test_end, 1] += 0.05      # ...trial 1 in every test window (must not leak)
    for f in folds:
        R[f.train_start:f.train_end, 1] -= 0.05
    E = np.ones((T, 2))
    chosen, _ = select_trials(R, E, folds, min_trades=1)
    assert (chosen == 0).all()


def test_min_trades_and_stitching():
    T = 100
    R = np.random.default_rng(1).normal(0, 0.01, (T, 2))
    E = np.zeros((T, 2))
    E[::10, 1] = 1                                 # only trial 1 trades
    folds = make_folds(0, T, 40, 30)
    chosen, _ = select_trials(R, E, folds, min_trades=2)
    assert (chosen == 1).all()
    chosen_none, _ = select_trials(R, np.zeros((T, 2)), folds, min_trades=1)
    assert (chosen_none == -1).all()
    weights = {0: np.full((T, 1), 0.3), 1: np.full((T, 1), 0.7)}
    w = stitch(lambda k: weights[k], folds, np.array([1, -1]), (T, 1))
    assert (w[:40] == 0).all() and (w[40:70] == 0.7).all() and (w[70:] == 0).all()


def test_rotation_preserves_exposure_inside_universe_segments():
    w = np.zeros((50, 2))
    w[10:20, 0] = 0.5
    w[30:35, 1] = 0.2
    valid = np.zeros((50, 2), dtype=bool)
    valid[5:40, 0] = True
    valid[25:50, 1] = True
    out = rotate_weights(w, valid, slice(0, 50), np.random.default_rng(0))
    assert sorted(out[:, 0]) == sorted(w[:, 0]) and sorted(out[:, 1]) == sorted(w[:, 1])
    assert (out[~valid] == 0).all()
    assert null_pvalue(1.0, np.array([0.5, 2.0, 0.1])) == pytest.approx(0.5)


BASE = {
    "experiment": {"name": "synthetic", "family": "test-family"},
    "data": {"interval": "1d", "start": "2020-01-01"},
    "universe": {"symbols": ["AUSDT", "BUSDT", "CUSDT"]},
    "strategy": {"name": "tsmom", "params": {"target_vol": 0.3}, "grid": {"lookback_days": [20, 60, 120]}},
    "validation": {"train_days": 365, "test_days": 120, "bootstrap_reps": 50, "null_reps": 10, "spa_reps": 50},
}


def test_config_validation():
    cfg = parse_config(BASE)
    assert [t["lookback_days"] for t in cfg.trial_grid()] == [20, 60, 120]
    with pytest.raises(ValueError, match="unknown key"):
        parse_config({**BASE, "data": {"intreval": "1d"}})
    with pytest.raises(ValueError, match="unknown section"):
        parse_config({**BASE, "risk": {}})
    with pytest.raises(ValueError, match="fixed"):
        parse_config({**BASE, "validation": {"mode": "fixed"}})
    with pytest.raises(ValueError, match="1.0"):
        parse_config({**BASE, "validation": {"cost_multipliers": [0, 2]}})


@pytest.fixture
def synthetic_lake(tmp_path):
    lake = Lake(tmp_path / "lake")
    closes = random_walk(1400, 3, seed=9, drift=0.0004, vol=0.03)
    for j, sym in enumerate(["AUSDT", "BUSDT", "CUSDT"]):
        df = kline_frame(closes[:, j], start=START - timedelta(days=400), volume=5e4)
        keys = [df["open_time"].dt.year().alias("y"), df["open_time"].dt.month().alias("m")]
        for (year, month), part in df.group_by(keys):
            lake.write_month(lake.kline_dir("spot", "1d", sym), f"{year:04d}-{month:02d}", part, "open_time")
    return lake


def test_full_experiment_on_synthetic_lake(synthetic_lake, tmp_path):
    reg = tmp_path / "registry.jsonl"
    cfg = parse_config(BASE, "synthetic.toml", "abc")
    res = run_experiment(cfg, synthetic_lake, tmp_path / "out", quick=True, workers=2, registry_file=reg)
    assert res.verdict in ("PASS", "FAIL")
    assert {c["name"] for c in res.criteria} >= {"OOS Sharpe after costs", "Probability of backtest overfitting"}
    assert (tmp_path / "out" / "report.md").exists() and (tmp_path / "out" / "equity.png").exists()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["verdict"] == res.verdict and len(summary["oos_dates"]) == len(summary["oos_equity"]["strategy"])
    # out-of-sample starts one training window after `start`, and the stitched record has no NaN
    assert res.panel.dates[res.oos_rows.start - 1] >= np.datetime64("2020-12-30")
    assert np.isfinite(res.strategy.returns).all()
    # registry: identical rerun does not inflate the family's trial count; a new variant does
    assert len(read_registry(reg)) == 1 and len(family_trial_ids("test-family", reg)) == 3
    run_experiment(cfg, synthetic_lake, None, quick=True, workers=2, registry_file=reg)
    assert len(family_trial_ids("test-family", reg)) == 3
    wider = parse_config({**BASE, "strategy": {**BASE["strategy"], "grid": {"lookback_days": [20, 60, 120, 240]}}})
    res2 = run_experiment(wider, synthetic_lake, None, quick=True, workers=2, registry_file=reg)
    assert res2.statistics["trials_in_family"] == 4


def test_fixed_mode_has_no_selection(synthetic_lake, tmp_path):
    raw = {**BASE, "strategy": {"name": "donchian_ensemble", "params": {"lookbacks": [10, 30, 90]}},
           "validation": {**BASE["validation"], "mode": "fixed"}}
    res = run_experiment(parse_config(raw), synthetic_lake, None, quick=True, workers=2,
                         registry_file=tmp_path / "r.jsonl")
    assert res.oos_rows.start == res.eval_rows.start + 1 and len(res.folds) == 1
    assert res.statistics["pbo"] is None
    assert any(c["pass"] is None for c in res.criteria)   # PBO not applicable, not a failure
