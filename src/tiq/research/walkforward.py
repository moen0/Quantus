"""Walk-forward validation: choose parameters on the past, trade them on the next window, repeat.

    |---- train 1 ----|-- test 1 --|
         |---- train 2 ----|-- test 2 --|
              |---- train 3 ----|-- test 3 --|      (rolling; anchored keeps the start fixed)

Only the test windows count, and they are glued together into one out-of-sample track record.
The parameters chosen in each fold use nothing from that fold's test window or later, so the
stitched record is what the selection procedure would have produced in real time. Parameter
changes between folds are executed by the backtester like any other trade, including costs.

Because every strategy and the universe are causal (row t uses data <= t), each parameter set
is backtested once over the full period and sliced, which is equivalent to re-running it with
history ending at each fold boundary. tests/test_strategies.py checks the causality.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int   # exclusive; equals test_start
    test_start: int
    test_end: int    # exclusive


def make_folds(first_row: int, n_rows: int, train_bars: int, test_bars: int, anchored: bool = False) -> list[Fold]:
    if train_bars < 2 or test_bars < 1:
        raise ValueError("train window needs >= 2 bars and test window >= 1 bar")
    folds = []
    test_start = first_row + train_bars
    while test_start < n_rows:
        test_end = min(test_start + test_bars, n_rows)
        train_start = first_row if anchored else test_start - train_bars
        folds.append(Fold(train_start, test_start, test_start, test_end))
        test_start = test_end
    return folds


def _sharpe_columns(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, r.mean(axis=0) / sd, -np.inf)


def select_trials(
    trial_returns: np.ndarray, trial_entries: np.ndarray, folds: list[Fold], min_trades: int,
    score: Callable[[np.ndarray], np.ndarray] = _sharpe_columns,
) -> tuple[np.ndarray, np.ndarray]:
    """Per fold: index of the best trial on the training window (-1 = none qualified, stay flat),
    and that trial's training score (per-period Sharpe)."""
    chosen = np.full(len(folds), -1, dtype=np.int64)
    scores = np.full(len(folds), np.nan)
    for k, f in enumerate(folds):
        r = trial_returns[f.train_start:f.train_end]
        trades = trial_entries[f.train_start:f.train_end].sum(axis=0)
        s = score(r).astype(float)
        s[trades < min_trades] = -np.inf
        if np.isfinite(s).any() and s.max() > -np.inf:
            chosen[k] = int(np.argmax(s))  # ties -> first in grid order
            scores[k] = s[chosen[k]]
    return chosen, scores


def stitch(weights_for: Callable[[int], np.ndarray], folds: list[Fold], chosen: np.ndarray,
           shape: tuple[int, int]) -> np.ndarray:
    """Target weights that follow each fold's chosen trial inside that fold's test window, flat elsewhere."""
    out = np.zeros(shape)
    cache: dict[int, np.ndarray] = {}
    for f, k in zip(folds, chosen, strict=True):
        if k < 0:
            continue
        if k not in cache:
            cache[k] = weights_for(int(k))
        out[f.test_start:f.test_end] = cache[k][f.test_start:f.test_end]
    return out
