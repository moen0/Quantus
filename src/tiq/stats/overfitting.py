"""Probability of Backtest Overfitting via combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado, Zhu (2017), "The probability of backtest overfitting", Journal
of Computational Finance.

Take the (T x N) matrix of returns of all N parameter sets that were tried. Cut time into S
blocks. For every way of choosing S/2 blocks as "in-sample" (the rest is "out-of-sample"):
pick the trial with the best in-sample Sharpe and look up the rank of that trial's
out-of-sample Sharpe among all trials. PBO is the fraction of splits in which the in-sample
winner lands in the bottom half out-of-sample.

PBO near 0: whatever wins in-sample keeps doing well. Near 0.5: selection is no better than
picking at random. Above 0.5: selecting the best backtest actively picks noise.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    pbo: float                 # P(in-sample winner is below the out-of-sample median)
    logits: np.ndarray         # logit of the winner's relative OOS rank, one per split
    is_sharpe: np.ndarray      # winner's in-sample Sharpe per split (per period)
    oos_sharpe: np.ndarray     # winner's out-of-sample Sharpe per split (per period)
    degradation_slope: float   # regression slope of OOS on IS Sharpe across splits
    prob_oos_loss: float       # fraction of splits where the winner loses money out of sample
    n_splits: int
    n_blocks: int


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: np.ndarray) -> np.ndarray:
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, 0.0) * n / np.maximum(n - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0, mean / np.sqrt(var), 0.0)


def pbo_cscv(trial_returns: np.ndarray, n_blocks: int = 16, max_splits: int | None = None,
             seed: int = 0) -> PBOResult:
    x = np.asarray(trial_returns, dtype=float)
    T, N = x.shape
    if N < 2:
        raise ValueError("PBO needs at least two trials")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    block_len = T // n_blocks
    if block_len < 2:
        raise ValueError(f"{T} rows is too short for {n_blocks} blocks")
    x = x[: block_len * n_blocks].reshape(n_blocks, block_len, N)
    b1, b2 = x.sum(axis=1), (x**2).sum(axis=1)          # (S, N) per-block sums
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    member = np.zeros((len(combos), n_blocks))
    for k, c in enumerate(combos):
        member[k, list(c)] = 1.0
    n_is = member.sum(axis=1, keepdims=True) * block_len
    n_oos = (n_blocks - member.sum(axis=1, keepdims=True)) * block_len
    sr_is = _sharpe_from_sums(member @ b1, member @ b2, n_is)
    sr_oos = _sharpe_from_sums((1 - member) @ b1, (1 - member) @ b2, n_oos)
    best = np.argmax(sr_is, axis=1)
    rows = np.arange(len(combos))
    chosen_oos = sr_oos[rows, best]
    # relative rank in (0, 1): ties count half, so a trial equal to all others sits at the middle
    below = (sr_oos < chosen_oos[:, None]).sum(axis=1)
    ties = (sr_oos == chosen_oos[:, None]).sum(axis=1)
    omega = (below + 0.5 * ties) / (N + 1.0) + 0.5 / (N + 1.0)
    logits = np.log(omega / (1.0 - omega))
    chosen_is = sr_is[rows, best]
    slope = float(np.polyfit(chosen_is, chosen_oos, 1)[0]) if np.ptp(chosen_is) > 0 else float("nan")
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)), logits=logits, is_sharpe=chosen_is, oos_sharpe=chosen_oos,
        degradation_slope=slope, prob_oos_loss=float(np.mean(chosen_oos < 0.0)),
        n_splits=len(combos), n_blocks=n_blocks,
    )


def suggested_blocks(n_rows: int, max_blocks: int = 16, min_block_len: int = 20) -> int:
    """Largest even block count <= max_blocks keeping every block at least min_block_len rows."""
    s = min(max_blocks, n_rows // max(min_block_len, 1))
    s -= s % 2
    return max(2, s) if n_rows >= 2 * min_block_len else 2


__all__ = ["PBOResult", "pbo_cscv", "suggested_blocks"]
