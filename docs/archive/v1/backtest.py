"""Backtest the z-score mean-reversion strategy on bars stored in ClickHouse.

    python backtest.py --symbol BTCUSDT --start "2026-10-14 00:00:00" --end "2026-10-15 00:00:00"
    python backtest.py --symbol BTCUSDT --start ... --end ... --sweep --cpus 0-15
"""
import argparse
import logging

import numba
import numpy as np
import polars as pl
import psutil

import signals as S

log = logging.getLogger("backtest")


def parse_cpus(spec: str) -> list[int]:
    """'0-7' -> [0..7];  '0,2,4' -> [0, 2, 4];  '0-3,8-11' works too."""
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def pin_to_cpus(cpus: list[int]) -> None:
    """Restrict this process to `cpus`. Call it BEFORE the first parallel Numba call: the worker
    threads are created lazily and inherit the affinity of the thread that creates them.
    psutil has no cpu_affinity on macOS, so this is a no-op there."""
    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        log.warning("cpu_affinity is not supported on this OS; not pinning")
        return
    proc.cpu_affinity(cpus)
    numba.set_num_threads(min(len(cpus), numba.config.NUMBA_NUM_THREADS))
    log.info("pinned to CPUs %s; Numba threads = %d", cpus, numba.get_num_threads())


def run_once(close, window, entry, exit_, fee, bars_per_year) -> dict:
    z = S.rolling_zscore(close, window)
    pos = S.positions_from_zscore(z, entry, exit_)
    total, sharpe, max_dd = S.summarize(S.strategy_returns(close, pos, fee), bars_per_year)
    return {"total_return": total, "sharpe": sharpe, "max_drawdown": max_dd, "trades": S.count_entries(pos)}


def fmt(res: dict) -> str:
    return (f"return {res['total_return']:+.2%} | sharpe {res['sharpe']:+.2f} | "
            f"max drawdown {res['max_drawdown']:.2%} | trades {res['trades']}")


def evaluate(bars: pl.DataFrame, args) -> None:
    close = bars["close"].to_numpy().astype(np.float64)   # float64: see GUIDE section on precision
    bars_per_year = 365 * 24 * 3600 / args.bar_seconds
    fee = args.fee_bps / 10_000
    log.info("%d bars of %ds; buy & hold over the period: %+.2f%%",
             len(close), args.bar_seconds, (close[-1] / close[0] - 1) * 100)

    if not args.sweep:
        print("strategy :", fmt(run_once(close, args.window, args.entry, args.exit, fee, bars_per_year)))
        return

    # Parameter sweep on the first part of the data, then judge the winner on the unseen rest.
    n_train = int(len(close) * args.split)
    train, test_start = close[:n_train], n_train
    combos = [(w, e, x) for w in (20, 40, 60, 120, 240) for e in (1.5, 2.0, 2.5, 3.0) for x in (0.0, 0.5)]
    windows = np.array([c[0] for c in combos], dtype=np.int64)
    entries = np.array([c[1] for c in combos])
    exits = np.array([c[2] for c in combos])
    out = S.sweep(train, windows, entries, exits, fee, bars_per_year)

    ranked = [i for i in np.argsort(-out[:, 1]) if out[i, 3] >= 10]   # ignore results with < 10 trades
    print(f"top 5 of {len(combos)} combinations IN-SAMPLE (first {args.split:.0%} of the data):")
    for i in ranked[:5]:
        print(f"  window={windows[i]:>3} entry={entries[i]:.1f} exit={exits[i]:.1f} -> "
              f"return {out[i, 0]:+.2%} | sharpe {out[i, 1]:+.2f} | trades {int(out[i, 3])}")
    if not ranked:
        print("no combination produced >= 10 trades; nothing to validate")
        return
    best = ranked[0]
    w = int(windows[best])
    # start `w` bars early so the rolling window is warm at the first out-of-sample bar (past data only)
    oos = close[test_start - w:]
    print("best combination OUT-OF-SAMPLE:", fmt(run_once(oos, w, entries[best], exits[best], fee, bars_per_year)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, help="UTC, e.g. '2026-10-14 00:00:00'")
    p.add_argument("--end", required=True)
    p.add_argument("--bar-seconds", type=int, default=60)
    p.add_argument("--window", type=int, default=60, help="z-score lookback in bars")
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--fee-bps", type=float, default=10.0, help="cost per unit traded, in basis points")
    p.add_argument("--sweep", action="store_true", help="grid search with an out-of-sample check")
    p.add_argument("--split", type=float, default=0.7, help="in-sample fraction for --sweep")
    p.add_argument("--cpus", help="pin to these logical CPUs, e.g. '0-15' (Linux/Windows only)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cpus:
        pin_to_cpus(parse_cpus(args.cpus))

    from bars import load_bars          # imported late so --help works without a database
    bars = load_bars(args.symbol, args.bar_seconds, args.start, args.end)
    if bars.height < 1000:
        raise SystemExit(f"only {bars.height} bars returned; collect more data first")
    evaluate(bars, args)


if __name__ == "__main__":
    main()
