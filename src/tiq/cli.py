"""Command-line interface: `tiq <command>` (or `python -m tiq <command>`).

    tiq symbols   --quote USDT                      list symbols in the archive
    tiq backfill  --symbols BTCUSDT ETHUSDT --interval 1d
    tiq backfill  --all --quote USDT --interval 1d  every eligible pair, including delisted ones
    tiq qc        --interval 1d                     data-quality summary of the lake
    tiq run       experiments/01_donchian_btc.toml  run a pre-registered experiment (several allowed)
    tiq registry                                    every run so far, newest last
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("tiq")


def parse_cpus(spec: str) -> list[int]:
    """'0-7' -> [0..7]; '0,2,4' -> [0, 2, 4]; '0-3,8-11' works too."""
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def pin_to_cpus(cpus: list[int]) -> None:
    """Restrict this process to `cpus` before any parallel work starts (Linux and Windows only).
    Numba worker threads are created lazily and inherit the affinity of the creating thread."""
    import numba
    import psutil

    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        log.warning("cpu_affinity is not supported on this OS; not pinning")
        return
    proc.cpu_affinity(cpus)
    numba.set_num_threads(min(len(cpus), numba.config.NUMBA_NUM_THREADS))
    log.info("pinned to CPUs %s; Numba threads = %d", cpus, numba.get_num_threads())


def cmd_symbols(args: argparse.Namespace) -> int:
    from tiq.data import binance_vision as bv
    from tiq.data.symbols import eligible_symbols

    with bv.make_client() as client:
        symbols = bv.list_symbols(client, args.market)
    quoted = [s for s in symbols if s.endswith(args.quote)]
    eligible = eligible_symbols(quoted, args.quote)
    print("\n".join(eligible if not args.include_excluded else quoted))
    print(f"# {len(quoted)} {args.quote} pairs in the archive, {len(eligible)} eligible crypto assets", file=sys.stderr)
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from tiq.data import binance_vision as bv
    from tiq.data.backfill import backfill_funding, backfill_klines
    from tiq.data.lake import Lake
    from tiq.data.symbols import eligible_symbols

    lake = Lake(args.lake) if args.lake else Lake()
    if args.all:
        with bv.make_client() as client:
            listed = [s for s in bv.list_symbols(client, args.market) if s.endswith(args.quote)]
        symbols = eligible_symbols(listed, args.quote)
        log.info("%d eligible %s pairs (of %d listed, delisted ones included)", len(symbols), args.quote, len(listed))
    else:
        symbols = [s.upper() for s in args.symbols]
    if not symbols:
        log.error("no symbols: pass --symbols or --all")
        return 2
    report = backfill_klines(symbols, args.interval, args.market, args.start, args.end, args.workers, lake)
    log.info("klines: %s", report.summary())
    failures = list(report.failures)
    if args.funding:
        funding = backfill_funding(symbols, args.start, args.workers, lake)
        log.info("funding: %s", funding.summary())
        failures += funding.failures
    for sym, month, err in failures[:20]:
        log.error("failed: %s %s: %s", sym, month, err)
    return 1 if failures else 0


def cmd_qc(args: argparse.Namespace) -> int:
    import polars as pl

    from tiq.data.lake import Lake
    from tiq.data.quality import qc_report

    lake = Lake(args.lake) if args.lake else Lake()
    df = qc_report(lake, args.market, args.interval, args.symbols or None)
    if df.is_empty():
        log.error("nothing stored for %s %s", args.market, args.interval)
        return 1
    with pl.Config(tbl_rows=args.rows, tbl_cols=20, tbl_width_chars=200):
        structural = df.filter((pl.col("duplicates") > 0) | (pl.col("misaligned") > 0) | (pl.col("gaps") > 0)
                               | (pl.col("ohlc_violations") > 0) | (pl.col("nonpositive_prices") > 0))
        extreme = df.filter(pl.col("extreme_moves") > 0)
        print(f"{df.height} symbols, {df['rows'].sum():,} bars checked. Structural findings in {structural.height}:")
        print(structural.sort("missing_bars", descending=True).select(
            "symbol", "rows", "first", "last", "gaps", "missing_bars", "longest_gap_bars", "duplicates",
            "misaligned", "ohlc_violations", "nonpositive_prices"))
        print(f"{extreme.height} symbols have very large single-bar moves ({extreme['extreme_moves'].sum():,} bars). "
              "These are usually real (listing pumps, collapses) and are kept; review them in the CSV.")
    if args.csv:
        df.write_csv(args.csv)
        print(f"full table written to {args.csv}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from tiq.data.lake import Lake
    from tiq.paths import reports_dir
    from tiq.research.config import load_config
    from tiq.research.experiment import run_experiment

    if args.cpus:
        pin_to_cpus(parse_cpus(args.cpus))
    if args.out and len(args.config) > 1:
        log.error("--out works with a single config only")
        return 2
    lake = Lake(args.lake) if args.lake else Lake()
    configs = [load_config(path) for path in args.config]  # validate every file before running any
    for cfg in configs:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) if args.out else reports_dir() / cfg.experiment.name / stamp
        res = run_experiment(cfg, lake, out_dir, quick=args.quick, workers=args.workers,
                             use_registry=not args.no_registry)
        ms = res.metrics["strategy"]
        print(f"\n{cfg.experiment.name}: {res.verdict}")
        print(f"  OOS {res.panel.dates[res.oos_rows.start].astype(str)[:10]} .. "
              f"{res.panel.dates[-1].astype(str)[:10]}: CAGR {ms.get('cagr', float('nan')):.1%}, "
              f"Sharpe {ms.get('sharpe', float('nan')):.2f}, max drawdown {ms.get('max_drawdown', float('nan')):.1%}")
        for c in res.criteria:
            mark = "n/a " if c["pass"] is None else ("pass" if c["pass"] else "FAIL")
            value = "n/a" if c["value"] is None else f"{c['value']:.3f}"
            print(f"  [{mark}] {c['name']}: {value} (required {c['op']} {c['threshold']})")
        print(f"  report: {out_dir / 'report.md'}")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    from tiq.research.registry import read_registry

    entries = read_registry()
    if not entries:
        print("no runs registered yet")
        return 0
    for e in entries[-args.last:]:
        sharpe = e.get("oos_sharpe")
        print(f"{e['timestamp']}  {e['verdict']:4}  {e['experiment']:<34} family={e['family']:<16} "
              f"trials={e['trials']:<4} OOS Sharpe={'n/a' if sharpe is None else f'{sharpe:.2f}'}"
              f"{'  (quick)' if e.get('quick') else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tiq", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("symbols", help="list symbols available in the archive")
    s.add_argument("--market", default="spot", choices=["spot", "um"])
    s.add_argument("--quote", default="USDT")
    s.add_argument("--include-excluded", action="store_true", help="also list stablecoins, leveraged tokens...")
    s.set_defaults(func=cmd_symbols)

    b = sub.add_parser("backfill", help="download history into the lake")
    b.add_argument("--symbols", nargs="*", default=[])
    b.add_argument("--all", action="store_true", help="every eligible pair for --quote, delisted ones included")
    b.add_argument("--quote", default="USDT")
    b.add_argument("--interval", default="1d")
    b.add_argument("--market", default="spot", choices=["spot", "um"])
    b.add_argument("--start", help="first month, YYYY-MM")
    b.add_argument("--end", help="last month, YYYY-MM")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--funding", action="store_true", help="also fetch USD-M perpetual funding rates")
    b.add_argument("--lake", help="lake directory (default: data/lake)")
    b.set_defaults(func=cmd_backfill)

    q = sub.add_parser("qc", help="data-quality summary")
    q.add_argument("--interval", default="1d")
    q.add_argument("--market", default="spot")
    q.add_argument("--symbols", nargs="*")
    q.add_argument("--rows", type=int, default=40)
    q.add_argument("--csv", help="write the full table to this CSV file")
    q.add_argument("--lake")
    q.set_defaults(func=cmd_qc)

    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config", nargs="+", help="one or more experiment TOML files, run in order")
    r.add_argument("--quick", action="store_true", help="few resampling reps, for a fast first look")
    r.add_argument("--workers", type=int, help="threads for parallel backtests (default: all cores)")
    r.add_argument("--cpus", help="pin to these logical CPUs first, e.g. '0-15' (Linux/Windows)")
    r.add_argument("--out", help="report directory (default: reports/<name>/<timestamp>)")
    r.add_argument("--no-registry", action="store_true", help="do not record this run (tests only)")
    r.add_argument("--lake")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("registry", help="list registered runs")
    g.add_argument("--last", type=int, default=50)
    g.set_defaults(func=cmd_registry)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
