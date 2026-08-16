"""CLI for FMP backfill: `python -m src.data backfill [...]`.

Default behavior (no args) is an INCREMENTAL update — safe to run on cron.
Each step is non-destructive: existing cached rows are never overwritten.

Examples:
    # First-time bulk backfill (~14-45 min depending on plan tier):
    python -m src.data backfill --regions US,UK,CA --since 2006-01-01

    # Daily refresh (incremental — extends cache forward only):
    python -m src.data backfill

    # Periodic fuller refresh — also re-pulls index membership (so index
    # adds/drops enter the universe) and tops up grades for already-cached
    # tickers (so UPDOWN1W_RATINGS picks up new rating events):
    python -m src.data backfill --regions US,UK,CA --refresh-membership --topup-grades

    # Rebuild a specific ticker (deletes its parquet first):
    python -m src.data backfill --tickers AAPL --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from src.config import load_config


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.data",
        description="FMP data backfill / incremental update.",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    bf = sub.add_parser("backfill", help="Download or incrementally update FMP caches.")
    bf.add_argument(
        "--regions",
        default="US",
        help="Comma-separated list of regions: US, UK, CA. Default: US.",
    )
    bf.add_argument(
        "--since",
        default=None,
        help="ISO start date for price/index history. Default: data.fmp_history_start from config.",
    )
    bf.add_argument(
        "--until",
        default=None,
        help="ISO end date. Default: today.",
    )
    bf.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker list to limit the run (skip universe step).",
    )
    bf.add_argument(
        "--force",
        action="store_true",
        help="Delete existing per-ticker parquets before re-fetching. Use with --tickers.",
    )
    bf.add_argument(
        "--refresh-membership",
        action="store_true",
        help=(
            "Re-fetch index membership from FMP instead of reading the cached "
            "constituent tables. Without this the universe is frozen at whatever "
            "the last refresh saw, so index adds/drops since then are missed."
        ),
    )
    bf.add_argument("--skip-prices", action="store_true", help="Skip the equity-price fetch step.")
    bf.add_argument("--skip-benchmarks", action="store_true", help="Skip the regional benchmark-index fetch.")
    bf.add_argument("--skip-fundamentals", action="store_true")
    bf.add_argument("--skip-grades", action="store_true")
    bf.add_argument(
        "--topup-grades",
        action="store_true",
        help=(
            "Also re-pull grades for tickers ALREADY in the cache, not just the "
            "missing ones. FMP's /grades endpoint has no 'from' cutoff, so without "
            "this an incremental run leaves every cached ticker frozen at whatever "
            "the last full fetch saw — new rating events since then never land, and "
            "UPDOWN1W_RATINGS goes silently stale. Costs one small call per ticker."
        ),
    )
    bf.add_argument("--verbose", "-v", action="store_true")

    val = sub.add_parser("validate", help="Audit cache for thin coverage and rename conflicts.")
    val.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Tickers with fewer rows of price history than this are flagged. Default 100.",
    )
    val.add_argument(
        "--out",
        default=None,
        help="Optional path to write the report as parquet. Default: ~/data_lake/fmp/audit/thin_tickers.parquet",
    )
    val.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args(argv)


def _delete_for_force(tickers: list[str]) -> None:
    """Delete per-ticker parquets before refetch when --force is used."""
    from src.data.fmp import _resolve_cache_dir, _safe_filename

    root = _resolve_cache_dir(None)
    subdirs = [
        root / "prices" / "by_symbol",
        root / "fundamentals" / "income_by_symbol",
        root / "fundamentals" / "balance_by_symbol",
    ]
    for t in tickers:
        for d in subdirs:
            f = d / f"{_safe_filename(t)}.parquet"
            if f.exists():
                f.unlink()
                logging.info(f"Deleted {f}")


def _build_universe(regions: list[str], cfg_data: dict, refresh: bool = False) -> list[str]:
    """Step 1-3: fetch membership for each region, return union of symbols."""
    from src.data.fmp import fetch_index_membership_fmp

    rate_limit = cfg_data.get("fmp_rate_limit_per_min", 240)
    region_to_index = {"US": "sp500", "UK": "ftse100", "CA": "tsx60"}

    all_symbols: set[str] = set()
    for region in regions:
        idx = region_to_index.get(region.upper())
        if not idx:
            raise ValueError(f"Unknown region: {region!r}. Use US, UK, or CA.")
        membership = fetch_index_membership_fmp(
            idx, rate_limit_per_min=rate_limit, refresh=refresh
        )
        all_symbols.update(membership["symbol"].astype(str).tolist())
        logging.info(f"  {region}: {len(membership)} membership rows ({membership['symbol'].nunique()} unique)")
    return sorted(all_symbols)


def cmd_backfill(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg_data = cfg["data"]
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    rate_limit = cfg_data.get("fmp_rate_limit_per_min", 240)
    since = args.since or cfg_data.get("fmp_history_start") or cfg_data.get("start_date")
    until = args.until or date.today().isoformat()

    logging.info(f"FMP backfill starting | regions={regions} | since={since} | until={until} | rate={rate_limit}/min")

    # Step 0: --force handling
    if args.tickers:
        tickers_arg = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers_arg = None

    if args.force:
        if not tickers_arg:
            print("--force requires --tickers (refusing to wipe entire cache).", file=sys.stderr)
            return 2
        _delete_for_force(tickers_arg)

    # Step 1-3: universe membership
    if tickers_arg is None:
        tickers = _build_universe(regions, cfg_data, refresh=args.refresh_membership)
        logging.info(f"Universe (union across {regions}): {len(tickers)} unique tickers")
    else:
        tickers = tickers_arg
        logging.info(f"Tickers (overridden via --tickers): {len(tickers)}")

    # Step 4: symbol changes (cheap; informational)
    from src.data.fmp import fetch_symbol_changes_fmp

    fetch_symbol_changes_fmp(rate_limit_per_min=rate_limit)

    # Step 5: company profiles
    from src.data.fmp import fetch_company_profiles_fmp

    fetch_company_profiles_fmp(tickers, rate_limit_per_min=rate_limit)

    # Step 6: prices
    if not args.skip_prices:
        from src.data.fmp import fetch_price_data_fmp

        fetch_price_data_fmp(tickers, since, until, rate_limit_per_min=rate_limit)

    # Step 7: benchmark indexes — independent of --skip-prices, since benchmarks
    # are tiny (3 calls) and the price-skip use case (refreshing fundamentals
    # only) almost always still wants the benchmark series current.
    benchmarks = cfg_data.get("benchmarks", {"US": "^GSPC", "UK": "^FTSE", "CA": "XIU.TO"})
    bench_symbols = [benchmarks[r] for r in regions if r in benchmarks]
    if bench_symbols and not args.skip_benchmarks:
        from src.data.fmp import fetch_price_data_fmp

        fetch_price_data_fmp(bench_symbols, since, until, rate_limit_per_min=rate_limit)
        logging.info(f"Fetched benchmarks: {bench_symbols}")

    # Step 8-9: fundamentals (income + balance, last)
    if not args.skip_fundamentals:
        from src.data.fmp import fetch_fundamentals_fmp

        fetch_fundamentals_fmp(
            tickers,
            rate_limit_per_min=rate_limit,
            pub_lag_days=cfg_data.get("fundamental_pub_lag_days", 90),
        )

    # Step 10: grades top-up (existing fetcher, already correct)
    if not args.skip_grades:
        from src.data.fetch import fetch_analyst_grades

        fetch_analyst_grades(
            tickers,
            rate_limit_per_min=rate_limit,
            topup_existing=args.topup_grades,
        )

    logging.info("FMP backfill complete.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Audit the cache: flag thin-coverage tickers and cross-reference root causes.

    For each ticker with fewer than `--min-rows` rows of price history, look up:
      - whether it has a symbol-change record (renamed → new symbol)
      - whether it appears in the SP500 historical events (membership context)
      - what the rename target looks like (does it have full coverage?)

    Output: a human-readable summary table to stdout, plus an optional parquet
    written to disk for downstream use (e.g., the universe loader can filter
    out thin tickers automatically).
    """
    import pandas as pd
    from src.data.fmp import _resolve_cache_dir

    root = _resolve_cache_dir(None)
    prices_dir = root / "prices" / "by_symbol"
    if not prices_dir.exists():
        print(f"No price cache found at {prices_dir}. Run `backfill` first.", file=sys.stderr)
        return 1

    logging.info(f"Auditing {prices_dir} for tickers with < {args.min_rows} rows of price history...")
    rows = []
    for parquet in sorted(prices_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet)
        if len(df) < args.min_rows:
            sym = parquet.stem
            mn = df["date"].min() if not df.empty else None
            mx = df["date"].max() if not df.empty else None
            rows.append({"symbol": sym, "rows": len(df), "first_date": mn, "last_date": mx})

    if not rows:
        print(f"All cached tickers have >= {args.min_rows} rows. No thin coverage detected.")
        return 0

    thin = pd.DataFrame(rows)
    logging.info(f"{len(thin)} thin-coverage tickers found")

    # Cross-reference symbol changes
    sc_path = root / "symbol_changes" / "symbol_changes_raw.parquet"
    if sc_path.exists():
        sc = pd.read_parquet(sc_path)
        # An oldSymbol can appear multiple times if the ticker was reused —
        # keep the most recent rename for context.
        sc_latest = sc.sort_values("date").drop_duplicates(subset=["oldSymbol"], keep="last")
        rename_lookup = sc_latest.set_index("oldSymbol")[["date", "newSymbol", "companyName"]].to_dict(orient="index")
    else:
        rename_lookup = {}

    # Cross-reference SP500 events for context
    events_path = root / "constituents" / "sp500_events_raw.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()

    def _sp500_context(sym: str) -> str:
        if events.empty:
            return ""
        hits = events[(events["symbol"] == sym) | (events["removedTicker"] == sym)]
        if hits.empty:
            return ""
        last = hits.iloc[-1]
        return f"{last['date'].date() if hasattr(last['date'],'date') else last['date']}: {last.get('reason','')}"

    # Coverage of rename targets
    def _target_coverage(new_sym: str) -> str:
        target = prices_dir / f"{new_sym}.parquet"
        if not target.exists():
            return "target NOT cached"
        t = pd.read_parquet(target)
        if t.empty:
            return "target empty"
        return f"target {len(t)} rows {t['date'].min().date()}→{t['date'].max().date()}"

    thin["rename"] = thin["symbol"].map(
        lambda s: f"→ {rename_lookup[s]['newSymbol']} ({rename_lookup[s]['date'].date()})"
        if s in rename_lookup else ""
    )
    thin["rename_target"] = thin["symbol"].map(
        lambda s: _target_coverage(rename_lookup[s]["newSymbol"]) if s in rename_lookup else ""
    )
    thin["sp500_context"] = thin["symbol"].map(_sp500_context)

    # Categorize each ticker
    def _classify(row) -> str:
        if row["rename"] and "target NOT cached" not in row["rename_target"] and "target empty" not in row["rename_target"]:
            return "rename — fetch target"
        if row["rename"]:
            return "rename — target also missing"
        if "Acquired" in row["sp500_context"] or "acquires" in row["sp500_context"].lower() or "merge" in row["sp500_context"].lower():
            return "M&A — data lost on FMP"
        if row["sp500_context"]:
            return "delisted/removed"
        return "unknown — not in SP500 events"

    thin["category"] = thin.apply(_classify, axis=1)

    # Print summary
    print()
    print(f"Thin-coverage report — {len(thin)} tickers with < {args.min_rows} rows")
    print("=" * 80)
    by_cat = thin["category"].value_counts()
    for cat, n in by_cat.items():
        print(f"  {n:3d} | {cat}")
    print()
    print("Detail:")
    print(thin[["symbol", "rows", "first_date", "last_date", "rename", "rename_target", "category"]].to_string(index=False))

    out_path = Path(args.out) if args.out else (root / "audit" / "thin_tickers.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thin.to_parquet(out_path)
    print(f"\nReport written to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Backward-compat: with no args, behave like the old single-shot price fetch.
    if not argv:
        argv = ["backfill"]

    args = _parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    if args.cmd == "backfill" or args.cmd is None:
        return cmd_backfill(args)
    if args.cmd == "validate":
        return cmd_validate(args)

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
