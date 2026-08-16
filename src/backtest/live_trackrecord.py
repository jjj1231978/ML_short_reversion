"""Live track record: weekly performance review + market commentary.

This is the human-facing weekly journal of the live strategy. Where
`forecast_picks_log.py` stores the *named picks* (one row per pick, with each
pick's realized return) and `forecast_track_record.py` rolls forecasts up into a
basket-return parquet, this module records, per resolved week, two things the
other two don't keep together:

  1. The two headline hit rates the desk reports each week:
       * single-stock hit rate — fraction of the individual long/short names
         that moved the right way (long up / short down).
       * equal-weight long/short portfolio return + whether it was positive
         (the "portfolio hit rate" accumulates as the share of weeks the
         dollar-neutral EW L/S basket made money).
  2. A free-text market-development commentary for the week, plus the headline
     index/vol moves it refers to.

Source of truth for the pick-level returns is the resolved picks ledger
(`forecast_picks.db`, table `forecast_picks`) — so run
`python -m src.backtest.forecast_picks_log` first (after a price refresh) to
resolve the week, then record it here.

Storage (under data/processed/):
    live_trackrecord.db
        weekly_performance  — one row per (signal_day, target_date, scope):
                              EW L/S return + single-stock hit rate.
        weekly_commentary   — one row per (signal_day, target_date):
                              market moves + commentary text.

Usage:
    # record a resolved week (metrics pulled from the picks ledger)
    python -m src.backtest.live_trackrecord --signal-day WED --target-date 2026-06-10 \
        --commentary-file /tmp/week.txt \
        --spx-ret -0.0380 --ndx-ret -0.0627 --vix-chg 6.16

    # show the accumulated track record
    python -m src.backtest.live_trackrecord --show
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

log = logging.getLogger(__name__)

PICKS_DB = PROCESSED_DIR / "forecast_picks.db"
DB_PATH = PROCESSED_DIR / "live_trackrecord.db"
PERF_TABLE = "weekly_performance"
COMMENT_TABLE = "weekly_commentary"

SCOPES = ("ALL", "US", "UK", "CA")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PERF_TABLE} (
            signal_day TEXT NOT NULL,
            target_date TEXT NOT NULL,
            as_of_date TEXT,
            scope TEXT NOT NULL,                 -- 'ALL' | 'US' | 'UK' | 'CA'
            n_long INTEGER,
            n_short INTEGER,
            long_ret REAL,                       -- EW mean realized return of longs
            short_ret REAL,                      -- EW mean realized return of shorts
            ls_ret REAL,                         -- long_ret - short_ret (dollar-neutral EW L/S)
            ls_positive INTEGER,                 -- 1 if ls_ret > 0 (feeds portfolio hit rate)
            n_picks INTEGER,                     -- resolved single-name picks
            n_correct INTEGER,                   -- picks whose signed return > 0
            single_stock_hit_rate REAL,          -- n_correct / n_picks
            bundle_retrain_date TEXT,            -- model vintage behind the picks
            recorded_at TEXT,
            PRIMARY KEY (signal_day, target_date, scope)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {COMMENT_TABLE} (
            signal_day TEXT NOT NULL,
            target_date TEXT NOT NULL,
            as_of_date TEXT,
            spx_ret REAL,                        -- S&P 500 as_of->target return
            ndx_ret REAL,                        -- Nasdaq Composite as_of->target return
            vix_chg REAL,                        -- VIX change in points over the week
            market_commentary TEXT,
            recorded_at TEXT,
            PRIMARY KEY (signal_day, target_date)
        )
        """
    )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Metrics from the resolved picks ledger
# ---------------------------------------------------------------------------

def _load_resolved_picks(signal_day: str, target_date: str, picks_db: Path) -> pd.DataFrame:
    if not picks_db.exists():
        raise FileNotFoundError(f"Picks ledger not found: {picks_db}. "
                                "Run `python -m src.backtest.forecast_picks_log` first.")
    with sqlite3.connect(picks_db) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM forecast_picks WHERE signal_day = ? AND target_date = ?",
            conn, params=(signal_day, target_date),
        )
    return df


def compute_scope_metrics(picks: pd.DataFrame, scope: str) -> dict | None:
    """Single-stock hit rate + EW L/S basket return for one scope, from the
    resolved picks of a single week. Returns None if no evaluable picks."""
    s = picks[(picks["scope"] == scope) & (picks["evaluable"] == 1)].copy()
    if s.empty:
        return None
    longs = s[s["side"] == "LONG"]
    shorts = s[s["side"] == "SHORT"]
    long_ret = float(longs["realized_ret"].mean()) if not longs.empty else np.nan
    short_ret = float(shorts["realized_ret"].mean()) if not shorts.empty else np.nan
    ls_ret = long_ret - short_ret
    n_correct = int((s["signed_ret"] > 0).sum())
    n_picks = int(len(s))
    return {
        "scope": scope,
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
        "long_ret": long_ret,
        "short_ret": short_ret,
        "ls_ret": ls_ret,
        "ls_positive": int(ls_ret > 0) if pd.notna(ls_ret) else None,
        "n_picks": n_picks,
        "n_correct": n_correct,
        "single_stock_hit_rate": float(n_correct / n_picks) if n_picks else np.nan,
    }


# ---------------------------------------------------------------------------
# Record / read
# ---------------------------------------------------------------------------

def record_week(
    signal_day: str,
    target_date: str,
    commentary: str | None = None,
    spx_ret: float | None = None,
    ndx_ret: float | None = None,
    vix_chg: float | None = None,
    picks_db: Path = PICKS_DB,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Record one resolved week's performance + (optional) market commentary.

    Pulls the named picks for (signal_day, target_date) from the picks ledger,
    aggregates per scope into EW L/S return and single-stock hit rate, and
    upserts into live_trackrecord.db. Returns the per-scope performance frame.
    """
    picks = _load_resolved_picks(signal_day, target_date, picks_db)
    if picks.empty:
        raise ValueError(f"No picks logged for {signal_day} target {target_date}. "
                         "Run a forecast / rebuild the ledger first.")
    n_eval = int((picks["evaluable"] == 1).sum())
    if n_eval == 0:
        raise ValueError(f"{signal_day} target {target_date} has no resolved picks yet "
                         "(target close not in the price cache). Refresh prices and rebuild "
                         "the ledger before recording.")

    as_of = str(picks["as_of_date"].iloc[0])
    bundle = picks["bundle_retrain_date"].dropna()
    bundle = str(bundle.iloc[0]) if not bundle.empty else None

    rows: list[dict] = []
    for scope in SCOPES:
        m = compute_scope_metrics(picks, scope)
        if m is None:
            continue
        m.update({
            "signal_day": signal_day,
            "target_date": target_date,
            "as_of_date": as_of,
            "bundle_retrain_date": bundle,
            "recorded_at": _utcnow(),
        })
        rows.append(m)
    perf = pd.DataFrame(rows)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    perf_cols = [
        "signal_day", "target_date", "as_of_date", "scope", "n_long", "n_short",
        "long_ret", "short_ret", "ls_ret", "ls_positive", "n_picks", "n_correct",
        "single_stock_hit_rate", "bundle_retrain_date", "recorded_at",
    ]
    with sqlite3.connect(db_path) as conn:
        _ensure_tables(conn)
        placeholders = ", ".join("?" for _ in perf_cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in perf_cols
                            if c not in ("signal_day", "target_date", "scope"))
        sql = (f"INSERT INTO {PERF_TABLE} ({', '.join(perf_cols)}) VALUES ({placeholders}) "
               f"ON CONFLICT(signal_day, target_date, scope) DO UPDATE SET {updates}")
        conn.executemany(sql, [tuple(_sqlval(r.get(c)) for c in perf_cols)
                               for r in rows])

        if commentary is not None or any(v is not None for v in (spx_ret, ndx_ret, vix_chg)):
            ccols = ["signal_day", "target_date", "as_of_date", "spx_ret", "ndx_ret",
                     "vix_chg", "market_commentary", "recorded_at"]
            cupd = ", ".join(f"{c}=excluded.{c}" for c in ccols
                             if c not in ("signal_day", "target_date"))
            conn.execute(
                f"INSERT INTO {COMMENT_TABLE} ({', '.join(ccols)}) "
                f"VALUES ({', '.join('?' for _ in ccols)}) "
                f"ON CONFLICT(signal_day, target_date) DO UPDATE SET {cupd}",
                (signal_day, target_date, as_of, _sqlval(spx_ret), _sqlval(ndx_ret),
                 _sqlval(vix_chg), commentary, _utcnow()),
            )
        conn.commit()
    return perf


def _sqlval(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    return v


def load_performance(signal_day: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        q = f"SELECT * FROM {PERF_TABLE}"
        params = ()
        if signal_day:
            q += " WHERE signal_day = ?"
            params = (signal_day,)
        q += " ORDER BY target_date, scope"
        return pd.read_sql_query(q, conn, params=params)


def load_commentary(signal_day: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        q = f"SELECT * FROM {COMMENT_TABLE}"
        params = ()
        if signal_day:
            q += " WHERE signal_day = ?"
            params = (signal_day,)
        q += " ORDER BY target_date"
        return pd.read_sql_query(q, conn, params=params)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_track_record(signal_day: str | None) -> None:
    perf = load_performance(signal_day)
    if perf.empty:
        print("live_trackrecord is empty — record a week first.")
        return
    print(f"\n=== Live track record  (DB: {DB_PATH}) ===")
    for sd, sgrp in perf.groupby("signal_day"):
        print(f"\n[{sd}]")
        weeks = sgrp[sgrp["scope"] == "ALL"]["target_date"].nunique()
        for scope in SCOPES:
            sc = sgrp[sgrp["scope"] == scope]
            if sc.empty:
                continue
            n_weeks = len(sc)
            port_hit = float((sc["ls_positive"] == 1).mean())
            cum_ls = float((1 + sc["ls_ret"].fillna(0)).prod() - 1)
            mean_ls = float(sc["ls_ret"].mean())
            tot_picks = int(sc["n_picks"].sum())
            tot_correct = int(sc["n_correct"].sum())
            ss_hit = (tot_correct / tot_picks) if tot_picks else float("nan")
            print(f"  {scope:>3}: weeks={n_weeks:>2}  "
                  f"EW L/S mean={mean_ls:+.2%}  cum={cum_ls:+.2%}  "
                  f"portfolio-hit={port_hit:5.0%}  "
                  f"single-stock-hit={ss_hit:5.0%} ({tot_correct}/{tot_picks})")
    comm = load_commentary(signal_day)
    if not comm.empty:
        print("\n  Commentary on record:")
        for _, r in comm.iterrows():
            print(f"   [{r['signal_day']}] {r['target_date']}: "
                  f"{(r['market_commentary'] or '')[:90]}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Record / show the live weekly track record.")
    p.add_argument("--signal-day", default="WED", choices=["MON", "TUE", "WED", "THU", "FRI"])
    p.add_argument("--target-date", help="Resolved target week to record (YYYY-MM-DD).")
    p.add_argument("--commentary", help="Inline market-development commentary text.")
    p.add_argument("--commentary-file", type=Path, help="Read commentary from a file.")
    p.add_argument("--spx-ret", type=float, help="S&P 500 as_of->target return (decimal).")
    p.add_argument("--ndx-ret", type=float, help="Nasdaq as_of->target return (decimal).")
    p.add_argument("--vix-chg", type=float, help="VIX change over the week (points).")
    p.add_argument("--show", action="store_true", help="Print the accumulated track record and exit.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.show or not args.target_date:
        _print_track_record(args.signal_day if args.target_date else None)
        if not args.target_date:
            return 0

    commentary = args.commentary
    if args.commentary_file:
        commentary = args.commentary_file.read_text().strip()

    perf = record_week(
        signal_day=args.signal_day,
        target_date=args.target_date,
        commentary=commentary,
        spx_ret=args.spx_ret,
        ndx_ret=args.ndx_ret,
        vix_chg=args.vix_chg,
    )
    print(f"\nRecorded {args.signal_day} target {args.target_date} "
          f"({len(perf)} scopes) into {DB_PATH}")
    show = perf.copy()
    for c in ("long_ret", "short_ret", "ls_ret", "single_stock_hit_rate"):
        show[c] = show[c].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    print(show[["scope", "n_long", "n_short", "long_ret", "short_ret", "ls_ret",
                "single_stock_hit_rate"]].to_string(index=False))
    _print_track_record(args.signal_day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
