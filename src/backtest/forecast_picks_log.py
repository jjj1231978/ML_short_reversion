"""Named-pick ledger: a queryable record of every top/bottom forecast pick.

Where `forecast_track_record.py` rolls saved forecasts up into *aggregate*
per-region basket returns, this module records the *named* picks themselves —
one row per (signal_day, target_date, scope, side, rank, ticker) — so the live
track record can be sliced by ticker, region, side, or model vintage, and each
pick carries its realized close-to-close return once its target week resolves.

Scopes recorded per forecast: each region (US / UK / CA) plus the pooled "ALL"
ranking. Sides: top-N LONG and bottom-N SHORT (default N=10).

Storage (under data/processed/):
    forecast_picks.db   — SQLite, table `forecast_picks`, upsert-keyed on
                          (signal_day, target_date, scope, side, rank).
    forecast_picks.csv  — flat mirror of the table for eyeballing / Excel.

The DB is the source-of-record for accumulation; it is append/upsert friendly:
    * `update_for_forecast(...)` logs one forecast's picks (called automatically
      at the end of every `python -m src.predict` run).
    * `rebuild_ledger(...)` (the CLI) re-derives the whole ledger from the saved
      forecast parquets and re-evaluates realized returns against the latest
      price cache — run it periodically to resolve in-flight picks.

Usage:
    python -m src.backtest.forecast_picks_log                 # rebuild WED/THU/FRI
    python -m src.backtest.forecast_picks_log --signal-day WED
    python -m src.backtest.forecast_picks_log --top-n 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

log = logging.getLogger(__name__)

FORECASTS_ROOT = PROCESSED_DIR / "forecasts"
MODELS_ROOT = PROCESSED_DIR / "models"
DB_PATH = PROCESSED_DIR / "forecast_picks.db"
CSV_PATH = PROCESSED_DIR / "forecast_picks.csv"
TABLE = "forecast_picks"
VERSIONS_TABLE = "model_versions"

# Logical-name prefixes that mark a feature column as macro/regime-derived.
_MACRO_NAMES = {
    "SPX", "NDX", "RUT", "VIX", "OIL", "GOLD", "COPPER", "DXY",
    "US10Y", "US30Y", "US3M", "EURUSD", "USDJPY",
}

# Column order is also the table schema order (see _ensure_table).
COLUMNS = [
    "signal_day",
    "as_of_date",
    "target_date",
    "scope",            # 'US' | 'UK' | 'CA' | 'ALL'
    "side",             # 'LONG' | 'SHORT'
    "rank",             # 1 = strongest long / strongest short
    "ticker",
    "industry",
    "prediction",
    "bundle_retrain_date",  # model vintage that produced the pick (may be NULL)
    "realized_ret",     # close-to-close as_of→target; NULL until target resolves
    "signed_ret",       # LONG: +ret, SHORT: -ret (the pick's P&L contribution)
    "correct",          # 1 if signed_ret > 0, else 0; NULL until resolved
    "evaluable",        # 1 once the target close is in the price cache
    "logged_at",
]
_PK = ("signal_day", "target_date", "scope", "side", "rank")


# ---------------------------------------------------------------------------
# Extraction: forecast parquet -> tidy named picks
# ---------------------------------------------------------------------------

def extract_picks(forecast: pd.DataFrame, n_per_side: int = 10) -> pd.DataFrame:
    """Pull top-N long + bottom-N short per region and pooled from one forecast.

    `forecast` is a saved `forecasts/{day}/{target}.parquet` (ticker-indexed,
    with prediction/eligible/region/industry/as_of_date/target_date/signal_day).
    Returns a tidy frame with the ledger columns (minus realized/eval, which are
    filled by `evaluate_picks`).
    """
    if forecast.empty:
        return pd.DataFrame(columns=COLUMNS)

    as_of = pd.Timestamp(forecast["as_of_date"].iloc[0]).normalize().date().isoformat()
    target = pd.Timestamp(forecast["target_date"].iloc[0]).normalize().date().isoformat()
    signal_day = str(forecast["signal_day"].iloc[0])
    bundle = forecast["bundle_retrain_date"].iloc[0] if "bundle_retrain_date" in forecast.columns else None

    eligible = forecast[forecast["eligible"]].copy()
    rows: list[dict] = []

    def _add(grp: pd.DataFrame, scope: str) -> None:
        if grp.empty:
            return
        ranked = grp.sort_values("prediction", ascending=False)
        # Cap basket so longs and shorts never overlap on a thin cross-section.
        k = min(n_per_side, len(ranked) // 2)
        if k < 1:
            return
        longs = ranked.head(k)
        shorts = ranked.tail(k).iloc[::-1]  # rank 1 = most-negative prediction
        for side, picks in (("LONG", longs), ("SHORT", shorts)):
            for rank, (ticker, r) in enumerate(picks.iterrows(), start=1):
                rows.append({
                    "signal_day": signal_day,
                    "as_of_date": as_of,
                    "target_date": target,
                    "scope": scope,
                    "side": side,
                    "rank": rank,
                    "ticker": str(ticker),
                    "industry": r.get("industry", None),
                    "prediction": float(r["prediction"]),
                    "bundle_retrain_date": bundle,
                    "realized_ret": np.nan,
                    "signed_ret": np.nan,
                    "correct": pd.NA,
                    "evaluable": 0,
                    "logged_at": _utcnow(),
                })

    for region in ("US", "UK", "CA"):
        _add(eligible[eligible["region"] == region], region)
    _add(eligible, "ALL")  # pooled — note the cross-section isn't region-neutral

    return pd.DataFrame(rows, columns=COLUMNS)


def evaluate_picks(picks: pd.DataFrame, close: pd.DataFrame | None) -> pd.DataFrame:
    """Fill realized_ret / signed_ret / correct / evaluable from a close panel.

    Realized return is close-to-close from the latest trading day <= as_of to
    the latest trading day <= target, and only counts as evaluable once a close
    on/after the target date exists in the cache (no partial holding periods).
    """
    picks = picks.copy()
    if close is None or close.empty or picks.empty:
        return picks

    avail = close.index
    for (as_of_s, target_s), grp in picks.groupby(["as_of_date", "target_date"]):
        as_of = pd.Timestamp(as_of_s)
        target = pd.Timestamp(target_s)
        as_of_actual = avail[avail <= as_of].max() if (avail <= as_of).any() else None
        target_actual = avail[avail <= target].max() if (avail <= target).any() else None
        # Strict: need a real close on/after target, and the as-of close present.
        if target_actual is None or target_actual < target:
            continue
        if as_of_actual is None or as_of_actual < as_of:
            continue
        tickers = grp["ticker"].tolist()
        start = close.loc[as_of_actual].reindex(tickers)
        end = close.loc[target_actual].reindex(tickers)
        ret = (end / start) - 1.0
        ret.index = grp.index  # align to row index for assignment
        signed = np.where(grp["side"].values == "LONG", ret.values, -ret.values)
        picks.loc[grp.index, "realized_ret"] = ret.values
        picks.loc[grp.index, "signed_ret"] = signed
        picks.loc[grp.index, "correct"] = pd.array(
            [pd.NA if pd.isna(s) else int(s > 0) for s in signed], dtype="Int64"
        )
        picks.loc[grp.index, "evaluable"] = np.where(pd.isna(ret.values), 0, 1)
    return picks


# ---------------------------------------------------------------------------
# SQLite persistence (upsert-keyed; CSV mirror)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            signal_day TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            target_date TEXT NOT NULL,
            scope TEXT NOT NULL,
            side TEXT NOT NULL,
            rank INTEGER NOT NULL,
            ticker TEXT,
            industry TEXT,
            prediction REAL,
            bundle_retrain_date TEXT,
            realized_ret REAL,
            signed_ret REAL,
            correct INTEGER,
            evaluable INTEGER,
            logged_at TEXT,
            PRIMARY KEY (signal_day, target_date, scope, side, rank)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_target ON {TABLE} (target_date, signal_day)"
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_ticker ON {TABLE} (ticker)")


def _to_sql_value(col: str, v):
    """Coerce pandas NA/NaT/NaN to None and numpy scalars to Python scalars."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def upsert_picks(picks: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Insert-or-update rows keyed on (signal_day, target_date, scope, side, rank).

    Re-logging a forecast (e.g. after its return resolves) overwrites the prior
    row in place rather than duplicating it. Returns the number of rows written.
    """
    if picks.empty:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = COLUMNS
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in _PK)
    sql = (
        f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({', '.join(_PK)}) DO UPDATE SET {updates}"
    )
    records = [tuple(_to_sql_value(c, row[c]) for c in cols) for _, row in picks.iterrows()]
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        conn.executemany(sql, records)
        conn.commit()
    return len(records)


def write_csv_mirror(db_path: Path = DB_PATH, csv_path: Path = CSV_PATH) -> None:
    """Dump the whole table to CSV, newest target first."""
    if not db_path.exists():
        return
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
    if df.empty:
        csv_path.write_text("")
        return
    df = df.sort_values(
        ["target_date", "signal_day", "scope", "side", "rank"],
        ascending=[False, True, True, True, True],
    )
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Model version history (separate table; picks reference bundle_retrain_date)
# ---------------------------------------------------------------------------

def _ensure_versions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {VERSIONS_TABLE} (
            signal_day TEXT NOT NULL,
            bundle_retrain_date TEXT NOT NULL,
            members TEXT,
            combine_method TEXT,
            n_features INTEGER,
            macro_enabled INTEGER,
            hyperparams TEXT,        -- compact JSON: windows + per-member key params
            config_hash TEXT,        -- same hash => same model spec (version fingerprint)
            saved_at TEXT,
            logged_at TEXT,
            PRIMARY KEY (signal_day, bundle_retrain_date)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{VERSIONS_TABLE}_hash ON {VERSIONS_TABLE} (config_hash)"
    )


def _macro_enabled(feature_columns: list[str]) -> bool:
    """True if any feature column is macro/regime-derived."""
    for c in feature_columns:
        u = str(c).upper()
        if "HMM" in u or "REGIME" in u or "TREND" in u:
            return True
        if u.split("_", 1)[0] in _MACRO_NAMES:
            return True
    return False


def _hp_from_models(models: dict) -> dict:
    """Best-effort compact per-member hyperparameters from the fitted estimators.

    Reads what each library reliably exposes; anything unreadable is recorded as
    'unparsed' rather than failing. Keys are kept short and architecture-defining.
    """
    hp: dict = {}
    for name, m in (models or {}).items():
        try:
            if name == "random_forest":
                p = m.get_params()
                hp["rf"] = {"n_est": p.get("n_estimators"), "depth": p.get("max_depth"),
                            "min_leaf": p.get("min_samples_leaf"), "max_feat": p.get("max_features")}
            elif name == "lightgbm":
                p = getattr(m, "params", {}) or {}
                hp["lgb"] = {"leaves": p.get("num_leaves"), "depth": p.get("max_depth"),
                             "lr": p.get("learning_rate"), "rounds": getattr(m, "num_trees", lambda: None)()}
            elif name == "xgboost":
                cfg = json.loads(m.save_config())
                tp = cfg.get("learner", {}).get("gradient_booster", {}).get("tree_train_param", {})
                hp["xgb"] = {"depth": _intornone(tp.get("max_depth")), "eta": _floatornone(tp.get("eta")),
                             "rounds": getattr(m, "best_iteration", None)}
            elif name == "mlp":
                import torch.nn as nn
                dims = [layer.out_features for layer in getattr(m, "module", []) if isinstance(layer, nn.Linear)]
                hp["mlp"] = {"hidden": dims[:-1] if dims else None}
            else:
                hp[name] = "unparsed"
        except Exception:
            hp[name] = "unparsed"
    return hp


def _intornone(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _floatornone(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def model_signature(payload: dict) -> dict:
    """Compact, hashable signature of one model bundle (consistent across the
    predict path and the bundle-backfill path — both read the same payload)."""
    members = list(payload.get("members") or [])
    fcols = list(payload.get("feature_columns") or [])
    hyperparams = {
        "windows": {
            "train_weeks": payload.get("train_weeks"),
            "val_weeks": payload.get("val_weeks"),
            "retrain_freq": payload.get("retrain_freq"),
        },
        "members": _hp_from_models(payload.get("models") or {}),
    }
    sig = {
        "members": ",".join(members),
        "combine_method": payload.get("combine_method"),
        "n_features": len(fcols),
        "macro_enabled": int(_macro_enabled(fcols)),
        "hyperparams": hyperparams,
        "saved_at": payload.get("saved_at"),
    }
    # The config_hash must fingerprint the *spec* (architecture + configured
    # hyperparameters), not per-fit artifacts. Drop data-dependent fields like
    # the realized boosting `rounds` (early stopping varies each retrain) so that
    # consecutive retrains of the same model collapse to one version.
    hashed = {
        "members": sig["members"],
        "combine_method": sig["combine_method"],
        "n_features": sig["n_features"],
        "macro_enabled": sig["macro_enabled"],
        "hyperparams": _strip_volatile(hyperparams),
    }
    sig["config_hash"] = hashlib.sha1(
        json.dumps(hashed, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return sig


_VOLATILE_HP_KEYS = {"rounds", "best_iteration"}


def _strip_volatile(hp: dict) -> dict:
    """Recursively drop per-fit artifact keys so the spec hash is retrain-stable."""
    out: dict = {}
    for k, v in hp.items():
        if k in _VOLATILE_HP_KEYS:
            continue
        out[k] = _strip_volatile(v) if isinstance(v, dict) else v
    return out


def record_model_version(
    payload: dict,
    signal_day: str,
    retrain_date: str,
    db_path: Path = DB_PATH,
    ignore_existing: bool = False,
) -> str:
    """Upsert one model-version row. Returns its config_hash.

    `ignore_existing=True` only fills gaps (INSERT OR IGNORE) so an authoritative
    row written by predict.py is never clobbered by a later backfill.
    """
    sig = model_signature(payload)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    row = (
        signal_day, str(retrain_date), sig["members"], sig["combine_method"],
        sig["n_features"], sig["macro_enabled"], json.dumps(sig["hyperparams"], default=str),
        sig["config_hash"], sig["saved_at"], _utcnow(),
    )
    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    cols = ("signal_day", "bundle_retrain_date", "members", "combine_method", "n_features",
            "macro_enabled", "hyperparams", "config_hash", "saved_at", "logged_at")
    conflict = "" if ignore_existing else (
        " ON CONFLICT(signal_day, bundle_retrain_date) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("signal_day", "bundle_retrain_date"))
    )
    with sqlite3.connect(db_path) as conn:
        _ensure_versions_table(conn)
        conn.execute(
            f"{verb} INTO {VERSIONS_TABLE} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}){conflict}",
            row,
        )
        conn.commit()
    return sig["config_hash"]


def backfill_model_versions(signal_days: list[str], db_path: Path = DB_PATH) -> int:
    """Populate model_versions from every saved bundle joblib (gap-fill only).

    Loads each bundle sequentially (they carry the fitted models), so this is the
    heavier path — run it occasionally, not per-forecast. Existing rows (e.g.
    written authoritatively by predict.py) are preserved.
    """
    import joblib

    n = 0
    for sd in signal_days:
        d = MODELS_ROOT / sd.upper()
        if not d.exists():
            continue
        for path in sorted(d.glob("*.joblib")):
            try:
                b = joblib.load(path)
                record_model_version(
                    b, signal_day=b.get("signal_day", sd),
                    retrain_date=b.get("retrain_date", path.stem),
                    db_path=db_path, ignore_existing=True,
                )
                n += 1
            except Exception as e:
                log.warning(f"version backfill skipped {path.name}: {e}")
    return n


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def update_for_forecast(
    forecast: pd.DataFrame,
    close: pd.DataFrame | None = None,
    n_per_side: int = 10,
    db_path: Path = DB_PATH,
) -> int:
    """Log one forecast's named picks into the ledger (called from predict.py).

    `close` (if supplied) lets newly-resolvable picks carry their realized
    return immediately; for a fresh forecast the target is in the future, so the
    pick is logged with realized=NULL and filled later by `rebuild_ledger`.
    """
    picks = extract_picks(forecast, n_per_side=n_per_side)
    if picks.empty:
        return 0
    picks = evaluate_picks(picks, close)
    n = upsert_picks(picks, db_path=db_path)
    write_csv_mirror(db_path=db_path)
    log.info(f"Picks ledger: upserted {n} rows for target {picks['target_date'].iloc[0]} ({db_path})")
    return n


def rebuild_ledger(
    signal_days: list[str] | None = None,
    n_per_side: int = 10,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Re-derive the full ledger from saved forecast parquets and re-evaluate
    realized returns against the latest price cache. Idempotent."""
    from src.backtest.forecast_track_record import _collect_forecast_tickers, _load_close_panel

    signal_days = signal_days or ["WED", "THU", "FRI"]
    all_picks: list[pd.DataFrame] = []
    for sd in signal_days:
        d = FORECASTS_ROOT / sd.upper()
        if not d.exists():
            continue
        files = sorted(d.glob("*.parquet"))
        if not files:
            continue
        tickers = _collect_forecast_tickers(sd)
        try:
            close = _load_close_panel(None, tickers)
        except Exception as e:
            log.warning(f"{sd}: could not load close panel ({e}); realized returns left blank")
            close = None
        for path in files:
            fc = pd.read_parquet(path)
            picks = extract_picks(fc, n_per_side=n_per_side)
            picks = evaluate_picks(picks, close)
            all_picks.append(picks)

    if not all_picks:
        log.warning("No forecasts found — nothing to rebuild.")
        return pd.DataFrame(columns=COLUMNS)

    ledger = pd.concat(all_picks, ignore_index=True)
    upsert_picks(ledger, db_path=db_path)
    write_csv_mirror(db_path=db_path)
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the named-pick forecast ledger.")
    parser.add_argument("--signal-day", default=None,
                        choices=["MON", "TUE", "WED", "THU", "FRI"],
                        help="Limit to one signal day (default: WED, THU, FRI).")
    parser.add_argument("--top-n", type=int, default=10, help="Picks per side per scope. Default 10.")
    parser.add_argument("--no-versions", action="store_true",
                        help="Skip the model_versions backfill (the heavier bundle scan).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sds = [args.signal_day] if args.signal_day else ["WED", "THU", "FRI"]
    ledger = rebuild_ledger(signal_days=sds, n_per_side=args.top_n)

    if not args.no_versions:
        log.info("Backfilling model_versions from saved bundles (gap-fill)...")
        nv = backfill_model_versions(sds)
        log.info(f"model_versions: processed {nv} bundle(s)")

    if ledger.empty:
        return 0

    n_weeks = ledger["target_date"].nunique()
    n_resolved = int((ledger["evaluable"] == 1).sum())
    print(f"\n=== Forecast picks ledger ===")
    print(f"  rows: {len(ledger):,} | target weeks: {n_weeks} | resolved picks: {n_resolved:,}")
    print(f"  DB:  {DB_PATH}")
    print(f"  CSV: {CSV_PATH}")

    # Per-scope hit rate + mean signed return over resolved picks.
    resolved = ledger[ledger["evaluable"] == 1]
    if not resolved.empty:
        print("\n  Resolved-pick performance by scope:")
        for scope in ("ALL", "US", "UK", "CA"):
            s = resolved[resolved["scope"] == scope]
            if s.empty:
                continue
            hit = float((s["signed_ret"] > 0).mean())
            mean = float(s["signed_ret"].mean())
            print(f"    {scope:>3}: n={len(s):>4}  hit-rate={hit:5.1%}  mean signed ret={mean:+.2%}")

    # Model version-change history: collapse consecutive retrains with the same
    # config_hash so only actual spec changes show up.
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            vt = pd.read_sql_query(
                f"SELECT signal_day, bundle_retrain_date, n_features, macro_enabled, config_hash, members "
                f"FROM {VERSIONS_TABLE} ORDER BY signal_day, bundle_retrain_date", conn,
            )
        if not vt.empty:
            print(f"\n  Model version history ({len(vt)} vintages, "
                  f"{vt['config_hash'].nunique()} distinct specs):")
            for sd, grp in vt.groupby("signal_day"):
                grp = grp.sort_values("bundle_retrain_date")
                changes = grp[grp["config_hash"].ne(grp["config_hash"].shift())]
                print(f"    [{sd}] spec changes:")
                for _, r in changes.iterrows():
                    macro = "macro-on" if r["macro_enabled"] else "macro-off"
                    print(f"      {r['bundle_retrain_date']}  {r['config_hash']}  "
                          f"{r['n_features']:>3} feat  {macro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
