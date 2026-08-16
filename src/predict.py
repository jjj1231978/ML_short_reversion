"""Live-inference entry point: forecast the next signal day's long/short lists.

The traded signal day is **Thursday** — the default, the day the backtest is fit
on, and the only day whose bundles and forecasts are maintained. `--signal-day`
still accepts the other weekdays for research (each keeps its own bundle
namespace), but nothing routine generates them.

Where the signal day falls on an exchange holiday the book trades the session
before it (Wednesday), or the one after (Friday) if that week opens with the
closure. That resolution is per region — a US closure does not move the UK or
Canadian legs — and is recorded per row as `target_trade_date`.

Usage:
    python -m src.predict                         # forecast next Thu, retrain if cadence elapsed
    python -m src.predict --no-retrain            # never retrain (load latest bundle, error if none)
    python -m src.predict --force-retrain         # always refit before predicting
    python -m src.predict --top-n 10              # how many names per side (default 10)
    python -m src.predict --config default        # which config to load
    python -m src.predict --output /path/to.parquet

Lifecycle vs `src.main`:
  - `python -m src.main` runs the full historical backtest end-to-end and
    persists a model bundle at every retrain step (under
    `data/processed/models/`).
  - `python -m src.predict` does NOT replay history. It pulls fresh FMP data,
    builds features through the most recent signal-day close, then either:
      * loads the latest saved bundle and scores that one cross-section, or
      * (if `retrain_freq` weeks have elapsed since the last persisted retrain)
        retrains all configured ensemble members on the most-recent train/val
        slice, saves a new bundle, then scores.
  - Output: top-N long + top-N short ranked by ensemble prediction, with the
    same eligibility filter (ADV floor + price floor + PIT US membership +
    ex-financials) that the backtest applies. Persisted to
    `data/processed/forecasts/{signal_day}/{target_date}.parquet`.

A cold start (no bundles on disk) requires either `python -m src.main` first,
or `--force-retrain` (will fit on the latest window and persist).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, load_config
from src.data.fetch import (
    fetch_analyst_grades,
    fetch_fundamentals,
    fetch_market_close,
    fetch_price_data,
)
from src.data.fmp import _resolve_cache_dir, fetch_index_membership_fmp
from src.data.macro import DEFAULT_MACROS, fetch_macro_data
from src.data.universe import (
    build_eligibility_mask,
    filter_by_adv,
    filter_by_gics_sector,
)
from src.features.build import build_feature_matrix
from src.features.neutralize import neutralize_stacked
from src.features.regime import build_and_save_macro_regimes, load_macro_regimes
from src.lib.trading_days import resolve_session, sessions_by_group
from src.model.ensemble import combine_predictions
from src.model.persistence import (
    list_bundles,
    load_latest_bundle,
    needs_retrain,
    read_state,
    save_bundle,
)
from src.model.train import DISPATCH, _resolve_members

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORECASTS_ROOT = PROCESSED_DIR / "forecasts"

# Map between pandas/numpy weekday() index and the codes used by build_feature_matrix.
WEEKDAY_CODE_TO_INDEX = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
WEEKDAY_INDEX_TO_NAME = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _next_signal_day(today: pd.Timestamp, signal_day: str) -> pd.Timestamp:
    """Most recent or upcoming target weekday >= today. If today IS that weekday, use today.

    Pure calendar arithmetic — the result can land on an exchange holiday. Use
    `_resolve_region_targets` to turn a nominal date into a tradeable one.
    """
    target_idx = WEEKDAY_CODE_TO_INDEX[signal_day.upper()]
    offset = (target_idx - today.weekday()) % 7
    return (today + pd.Timedelta(days=offset)).normalize()


def _resolve_region_targets(
    close: pd.DataFrame, region_map: dict, nominal: pd.Timestamp
) -> dict[str, pd.Timestamp]:
    """Per-region tradeable date for a nominal signal-day target.

    The target is a weekly-bin label produced by calendar arithmetic, so it can
    fall on a day a given exchange is shut — US Thanksgiving, a UK bank holiday,
    Canada Day. Each region resolves independently against its own observed
    sessions, because a book spanning US/UK/CA has no single calendar: the US
    can be closed on a Thursday the LSE and TSX both trade through.

    Falls back to the session before the target (i.e. Wednesday for a shut
    Thursday) and only then to the one after (Friday), per `resolve_session`.

    Regions whose target cannot be resolved inside that window — including the
    normal case of a target in the future, whose bars do not exist yet — keep
    the nominal date. That is the honest answer: the forecast is *for* the
    nominal bin, and a genuine closure will be picked up when the picks are
    evaluated against real bars.
    """
    if close is None or close.empty or not region_map:
        return {}

    nominal = pd.Timestamp(nominal).normalize()
    per_region = sessions_by_group(close, region_map)
    resolved: dict[str, pd.Timestamp] = {}
    for region, days in per_region.items():
        # Only treat this as a closure if the panel actually covers the target
        # week; an unresolvable future date just means the bars are not in yet.
        if len(days) == 0 or days.max() < nominal:
            resolved[region] = nominal
            continue
        actual = resolve_session(nominal, days)
        if actual is None:
            log.warning(
                f"{region}: no session within ±3d of target {nominal.date()}; "
                "keeping the nominal date."
            )
            resolved[region] = nominal
        else:
            resolved[region] = actual
            if actual != nominal:
                log.info(
                    f"{region}: target {nominal.date()} is not a session — "
                    f"trading date resolves to {actual.date()}."
                )
    return resolved


# ---------------------------------------------------------------------------
# Feature preparation (mirrors src/main.py Phase 1+2)
# ---------------------------------------------------------------------------

def _load_universe(cfg: dict, refresh_membership: bool = True) -> tuple[list[str], dict, dict, dict, Path]:
    """Resolve region union, exclude financials. Returns (tickers, sector_map,
    industry_map, region_map, cache_root).

    If `refresh_membership` is True, re-pulls FMP's add/drop event log for each
    region before reading the cached parquet so recent index reshuffles are
    captured. Costs ~30s of API calls.
    """
    regions = tuple(cfg["universe"].get("regions", ["US"]))
    backtest_start = pd.Timestamp(cfg["data"]["start_date"])
    backtest_end = pd.Timestamp(cfg["data"]["end_date"])

    cache_root = _resolve_cache_dir(None) / "constituents"
    region_files = {
        "US": "sp500_membership.parquet",
        "UK": "ftse100_current.parquet",
        "CA": "tsx60_current.parquet",
    }
    region_index = {"US": "sp500", "UK": "ftse100", "CA": "tsx60"}
    PIT_REGIONS = {"US"}

    if refresh_membership:
        for region in regions:
            r = region.upper()
            idx = region_index.get(r)
            if idx is None:
                continue
            log.info(f"Refreshing {r} membership from FMP (refresh=True)...")
            try:
                fetch_index_membership_fmp(
                    index=idx,
                    refresh=True,
                    rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
                )
            except Exception as e:
                log.warning(f"Membership refresh failed for {r}: {e}; falling back to cache")

    frames = []
    for region in regions:
        region = region.upper()
        fname = region_files.get(region)
        if fname is None:
            log.warning(f"Unknown region {region!r}; skipping")
            continue
        path = cache_root / fname
        if not path.exists():
            log.warning(f"Membership cache missing for {region}: {path}")
            continue
        df = pd.read_parquet(path)
        if region in PIT_REGIONS:
            overlap = df[
                (df["date_removed"].isna() | (df["date_removed"] >= backtest_start))
                & (df["date_added"].isna() | (df["date_added"] <= backtest_end))
            ]
        else:
            overlap = df
        frames.append(overlap[["symbol", "sector", "sub_industry", "region", "exchange"]])

    constituents = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])
    sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
    industry_map = dict(zip(constituents["symbol"], constituents["sub_industry"]))
    region_map = dict(zip(constituents["symbol"], constituents["region"]))
    tickers = constituents["symbol"].tolist()

    excl = cfg["universe"].get("exclude_sectors", ["Financial Services"])
    tickers = filter_by_gics_sector(tickers, sector_map, exclude_sectors=excl)
    log.info(f"Universe: {len(tickers):,} tickers across {regions} (post sector filter)")
    return tickers, sector_map, industry_map, region_map, cache_root


def _build_features_and_close(
    cfg: dict,
    tickers: list[str],
    industry_map: dict,
    region_map: dict,
    topup_grades: bool = True,
    signal_day: str = "WED",
    skip_price_fetch: bool = False,
    skip_fundamentals_fetch: bool = False,
):
    """Replicates main.py Phase 1a-2b. Returns (features_neutralized, close,
    volume, market_close, kept_tickers, kept_region_map, kept_industry_map)."""
    price_source = cfg["data"].get("price_source", "fmp")
    price_kwargs = {}
    if price_source == "databento":
        price_kwargs["dataset"] = cfg["data"]["databento_dataset"]

    if skip_price_fetch:
        log.info("Loading prices from cache only (skip_fetch=True)...")
        if price_source == "fmp":
            price_kwargs["skip_fetch"] = True
        else:
            log.warning(
                f"--no-refresh-prices only supported for price_source='fmp' "
                f"(got {price_source!r}); falling back to normal fetch."
            )
    else:
        log.info("Fetching prices (cache-aware)...")
    price_data = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        **price_kwargs,
    )
    close = price_data["Close"]
    volume = price_data["Volume"]
    log.info(f"Price data: {close.shape[0]} days × {close.shape[1]} tickers")

    benchmarks = cfg["data"].get("benchmarks", {})
    market_ticker = benchmarks.get("US", "SPY") if price_source == "fmp" else "SPY"
    # The benchmark is a single symbol; fetch_market_close does NOT accept
    # `skip_fetch` (that's an equity-path kwarg), and a cache-aware single-symbol
    # fetch is cheap regardless. Drop skip_fetch so the market series still loads
    # under --no-refresh-prices — otherwise BETA6M / IREV1W (beta-adjusted
    # residual reversal) silently zero out.
    market_kwargs = {k: v for k, v in price_kwargs.items() if k != "skip_fetch"}
    try:
        market_close = fetch_market_close(
            market_ticker,
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            **market_kwargs,
        )
    except Exception as e:
        log.warning(f"Market close fetch failed: {e}")
        market_close = None

    # ADV filter — keep tickers passing the floor as of the most recent day.
    min_adv_by_region = cfg["universe"].get("min_adv_6m_by_region")
    if min_adv_by_region:
        adv_mask = filter_by_adv(
            volume, close, min_adv=min_adv_by_region, window=126, region_map=region_map
        )
    else:
        adv_mask = filter_by_adv(volume, close, min_adv=cfg["universe"]["min_adv_6m"], window=126)
    if not adv_mask.empty:
        passing = adv_mask.iloc[-1].fillna(False)
        keep = passing[passing].index.tolist()
        close = close[keep]
        volume = volume[keep]
        tickers = [t for t in tickers if t in keep]
        industry_map = {t: i for t, i in industry_map.items() if t in keep}
        region_map = {t: r for t, r in region_map.items() if t in keep}
        log.info(f"ADV filter: kept {len(keep)} tickers")

    fund_source = cfg["data"].get("fundamental_source")
    if skip_fundamentals_fetch:
        log.info("Loading fundamentals from cache only (skip_fetch=True)...")
    else:
        log.info("Fetching fundamentals...")
    fund_kwargs = {}
    if fund_source == "fmp":
        fund_kwargs["price_data"] = price_data
        if skip_fundamentals_fetch:
            fund_kwargs["skip_fetch"] = True
    elif skip_fundamentals_fetch:
        log.warning(
            f"--no-refresh-fundamentals only supported for fundamental_source='fmp' "
            f"(got {fund_source!r}); falling back to normal fetch."
        )
    fundamentals = fetch_fundamentals(
        tickers=tickers,
        pub_lag_days=cfg["data"]["fundamental_pub_lag_days"],
        **fund_kwargs,
    )

    log.info(f"Fetching analyst grades (topup_existing={topup_grades})...")
    try:
        grade_events = fetch_analyst_grades(
            tickers=tickers,
            rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
            topup_existing=topup_grades,
        )
    except Exception as e:
        log.warning(f"Grades fetch failed: {e}; UPDOWN1W_RATINGS will be NaN")
        grade_events = None

    macro_df = None
    macro_regimes_df = None
    macro_cfg = cfg.get("macro") or {}
    if macro_cfg.get("enabled", False):
        try:
            macro_df = fetch_macro_data(
                symbols=macro_cfg.get("symbols") or DEFAULT_MACROS,
                start=cfg["data"]["start_date"],
                end=cfg["data"]["end_date"],
                rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
            )
        except Exception as e:
            log.warning(f"Macro fetch failed: {e}")
        if macro_df is not None and macro_cfg.get("hmm_enabled", True):
            regimes_path = PROCESSED_DIR / "macro_regimes.parquet"
            macro_regimes_df = load_macro_regimes(
                regimes_path,
                expected_macros=set(macro_df.columns),
                expected_end=macro_df.index.max(),
            )
            if macro_regimes_df is None or macro_cfg.get("hmm_refit", False):
                try:
                    macro_regimes_df = build_and_save_macro_regimes(
                        macro_df, regimes_path,
                        n_states=macro_cfg.get("hmm_states", 2),
                        fit_mode=macro_cfg.get("hmm_fit_mode", "walk_forward"),
                        refit_every=macro_cfg.get("hmm_refit_every", 60),
                        min_obs=macro_cfg.get("hmm_min_obs", 252),
                        n_iter=macro_cfg.get("hmm_iter", 200),
                        random_state=macro_cfg.get("hmm_seed", 42),
                    )
                except Exception as e:
                    log.warning(f"HMM regime fit failed: {e}")
                    macro_regimes_df = None

    log.info(f"Building feature matrix (signal_day={signal_day})...")
    features, passthrough_cols = build_feature_matrix(
        close, volume,
        market_close=market_close,
        fundamentals=fundamentals,
        grade_events=grade_events,
        macro=macro_df,
        macro_regimes=macro_regimes_df,
        cfg=cfg,
        signal_day=signal_day,
    )

    log.info("Neutralizing features...")
    n_cfg = cfg["neutralization"]
    features = neutralize_stacked(
        features,
        industry_map=industry_map,
        winsorize_lower=n_cfg["winsorize_lower"],
        winsorize_upper=n_cfg["winsorize_upper"],
        zscore_iterations=n_cfg["zscore_iterations"],
        zscore_cap=n_cfg["zscore_cap"],
        max_missing_factors=n_cfg["max_missing_factors"],
        passthrough_cols=passthrough_cols,
    )
    log.info(f"Feature matrix (neutralized): {features.shape}")
    return features, close, volume, market_close, tickers, region_map, industry_map


# ---------------------------------------------------------------------------
# Retrain helper
# ---------------------------------------------------------------------------

def _retrain_with_close(
    cfg: dict,
    features: pd.DataFrame,
    close: pd.DataFrame,
    latest_wed: pd.Timestamp,
    signal_day: str = "WED",
) -> tuple[dict, list[str], str, pd.Timestamp]:
    """Refit each ensemble member on most-recent train/val slice. Returns
    (fitted_models, members, combine_method, retrain_date_used)."""
    from src.features.build import build_target

    model_cfg = cfg["model"]
    train_weeks = model_cfg["train_weeks"]
    val_weeks = model_cfg["val_weeks"]
    members = _resolve_members(cfg)
    combine_method = model_cfg.get("ensemble", {}).get("combine", "rank_mean")

    target = build_target(
        close, signal_day=signal_day,
        exec_lag_days=cfg["backtest"].get("execution_lag_days", 1),
    )
    dates = features.index.get_level_values("date").unique().sort_values()
    if latest_wed not in dates:
        prior = dates[dates <= latest_wed]
        if len(prior) == 0:
            raise RuntimeError(f"No feature dates at or before {latest_wed.date()}")
        latest_wed = prior[-1]

    i = dates.get_loc(latest_wed)
    if i < train_weeks + val_weeks:
        raise RuntimeError(
            f"Insufficient history at {latest_wed.date()}: need "
            f"{train_weeks + val_weeks} prior weeks, have {i}."
        )
    train_start = dates[i - train_weeks - val_weeks]
    train_end = dates[i - val_weeks - 1]
    val_start = dates[i - val_weeks]
    val_end = dates[i - 1]

    train_mask = features.index.get_level_values("date").to_series().between(
        train_start, train_end
    ).values
    val_mask = features.index.get_level_values("date").to_series().between(
        val_start, val_end
    ).values
    X_train = features.loc[train_mask].fillna(0.0)
    y_train = target["target"].reindex(X_train.index)
    X_val = features.loc[val_mask].fillna(0.0)
    y_val = target["target"].reindex(X_val.index)

    valid_train = y_train.notna()
    valid_val = y_val.notna()
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_val = X_val[valid_val]
    y_val = y_val[valid_val]

    log.info(
        f"Retrain @ {latest_wed.date()}: train={len(X_train):,} rows "
        f"({train_start.date()}→{train_end.date()}), "
        f"val={len(X_val):,} rows ({val_start.date()}→{val_end.date()})"
    )
    models: dict = {}
    for m in members:
        fit_fn, _ = DISPATCH[m]
        models[m] = fit_fn(model_cfg[m], X_train, y_train, X_val, y_val)
    return models, members, combine_method, latest_wed


# ---------------------------------------------------------------------------
# Prediction + eligibility
# ---------------------------------------------------------------------------

def _predict_cross_section(
    models: dict, members: list[str], combine_method: str,
    week_features: pd.DataFrame, feature_columns: list[str],
) -> pd.Series:
    """Score one week. `week_features` is the (ticker)-indexed slice for the
    target Wednesday. Reindexed to bundle's feature column order before
    predicting."""
    X = week_features.reindex(columns=feature_columns).fillna(0.0)
    per_model: dict[str, pd.Series] = {}
    for m in members:
        _, pred_fn = DISPATCH[m]
        preds = pd.Series(pred_fn(models[m], X), index=X.index, name="prediction")
        per_model[m] = preds
    if len(members) > 1:
        # ir_weighted needs per-week validation ICs and stack needs realized
        # target history; neither exists when scoring a single forward week, so
        # the live forecast degrades to the equal-weight rank mean.
        if combine_method in ("ir_weighted", "stack"):
            log.warning(
                "combine=%s unsupported in live single-week scoring; "
                "using rank_mean for the forecast.", combine_method
            )
            combine_method = "rank_mean"
        return combine_predictions(per_model, method=combine_method)
    return per_model[members[0]].copy()


def _build_target_eligibility(
    cfg: dict,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    region_map: dict,
    cache_root: Path,
    target_wed: pd.Timestamp,
    signal_day: str = "WED",
) -> pd.Series:
    """Return a boolean Series indexed by ticker indicating eligibility on
    `target_wed` (the most recent signal-day close)."""
    from src.features.build import resample_to_weekday
    adv_daily = (close * volume).rolling(window=126, min_periods=63).mean()
    weekly_close = resample_to_weekday(close, signal_day)
    weekly_adv = resample_to_weekday(adv_daily, signal_day)

    pit_us = None
    if "US" in [r.upper() for r in cfg["universe"].get("regions", ["US"])]:
        pit_path = cache_root / "sp500_membership.parquet"
        if pit_path.exists():
            pit_us = pd.read_parquet(pit_path)

    eligibility = build_eligibility_mask(
        weekly_close=weekly_close,
        weekly_adv=weekly_adv,
        region_map=region_map,
        min_price_by_region=cfg["universe"].get("min_price_by_region", {}),
        min_adv_by_region=cfg["universe"].get("min_adv_6m_by_region", {}),
        pit_membership_us=pit_us,
    )
    # Snap target_wed to the closest Wednesday index in `eligibility`
    if target_wed not in eligibility.index:
        prior = eligibility.index[eligibility.index <= target_wed]
        if len(prior) == 0:
            return pd.Series(False, index=close.columns)
        target_wed = prior[-1]
    out = eligibility.loc[target_wed]

    # Drop preferred-share lines from the tradeable set. Applied at SELECTION
    # time only — the model is not retrained, it simply never picks them. See
    # universe.is_preferred_share for why they are unsuitable (fixed par, bond
    # -like yields, parent-issuer market cap).
    if cfg["universe"].get("exclude_preferred", True):
        from src.data.universe import is_preferred_share

        pref = pd.Index([t for t in out.index if is_preferred_share(t)])
        if len(pref):
            out.loc[pref] = False
            log.info(
                f"Excluded {len(pref)} preferred-share lines from the eligible "
                f"universe: {', '.join(sorted(pref)[:12])}"
                + (" ..." if len(pref) > 12 else "")
            )
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast top long/short for the next rebalance weekday."
    )
    parser.add_argument("--config", default="default", help="Config name (default: default)")
    parser.add_argument("--top-n", type=int, default=10, help="Names per side (default: 10)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Never retrain — error if no bundle exists or cadence elapsed.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Refit before predicting regardless of cadence.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output parquet path.")
    parser.add_argument("--no-refresh-grades", action="store_true",
                        help="Skip the per-ticker grades top-up "
                             "(default: top-up on every run to catch new rating events).")
    parser.add_argument("--no-refresh-membership", action="store_true",
                        help="Skip the index membership refresh "
                             "(default: re-pull FMP add/drop events on every run).")
    parser.add_argument("--no-refresh-prices", action="store_true",
                        help="Skip the per-ticker price refresh — load whatever is "
                             "in the cache. Fast path for iterating Wed/Thu/Fri runs "
                             "after the day's first fetch.")
    parser.add_argument("--no-refresh-fundamentals", action="store_true",
                        help="Skip the per-ticker fundamentals refresh — compute "
                             "ratios from cached quarterly parquets only.")
    parser.add_argument("--signal-day", default="THU",
                        choices=["MON", "TUE", "WED", "THU", "FRI"],
                        help="Rebalance weekday (default THU — the day this book "
                             "is traded and the day the backtest is fit on). Each "
                             "value gets its own model bundle namespace "
                             "(data/processed/models/{day}/) and forecast output "
                             "directory (data/processed/forecasts/{day}/). Other "
                             "days remain runnable for research, but are not "
                             "maintained.")
    args = parser.parse_args(argv)

    signal_day = args.signal_day.upper()
    cfg = load_config(args.config)
    today = pd.Timestamp(datetime.now().date())
    log.info(
        f"Today {today.date()}, signal_day={signal_day} "
        f"({WEEKDAY_INDEX_TO_NAME[WEEKDAY_CODE_TO_INDEX[signal_day]]}); "
        f"resolving as-of and target dates from features..."
    )

    # Override config end_date so we pull data up to today.
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["end_date"] = today.strftime("%Y-%m-%d")

    tickers, sector_map, industry_map, region_map, cache_root = _load_universe(
        cfg, refresh_membership=not args.no_refresh_membership
    )
    features, close, volume, market_close, tickers, region_map, industry_map = (
        _build_features_and_close(
            cfg, tickers, industry_map, region_map,
            topup_grades=not args.no_refresh_grades,
            signal_day=signal_day,
            skip_price_fetch=args.no_refresh_prices,
            skip_fundamentals_fetch=args.no_refresh_fundamentals,
        )
    )

    dates = features.index.get_level_values("date").unique().sort_values()
    if len(dates) == 0:
        log.error("Feature matrix is empty — nothing to forecast.")
        return 2
    # Use the most recent COMPLETED weekly bin as the as-of date — i.e. the
    # latest bin whose label-date is ≤ today. pandas' resample("W-WED") labels
    # the in-progress bin by its future end date; using it as as-of would
    # double-count days and break the Wed→Wed forecast horizon.
    completed = dates[dates <= today]
    if len(completed) == 0:
        log.error(f"No completed weekly bin <= today ({today.date()}).")
        return 2
    latest_wed = completed[-1]
    # Target is the next signal-day after the as-of: that defines the Wed→Wed
    # holding period the model was trained to forecast.
    target_wed = latest_wed + pd.Timedelta(days=7)
    log.info(
        f"As-of (latest completed {signal_day} bin): {latest_wed.date()}  →  "
        f"target {signal_day} {target_wed.date()}"
    )
    # The target is a bin label and may fall on an exchange holiday. Resolve the
    # date each region can actually trade (target → prior session → next), and
    # carry it alongside the nominal date rather than replacing it: the nominal
    # date identifies the forecast (filename, ledger key) and must stay stable
    # and comparable across regions.
    region_target = _resolve_region_targets(close, region_map, target_wed)

    # Decide retrain vs load
    retrain_freq = cfg["model"]["retrain_freq"]
    bundle = None
    if not args.force_retrain:
        bundle = load_latest_bundle(signal_day)

    do_retrain = args.force_retrain or bundle is None or needs_retrain(latest_wed, retrain_freq, signal_day)
    if args.no_retrain and do_retrain:
        if bundle is None:
            log.error("No persisted bundle and --no-retrain set. "
                      "Run `python -m src.main` first or drop --no-retrain.")
            return 3
        log.warning(
            f"Cadence elapsed (latest bundle {bundle['retrain_date']}) but "
            "--no-retrain set; reusing stale bundle."
        )
        do_retrain = False

    if do_retrain:
        log.info("Retraining ensemble on most recent train/val window...")
        models, members, combine_method, retrain_date_used = _retrain_with_close(
            cfg, features, close, latest_wed, signal_day=signal_day
        )
        saved = save_bundle(
            retrain_date=retrain_date_used,
            models=models,
            members=members,
            combine_method=combine_method,
            feature_columns=list(features.columns),
            train_weeks=cfg["model"]["train_weeks"],
            val_weeks=cfg["model"]["val_weeks"],
            retrain_freq=retrain_freq,
            signal_day=signal_day,
        )
        feature_columns = list(features.columns)
        model_saved_at = saved.get("saved_at")
    else:
        models = bundle["models"]
        members = bundle["members"]
        combine_method = bundle["combine_method"]
        feature_columns = bundle["feature_columns"]
        model_saved_at = bundle.get("saved_at")
        log.info(f"Reusing bundle from {bundle['retrain_date']} "
                 f"({len(members)} members: {members})")

    # Score the latest Wednesday cross-section. Keep the (date, ticker)
    # MultiIndex during prediction so `combine_predictions` can group by date
    # for the rank-mean ensemble; drop the date level afterward to build the
    # ticker-indexed output table.
    week_features = features.loc[
        features.index.get_level_values("date") == latest_wed
    ]
    preds = _predict_cross_section(
        models, members, combine_method,
        week_features, feature_columns,
    )
    preds = preds.droplevel("date")
    preds.name = "prediction"

    # Per-pick feature attribution: name the 1-2 factors that moved each score
    # the most (SHAP over the tree members). Best-effort — a failure here must
    # not block the forecast.
    try:
        from src.diagnostics.explain import explain_cross_section

        X_explain = (
            week_features.reindex(columns=feature_columns)
            .fillna(0.0)
            .droplevel("date")
        )
        commentary = explain_cross_section(models, members, X_explain, top_k=2)
        log.info("Computed per-pick commentary for %d tickers", len(commentary))
    except Exception as e:  # noqa: BLE001
        log.warning(f"Commentary generation failed: {e}")
        commentary = pd.Series("", index=preds.index, name="commentary")

    # Eligibility filter for the target Wednesday (use latest_wed as the
    # eligibility key since that's the as-of date for the position decision).
    eligible = _build_target_eligibility(
        cfg, close, volume, region_map, cache_root, latest_wed,
        signal_day=signal_day,
    )
    eligible = eligible.reindex(preds.index, fill_value=False)

    df = pd.DataFrame({
        "prediction": preds,
        "eligible": eligible,
        "region": [region_map.get(t, "?") for t in preds.index],
        "industry": [industry_map.get(t, "?") for t in preds.index],
        "commentary": commentary.reindex(preds.index).fillna(""),
    })
    df.index.name = "ticker"

    # Per-region top-N: each region is ranked independently so a top-10 long
    # basket from US doesn't crowd out the UK/CA winners. This is the default
    # because pooled cross-sectional scores carry strong region-level bias
    # (the universe isn't region-mean neutralized — see neutralize_stacked).
    n = args.top_n
    df_eligible = df[df["eligible"]].copy()
    longs_by_region: dict[str, pd.DataFrame] = {}
    shorts_by_region: dict[str, pd.DataFrame] = {}
    for region, grp in df_eligible.groupby("region"):
        g = grp.sort_values("prediction", ascending=False)
        longs = g.head(n).copy()
        longs["side"] = "LONG"
        longs["rank"] = range(1, len(longs) + 1)
        longs_by_region[region] = longs
        shorts = g.tail(n).copy().iloc[::-1]
        shorts["side"] = "SHORT"
        shorts["rank"] = range(1, len(shorts) + 1)
        shorts_by_region[region] = shorts

    # Persist full + per-region rank columns. Streamlit page reads these.
    out_dir = FORECASTS_ROOT / signal_day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"{target_wed.strftime('%Y-%m-%d')}.parquet"
    bundle_retrain_date = (
        bundle["retrain_date"] if (bundle and not do_retrain)
        else retrain_date_used.strftime("%Y-%m-%d")
    )
    full = df.copy()
    full["as_of_date"] = latest_wed
    full["target_date"] = target_wed
    # Per-region tradeable date. Equals target_date except where that region's
    # exchange is shut on the target, in which case it points at the session
    # actually used to close the position.
    full["target_trade_date"] = (
        full["region"].map(region_target).fillna(target_wed)
        if "region" in full.columns else target_wed
    )
    full["signal_day"] = signal_day
    full["bundle_retrain_date"] = bundle_retrain_date  # model vintage / provenance
    # Rank within (region, eligible-only): 1 = top long, N = bottom short. NaN
    # for ineligible rows.
    full["region_rank"] = (
        df_eligible.groupby("region")["prediction"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    full.to_parquet(out_path)
    log.info(f"Wrote full predictions: {out_path} ({len(full):,} rows)")

    # Append the named top/bottom picks to the accumulating ledger (SQLite + CSV).
    # `close` lets any now-resolvable picks carry realized returns immediately;
    # this forecast's own target is in the future, so it logs with realized=NULL
    # and is filled later by `python -m src.backtest.forecast_picks_log`.
    try:
        from src.backtest.forecast_picks_log import record_model_version, update_for_forecast

        update_for_forecast(full, close=close, n_per_side=args.top_n)
        # Record the model vintage that produced these picks (separate table;
        # picks reference bundle_retrain_date). Authoritative — overwrites any
        # gap-filled backfill row for this vintage.
        vhash = record_model_version(
            {
                "members": members,
                "combine_method": combine_method,
                "feature_columns": feature_columns,
                "train_weeks": cfg["model"]["train_weeks"],
                "val_weeks": cfg["model"]["val_weeks"],
                "retrain_freq": retrain_freq,
                "models": models,
                "saved_at": model_saved_at,
            },
            signal_day=signal_day,
            retrain_date=bundle_retrain_date,
        )
        log.info(f"Model version recorded: {bundle_retrain_date} (config_hash={vhash})")
    except Exception as e:
        log.warning(f"Picks-ledger update failed (forecast parquet still written): {e}")

    # Pretty-print per-region top/bottom.
    weekday_name = WEEKDAY_INDEX_TO_NAME[WEEKDAY_CODE_TO_INDEX[signal_day]]
    print("\n" + "=" * 78)
    print(f"FORECAST  {weekday_name}  as-of {latest_wed.date()}  →  target {target_wed.date()}")
    print(f"  bundle retrain_date: "
          f"{(bundle['retrain_date'] if bundle and not do_retrain else latest_wed.strftime('%Y-%m-%d'))}"
          f"  members: {members}  combine: {combine_method}")
    print(f"  eligible universe: {int(df['eligible'].sum())} / {len(df)}")
    region_counts = df_eligible["region"].value_counts().to_dict()
    print(f"  per-region eligible: {region_counts}")
    print("=" * 78)
    for region in sorted(longs_by_region.keys()):
        _print_side(f"TOP {n} LONG — {region}", longs_by_region[region])
    for region in sorted(shorts_by_region.keys()):
        _print_side(f"TOP {n} SHORT — {region}", shorts_by_region[region])
    return 0


def _print_side(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(f"  {'rank':>4} {'ticker':<10} {'region':<4} {'pred':>10}  industry")
    for ticker, row in df.iterrows():
        print(
            f"  {row['rank']:>4} {ticker:<10} {row['region']:<4} "
            f"{row['prediction']:>10.4f}  {row['industry']}"
        )
        comment = row.get("commentary", "")
        if comment:
            print(f"       ↳ {comment}")


if __name__ == "__main__":
    raise SystemExit(main())
