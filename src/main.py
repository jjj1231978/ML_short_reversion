"""Main pipeline orchestrator: data -> features -> model -> backtest -> diagnostics."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT, load_config
from src.data.fetch import (
    fetch_price_data,
    fetch_market_close,
    fetch_fundamentals,
    fetch_analyst_grades,
)
from src.data.macro import DEFAULT_MACROS, fetch_macro_data
from src.data.universe import (
    get_sp500_constituents,
    filter_by_gics_sector,
    filter_by_adv,
    get_gics_sector_map,
    build_eligibility_mask,
)
from src.features.build import build_feature_matrix, build_target
from src.features.neutralize import neutralize_stacked
from src.features.regime import (
    build_and_save_macro_regimes,
    load_macro_regimes,
)
from src.model.train import rolling_train_predict_ensemble
from src.backtest.portfolio import (
    build_long_short_portfolio,
    compute_performance_metrics,
)
from src.backtest.baselines import r1w_reversal_predictions
from src.diagnostics.shap_analysis import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    alpha_decay_analysis,
    weekday_effect_analysis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(config_name: str = "default") -> dict:
    """Execute the full pipeline end-to-end."""
    cfg = load_config(config_name)

    # --- Phase 1: Data Acquisition ---
    # Universe: FMP point-in-time membership across configured regions.
    # US (SP500) is genuinely point-in-time back to 1957 with date_added /
    # date_removed events. UK (FTSE100) and CA (TSX60) are current-snapshot
    # only on FMP — accepted survivorship-bias residual for those regions.
    log.info("Phase 1: Building point-in-time multi-region universe from FMP membership...")
    regions = tuple(cfg["universe"].get("regions", ["US"]))
    backtest_start = pd.Timestamp(cfg["data"]["start_date"])
    backtest_end = pd.Timestamp(cfg["data"]["end_date"])

    from src.data.fmp import _resolve_cache_dir
    cache_root = _resolve_cache_dir(None) / "constituents"
    region_files = {
        "US": "sp500_membership.parquet",
        "UK": "ftse100_current.parquet",
        "CA": "tsx60_current.parquet",
    }

    # US has true PIT membership; UK/CA are current-snapshot only on FMP.
    # For snapshot regions, treat all members as always-active during the
    # backtest window (accepted survivorship-bias residual; see 001-overview.md).
    PIT_REGIONS = {"US"}

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
            # PIT: keep tickers whose [date_added, date_removed) overlaps the window
            overlap = df[
                (df["date_removed"].isna() | (df["date_removed"] >= backtest_start))
                & (df["date_added"].isna() | (df["date_added"] <= backtest_end))
            ]
            log.info(f"  {region} (PIT): {len(overlap):,} tickers active during window")
        else:
            # Snapshot region: include all members as always-active
            overlap = df
            log.info(f"  {region} (snapshot, survivorship-biased): {len(overlap):,} tickers")
        frames.append(overlap[["symbol", "sector", "sub_industry", "region", "exchange"]])

    constituents = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"])
    sector_map = dict(zip(constituents["symbol"], constituents["sector"]))
    industry_map = dict(zip(constituents["symbol"], constituents["sub_industry"]))
    region_map = dict(zip(constituents["symbol"], constituents["region"]))
    tickers = constituents["symbol"].tolist()
    log.info(f"Multi-region union: {len(tickers):,} tickers across {regions}")

    # Exclude financials. FMP profile uses "Financial Services" (not GICS "Financials").
    excl = cfg["universe"].get("exclude_sectors", ["Financial Services"])
    tickers = filter_by_gics_sector(tickers, sector_map, exclude_sectors=excl)
    log.info(f"After excluding {excl}: {len(tickers):,} tickers")

    log.info("Phase 1a: Fetching price data...")
    price_source = cfg["data"].get("price_source", "databento")
    price_kwargs = {}
    if price_source == "databento":
        price_kwargs["dataset"] = cfg["data"]["databento_dataset"]
    price_data = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        **price_kwargs,
    )
    close = price_data["Close"]
    volume = price_data["Volume"]
    log.info(f"Price data: {close.shape[0]} days x {close.shape[1]} tickers")

    # Market benchmark for BETA6M and IREV1W. Region-aware via configured benchmarks;
    # falls back to SPY (databento) for legacy compatibility.
    benchmarks = cfg["data"].get("benchmarks", {})
    market_ticker = benchmarks.get("US", "SPY") if price_source == "fmp" else "SPY"
    try:
        market_close = fetch_market_close(
            market_ticker,
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            **price_kwargs,
        )
        log.info(f"Market close ({market_ticker}): {len(market_close)} days")
    except Exception as e:
        log.warning(f"Failed to fetch {market_ticker}: {e}. BETA6M / IREV1W will use fallback.")
        market_close = None

    # ADV filter: drop tickers below the 6-month rolling ADV floor. Per-region
    # native-currency thresholds when configured; falls back to the legacy
    # scalar `min_adv_6m`. UK uses GBp pence; CA uses CAD; US uses USD.
    min_adv_by_region = cfg["universe"].get("min_adv_6m_by_region")
    if min_adv_by_region:
        log.info(
            f"Phase 1c: Applying per-region ADV filter (native currency): "
            f"{min_adv_by_region}"
        )
        adv_mask = filter_by_adv(
            volume, close, min_adv=min_adv_by_region,
            window=126, region_map=region_map,
        )
    else:
        min_adv = cfg["universe"]["min_adv_6m"]
        log.info(f"Phase 1c: Applying ADV filter (>= ${min_adv:,.0f} 6m rolling)...")
        adv_mask = filter_by_adv(volume, close, min_adv=min_adv, window=126)
    if not adv_mask.empty:
        # A ticker passes if it satisfies the floor on the most recent date.
        passing = adv_mask.iloc[-1].fillna(False)
        keep = passing[passing].index.tolist()
        dropped = sorted(set(close.columns) - set(keep))
        if min_adv_by_region:
            # Log per-region drop counts for transparency
            from collections import Counter
            dropped_by_region = Counter(region_map.get(t, "?") for t in dropped)
            kept_by_region = Counter(region_map.get(t, "?") for t in keep)
            log.info(
                f"ADV filter: keeping {len(keep)} of {close.shape[1]}; "
                f"per-region kept={dict(kept_by_region)}, dropped={dict(dropped_by_region)}"
            )
        else:
            log.info(f"ADV filter: keeping {len(keep)} of {close.shape[1]} tickers; dropped {len(dropped)}")
        close = close[keep]
        volume = volume[keep]
        tickers = [t for t in tickers if t in keep]
        sector_map = {t: s for t, s in sector_map.items() if t in keep}
        industry_map = {t: i for t, i in industry_map.items() if t in keep}
        region_map = {t: r for t, r in region_map.items() if t in keep}

    log.info("Phase 1b: Fetching fundamentals...")
    fund_kwargs = {}
    if cfg["data"].get("fundamental_source") == "fmp":
        # Pass already-fetched prices so the FMP path doesn't re-load them
        fund_kwargs["price_data"] = price_data
    fundamentals = fetch_fundamentals(
        tickers=tickers,
        pub_lag_days=cfg["data"]["fundamental_pub_lag_days"],
        **fund_kwargs,
    )

    log.info("Phase 1d: Fetching analyst grades from FMP (for UPDOWN1W_RATINGS)...")
    try:
        grade_events = fetch_analyst_grades(
            tickers=tickers,
            rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
        )
        log.info(f"Grade events: {len(grade_events):,}")
    except EnvironmentError as e:
        log.warning(f"FMP grades skipped: {e}")
        grade_events = None
    except Exception as e:
        log.warning(f"FMP grades fetch failed: {e}", exc_info=True)
        grade_events = None

    # Align fundamental columns to price columns (handle ticker format differences)
    common_tickers = close.columns.intersection(
        fundamentals.get("PE", pd.DataFrame()).columns
    )
    if len(common_tickers) > 0:
        log.info(f"Fundamental/price ticker overlap: {len(common_tickers)}")
    else:
        log.warning("No ticker overlap between price and fundamental data — check ticker formats")

    # --- Phase 1e: Macro & HMM regimes ---
    macro_cfg = cfg.get("macro") or {}
    macro_df: pd.DataFrame | None = None
    macro_regimes_df: pd.DataFrame | None = None
    if macro_cfg.get("enabled", True):
        log.info("Phase 1e: Fetching macro data (FMP)...")
        try:
            macro_df = fetch_macro_data(
                symbols=macro_cfg.get("symbols") or DEFAULT_MACROS,
                start=cfg["data"]["start_date"],
                end=cfg["data"]["end_date"],
                rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
            )
            log.info(f"Macro data: {macro_df.shape[0]} dates × {macro_df.shape[1]} series")
        except Exception as e:
            log.warning(f"Macro fetch failed: {e}; proceeding without macro factors", exc_info=True)
            macro_df = None

        if macro_df is not None and macro_cfg.get("hmm_enabled", True):
            regimes_path = PROCESSED_DIR / "macro_regimes.parquet"
            macro_regimes_df = load_macro_regimes(regimes_path)
            if macro_regimes_df is None or macro_cfg.get("hmm_refit", False):
                log.info("Phase 1f: Fitting HMM regime posteriors (once over full history)...")
                try:
                    macro_regimes_df = build_and_save_macro_regimes(
                        macro_df,
                        regimes_path,
                        n_states=macro_cfg.get("hmm_states", 2),
                        fit_mode=macro_cfg.get("hmm_fit_mode", "walk_forward"),
                        refit_every=macro_cfg.get("hmm_refit_every", 60),
                        min_obs=macro_cfg.get("hmm_min_obs", 252),
                        n_iter=macro_cfg.get("hmm_iter", 200),
                        random_state=macro_cfg.get("hmm_seed", 42),
                    )
                except Exception as e:
                    log.warning(f"HMM regime fit failed: {e}; proceeding without HMM features", exc_info=True)
                    macro_regimes_df = None

    # --- Phase 2: Feature Engineering ---
    log.info("Phase 2: Building features...")
    features, passthrough_cols = build_feature_matrix(
        close, volume,
        market_close=market_close,
        fundamentals=fundamentals,
        grade_events=grade_events,
        macro=macro_df,
        macro_regimes=macro_regimes_df,
        cfg=cfg,
    )
    target = build_target(close)
    log.info(
        f"Feature matrix: {features.shape} "
        f"({len(passthrough_cols)} passthrough macro/regime cols)"
    )

    # Neutralize features (cross-sectional per (date, factor); industry-median per GICS sub-industry).
    # Broadcast macro/regime columns bypass the cross-sectional pipeline via passthrough_cols
    # (their std=0 per date would otherwise zero them out).
    log.info("Phase 2b: Neutralizing features (winsorize → iterative z-score → cap → industry median)...")
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
    log.info(f"Post-neutralization feature matrix: {features.shape}")

    # --- Phase 3: Model Training ---
    log.info("Phase 3: Training ensemble (xgboost + lightgbm + random_forest)...")
    per_model_preds, ensemble_preds, last_models, last_test_X = (
        rolling_train_predict_ensemble(features, target, cfg=cfg)
    )
    log.info(
        f"Predictions: ensemble={len(ensemble_preds):,} obs; "
        f"per-member={ {m: len(s) for m, s in per_model_preds.items()} }"
    )
    # Headline predictions (consumed by backtest, alpha decay, Streamlit) = ensemble.
    predictions = ensemble_preds

    # --- Phase 4: Backtesting ---
    log.info("Phase 4: Running backtest...")
    from src.features.build import resample_to_wednesday

    # Execution lag: signal computed Wed close, trades fill `execution_lag_days`
    # later. Shift daily closes back by the lag before resampling, so the weekly
    # return series measures (Wed+lag) → (next Wed+lag) holding period.
    # Lag < 1 is leakage (you cannot trade at the close that produced the signal);
    # the production backtest refuses it. See .specify/004-constraints.md.
    exec_lag = cfg["backtest"].get("execution_lag_days", 1)
    if exec_lag < 1:
        raise ValueError(
            f"backtest.execution_lag_days must be >= 1 (got {exec_lag}). "
            "Lag = 0 trades at the same close that produced the signal — leakage. "
            "Use alpha_decay_analysis for the lag-0 diagnostic curve only."
        )
    log.info(f"Applying {exec_lag}-day execution lag to backtest returns")
    shifted_close = close.shift(-exec_lag)
    weekly_returns = resample_to_wednesday(shifted_close).pct_change().shift(-1)

    # Cap individual stock weekly returns to suppress tail-event blowups
    # (e.g., 2009-04 post-GFC penny-stock reflation pops). Conservative cap.
    return_cap = cfg["backtest"].get("weekly_return_cap")
    if return_cap is not None:
        log.info(f"Capping individual stock weekly returns at ±{return_cap:.0%}")
        weekly_returns = weekly_returns.clip(lower=-return_cap, upper=return_cap)

    # ADV (6-month rolling avg dollar volume) for position scaling per paper p.10
    adv_daily = (close * volume).rolling(window=126, min_periods=63).mean()
    adv_weekly = resample_to_wednesday(adv_daily)

    # Per-week eligibility: drops penny-stock weeks, illiquid weeks, and (for
    # US) weeks where the ticker wasn't in the SP500 yet / had been removed.
    # Without this the model trades pre-IPO names, post-delisting zombie names,
    # and reflation penny stocks — breaking the long-short construct in tails.
    log.info("Phase 4a: Building per-week eligibility mask...")
    weekly_close = resample_to_wednesday(close)
    pit_us = None
    if "US" in regions:
        pit_us = pd.read_parquet(cache_root / "sp500_membership.parquet")
    eligibility = build_eligibility_mask(
        weekly_close=weekly_close,
        weekly_adv=adv_weekly,
        region_map=region_map,
        min_price_by_region=cfg["universe"].get("min_price_by_region", {}),
        min_adv_by_region=cfg["universe"].get("min_adv_6m_by_region", {}),
        pit_membership_us=pit_us,
    )
    log.info(
        f"Eligibility: {eligibility.sum().sum():,} of {eligibility.size:,} cells active "
        f"({100*eligibility.sum().sum()/eligibility.size:.1f}%)"
    )

    # Mask predictions: stack eligibility to (date, ticker) and apply
    elig_stack = eligibility.stack(future_stack=True)
    elig_stack.index.names = ["date", "ticker"]

    def _mask(s: pd.Series) -> pd.Series:
        return s.where(elig_stack.reindex(s.index, fill_value=False))

    masked_preds = {m: _mask(s) for m, s in per_model_preds.items()}
    masked_preds["ensemble"] = _mask(ensemble_preds)
    predictions = masked_preds["ensemble"]
    log.info(f"Predictions after eligibility mask: {predictions.notna().sum():,} non-NaN")

    # Per-model backtest fan-out — same weekly_returns + ADV + region for fairness.
    log.info("Phase 4a: Running backtest per model...")
    model_comparison: dict[str, dict] = {}
    portfolios: dict[str, pd.DataFrame] = {}
    for member, preds in masked_preds.items():
        portf = build_long_short_portfolio(
            preds, weekly_returns, adv=adv_weekly, cfg=cfg, region_map=region_map
        )
        portfolios[member] = portf
        model_comparison[member] = compute_performance_metrics(portf)
        ir = model_comparison[member].get("information_ratio", float("nan"))
        log.info(f"  {member}: IR={ir:.3f}")
    portfolio = portfolios["ensemble"]  # headline
    metrics = model_comparison["ensemble"]

    # Baseline: sector-relative R1W reversal (paper p.5 baseline strategy a).
    # Earnings-filtered baseline gated on Phase 2 UPDOWN1W availability.
    log.info("Phase 4b: R1W reversal baseline...")
    baseline_preds = r1w_reversal_predictions(close, sector_map=sector_map)
    baseline_portfolio = build_long_short_portfolio(
        baseline_preds, weekly_returns, adv=adv_weekly, cfg=cfg, region_map=region_map
    )
    baseline_metrics = compute_performance_metrics(baseline_portfolio)
    log.info(f"Baseline R1W IR: {baseline_metrics['information_ratio']:.3f}")

    log.info("=== Backtest Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info(f"  {k}: {v:.4f}")
        else:
            log.info(f"  {k}: {v}")

    # --- Phase 5: Save Results ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(PROCESSED_DIR / "portfolio_returns.parquet")
    predictions.to_frame().to_parquet(PROCESSED_DIR / "predictions.parquet")
    baseline_portfolio.to_parquet(PROCESSED_DIR / "baseline_r1w_returns.parquet")

    # Per-model artifacts: predictions in long format + comparison metrics dict.
    per_model_long = pd.concat(
        [s.to_frame("prediction").assign(model=m) for m, s in masked_preds.items()]
    )
    per_model_long.to_parquet(PROCESSED_DIR / "predictions_per_model.parquet")

    def _jsonable(d):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in d.items()}
    (PROCESSED_DIR / "model_comparison.json").write_text(
        json.dumps({m: _jsonable(d) for m, d in model_comparison.items()}, indent=2)
    )

    # SHAP fan-out: run on each member's most-recent fitted model. XGBoost
    # output keeps the legacy filename so Streamlit / pack consumers stay
    # unchanged; the other two members write member-suffixed files.
    log.info("Phase 5a: Computing SHAP values per member...")
    n_weeks = cfg.get("diagnostics", {}).get("shap_test_weeks", 26)
    all_test_dates = predictions.index.get_level_values("date").unique().sort_values()
    recent_dates = all_test_dates[-n_weeks:]
    shap_X = features.loc[
        features.index.get_level_values("date").isin(recent_dates)
    ].dropna()
    log.info(f"SHAP input: {shap_X.shape[0]:,} obs over last {len(recent_dates)} weeks")
    shap_X.to_parquet(PROCESSED_DIR / "shap_features.parquet")
    figdir = PROJECT_ROOT / "notebooks" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # File naming: xgboost stays at canonical paths (back-compat); other
    # members get suffixed paths. Explainer choice dispatches on model type:
    #   tree members (xgb / lgb / rf) → TreeExplainer (exact, fast)
    #   MLP                            → GradientExplainer in standardized space
    #                                    (operate on the scaler-transformed input
    #                                    so SHAP values are in the same units the
    #                                    net actually saw — then we attribute back
    #                                    to the named columns)
    canonical_member = "xgboost"

    def _shap_for_member(member: str, model: object, X: pd.DataFrame):
        """Return (shap_array, X_used) where shap_array.shape == X_used.shape."""
        import shap as _shap

        if member == "mlp":
            import numpy as _np
            import torch as _torch
            # Standardize X with the wrapper's scaler so SHAP runs in the same
            # space the net was trained on. Sample a background set for the
            # gradient explainer baseline; 100 rows is enough and keeps the
            # call sub-second on CPU.
            arr = _np.nan_to_num(X.values)
            Xs = model.scaler.transform(arr).astype(_np.float32)
            bg_n = min(100, Xs.shape[0])
            bg_idx = _np.random.default_rng(0).choice(Xs.shape[0], size=bg_n, replace=False)
            bg = _torch.from_numpy(Xs[bg_idx]).to(model.device)
            expl = _shap.GradientExplainer(model.module, bg)
            t = _torch.from_numpy(Xs).to(model.device)
            sv = expl.shap_values(t)
            # GradientExplainer returns a list (one entry per output) for
            # multi-output nets; our net is scalar so unwrap if so.
            if isinstance(sv, list):
                sv = sv[0]
            sv = _np.asarray(sv).reshape(Xs.shape)
            return sv, X

        explainer = _shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        return vals, X

    for member, model in last_models.items():
        try:
            log.info(f"  SHAP for {member}...")
            vals, X_used = _shap_for_member(member, model, shap_X)
            suffix = "" if member == canonical_member else f"_{member}"
            shap_path = PROCESSED_DIR / f"shap_values{suffix}.parquet"
            imp_path = PROCESSED_DIR / f"feature_importance{suffix}.parquet"
            pd.DataFrame(vals, index=X_used.index, columns=X_used.columns).to_parquet(shap_path)
            importance = (
                pd.Series(np.abs(vals).mean(axis=0), index=X_used.columns)
                .sort_values(ascending=False)
                .to_frame("mean_abs_shap")
            )
            importance.to_parquet(imp_path)
            plot_shap_summary(vals, X_used, save_path=str(figdir / f"shap_summary{suffix}.png"))
            plot_shap_bar(vals, X_used, save_path=str(figdir / f"shap_bar{suffix}.png"))
            log.info(f"    top: {importance.head(5).index.tolist()}")
        except Exception as e:
            log.warning(f"SHAP failed for {member}: {e}", exc_info=True)

    # Top-line metrics summary for the reporting layer (= ensemble's metrics)
    summary = _jsonable(metrics)
    (PROCESSED_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    # --- Phase 6: Diagnostics ---
    diag_cfg = cfg.get("diagnostics", {})

    log.info("Phase 6a: Alpha decay (re-runs backtest at lags 0-4d)...")
    try:
        decay = alpha_decay_analysis(
            close, predictions,
            lags=diag_cfg.get("alpha_decay_lags", [0, 1, 2, 3, 4]),
            adv=adv_weekly,
            region_map=region_map,
            weekly_return_cap=cfg["backtest"].get("weekly_return_cap"),
            cfg=cfg,
        )
        decay.to_parquet(PROCESSED_DIR / "alpha_decay.parquet")
        log.info(f"Alpha decay (IR by lag): {decay['information_ratio'].round(3).to_dict()}")
    except Exception as e:
        log.warning(f"Alpha decay step failed: {e}", exc_info=True)

    log.info("Phase 6b: Weekday effect (5 full retrains, ~5-10 min)...")
    if diag_cfg.get("run_weekday_effect", True):
        try:
            we = weekday_effect_analysis(
                close, volume, market_close, fundamentals, industry_map, cfg=cfg,
            )
            we.to_parquet(PROCESSED_DIR / "weekday_effect.parquet")
            log.info(f"Weekday IR: {we['information_ratio'].round(3).to_dict()}")
        except Exception as e:
            log.warning(f"Weekday effect step failed: {e}", exc_info=True)
    else:
        log.info("Skipping weekday-effect (diagnostics.run_weekday_effect=false)")

    log.info(f"Results saved to {PROCESSED_DIR}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
