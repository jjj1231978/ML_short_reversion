"""Orchestrate feature matrix construction from raw data."""

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, load_config
from src.features import factors


def resample_to_weekday(df: pd.DataFrame, weekday: str = "WED") -> pd.DataFrame:
    """Resample daily data to weekly closes anchored on a specific weekday.

    Args:
        df: Daily-indexed DataFrame.
        weekday: Three-letter pandas weekday code: MON, TUE, WED, THU, FRI.
    """
    return df.resample(f"W-{weekday}").last()


def resample_to_wednesday(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compat alias for `resample_to_weekday(df, 'WED')`."""
    return resample_to_weekday(df, "WED")


def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    market_close: pd.Series | None = None,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    grade_events: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    macro_regimes: pd.DataFrame | None = None,
    cfg: dict | None = None,
    signal_day: str = "WED",
) -> tuple[pd.DataFrame, list[str]]:
    """Build weekly feature matrix from daily price, fundamental, and macro data.

    Args:
        close: Daily close prices, columns = tickers.
        volume: Daily volume, columns = tickers.
        market_close: Market index close (e.g. SPY) for beta calculation.
        fundamentals: Dict from fetch_fundamentals_* with keys
            'PE', 'PB', 'ROE', 'GPOA' — each a daily DataFrame (dates × tickers).
        grade_events: Long-format analyst grade events for UPDOWN1W_RATINGS.
        macro: Wide DataFrame from `src.data.macro.fetch_macro_data` — daily
            macro levels (oil, yields, FX, vol). If provided, builds three
            macro factor families per `cfg.macro`:
              (a) per-stock rolling betas vs selected macros (cross-sectional)
              (b) threshold regime flags from rolling z-scores (passthrough)
              (c) raw broadcast macro returns/changes (passthrough)
        macro_regimes: Wide DataFrame from `src.features.regime.load_macro_regimes`
            — HMM posteriors per macro. All columns are passthrough.
        cfg: Config dict.

    Returns:
        (features, passthrough_cols)
            features: MultiIndex (date, ticker) → feature columns.
            passthrough_cols: feature column names that are constant per date
                across tickers (must be excluded from cross-sectional z-score).
    """
    if cfg is None:
        cfg = load_config()

    # Defensive index normalization: strip timezone from all inputs so concat
    # below doesn't trip on a tz-naive vs tz-aware mix (Databento returns
    # tz-aware UTC; SimFin returns tz-naive).
    def _strip_tz(idx: pd.Index) -> pd.Index:
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    close.index = _strip_tz(close.index)
    volume.index = _strip_tz(volume.index)
    if market_close is not None:
        market_close.index = _strip_tz(market_close.index)
    if fundamentals:
        fundamentals = {k: v.copy() for k, v in fundamentals.items()}
        for v in fundamentals.values():
            v.index = _strip_tz(v.index)

    def _w(df: pd.DataFrame) -> pd.DataFrame:
        return resample_to_weekday(df, signal_day)

    feature_frames = {
        # Price reversals
        "R1W": _w(factors.r1w(close)),
        "RSI5D": _w(factors.rsi(close, window=5)),
        "RSI14": _w(factors.rsi(close, window=14)),
        "RSI30": _w(factors.rsi(close, window=30)),
        # Price momentum
        "R1M": _w(factors.momentum(close, lookback=21, skip=0)),
        "R3M1M": _w(factors.momentum(close, lookback=63, skip=21)),
        "R6M1M": _w(factors.momentum(close, lookback=126, skip=21)),
        "R12M1M": _w(factors.momentum(close, lookback=252, skip=21)),
        # Low risk
        "VOL6M": _w(factors.rolling_volatility(close, window=126)),
        "VOL12M": _w(factors.rolling_volatility(close, window=252)),
        "DOWNVOL6M": _w(factors.downside_volatility(close, window=126)),
    }

    # IREV1W/IREV1M, BETA6M, and IVOL6M/12M all need market returns. Without
    # market_close we fall back to plain R1W/R1M for IREV and skip BETA/IVOL
    # entirely. Real runs always pass market_close (see main.py).
    if market_close is not None:
        beta_daily = factors.rolling_beta(close, market_close, window=126)
        beta_12m = factors.rolling_beta(close, market_close, window=252)
        feature_frames["BETA6M"] = _w(beta_daily)
        feature_frames["BETA12M"] = _w(beta_12m)
        feature_frames["IREV1W"] = _w(factors.residual_return(close, market_close, beta_daily, period=5))
        feature_frames["IREV1M"] = _w(factors.residual_return(close, market_close, beta_daily, period=21))
        feature_frames["IVOL6M"] = _w(factors.idiosyncratic_volatility(close, market_close, window=126))
        feature_frames["IVOL12M"] = _w(factors.idiosyncratic_volatility(close, market_close, window=252))

        # Tier-1 paper extensions (page 13–14):
        # IMOM* = residual return at longer horizon lagged 1 month (21 trading days).
        # The residualization beta is the 6-month rolling beta — same as IREV* —
        # for consistency across horizons.
        imom_12m = factors.residual_return(close, market_close, beta_daily, period=252).shift(21)
        imom_3m = factors.residual_return(close, market_close, beta_daily, period=63).shift(21)
        feature_frames["IMOM12M1M"] = _w(imom_12m)
        feature_frames["IMOM3M1M"] = _w(imom_3m)

        # Risk-adjusted variants: residual return divided by residual volatility.
        # Numerator/denominator horizons follow paper convention (numerator window
        # for momentum factors; IVOL6M for short-horizon reversal denominators
        # since residual std over a 5d/21d window is too noisy to be stable).
        ivol_12m_d = factors.idiosyncratic_volatility(close, market_close, window=252)
        ivol_6m_d = factors.idiosyncratic_volatility(close, market_close, window=126)
        feature_frames["IMOMVOL12M1M"] = _w(imom_12m / ivol_12m_d.replace(0, np.nan))
        feature_frames["IMOMVOL3M1M"] = _w(imom_3m / ivol_6m_d.replace(0, np.nan))
        feature_frames["IREVVOL1W"] = _w(
            factors.residual_return(close, market_close, beta_daily, period=5)
            / ivol_6m_d.replace(0, np.nan)
        )
        feature_frames["IREVVOL1M"] = _w(
            factors.residual_return(close, market_close, beta_daily, period=21)
            / ivol_6m_d.replace(0, np.nan)
        )
    else:
        feature_frames["IREV1W"] = _w(factors.r1w(close))
        feature_frames["IREV1M"] = _w(factors.momentum(close, lookback=21, skip=0))

    # Fundamental factors (daily-aligned from FMP/SimFin, resample to signal_day).
    # Reindex to the price date range first — the daily index from FMP/SimFin
    # may extend further back than our backtest window. Without this reindex,
    # the concat below would extend the feature matrix and the target lookup
    # in training would fail on those phantom dates.
    if fundamentals:
        # Phase 1 set (always present from both data sources) + Phase 2 extras
        # (only present from the FMP path's `_compute_fundamental_ratios`).
        # Missing keys are silently skipped — the build remains valid on
        # either fundamentals backend.
        # Factors dropped due to sparse FMP coverage on the 1214-ticker
        # multi-region universe (measured post-2018 OOS; 70-78% coverage
        # forces tickers over the max_missing_factors=10 threshold and
        # collapses the per-week universe to median ~24):
        #   POCF, PFCF       (72%) — TTM OCF/FCF over MktCap
        #   CAPEXPPE          (76%)
        #   FCFROIC, EVOCF, EVFCF (78%)
        # All 6 depend on cashflow-statement coverage that's noticeably worse
        # than income/balance for FMP. Kept the 79-81% factors (PE/PB/PSALES/
        # PTB/SIZE) because they are core paper Value/Low-Risk signals.
        for name in [
            # Phase 1
            "PE", "PB", "ROE", "GPOA",
            # Phase 2 (existing FMP extras)
            "DE", "NDE", "EBIT_COVER", "ACCRUALS", "SALES_GROWTH_YOY",
            # Tier-2 — Value (cashflow-derived POCF/PFCF/EVOCF/EVFCF dropped)
            "PSALES", "EVSALES", "EVEBIT", "EVEBITDA", "PTB",
            # Tier-2 — Profitability (FCFROIC dropped)
            "ROA", "GROSSMARGIN", "ROIC", "OPRDA", "OPRDIC", "GBROC",
            # Tier-2 — Leverage (additional)
            "DEBITDA", "LTDE", "LTDEBITDA", "NDEBITDA",
            # Tier-2 — Capital Allocation (CAPEXPPE dropped)
            "STKP",
            # Tier-1 (price-derived, computed inside FMP fundamentals path b/c
            # it consumes shares-outstanding × price)
            "SIZE",
        ]:
            if name in fundamentals and not fundamentals[name].empty:
                aligned = fundamentals[name].reindex(close.index)
                feature_frames[name] = _w(aligned)

    # Earnings/analyst-revision factor (UPDOWN1W proxy via FMP analyst grades).
    # Per paper p.7 this should be the #2 SHAP feature when using EPS revisions;
    # we use rating revisions because FMP Starter doesn't expose historical
    # EPS estimate revisions. Distinct name `UPDOWN1W_RATINGS` makes this honest.
    if grade_events is not None and not grade_events.empty:
        updown_daily = factors.updown_ratings_1w(
            grade_events, close.index, list(close.columns)
        )
        feature_frames["UPDOWN1W_RATINGS"] = _w(updown_daily)

    # --- Macro factors ---------------------------------------------------
    # Three families per cfg.macro:
    #   (a) per-stock rolling betas vs selected macros — cross-sectional
    #   (b) threshold regime flags — passthrough (broadcast)
    #   (c) raw macro changes/returns — passthrough (broadcast)
    # Plus HMM regime posteriors loaded from `macro_regimes` — passthrough.
    passthrough_cols: list[str] = []
    macro_cfg = (cfg or {}).get("macro") or {}
    tickers = list(close.columns)

    if macro is not None and not macro.empty:
        macro = macro.copy()
        macro.index = _strip_tz(macro.index)

        # Reindex macro to the equity daily calendar before transforms so the
        # broadcast aligns 1-to-1 with the rest of `feature_frames`.
        macro_daily = macro.reindex(close.index).ffill()

        # (a) Per-stock betas — only for macros listed under `macro.beta_macros`.
        # Default to ['OIL','DXY','US10Y','VIX'] when configured, else skip betas.
        beta_macros = macro_cfg.get("beta_macros", [])
        beta_window = int(macro_cfg.get("beta_window", 126))
        for name in beta_macros:
            if name not in macro_daily.columns:
                continue
            b = factors.stock_macro_beta(close, macro_daily[name], window=beta_window)
            feature_frames[f"BETA_{name}"] = _w(b)

        # (b) Threshold regime flags — broadcast.
        flag_macros = macro_cfg.get("flag_macros", [])
        flag_window = int(macro_cfg.get("flag_window", 126))
        flag_threshold = float(macro_cfg.get("flag_threshold", 1.0))
        if flag_macros:
            flags = factors.macro_regime_flags(
                macro_daily[[m for m in flag_macros if m in macro_daily.columns]],
                window=flag_window,
                hi_threshold=flag_threshold,
            )
            for col in flags.columns:
                broad = factors._broadcast_to_tickers(flags[col], tickers)
                feature_frames[col] = _w(broad)
                passthrough_cols.append(col)

        # (c) Raw macro returns/changes — broadcast.
        # config.macro.raw: {<NAME>: "returns" | "changes", periods: int}
        raw_spec = macro_cfg.get("raw", {})
        raw_periods = int(macro_cfg.get("raw_periods", 5))  # default = 1-week
        for name, kind in raw_spec.items():
            if name not in macro_daily.columns:
                continue
            if kind == "returns":
                ts = macro_daily[name].pct_change(raw_periods)
            elif kind == "changes":
                ts = macro_daily[name].diff(raw_periods)
            else:
                continue
            broad = factors._broadcast_to_tickers(ts, tickers)
            colname = f"{name}_{'R' if kind == 'returns' else 'D'}{raw_periods}"
            feature_frames[colname] = _w(broad)
            passthrough_cols.append(colname)

    # HMM regime posteriors — all columns broadcast (passthrough).
    if macro_regimes is not None and not macro_regimes.empty:
        regimes = macro_regimes.copy()
        regimes.index = _strip_tz(regimes.index)
        regimes_daily = regimes.reindex(close.index).ffill()
        for col in regimes_daily.columns:
            broad = factors._broadcast_to_tickers(regimes_daily[col], tickers)
            feature_frames[col] = _w(broad)
            passthrough_cols.append(col)

    # Optional feature whitelist: when `features.whitelist` is set in cfg, drop
    # all factor frames not in the list. Used by configs/may9.yaml to replay
    # the May-9 13-factor baseline on the current (bug-fixed) pipeline so we
    # can attribute the regression cleanly between bug-driven and factor-set-
    # driven effects. Whitelisted names that aren't computed are silently
    # skipped (e.g., asking for IREV1W with market_close=None just won't appear).
    features_cfg = (cfg or {}).get("features") or {}
    whitelist = features_cfg.get("whitelist")
    if whitelist:
        whitelist_set = set(whitelist)
        kept = {k: v for k, v in feature_frames.items() if k in whitelist_set}
        dropped = [k for k in feature_frames if k not in whitelist_set]
        feature_frames = kept
        passthrough_cols = [c for c in passthrough_cols if c in whitelist_set]
        if dropped:
            import logging
            logging.getLogger(__name__).info(
                f"features.whitelist: kept {sorted(kept.keys())}; "
                f"dropped {len(dropped)} non-whitelisted factors"
            )

    # Stack into (date, ticker) -> features
    combined = pd.concat(feature_frames, axis=1)
    # combined has columns = MultiIndex (feature_name, ticker)
    # Reshape to (date, ticker) rows x feature columns
    stacked = combined.stack(level=-1, future_stack=True)
    stacked.index.names = ["date", "ticker"]

    return stacked, passthrough_cols


def build_target(close: pd.DataFrame, signal_day: str = "WED") -> pd.DataFrame:
    """Forward 1-week return (signal_day → next signal_day), cross-sectionally z-scored."""
    weekly_close = resample_to_weekday(close, signal_day)
    fwd_ret = weekly_close.pct_change().shift(-1)  # forward return

    # Cross-sectional z-score
    mean = fwd_ret.mean(axis=1)
    std = fwd_ret.std(axis=1).replace(0, float("nan"))
    zscored = fwd_ret.sub(mean, axis=0).div(std, axis=0)

    # Stack to (date, ticker)
    stacked = zscored.stack()
    stacked.name = "target"
    stacked.index.names = ["date", "ticker"]
    return stacked.to_frame()


if __name__ == "__main__":
    cfg = load_config()
    print("Feature build module — run via src.main or import directly.")
