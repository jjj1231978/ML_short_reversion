"""SHAP-based model interpretability and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def compute_shap_values(
    model: xgb.Booster, X: pd.DataFrame
) -> np.ndarray:
    """Compute SHAP values for a trained XGBoost model."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str | None = None,
    max_display: int = 10,
) -> None:
    """Paper-p.7-style SHAP beeswarm: horizontal lanes, blue-low/red-high gradient.

    `max_display=10` matches the paper's top-10 feature list. `shap.summary_plot`
    handles marker alpha and density automatically.
    """
    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
        plot_size=None,  # use the figure size we set above
    )
    plt.title("SHAP values (impact on model output) — top features", fontsize=12, pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Mean absolute SHAP value bar chart (feature importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot.barh(ax=ax)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def alpha_decay_analysis(
    close: pd.DataFrame,
    predictions: pd.Series,
    lags: list[int],
    adv: pd.DataFrame | None = None,
    region_map: dict[str, str] | None = None,
    weekly_return_cap: float | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Test alpha decay by measuring IR at different execution lags.

    Mirrors the main backtest's portfolio construction so the lag-vs-IR
    sweep is comparable: same ADV-scaled weighting, same per-region setup,
    same defensive return cap. Without these the diagnostic produces
    wildly different numbers than the main run (penny-stock blowups, etc.).

    Args:
        close: Daily close prices (tickers as columns).
        predictions: Model predictions indexed by (date, ticker). Should
            already be eligibility-masked when called from main.py.
        lags: List of execution lag days to test (e.g. [0, 1, 2, 3, 4]).
        adv: Optional weekly ADV (date × ticker) for per-stock weight scaling.
        region_map: Optional {ticker: region} for per-region scaling thresholds.
        weekly_return_cap: Optional ±cap applied to per-stock weekly returns.
        cfg: Optional config dict forwarded to build_long_short_portfolio.

    Returns:
        DataFrame indexed by lag with metrics from compute_performance_metrics.
    """
    from src.backtest.portfolio import build_long_short_portfolio, compute_performance_metrics
    from src.features.build import resample_to_wednesday

    results = []
    for lag in lags:
        # Shift close prices by lag days before computing returns
        shifted_close = close.shift(-lag)
        weekly_ret = resample_to_wednesday(shifted_close).pct_change().shift(-1)
        if weekly_return_cap is not None:
            weekly_ret = weekly_ret.clip(lower=-weekly_return_cap, upper=weekly_return_cap)

        portfolio = build_long_short_portfolio(
            predictions, weekly_ret, adv=adv, cfg=cfg, region_map=region_map
        )
        metrics = compute_performance_metrics(portfolio)
        metrics["lag"] = lag
        results.append(metrics)

    return pd.DataFrame(results).set_index("lag")


def weekday_effect_analysis(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    market_close: pd.Series | None,
    fundamentals: dict[str, pd.DataFrame] | None,
    industry_map: dict[str, str],
    cfg: dict,
    signal_days: list[str] | None = None,
) -> pd.DataFrame:
    """Per-weekday signal backtests (paper p.8: Thu best, Mon worst).

    For each weekday in `signal_days`, rebuild the features and target keyed
    off that weekday, retrain the model, run the backtest, and collect metrics.

    Returns:
        DataFrame indexed by signal_day name with one column per metric from
        compute_performance_metrics.
    """
    from src.backtest.portfolio import (
        build_long_short_portfolio,
        compute_performance_metrics,
    )
    from src.features.build import (
        build_feature_matrix,
        build_target,
        resample_to_weekday,
    )
    from src.features.neutralize import neutralize_stacked
    from src.model.train import rolling_train_predict

    if signal_days is None:
        signal_days = cfg.get("diagnostics", {}).get(
            "weekday_signals",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        )

    weekday_codes = {
        "Monday": "MON", "Tuesday": "TUE", "Wednesday": "WED",
        "Thursday": "THU", "Friday": "FRI",
    }

    n_cfg = cfg["neutralization"]
    # Production diagnostic — same tradeable-only constraint as main.py.
    # Lag < 1 is leakage; alpha_decay_analysis is the only place lag-0 is allowed
    # (and only as a labeled upper-bound point on the decay curve).
    exec_lag = cfg["backtest"].get("execution_lag_days", 1)
    if exec_lag < 1:
        raise ValueError(
            f"backtest.execution_lag_days must be >= 1 (got {exec_lag}). "
            "weekday_effect_analysis reports tradeable per-weekday performance; "
            "use alpha_decay_analysis for the lag-0 diagnostic only."
        )
    adv_daily = (close * volume).rolling(126, min_periods=63).mean()

    rows = []
    for day_name in signal_days:
        code = weekday_codes[day_name]
        features, passthrough_cols = build_feature_matrix(
            close, volume, market_close=market_close,
            fundamentals=fundamentals, cfg=cfg, signal_day=code,
        )
        target = build_target(close, signal_day=code)
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
        predictions = rolling_train_predict(features, target, cfg=cfg)

        shifted_close = close.shift(-exec_lag)
        weekly_returns = resample_to_weekday(shifted_close, code).pct_change().shift(-1)
        adv_weekly = resample_to_weekday(adv_daily, code)

        portfolio = build_long_short_portfolio(
            predictions, weekly_returns, adv=adv_weekly, cfg=cfg
        )
        metrics = compute_performance_metrics(portfolio)
        metrics["signal_day"] = day_name
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("signal_day")
