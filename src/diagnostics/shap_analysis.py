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
) -> None:
    """Generate SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
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
) -> pd.DataFrame:
    """Test alpha decay by measuring IR at different execution lags.

    Args:
        close: Daily close prices (tickers as columns).
        predictions: Model predictions indexed by (date, ticker).
        lags: List of execution lag days to test (e.g. [0, 1, 2, 3, 4]).

    Returns:
        DataFrame with columns: lag, annualized_return, annualized_vol, IR.
    """
    from src.backtest.portfolio import build_long_short_portfolio, compute_performance_metrics
    from src.features.build import resample_to_wednesday

    results = []
    for lag in lags:
        # Shift close prices by lag days before computing returns
        shifted_close = close.shift(-lag)
        weekly_ret = resample_to_wednesday(shifted_close).pct_change().shift(-1)

        portfolio = build_long_short_portfolio(predictions, weekly_ret)
        metrics = compute_performance_metrics(portfolio)
        metrics["lag"] = lag
        results.append(metrics)

    return pd.DataFrame(results).set_index("lag")


def weekday_effect_analysis(
    close: pd.DataFrame,
    features_builder,
    model_trainer,
    signal_days: list[str],
) -> pd.DataFrame:
    """Run separate backtests keyed off different signal days.

    This is a placeholder — full implementation requires rebuilding features
    and targets for each signal day.
    """
    # TODO: Implement per-weekday signal backtests
    raise NotImplementedError("Weekday effect analysis requires per-day feature rebuilds")
