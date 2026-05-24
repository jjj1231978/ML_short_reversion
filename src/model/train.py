"""Rolling-window training for XGBoost, LightGBM, and RandomForest.

The rolling loop trains all configured ensemble members at each retrain step
on the same (X_train, y_train, X_val, y_val) slice, then predicts the next
`retrain_freq` weeks with each member. Final predictions are combined via
`src.model.ensemble.combine_predictions`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.model.ensemble import combine_predictions

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-member fit / predict
# ---------------------------------------------------------------------------

def _fit_xgb(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    params = dict(member_cfg["params"])
    n_estimators = params.pop("n_estimators", 2000)
    early_stop = member_cfg.get("early_stopping_rounds", 50)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )


def _predict_xgb(model, X) -> np.ndarray:
    return model.predict(xgb.DMatrix(X))


def _fit_lgb(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    import lightgbm as lgb
    params = dict(member_cfg["params"])
    n_estimators = params.pop("n_estimators", 2000)
    early_stop = member_cfg.get("early_stopping_rounds", 50)
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )


def _predict_lgb(model, X) -> np.ndarray:
    return model.predict(X.values, num_iteration=model.best_iteration)


def _fit_rf(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    """Fit RandomForest. Concatenates train+val since RF has no early stopping
    and otherwise the val rows are unused (asymmetric vs xgb/lgb)."""
    from sklearn.ensemble import RandomForestRegressor
    params = dict(member_cfg["params"])
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    model = RandomForestRegressor(**params)
    model.fit(X_all.values, y_all.values)
    return model


def _predict_rf(model, X) -> np.ndarray:
    return model.predict(X.values)


def _fit_mlp(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    """Fit the PyTorch MLP member. The wrapper holds scaler + module + device."""
    from src.model.mlp import fit_from_member_cfg
    return fit_from_member_cfg(member_cfg, X_train, y_train, X_val, y_val)


def _predict_mlp(model, X) -> np.ndarray:
    return model.predict(X)


DISPATCH: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (_fit_xgb, _predict_xgb),
    "lightgbm": (_fit_lgb, _predict_lgb),
    "random_forest": (_fit_rf, _predict_rf),
    "mlp": (_fit_mlp, _predict_mlp),
}


# ---------------------------------------------------------------------------
# Rolling loop
# ---------------------------------------------------------------------------

def _resolve_members(cfg: dict) -> list[str]:
    """Read which members to train from cfg.model. type=ensemble runs all
    listed in ensemble.members; bare type runs that single member."""
    model_cfg = cfg["model"]
    t = model_cfg.get("type", "xgboost")
    if t == "ensemble":
        return list(model_cfg["ensemble"]["members"])
    if t in DISPATCH:
        return [t]
    raise ValueError(f"unknown model.type: {t}")


def rolling_train_predict_ensemble(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[dict[str, pd.Series], pd.Series, dict[str, Any], pd.DataFrame]:
    """Train all ensemble members on rolling windows and combine predictions.

    Returns:
        per_model_predictions: {member: Series of out-of-sample predictions}
        ensemble_predictions: Series — combine_predictions over per_model
        last_models: {member: most-recent fitted model object} (for SHAP)
        last_test_X: DataFrame fed to the most recent prediction call
    """
    if cfg is None:
        cfg = load_config()

    model_cfg = cfg["model"]
    train_weeks = model_cfg["train_weeks"]
    val_weeks = model_cfg["val_weeks"]
    retrain_freq = model_cfg["retrain_freq"]
    members = _resolve_members(cfg)
    combine_method = model_cfg.get("ensemble", {}).get("combine", "rank_mean")

    log.info(f"Training members: {members} (combine={combine_method})")

    dates = features.index.get_level_values("date").unique().sort_values()
    min_start = train_weeks + val_weeks
    if len(dates) <= min_start:
        raise ValueError(
            f"Not enough dates: have {len(dates)}, need at least {min_start + 1}"
        )

    # Per-member running state and prediction buckets
    current_models: dict[str, Any] = {m: None for m in members}
    per_model_preds: dict[str, list[pd.Series]] = {m: [] for m in members}
    last_test_X: pd.DataFrame | None = None

    for i in range(min_start, len(dates)):
        t = dates[i]

        # Retrain all members periodically on the same train/val slice
        if current_models[members[0]] is None or (i - min_start) % retrain_freq == 0:
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

            # Paper convention (page 13): "missing z-scores are replaced with
            # zero" after tickers with >10 missing factors are dropped.
            # neutralize_stacked handles the >10 drop AND fills the rest to 0
            # for non-passthrough columns, so this fillna(0) is a no-op for
            # current configs (no passthroughs). Kept explicit so that any
            # future passthrough column (e.g., re-enabled macro) cannot silently
            # gut the universe via row-level dropna — the only legitimate row
            # filter is on the target (y), enforced below.
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
                f"  retrain @ {t.date()}: train={len(X_train):,} rows "
                f"({train_start.date()}→{train_end.date()}), "
                f"val={len(X_val):,} rows ({val_start.date()}→{val_end.date()})"
            )
            for m in members:
                fit_fn, _ = DISPATCH[m]
                current_models[m] = fit_fn(model_cfg[m], X_train, y_train, X_val, y_val)

            if model_cfg.get("persist_retrains", True):
                # main.py's rolling loop is WED-keyed; per-weekday production
                # bundles are written by src/predict.py via direct save_bundle
                # calls. Diagnostic re-runs (weekday_effect_analysis) currently
                # overwrite the WED bundle — acceptable since those are throwaway.
                from src.model.persistence import save_bundle
                save_bundle(
                    retrain_date=t,
                    models=current_models,
                    members=members,
                    combine_method=combine_method,
                    feature_columns=list(features.columns),
                    train_weeks=train_weeks,
                    val_weeks=val_weeks,
                    retrain_freq=retrain_freq,
                    signal_day="WED",
                )

        # Predict for this week with each member. Match the training-time
        # convention: fill NaN with 0 (paper page 13) instead of dropping rows,
        # so that a single sparse passthrough column or a one-off NaN feature
        # cannot mask out the entire week's universe.
        week_data = features.loc[features.index.get_level_values("date") == t].fillna(0.0)
        if len(week_data) == 0:
            continue
        last_test_X = week_data
        for m in members:
            _, pred_fn = DISPATCH[m]
            preds = pd.Series(
                pred_fn(current_models[m], week_data),
                index=week_data.index,
                name="prediction",
            )
            per_model_preds[m].append(preds)

    per_model_series: dict[str, pd.Series] = {
        m: pd.concat(parts) for m, parts in per_model_preds.items()
    }

    if len(members) > 1:
        ensemble_preds = combine_predictions(per_model_series, method=combine_method)
    else:
        ensemble_preds = per_model_series[members[0]].copy()
        ensemble_preds.name = "prediction"

    return per_model_series, ensemble_preds, current_models, last_test_X


def rolling_train_predict(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict | None = None,
    return_artifacts: bool = False,
):
    """Back-compat shim — returns the XGBoost member's predictions.

    Existing callers (diagnostics weekday_effect_analysis, tests) get the
    XGBoost-only predictions matching the legacy single-model API.
    """
    if cfg is None:
        cfg = load_config()

    # Force a single-member run regardless of cfg.model.type — callers of this
    # function want one model, not an ensemble.
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["type"] = "xgboost"

    per_model, _, last_models, last_test_X = rolling_train_predict_ensemble(
        features, target, cfg=cfg
    )
    predictions = per_model["xgboost"]

    if return_artifacts:
        return predictions, last_models["xgboost"], last_test_X
    return predictions


if __name__ == "__main__":
    print("Model training module — run via src.main or import directly.")
