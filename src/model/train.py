"""Rolling-window model training with XGBoost/LightGBM."""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config


def rolling_train_predict(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.Series:
    """Train models on rolling windows and collect out-of-sample predictions.

    Args:
        features: MultiIndex (date, ticker) DataFrame of neutralized factors.
        target: MultiIndex (date, ticker) DataFrame with 'target' column.
        cfg: Config dict; uses model and training window settings.

    Returns:
        Series of out-of-sample predictions indexed like target.
    """
    if cfg is None:
        cfg = load_config()

    model_cfg = cfg["model"]
    train_weeks = model_cfg["train_weeks"]
    val_weeks = model_cfg["val_weeks"]
    retrain_freq = model_cfg["retrain_freq"]
    params = model_cfg["params"]
    early_stop = model_cfg["early_stopping_rounds"]

    # Get unique weekly dates
    dates = features.index.get_level_values("date").unique().sort_values()

    min_start = train_weeks + val_weeks
    if len(dates) <= min_start:
        raise ValueError(
            f"Not enough dates: have {len(dates)}, need at least {min_start + 1}"
        )

    all_predictions = []
    current_model = None

    for i in range(min_start, len(dates)):
        t = dates[i]

        # Retrain model periodically
        if current_model is None or (i - min_start) % retrain_freq == 0:
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

            X_train = features.loc[train_mask].dropna()
            y_train = target.loc[X_train.index, "target"]

            X_val = features.loc[val_mask].dropna()
            y_val = target.loc[X_val.index, "target"]

            # Drop any remaining NaN targets
            valid_train = y_train.notna()
            valid_val = y_val.notna()

            dtrain = xgb.DMatrix(X_train[valid_train], label=y_train[valid_train])
            dval = xgb.DMatrix(X_val[valid_val], label=y_val[valid_val])

            current_model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get("n_estimators", 2000),
                evals=[(dval, "val")],
                early_stopping_rounds=early_stop,
                verbose_eval=False,
            )

        # Predict for this week
        week_data = features.loc[features.index.get_level_values("date") == t].dropna()
        if len(week_data) > 0:
            dtest = xgb.DMatrix(week_data)
            preds = pd.Series(
                current_model.predict(dtest),
                index=week_data.index,
                name="prediction",
            )
            all_predictions.append(preds)

    return pd.concat(all_predictions)


if __name__ == "__main__":
    print("Model training module — run via src.main or import directly.")
