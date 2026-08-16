"""Per-pick feature attribution for the live forecaster.

Given the fitted ensemble members and the scored cross-section, produce a short
plain-English commentary for each ticker naming the 1-2 features that moved its
score the most. Uses SHAP TreeExplainer on the tree-based members (xgboost,
lightgbm, random_forest) — the MLP is skipped (no fast exact explainer), so the
commentary reflects the tree majority of the ensemble.

The commentary is intentionally terse so it fits a Streamlit table cell, e.g.:

    "1-week reversal (R1W) high (pushes score down); 14-day RSI high (pushes score down)"

Read as: the stock's neutralized R1W z-score is high, and the model learned that
high recent return drags the forward-return score down (classic reversion).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Members we can explain exactly + cheaply with shap.TreeExplainer.
_TREE_MEMBERS = ("xgboost", "lightgbm", "random_forest")

# Human-readable labels for the common factors. Anything not listed falls back
# to the raw column name (which is already a reasonable mnemonic) via
# `friendly_label`.
FEATURE_LABELS: dict[str, str] = {
    # Price reversal
    "R1W": "1-week reversal (R1W)",
    "IREV1W": "1-week residual reversal",
    "IREV1M": "1-month residual reversal",
    "IREVVOL1W": "vol-scaled 1-week reversal",
    "IREVVOL1M": "vol-scaled 1-month reversal",
    "RSI5D": "5-day RSI",
    "RSI14": "14-day RSI",
    "RSI30": "30-day RSI",
    # Price momentum
    "R1M": "1-month return",
    "R3M1M": "3-month momentum",
    "R6M1M": "6-month momentum",
    "R12M1M": "12-month momentum",
    "IMOM12M1M": "12-month residual momentum",
    "IMOM3M1M": "3-month residual momentum",
    "IMOMVOL12M1M": "vol-scaled 12-month momentum",
    "IMOMVOL3M1M": "vol-scaled 3-month momentum",
    # Size
    "SIZE": "size (market cap)",
    # Low risk
    "VOL6M": "6-month volatility",
    "VOL12M": "12-month volatility",
    "IVOL6M": "6-month idiosyncratic vol",
    "IVOL12M": "12-month idiosyncratic vol",
    "DOWNVOL6M": "6-month downside vol",
    "BETA6M": "6-month beta",
    "BETA12M": "12-month beta",
    "BETA_VIX": "VIX beta",
    "BETA_US10Y": "10Y-yield beta",
    "BETA_OIL": "oil beta",
    "BETA_DXY": "dollar beta",
    # Value
    "PE": "P/E",
    "PB": "P/B",
    "PTB": "price-to-book",
    "PSALES": "P/S",
    "EVEBIT": "EV/EBIT",
    "EVEBITDA": "EV/EBITDA",
    "EVSALES": "EV/sales",
    # Profitability / quality / leverage
    "ROE": "return on equity",
    "ROA": "return on assets",
    "ROIC": "return on invested capital",
    "GPOA": "gross profit / assets",
    "GBROC": "gross profit / capital",
    "GROSSMARGIN": "gross margin",
    "OPRDIC": "operating profit / capital",
    "OPRDA": "operating profit / assets",
    "EBIT_COVER": "interest coverage",
    "ACCRUALS": "accruals",
    "SALES_GROWTH_YOY": "YoY sales growth",
    "DE": "debt / equity",
    "DEBITDA": "debt / EBITDA",
    "NDE": "net debt / equity",
    "NDEBITDA": "net debt / EBITDA",
    "LTDE": "long-term debt / equity",
    "LTDEBITDA": "long-term debt / EBITDA",
    # Analyst
    "UPDOWN1W_RATINGS": "recent analyst rating changes",
}


def friendly_label(name: str) -> str:
    """Map a raw factor column to a readable label, falling back to the raw name.

    Macro/regime passthrough columns keep their raw names (already descriptive,
    e.g. ``OIL_R5``, ``VIX_S1``) so the comment stays honest about what drove it.
    """
    return FEATURE_LABELS.get(name, name)


def _stack_tree_shap(models: dict, members: list[str], X: pd.DataFrame) -> np.ndarray | None:
    """Average per-row SHAP across the tree members, normalized so each member
    contributes comparably. Returns an (n_rows, n_features) array or None if no
    tree member could be explained."""
    # tree_shap_values additionally handles the LightGBM random-ensemble bag
    # (averages TreeExplainer attributions across its boosters). A bare
    # TreeExplainer raises on an LGBBag, and since this loop swallows failures
    # per member, using one here would silently drop lightgbm — the strongest
    # member — from every live forecast's explanation.
    from src.model.train import tree_shap_values  # local — heavy (shap) import

    mats: list[np.ndarray] = []
    for m in members:
        if m not in _TREE_MEMBERS:
            continue
        try:
            sv = np.asarray(tree_shap_values(models[m], X), dtype=float)
            if sv.ndim != 2 or sv.shape != X.shape:
                log.warning("Unexpected SHAP shape for %s: %s; skipping", m, sv.shape)
                continue
            # Equalize each member's overall magnitude before averaging so one
            # member's larger raw scale doesn't dominate the attribution.
            scale = np.abs(sv).mean()
            if scale > 0:
                sv = sv / scale
            mats.append(sv)
        except Exception as e:  # noqa: BLE001 — explanation is best-effort
            log.warning("SHAP failed for member %s: %s", m, e)
            continue

    if not mats:
        return None
    return np.mean(mats, axis=0)


def explain_cross_section(
    models: dict,
    members: list[str],
    X: pd.DataFrame,
    top_k: int = 2,
) -> pd.Series:
    """Return a per-ticker commentary Series for one scored cross-section.

    Args:
        models: fitted member -> model object (same dict scored by predict.py).
        members: ensemble member names in bundle order.
        X: ticker-indexed feature matrix actually fed to the models (neutralized
           z-scores, reindexed to the bundle's feature columns, NaN->0).
        top_k: how many drivers to name per pick (default 2).

    Returns:
        Series indexed like ``X.index`` with a short commentary string (empty if
        no tree member was explainable).
    """
    mean_shap = _stack_tree_shap(models, members, X)
    if mean_shap is None:
        return pd.Series("", index=X.index, name="commentary")

    cols = list(X.columns)
    x_vals = X.values
    out: list[str] = []
    for i in range(mean_shap.shape[0]):
        row = mean_shap[i]
        order = np.argsort(np.abs(row))[::-1][:top_k]
        pieces: list[str] = []
        for j in order:
            contrib = row[j]
            if not np.isfinite(contrib) or contrib == 0:
                continue
            z = x_vals[i, j]
            zword = "high" if z >= 0 else "low"
            effect = "pushes score up" if contrib > 0 else "pushes score down"
            pieces.append(f"{friendly_label(cols[j])} {zword} ({effect})")
        out.append("; ".join(pieces))
    return pd.Series(out, index=X.index, name="commentary")
