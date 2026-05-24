"""Ensemble combination of per-model cross-sectional predictions."""

from __future__ import annotations

import pandas as pd


def combine_predictions(
    per_model: dict[str, pd.Series],
    method: str = "rank_mean",
) -> pd.Series:
    """Combine per-member predictions into a single ensemble Series.

    Args:
        per_model: dict from member name to a Series indexed by (date, ticker).
        method: "rank_mean" (per-date cross-sectional rank then mean — scale
            invariant) or "mean" (raw average of predictions).

    Returns:
        Series with the union of all members' (date, ticker) indices. Cells
        covered by only some members get the mean of those members.
    """
    if not per_model:
        raise ValueError("per_model is empty")

    if method == "rank_mean":
        # rank(pct=True) yields [0, 1] ranks per date — scale invariant across
        # the three boosters, which output very different prediction magnitudes.
        ranked = {
            m: s.groupby(level="date").rank(pct=True)
            for m, s in per_model.items()
        }
        df = pd.concat(ranked, axis=1)
    elif method == "mean":
        df = pd.concat(per_model, axis=1)
    else:
        raise ValueError(f"unknown combine method: {method}")

    out = df.mean(axis=1)
    out.name = "prediction"
    return out
