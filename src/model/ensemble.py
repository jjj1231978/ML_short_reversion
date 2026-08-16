"""Ensemble combination of per-model cross-sectional predictions.

Four combination methods, selected via ``model.ensemble.combine``:

- ``mean``        — raw average of member predictions.
- ``rank_mean``   — per-date cross-sectional rank then equal-weight mean
                    (scale-invariant; the historical default).
- ``ir_weighted`` — per-date rank, then weight each member by its validation
                    rank-IC for the retrain period covering that week (skill
                    weighting; down-weights weak members instead of dragging
                    toward them).
- ``stack``       — walk-forward meta-learner: ridge regression of the realized
                    target on the members' cross-sectional ranks, fit only on
                    weeks whose target is already realized, applied to the
                    current week. Falls back to ``rank_mean`` until enough
                    history accrues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ranked(per_model: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-date cross-sectional percentile rank of each member, as a
    (date, ticker) × member frame. rank(pct=True) maps to [0, 1] so the members'
    very different raw prediction scales become comparable."""
    ranked = {
        m: s.groupby(level="date").rank(pct=True) for m, s in per_model.items()
    }
    return pd.concat(ranked, axis=1)


def combine_predictions(
    per_model: dict[str, pd.Series],
    method: str = "rank_mean",
    val_ic: pd.DataFrame | None = None,
    target: pd.DataFrame | pd.Series | None = None,
) -> pd.Series:
    """Combine per-member predictions into a single ensemble Series.

    Args:
        per_model: dict from member name to a Series indexed by (date, ticker).
        method: one of "mean", "rank_mean", "ir_weighted", "stack".
        val_ic: (date × member) frame of validation rank-IC per OOS week —
            required for "ir_weighted".
        target: (date, ticker) realized target — required for "stack".

    Returns:
        Series with the union of all members' (date, ticker) indices. For "mean"
        / "rank_mean", cells covered by only some members get those members'
        mean.
    """
    if not per_model:
        raise ValueError("per_model is empty")

    if method == "mean":
        out = pd.concat(per_model, axis=1).mean(axis=1)
        out.name = "prediction"
        return out

    if method == "rank_mean":
        out = _ranked(per_model).mean(axis=1)
        out.name = "prediction"
        return out

    if method == "ir_weighted":
        if val_ic is None:
            raise ValueError("ir_weighted requires val_ic")
        return _combine_ir_weighted(per_model, val_ic)

    if method == "stack":
        if target is None:
            raise ValueError("stack requires target")
        return _combine_stack(per_model, target)

    raise ValueError(f"unknown combine method: {method}")


def _combine_ir_weighted(
    per_model: dict[str, pd.Series], val_ic: pd.DataFrame
) -> pd.Series:
    """Per-date rank, weighted by each member's clipped validation rank-IC.

    Weights = max(val_ic, 0) normalized per date; a date whose members all have
    non-positive IC falls back to equal weights (avoids a zero-weight week)."""
    ranked = _ranked(per_model)
    members = list(ranked.columns)

    w = val_ic.reindex(columns=members).clip(lower=0.0)
    row_sum = w.sum(axis=1)
    w = w.div(row_sum, axis=0)
    # Equal-weight fallback for dates with no positive-IC member (or no val_ic).
    w.loc[row_sum == 0] = 1.0 / len(members)
    w = w.fillna(1.0 / len(members))

    dates = ranked.index.get_level_values("date")
    W = w.reindex(dates).to_numpy()  # (n_rows, n_members), date→rows broadcast
    out = pd.Series(
        np.nansum(ranked.to_numpy() * W, axis=1),
        index=ranked.index,
        name="prediction",
    )
    return out


def _combine_stack(
    per_model: dict[str, pd.Series],
    target: pd.DataFrame | pd.Series,
    min_train_weeks: int = 26,
    alpha: float = 1.0,
) -> pd.Series:
    """Walk-forward ridge stack of member ranks onto the realized target.

    For each week d, fit ridge on all weeks strictly before d (whose forward
    target is realized by d, so there is no look-ahead) using the members'
    cross-sectional ranks as features, then score week d. Until min_train_weeks
    of history exist, fall back to the equal-weight rank mean for that week.
    Features/target are demeaned over the training slice so the ridge needs no
    intercept; constant per-date shifts are irrelevant to the downstream
    cross-sectional ranking anyway.
    """
    ranked = _ranked(per_model)
    members = list(ranked.columns)
    y = target["target"] if isinstance(target, pd.DataFrame) else target
    y = y.reindex(ranked.index)

    dates = ranked.index.get_level_values("date")
    unique_dates = pd.Index(sorted(dates.unique()))
    date_codes = pd.factorize(dates, sort=True)[0]  # aligns to unique_dates order

    X_all = ranked.to_numpy(dtype=float)
    y_all = y.to_numpy(dtype=float)
    eye = np.eye(len(members)) * alpha
    rank_mean = ranked.mean(axis=1).to_numpy()
    out = np.empty(len(ranked), dtype=float)

    for di in range(len(unique_dates)):
        test_rows = date_codes == di
        train_rows = (date_codes < di) & np.isfinite(y_all)
        if di < min_train_weeks or train_rows.sum() < len(members) + 1:
            out[test_rows] = rank_mean[test_rows]
            continue
        Xtr, ytr = X_all[train_rows], y_all[train_rows]
        xm, ym = Xtr.mean(axis=0), ytr.mean()
        Xc, yc = Xtr - xm, ytr - ym
        beta = np.linalg.solve(Xc.T @ Xc + eye, Xc.T @ yc)
        out[test_rows] = (X_all[test_rows] - xm) @ beta

    return pd.Series(out, index=ranked.index, name="prediction")
