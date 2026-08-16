"""Derive actual trading sessions from an observed price panel.

Domain-agnostic. Takes a wide DataFrame (dates x series) plus an optional
mapping from column -> group, and reports which dates each group actually
traded on.

Why derive the calendar from observed bars instead of a holiday library:

- No new dependency, and nothing to keep patched as exchanges publish schedule
  changes. The bars ARE the ground truth for whatever venues the data covers.
- It generalises to every region in the panel at once. A panel spanning several
  countries has no single calendar; per-group derivation gives each its own.
- It captures one-off closures (weather, national mourning, outages) that
  published calendars routinely miss.

The cost is that it cannot see into the future: a session is only known once
its bars exist. Callers resolving a *forward* date must handle `None`.
"""

import pandas as pd

__all__ = ["sessions", "sessions_by_group", "resolve_session"]


def sessions(
    panel: pd.DataFrame,
    min_frac: float = 0.2,
    reference_window: int = 21,
) -> pd.DatetimeIndex:
    """Dates on which `panel` actually traded.

    A date counts as a session when the number of columns reporting a value is
    at least `min_frac` of the local norm. The norm is a rolling median rather
    than a global one because coverage drifts: early in a long history most
    series are not yet listed, so a fixed fraction of the column count would
    misread genuine early sessions as closures.

    On a real closure essentially nothing prints, so the observed counts are
    ~0 against a norm of hundreds — the threshold is not delicate.

    Args:
        panel: Wide frame, DatetimeIndex x series.
        min_frac: Fraction of the local norm required to count as a session.
        reference_window: Observations in the rolling median defining the norm.
    """
    if panel is None or panel.empty:
        return pd.DatetimeIndex([])

    counts = panel.notna().sum(axis=1)
    norm = counts.rolling(reference_window, center=True, min_periods=1).median()
    open_ = counts >= (norm * min_frac)
    # A zero-coverage date is a closure regardless of what the norm suggests
    # (guards the degenerate case of a long all-empty stretch dragging the
    # median to zero, which would make `counts >= 0` true everywhere).
    open_ &= counts > 0
    return pd.DatetimeIndex(counts.index[open_])


def sessions_by_group(
    panel: pd.DataFrame,
    group_map: dict,
    min_frac: float = 0.2,
    reference_window: int = 21,
) -> dict:
    """Per-group session calendars. See `sessions` for the detection rule.

    Args:
        panel: Wide frame, DatetimeIndex x series.
        group_map: column -> group label. Columns absent from the map are
            ignored; groups with no columns present in the panel are omitted.
    """
    out: dict = {}
    if panel is None or panel.empty:
        return out

    by_group: dict = {}
    for col in panel.columns:
        g = group_map.get(col)
        if g is not None:
            by_group.setdefault(g, []).append(col)

    for g, cols in by_group.items():
        out[g] = sessions(
            panel[cols], min_frac=min_frac, reference_window=reference_window
        )
    return out


def resolve_session(
    nominal: pd.Timestamp,
    days: pd.DatetimeIndex,
    max_back: int = 3,
    max_forward: int = 3,
) -> pd.Timestamp | None:
    """Snap `nominal` to a real session, preferring the day itself.

    Search order is the nominal date, then backwards day by day, then forwards.
    Backwards wins ties because the prior session is the last one whose close is
    already known when the nominal date arrives; falling forward commits to a
    price that does not exist yet.

    The windows are deliberately tight. A weekly cadence means anything more
    than a few days out is a different rebalance, so a run of closures wider
    than the window should surface as `None` and be handled explicitly rather
    than silently resolving into an adjacent week.

    Returns:
        The resolved session, or None if no session falls in the window.
    """
    if days is None or len(days) == 0:
        return None
    nominal = pd.Timestamp(nominal).normalize()
    available = set(pd.DatetimeIndex(days).normalize())

    if nominal in available:
        return nominal
    for step in range(1, max_back + 1):
        cand = nominal - pd.Timedelta(step, "D")
        if cand in available:
            return cand
    for step in range(1, max_forward + 1):
        cand = nominal + pd.Timedelta(step, "D")
        if cand in available:
            return cand
    return None
