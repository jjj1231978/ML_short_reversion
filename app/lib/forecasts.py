"""Shared forecast-file discovery for the viewer pages.

Both the Live forecast and Deep dive pages need the same sidebar selectors —
which signal day, which target date — so the discovery logic lives here rather
than being duplicated per page.
"""

from __future__ import annotations

from pathlib import Path

from src.config import PROCESSED_DIR

FORECASTS_ROOT = PROCESSED_DIR / "forecasts"

SIGNAL_DAY_LABELS = {
    "WED": "Wednesday",
    "THU": "Thursday",
    "FRI": "Friday",
}


def list_forecasts(signal_day: str) -> list[Path]:
    """Parquet files for a given signal day, newest target date first."""
    d = FORECASTS_ROOT / signal_day.upper()
    if not d.exists():
        return []
    return sorted(d.glob("*.parquet"), reverse=True)


def default_day_index() -> int:
    """Index of the signal day holding the most recent forecast.

    A reload lands on the day most recently run rather than snapping back to
    Wednesday every time.
    """
    codes = list(SIGNAL_DAY_LABELS.keys())
    latest = {code: fs[0].stem for code in codes if (fs := list_forecasts(code))}
    if not latest:
        return 0
    return codes.index(max(latest, key=latest.get))


def code_for_label(label: str) -> str:
    return {v: k for k, v in SIGNAL_DAY_LABELS.items()}[label]
