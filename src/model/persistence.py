"""Model bundle persistence for live-inference reuse.

Every retrain step in `rolling_train_predict_ensemble` can dump a bundle
containing the fitted member models plus the metadata needed to score a fresh
cross-section: feature column order, ensemble combine method, train/val window
parameters used. `src/predict.py` reads these to skip retraining when the
cadence threshold has not yet elapsed.

Bundles live at `data/processed/models/{retrain_date}.joblib`. A single
`state.json` next to them records the most recent retrain date and the
configured cadence so the inference job can decide whether to refit.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import PROCESSED_DIR

log = logging.getLogger(__name__)

MODELS_ROOT = PROCESSED_DIR / "models"


def _signal_dir(signal_day: str) -> Path:
    """Per-signal-day subdir. WED features and THU features yield different
    model bundles — must not be co-mingled."""
    return MODELS_ROOT / signal_day.upper()


def _state_path(signal_day: str) -> Path:
    return _signal_dir(signal_day) / "state.json"


def _ensure_dir(signal_day: str) -> None:
    _signal_dir(signal_day).mkdir(parents=True, exist_ok=True)


def _format_date(d: pd.Timestamp | date | str) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d")


def save_bundle(
    retrain_date: pd.Timestamp,
    models: dict[str, Any],
    members: list[str],
    combine_method: str,
    feature_columns: list[str],
    train_weeks: int,
    val_weeks: int,
    retrain_freq: int,
    signal_day: str = "WED",
    extra: dict | None = None,
) -> dict:
    """Persist a retrain artifact and update state.json.

    `feature_columns` is the exact column order the models were fit on — the
    inference path must reindex to this before predicting so column drift
    between training and live data cannot silently corrupt predictions.

    Per-`signal_day` namespacing keeps Wed/Thu/Fri bundles separate (their
    feature distributions differ; one model cannot score another's features).

    Returns the persisted payload (including the generated ``saved_at`` and the
    on-disk ``path``) so callers can record provenance without re-reading the
    joblib — e.g. predict.py logs ``saved_at`` into the model_versions table.
    """
    sd = signal_day.upper()
    _ensure_dir(sd)
    date_str = _format_date(retrain_date)
    path = _signal_dir(sd) / f"{date_str}.joblib"
    payload = {
        "retrain_date": date_str,
        "signal_day": sd,
        "members": list(members),
        "combine_method": combine_method,
        "feature_columns": list(feature_columns),
        "train_weeks": int(train_weeks),
        "val_weeks": int(val_weeks),
        "retrain_freq": int(retrain_freq),
        "models": dict(models),
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if extra:
        payload["extra"] = extra
    joblib.dump(payload, path)
    _write_state(date_str, retrain_freq, sd)
    log.info(f"Saved model bundle: {path}")
    # `path` added after dump so it lives only on the returned in-memory dict,
    # never persisted into the joblib (where it could go stale if moved).
    payload["path"] = path
    return payload


def _write_state(latest_date: str, retrain_freq: int, signal_day: str) -> None:
    state = {
        "latest_retrain_date": latest_date,
        "retrain_freq_weeks": int(retrain_freq),
        "signal_day": signal_day,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _state_path(signal_day).write_text(json.dumps(state, indent=2))


def read_state(signal_day: str = "WED") -> dict | None:
    p = _state_path(signal_day.upper())
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_bundles(signal_day: str = "WED") -> list[Path]:
    sd = _signal_dir(signal_day.upper())
    if not sd.exists():
        return []
    return sorted(sd.glob("*.joblib"))


def load_latest_bundle(signal_day: str = "WED") -> dict | None:
    """Return the most-recent bundle's payload, or None if no bundles exist."""
    bundles = list_bundles(signal_day)
    if not bundles:
        return None
    latest = bundles[-1]
    log.info(f"Loading model bundle: {latest}")
    return joblib.load(latest)


def load_matching_bundle(
    retrain_date: pd.Timestamp | date | str,
    signal_day: str,
    *,
    feature_columns: list[str],
    members: list[str],
    train_weeks: int,
    val_weeks: int,
    retrain_freq: int,
    fingerprint: str | None = None,
    allow_structural_restamp: bool = False,
) -> dict | None:
    """Return the saved bundle for ``retrain_date`` IFF it was trained under a
    matching configuration — else None (caller refits).

    This is the per-retrain checkpoint that makes a killed backtest resumable:
    on restart, every retrain step whose bundle already exists on disk and
    matches the current run is reloaded instead of refit (the dominant cost),
    while the cheap weekly scoring is replayed.

    Matching is deliberately strict so a stale bundle from a *different* config
    (e.g. a different universe / feature set / window left in the same dir) is
    never silently reused:

    - If both the on-disk bundle and the caller carry a ``fingerprint`` (newer
      bundles store one under ``extra``), they must be equal. The fingerprint
      also pins the data extent (date index), so a price/data refresh that adds
      weeks invalidates the reuse and forces a clean refit.
    - Legacy bundles saved before fingerprints existed fall back to matching on
      feature column order, member set, and the train/val/retrain windows. That
      distinguishes configs but cannot detect a same-config *data* change, so a
      one-line warning is logged; reuse here assumes the data is unchanged
      (true for a cache-only resume of an interrupted run).
    - ``allow_structural_restamp`` (manual resume escape hatch): when set, a
      fingerprint *mismatch* falls through to the same structural match instead
      of forcing a refit. Use ONLY when the data is known-unchanged out-of-band
      and the fingerprint delta is a benign config change (the caller re-saves
      the bundle under the new fingerprint). Off by default — normal runs keep
      the strict guard.
    """
    date_str = _format_date(retrain_date)
    path = _signal_dir(signal_day.upper()) / f"{date_str}.joblib"
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as e:  # corrupt/partial bundle from a hard kill → refit
        log.warning(f"Could not load bundle {path} ({e}); will refit.")
        return None

    def _structural_match() -> bool:
        return (
            list(payload.get("feature_columns", [])) == list(feature_columns)
            and set(payload.get("members", [])) == set(members)
            and int(payload.get("train_weeks", -1)) == int(train_weeks)
            and int(payload.get("val_weeks", -1)) == int(val_weeks)
            and int(payload.get("retrain_freq", -1)) == int(retrain_freq)
        )

    saved_fp = (payload.get("extra") or {}).get("fingerprint")
    if fingerprint is not None and saved_fp is not None:
        if saved_fp == fingerprint:
            return payload
        if allow_structural_restamp and _structural_match():
            log.warning(
                f"RESUME_RESTAMP: reusing {path.name} despite fingerprint "
                f"mismatch ({saved_fp} != {fingerprint}) — structural match holds; "
                "assuming input data unchanged. Bundle will be re-saved under the "
                "current fingerprint."
            )
            return payload
        return None

    # Legacy fallback: no fingerprint stored. Match on the structural fields the
    # bundle does carry.
    if _structural_match():
        log.warning(
            f"Reusing legacy bundle {path.name} by feature/member/window match "
            "(no fingerprint stored); assuming the input data is unchanged."
        )
        return payload
    return None


def needs_retrain(as_of: pd.Timestamp, retrain_freq_weeks: int, signal_day: str = "WED") -> bool:
    """True if there is no saved state, or the cadence threshold has elapsed.

    `as_of` is typically the target signal-day date. The comparison is in days
    (retrain_freq_weeks * 7) for simplicity — week-of-year arithmetic adds
    no value here.
    """
    state = read_state(signal_day)
    if state is None:
        return True
    last = pd.Timestamp(state["latest_retrain_date"])
    threshold_days = int(retrain_freq_weeks) * 7
    return (as_of - last).days >= threshold_days
