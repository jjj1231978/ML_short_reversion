"""Cached parquet/json loaders for Streamlit pages.

All loaders use `st.cache_data` keyed by file mtime so pipeline reruns
auto-invalidate. Returns None when a file is absent so pages can render a
"not yet produced" message instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR

REPORTS_DIR = PROCESSED_DIR / "reports"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_parquet(name: str, mtime_key: float | None = None) -> pd.DataFrame | None:
    """Load a parquet from `data/processed/` by base name (no extension).

    `mtime_key` must NOT be underscore-prefixed: st.cache_data skips hashing
    underscore-prefixed arguments, so the previous `_mtime_key` never entered the
    cache key and a rerun of the pipeline left every page serving the artifact it
    had first read.
    """
    p = PROCESSED_DIR / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_json(name: str, mtime_key: float | None = None) -> dict | None:
    p = PROCESSED_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def get(name: str, kind: str = "parquet"):
    """Convenience wrapper that passes the file mtime as a cache key."""
    p = PROCESSED_DIR / f"{name}.{kind}"
    key = _mtime(p)
    if kind == "parquet":
        return load_parquet(name, mtime_key=key)
    if kind == "json":
        return load_json(name, mtime_key=key)
    raise ValueError(f"Unknown kind: {kind}")


def list_report_runs() -> list[str]:
    """Return run_id directories under data/processed/reports/, newest first."""
    if not REPORTS_DIR.exists():
        return []
    runs = [p.name for p in REPORTS_DIR.iterdir() if p.is_dir() and p.name != "_cache" and not p.is_symlink()]
    return sorted(runs, reverse=True)


def load_report_run(run_id: str | None = None) -> dict:
    """Load draft.md / critique.md / final.md / meta.json for a report run."""
    if not REPORTS_DIR.exists():
        return {}
    target = REPORTS_DIR / (run_id or "latest")
    if not target.exists():
        return {}
    out: dict = {"run_id": target.name}
    for name in ("draft", "critique", "final"):
        f = target / f"{name}.md"
        out[name] = f.read_text() if f.exists() else None
    meta = target / "meta.json"
    out["meta"] = json.loads(meta.read_text()) if meta.exists() else None
    return out


def render_no_data_warning(artifact: str) -> None:
    """Standard placeholder rendered when a parquet/json is missing."""
    st.warning(
        f"`{artifact}` not found in `{PROCESSED_DIR}`. "
        "Run the pipeline first: `python -m src.main`."
    )
