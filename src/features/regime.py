"""Project-specific macro regime wrapper around `src.lib.regime_hmm`.

Thin layer: picks emission types per macro (rates/vol get `changes`, prices
get `returns`), reads HMM hyperparameters from config, writes the posteriors
to `data/processed/macro_regimes.parquet`. The generic HMM lives in
`src/lib/regime_hmm.py` and is consumed unchanged.

Run as a script to (re)generate the cached regimes file:

    python -m src.features.regime
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.lib.regime_hmm import fit_regimes

log = logging.getLogger(__name__)


# Macros whose emission distribution lives in level-change space (rates, vol),
# not return space. Names match the keys in `src.data.macro.DEFAULT_MACROS`.
DEFAULT_CHANGE_EMISSION = ("VIX", "US10Y", "US2Y", "US30Y", "US3M")


def build_macro_regimes(
    macro: pd.DataFrame,
    n_states: int = 2,
    fit_mode: str = "walk_forward",
    refit_every: int = 60,
    min_obs: int = 252,
    n_iter: int = 200,
    random_state: int = 42,
    change_emission_for: tuple[str, ...] = DEFAULT_CHANGE_EMISSION,
) -> pd.DataFrame:
    """Compute HMM regime posteriors for each macro column.

    Default is walk-forward (causal, no look-ahead). The "full" mode is
    available for diagnostics but should not be used for backtest features.

    Args:
        macro: wide DataFrame from `src.data.macro.fetch_macro_data`.
        n_states: HMM states (default 2 per project spec).
        fit_mode: "walk_forward" (default, causal) or "full" (diagnostic only).
        refit_every: walk-forward refit cadence in business days.
        min_obs: warmup window before any decoding (causal-only rows start here).
        change_emission_for: macro names that should use diff() emissions
            instead of pct_change().

    Returns:
        DataFrame indexed by date with columns
            {NAME}_HMM_S0_P, {NAME}_HMM_S1_P, {NAME}_HMM_TRANS_P
        per macro. S0 is canonicalized to lowest-variance state ("calm");
        S1 is highest-variance ("stress"). Warmup rows are NaN in
        walk-forward mode (no causal posterior available yet).
    """
    emission_map = {
        c: ("changes" if c.upper() in {x.upper() for x in change_emission_for} else "returns")
        for c in macro.columns
    }
    log.info(
        f"HMM regimes: {len(macro.columns)} macros, n_states={n_states}, "
        f"fit_mode={fit_mode}, min_obs={min_obs}, refit_every={refit_every}, "
        f"changes-emission={[c for c, e in emission_map.items() if e=='changes']}"
    )
    return fit_regimes(
        data=macro,
        n_states=n_states,
        emission=emission_map,
        fit_mode=fit_mode,
        refit_every=refit_every,
        min_obs=min_obs,
        n_iter=n_iter,
        random_state=random_state,
    )


def build_and_save_macro_regimes(
    macro: pd.DataFrame,
    out_path: Path,
    **kwargs,
) -> pd.DataFrame:
    """Build regime posteriors and write to `out_path` (parquet)."""
    regimes = build_macro_regimes(macro, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(out_path)
    log.info(
        f"Saved macro regimes: {regimes.shape[0]} dates × {regimes.shape[1]} cols → {out_path}"
    )
    return regimes


def load_macro_regimes(path: Path) -> pd.DataFrame | None:
    """Load cached regime posteriors if present; return None if missing."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    log.info(f"Loaded macro regimes from {path}: {df.shape[0]} dates × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.config import PROCESSED_DIR, load_config
    from src.data.macro import DEFAULT_MACROS, fetch_macro_data

    cfg = load_config()
    macro_cfg = cfg.get("macro", {})
    syms = macro_cfg.get("symbols") or DEFAULT_MACROS

    macro = fetch_macro_data(
        symbols=syms,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        rate_limit_per_min=cfg["data"].get("fmp_rate_limit_per_min", 240),
    )

    out_path = PROCESSED_DIR / "macro_regimes.parquet"
    build_and_save_macro_regimes(
        macro,
        out_path,
        n_states=macro_cfg.get("hmm_states", 2),
        fit_mode=macro_cfg.get("hmm_fit_mode", "walk_forward"),
        refit_every=macro_cfg.get("hmm_refit_every", 60),
        min_obs=macro_cfg.get("hmm_min_obs", 252),
        n_iter=macro_cfg.get("hmm_iter", 200),
        random_state=macro_cfg.get("hmm_seed", 42),
    )
