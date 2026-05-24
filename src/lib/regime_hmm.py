"""Generic Hidden-Markov-Model regime detection for univariate time series.

Domain-agnostic. Takes a wide DataFrame of time series (any columns) and
returns a wide DataFrame of state posteriors and switching probabilities.
Designed to be lifted into a standalone package later — no imports from
project-specific modules, no project paths, no domain vocabulary in the API.

Key design choices:
- One independent HMM per input column. Multivariate joint HMMs are a
  separate function/class entirely; mixing the two in one API is confusing.
- Canonicalized state ordering: states are sorted by emission variance
  ascending, so state 0 is always "lowest-variance" and state N-1 is always
  "highest-variance" regardless of EM init. Output column names are stable
  across reruns.
- Two emission flavors: "returns" (pct_change) for prices/levels, "changes"
  (diff) for rates/vol. Choice is per-column via `emission`.
- Fit modes:
    "walk_forward"  — refit on the expanding window every `refit_every` rows,
                      decode forward only. STRICTLY CAUSAL: warmup rows
                      (before `min_obs`) are returned as NaN, not in-sample
                      posteriors. Default and recommended.
    "full"          — fit once on the whole series. Cheap and deterministic,
                      but state geometry inherits future-date information.
                      Useful for diagnostics; never for backtest features.

Public API:
    fit_regimes(data, ...)  -> regimes (wide DataFrame)
    HMMRegime  (lower-level class if you want the fitted model objects)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def fit_regimes(
    data: pd.DataFrame,
    n_states: int = 2,
    emission: str | dict[str, str] = "returns",
    fit_mode: str = "walk_forward",
    refit_every: int = 60,
    min_obs: int = 252,
    n_iter: int = 200,
    random_state: int = 42,
    covariance_type: str = "diag",
) -> pd.DataFrame:
    """Fit an HMM per column and return state posteriors + switching probs.

    Args:
        data: wide DataFrame, datetime-indexed, one time series per column.
        n_states: number of hidden states.
        emission: "returns" (pct_change), "changes" (diff), or "raw" (as-is).
            A dict overrides per-column, e.g. {"VIX": "changes", "OIL": "returns"}.
        fit_mode: "full" (one fit on whole series, mild look-ahead) or
            "walk_forward" (refit every `refit_every` rows, no look-ahead).
        refit_every: walk-forward refit cadence in observations.
        min_obs: minimum non-NaN observations required to attempt a fit.
        n_iter: max EM iterations per fit.
        random_state: seed for reproducibility.
        covariance_type: passed to hmmlearn ("diag" recommended; "full" is
            overkill for univariate, "spherical" equivalent to "diag" here).

    Returns:
        Wide DataFrame indexed by the input dates, with `(2 * n_states + 1)`
        columns per input column:
            {NAME}_HMM_S{k}_P     — posterior P(state k | observations up to t)
                                    for k in 0..n_states-1
            {NAME}_HMM_TRANS_P    — posterior-weighted switching probability
                                    (probability of changing state next step)

        States are canonicalized by emission variance ascending: S0 = lowest
        variance, S{n-1} = highest. Columns are dropped silently for input
        series whose fit fails (too few obs, EM divergence, etc.) — check
        the log warnings.
    """
    if fit_mode not in ("full", "walk_forward"):
        raise ValueError(f"fit_mode must be 'full' or 'walk_forward', got {fit_mode!r}")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")

    out: dict[str, pd.Series] = {}
    for col in data.columns:
        emission_for_col = (
            emission if isinstance(emission, str) else emission.get(col, "returns")
        )
        try:
            cols = _fit_one_column(
                data[col],
                column_name=col,
                n_states=n_states,
                emission=emission_for_col,
                fit_mode=fit_mode,
                refit_every=refit_every,
                min_obs=min_obs,
                n_iter=n_iter,
                random_state=random_state,
                covariance_type=covariance_type,
            )
        except _FitSkipped as e:
            log.warning(f"  {col}: skipped ({e})")
            continue
        for k, s in cols.items():
            out[k] = s

    if not out:
        raise RuntimeError("All HMM fits failed; no regime posteriors produced.")

    return pd.DataFrame(out).sort_index()


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


class _FitSkipped(Exception):
    """Internal signal that a column couldn't be fit (logged, not raised)."""


@dataclass
class _FitResult:
    """Internal: ordering and parameters of a fitted HMM after canonicalization."""
    order: np.ndarray            # permutation that sorts states by emission variance
    transmat: np.ndarray         # reordered transition matrix (n_states × n_states)
    emission_var: np.ndarray     # reordered emission variances


def _make_emission(s: pd.Series, kind: str) -> pd.Series:
    if kind == "returns":
        return s.pct_change()
    if kind == "changes":
        return s.diff()
    if kind == "raw":
        return s
    raise ValueError(f"emission must be 'returns'/'changes'/'raw', got {kind!r}")


def _fit_gaussian_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int,
    random_state: int,
    covariance_type: str,
):
    """Fit a Gaussian HMM. Imported lazily so hmmlearn is only required when
    HMM features are actually used."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError(
            "hmmlearn is required. Install with: pip install hmmlearn"
        ) from e

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    return model


def _canonicalize(model, n_states: int) -> _FitResult:
    """Sort states by emission variance ascending. Stable across reruns."""
    variances = model.covars_.reshape(n_states, -1).mean(axis=1)
    order = np.argsort(variances)
    transmat = model.transmat_[order][:, order]
    return _FitResult(order=order, transmat=transmat, emission_var=variances[order])


def _fit_one_column(
    s: pd.Series,
    column_name: str,
    n_states: int,
    emission: str,
    fit_mode: str,
    refit_every: int,
    min_obs: int,
    n_iter: int,
    random_state: int,
    covariance_type: str,
) -> dict[str, pd.Series]:
    obs = _make_emission(s, emission).dropna()
    if len(obs) < min_obs:
        raise _FitSkipped(f"only {len(obs)} obs (< min_obs={min_obs})")

    X = obs.values.reshape(-1, 1)

    if fit_mode == "full":
        try:
            model = _fit_gaussian_hmm(X, n_states, n_iter, random_state, covariance_type)
        except Exception as e:
            raise _FitSkipped(f"HMM fit failed: {type(e).__name__}: {e}")
        fr = _canonicalize(model, n_states)
        post = model.predict_proba(X)[:, fr.order]
        trans_p = _per_obs_switch_prob(post, fr.transmat)
        index = obs.index

    elif fit_mode == "walk_forward":
        post, trans_p, index = _walk_forward_decode(
            X, obs.index, n_states, refit_every, min_obs,
            n_iter, random_state, covariance_type,
        )
        if post is None:
            raise _FitSkipped(f"walk-forward HMM produced no usable rows")

    else:  # already validated above
        raise AssertionError(f"unreachable: fit_mode={fit_mode!r}")

    cols: dict[str, pd.Series] = {}
    for k in range(n_states):
        cols[f"{column_name}_HMM_S{k}_P"] = pd.Series(post[:, k], index=index)
    cols[f"{column_name}_HMM_TRANS_P"] = pd.Series(trans_p, index=index)

    log.info(
        f"  {column_name}: emission={emission}, mode={fit_mode}, "
        f"emission_var={np.round(_emission_var_from_obs(obs.values, post), 5).tolist()}"
    )
    return cols


def _emission_var_from_obs(obs: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Posterior-weighted variance per state — sanity check that the
    canonicalized state-0 really is the lower-variance regime."""
    out = []
    for k in range(post.shape[1]):
        w = post[:, k]
        mean = (w * obs).sum() / max(w.sum(), 1e-9)
        var = (w * (obs - mean) ** 2).sum() / max(w.sum(), 1e-9)
        out.append(var)
    return np.array(out)


def _per_obs_switch_prob(post: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Posterior-weighted probability of moving to a different state next step."""
    diag = np.diag(transmat)
    switch_per_state = 1.0 - diag
    return post @ switch_per_state


def _walk_forward_decode(
    X: np.ndarray,
    index: pd.Index,
    n_states: int,
    refit_every: int,
    min_obs: int,
    n_iter: int,
    random_state: int,
    covariance_type: str,
) -> tuple[np.ndarray | None, np.ndarray | None, pd.Index | None]:
    """Refit every `refit_every` rows on the expanding window, decode forward
    only. STRICTLY CAUSAL: rows before `min_obs` are returned as NaN — they
    cannot be decoded without seeing their own data in the fit, so we refuse
    to emit a posterior for them. The first usable output row is at index
    `min_obs` (decoded by a model fit on `X[:min_obs]` and then queried for
    row `min_obs` alone via the forward algorithm).
    """
    T = X.shape[0]
    if T <= min_obs:
        return None, None, None

    post_all = np.full((T, n_states), np.nan)
    trans_p_all = np.full(T, np.nan)

    # First fit uses the warmup window [0:min_obs). Warmup rows themselves
    # stay NaN — they were part of the fit, so a posterior on them would
    # leak. The first row we DECODE is t=min_obs, on the model fit at t-1.
    try:
        model = _fit_gaussian_hmm(
            X[:min_obs], n_states, n_iter, random_state, covariance_type
        )
        fr = _canonicalize(model, n_states)
    except Exception as e:
        log.warning(f"    walk-forward initial fit failed: {e}")
        return None, None, None

    last_refit = min_obs
    for t in range(min_obs, T):
        if t - last_refit >= refit_every:
            try:
                model = _fit_gaussian_hmm(
                    X[:t], n_states, n_iter, random_state, covariance_type
                )
                fr = _canonicalize(model, n_states)
            except Exception as e:
                log.warning(f"    walk-forward refit at t={t} failed: {e}; keeping prior model")
            last_refit = t

        # Online filter via predict_proba on [:t+1] — t is now in the data
        # but the *model* it's being decoded against was fit at last_refit ≤ t.
        # This is the standard walk-forward decoding pattern.
        post_window = model.predict_proba(X[: t + 1])[:, fr.order]
        post_all[t] = post_window[-1]
        trans_p_all[t] = _per_obs_switch_prob(post_window[-1:], fr.transmat)[0]

    return post_all, trans_p_all, index


__all__ = ["fit_regimes"]
