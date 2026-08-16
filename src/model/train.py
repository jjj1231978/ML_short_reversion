"""Rolling-window training for XGBoost, LightGBM, and RandomForest.

The rolling loop trains all configured ensemble members at each retrain step
on the same (X_train, y_train, X_val, y_val) slice, then predicts the next
`retrain_freq` weeks with each member. Final predictions are combined via
`src.model.ensemble.combine_predictions`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.model.ensemble import combine_predictions

log = logging.getLogger(__name__)


def _rss_gb() -> float:
    """Current process resident memory in GB (Linux /proc/self/statm).

    Logged per retrain so the backtest's memory curve is visible: a capped-
    expanding window should plateau once the cap binds rather than ratchet up.
    """
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * 4096 / 1e9
    except Exception:
        return float("nan")


def _run_fingerprint(
    feature_columns: list[str],
    members: list[str],
    train_weeks: int,
    val_weeks: int,
    retrain_freq: int,
    window: str,
    signal_day: str,
    dates: pd.DatetimeIndex,
    max_train_weeks: int | None = None,
    member_params: dict | None = None,
) -> str:
    """Stable hash identifying a backtest configuration + its input data extent.

    Stored on each saved retrain bundle so a resumed run can reuse a bundle ONLY
    when the config and the date index are byte-for-byte the same. Including the
    full date index means any data refresh that adds/removes weeks invalidates
    every reused fit, forcing a clean rebuild rather than mixing vintages.

    `member_params` covers the per-member hyperparameters (the `model.<member>`
    config blocks). Without it the hash is blind to the thing you are most likely
    to change deliberately: flipping `lightgbm.random_ensemble.enabled`, retuning
    a learning rate, or resizing the bag all leave the fingerprint identical, so a
    re-run silently reloads the OLD models and reproduces the previous result. An
    A/B on hyperparameters would then look like it ran while actually comparing a
    config against itself.
    """
    parts = [
        "|".join(feature_columns),
        "|".join(sorted(members)),
        f"{int(train_weeks)},{int(val_weeks)},{int(retrain_freq)}",
        str(window),
        f"cap={'' if max_train_weeks is None else int(max_train_weeks)}",
        signal_day.upper(),
        "|".join(pd.DatetimeIndex(dates).strftime("%Y-%m-%d")),
        # sort_keys → stable across dict ordering; default=str tolerates any
        # non-JSON scalars (e.g. None/tuples) that land in a member config.
        json.dumps(
            {m: (member_params or {}).get(m) for m in sorted(members)},
            sort_keys=True,
            default=str,
        ),
    ]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-member fit / predict
# ---------------------------------------------------------------------------

def _fit_xgb(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    params = dict(member_cfg["params"])
    n_estimators = params.pop("n_estimators", 2000)
    early_stop = member_cfg.get("early_stopping_rounds", 50)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )


def _predict_xgb(model, X) -> np.ndarray:
    return model.predict(xgb.DMatrix(X))


# --- LightGBM random-hyperparameter ensemble ------------------------------
# Integer-valued and log-sampled LightGBM params, so a [lo, hi] range in the
# config is drawn on the right scale (uniform in log space for the learning
# rate, rounded to int for leaf/sample counts).
_LGB_INT_PARAMS = frozenset(
    {"num_leaves", "min_child_samples", "max_depth", "min_data_in_leaf", "bagging_freq"}
)
_LGB_LOG_PARAMS = frozenset({"learning_rate"})


class LGBBag:
    """A bag of LightGBM boosters from the random-hyperparameter ensemble.

    Holds the ``n_select`` boosters that scored best by validation rank-IC out
    of ``n_candidates`` trained on the same train/val slice. ``predict`` returns
    the simple mean across boosters (each at its own ``best_iteration``).
    ``boosters`` is ordered best-IC first, so ``boosters[0]`` is the single
    strongest member and ``val_ics[0]`` its validation rank-IC.
    """

    def __init__(self, boosters: list, val_ics: list[float]):
        self.boosters = list(boosters)
        self.val_ics = list(val_ics)

    def predict(self, X_values) -> np.ndarray:
        preds = []
        for b in self.boosters:
            # Locally-fit boosters carry best_iteration; boosters reloaded from a
            # Modal worker are already truncated to their best iteration and have
            # best_iteration == -1 (predict all trees == best). Normalize both.
            ni = getattr(b, "best_iteration", None)
            ni = ni if (ni and ni > 0) else None
            preds.append(b.predict(X_values, num_iteration=ni))
        return np.mean(preds, axis=0)


def _sample_lgb_params(base: dict, space: dict, rng) -> dict:
    """Draw one hyperparameter set: base params overlaid with a random sample
    from each [lo, hi] range in ``space`` (log-uniform / int-rounded per the
    sets above)."""
    params = dict(base)
    for name, bounds in space.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if name in _LGB_LOG_PARAMS:
            val = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            val = float(rng.uniform(lo, hi))
        if name in _LGB_INT_PARAMS:
            val = int(round(val))
        params[name] = val
    return params


def _rank_ic(pred: np.ndarray, y: np.ndarray, dates) -> float:
    """Mean cross-sectional Spearman rank-IC of ``pred`` vs ``y``, grouped by
    date. Weeks with fewer than 5 names are skipped (rank corr is noise there).
    Returns NaN if no week qualifies."""
    df = pd.DataFrame({"p": np.asarray(pred), "y": np.asarray(y)})
    # Group by an external key array so the grouping label is not a column of
    # the frame the lambda sees (avoids pandas' include_groups deprecation).
    ics = df.groupby(np.asarray(dates), sort=False).apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 5 else np.nan
    )
    vals = ics.to_numpy(dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _fit_lgb(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    import lightgbm as lgb
    base = dict(member_cfg["params"])
    n_estimators = base.pop("n_estimators", 2000)
    early_stop = member_cfg.get("early_stopping_rounds", 50)

    re_cfg = member_cfg.get("random_ensemble") or {}
    # Modal backend: fan the candidate fits out across CPU containers instead of
    # training all 200 locally. Falls through to the local loop below otherwise.
    if re_cfg.get("enabled", False) and re_cfg.get("backend", "local") == "modal":
        return fit_lgb_bag_modal(member_cfg, X_train, y_train, X_val, y_val)

    # feature_pre_filter=False: the dataset is binned once and reused across all
    # candidates, but each candidate randomizes min_child_samples — pre-filtering
    # would lock features to the first candidate's value and then error when a
    # later candidate lowers it.
    ds_params = {"feature_pre_filter": False}
    train_set = lgb.Dataset(X_train, label=y_train, params=ds_params)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, params=ds_params)

    def _train_one(params: dict):
        return lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )

    if not re_cfg.get("enabled", False):
        return _train_one(base)

    # Local randomized-hyperparameter ensemble: train n_candidates boosters, each
    # a random param draw + distinct seed, then keep the n_select best by
    # validation rank-IC (the strategy's cross-sectional ranking objective).
    n_candidates = int(re_cfg.get("n_candidates", 200))
    n_select = min(int(re_cfg.get("n_select", 50)), n_candidates)
    space = re_cfg.get("param_space") or {}
    rng = np.random.default_rng(int(re_cfg.get("seed", 42)))
    val_dates = X_val.index.get_level_values("date")
    y_val_arr = y_val.to_numpy()

    scored: list[tuple[float, int, Any]] = []
    for k in range(n_candidates):
        params = _sample_lgb_params(base, space, rng)
        params["seed"] = int(rng.integers(0, 2**31 - 1))
        booster = _train_one(params)
        pred = booster.predict(X_val.values, num_iteration=booster.best_iteration)
        scored.append((_rank_ic(pred, y_val_arr, val_dates), k, booster))

    # Best IC first; NaN ICs (degenerate val weeks) sink to the bottom.
    scored.sort(key=lambda t: t[0] if t[0] == t[0] else float("-inf"), reverse=True)
    top = scored[:n_select]
    log.info(
        f"    lgb random-ensemble: {n_candidates} trained, kept top {n_select} "
        f"by val rank-IC (best={top[0][0]:.4f}, worst-kept={top[-1][0]:.4f})"
    )
    return LGBBag([b for _, _, b in top], [ic for ic, _, _ in top])


def _predict_lgb(model, X) -> np.ndarray:
    if isinstance(model, LGBBag):
        return model.predict(X.values)
    return model.predict(X.values, num_iteration=model.best_iteration)


def tree_shap_values(model, X) -> np.ndarray:
    """SHAP attributions for a tree member. For the LightGBM bag, averages
    TreeExplainer attributions across its selected boosters so the importance
    reflects what the bagged prediction actually keys off; for a single model,
    explains it directly. Returns an (n_rows, n_features) float array."""
    import shap

    if isinstance(model, LGBBag):
        mats = [
            np.asarray(shap.TreeExplainer(b).shap_values(X), dtype=float)
            for b in model.boosters
        ]
        return np.mean(mats, axis=0)
    return np.asarray(shap.TreeExplainer(model).shap_values(X), dtype=float)


def fit_lgb_bag_modal(member_cfg: dict, X_train, y_train, X_val, y_val) -> "LGBBag":
    """Modal-backed LightGBM random ensemble: sample n_candidates param sets
    locally, fan the fits out across CPU containers, score each by validation
    rank-IC on the worker, and reassemble the global top n_select into an LGBBag.

    The train/val window is broadcast once to all workers (via .map kwargs); only
    the per-worker param chunk varies. Boosters come back as LightGBM model
    strings truncated to their best iteration."""
    import uuid

    import lightgbm as lgb
    import modal
    import pandas as pd

    from src.model.lgb_modal import app, train_chunk, vol

    base = dict(member_cfg["params"])
    n_estimators = base.pop("n_estimators", 2000)
    early_stop = member_cfg.get("early_stopping_rounds", 50)
    re_cfg = member_cfg["random_ensemble"]
    n_candidates = int(re_cfg.get("n_candidates", 200))
    n_select = min(int(re_cfg.get("n_select", 50)), n_candidates)
    n_workers = int(re_cfg.get("n_workers", 10))
    space = re_cfg.get("param_space") or {}
    rng = np.random.default_rng(int(re_cfg.get("seed", 42)))

    # Sample candidates locally so selection stays deterministic and identical to
    # the local backend; each gets a distinct seed and a stable index.
    candidates = []
    for _ in range(n_candidates):
        p = _sample_lgb_params(base, space, rng)
        p["seed"] = int(rng.integers(0, 2**31 - 1))
        p.setdefault("verbose", -1)
        candidates.append(p)

    # One task per worker: round-robin the indexed candidates for even load. The
    # run_id namespaces this retrain's booster files on the staging volume.
    run_id = uuid.uuid4().hex[:12]
    tasks = []
    for w in range(n_workers):
        items = [[i, candidates[i]] for i in range(w, n_candidates, n_workers)]
        if items:
            tasks.append({"run_id": run_id, "items": items})

    data_kwargs = dict(
        Xtr=X_train.to_numpy(np.float32),
        ytr=y_train.to_numpy(np.float32),
        Xval=X_val.to_numpy(np.float32),
        yval=y_val.to_numpy(np.float32),
        val_codes=pd.factorize(X_val.index.get_level_values("date"))[0].astype(np.int32),
        n_estimators=n_estimators,
        early_stop=early_stop,
    )

    log.info(
        f"    lgb modal: fanning {n_candidates} candidates over {len(tasks)} "
        f"workers (~{len(tasks[0]['items'])} each), run_id={run_id}..."
    )
    results: list[dict] = []
    with modal.enable_output(), app.run():
        for chunk_out in train_chunk.map(tasks, kwargs=data_kwargs):
            results.extend(chunk_out)

    # Global top-N by validation rank-IC (NaN ICs sink last); load just those
    # boosters back from the staging volume (small download, no result blobs).
    results.sort(key=lambda r: r["val_ic"] if r["val_ic"] == r["val_ic"] else float("-inf"), reverse=True)
    top = results[:n_select]
    boosters = [
        lgb.Booster(model_str=b"".join(vol.read_file(f"{run_id}/{r['idx']}.txt")).decode())
        for r in top
    ]
    log.info(
        f"    lgb modal: {len(results)} trained, kept top {n_select} by val "
        f"rank-IC (best={top[0]['val_ic']:.4f}, worst-kept={top[-1]['val_ic']:.4f})"
    )
    try:  # best-effort cleanup so the staging volume doesn't accumulate runs
        vol.remove_file(run_id, recursive=True)
    except Exception:  # noqa: BLE001
        pass
    return LGBBag(boosters, [r["val_ic"] for r in top])


def _fit_rf(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    """Fit RandomForest. Concatenates train+val since RF has no early stopping
    and otherwise the val rows are unused (asymmetric vs xgb/lgb).

    Because the val rows ARE training rows for this member, scoring them with
    `.predict()` is in-sample and inflates the validation rank-IC that
    `ir_weighted` uses as a skill weight — observed at roughly 4x the boosters'
    (0.068 vs 0.010-0.027), which would hand RF a dominant weight for a reason
    unrelated to skill. So we also stash out-of-bag predictions for the val
    slice: each OOB prediction averages only the trees that did not bootstrap
    that row, making it a fair-ish score comparable to the boosters' true
    holdout. Fitted predictions and hence the member's IR are unchanged — this
    only affects the weighting statistic.

    Caveat worth keeping in mind: OOB is not a pure holdout, since the trees
    that skipped a given row still saw OTHER val rows. With overlapping
    financial labels OOB is known to run optimistic, so treat this as "much
    better than in-sample", not "as clean as a true holdout".
    """
    from sklearn.ensemble import RandomForestRegressor
    params = dict(member_cfg["params"])
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    # oob_score needs bootstrap=True (sklearn's default). Honour an explicit
    # bootstrap=False rather than overriding the caller's config.
    want_oob = params.get("bootstrap", True) and len(X_val) > 0
    if want_oob:
        params.setdefault("oob_score", True)
    model = RandomForestRegressor(**params)
    model.fit(X_all.values, y_all.values)
    if want_oob:
        try:
            # X_val is the tail of X_all by construction of the concat above.
            oob = np.asarray(model.oob_prediction_, dtype=float)[-len(X_val):]
            # Rows never left out of any bootstrap come back NaN; only keep the
            # vector if it is mostly usable, else fall back to .predict().
            if np.isfinite(oob).sum() >= 0.5 * len(oob):
                model._oob_val_pred = oob
        except Exception as e:  # noqa: BLE001 — diagnostic only, never fatal
            log.warning(f"RF OOB val predictions unavailable ({e}); "
                        "validation rank-IC for random_forest will be in-sample.")
    return model


def _predict_rf(model, X) -> np.ndarray:
    return model.predict(X.values)


def _fit_mlp(member_cfg: dict, X_train, y_train, X_val, y_val) -> Any:
    """Fit the PyTorch MLP member. The wrapper holds scaler + module + device."""
    from src.model.mlp import fit_from_member_cfg
    return fit_from_member_cfg(member_cfg, X_train, y_train, X_val, y_val)


def _predict_mlp(model, X) -> np.ndarray:
    return model.predict(X)


DISPATCH: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (_fit_xgb, _predict_xgb),
    "lightgbm": (_fit_lgb, _predict_lgb),
    "random_forest": (_fit_rf, _predict_rf),
    "mlp": (_fit_mlp, _predict_mlp),
}


# ---------------------------------------------------------------------------
# Rolling loop
# ---------------------------------------------------------------------------

def _resolve_members(cfg: dict) -> list[str]:
    """Read which members to train from cfg.model. type=ensemble runs all
    listed in ensemble.members; bare type runs that single member."""
    model_cfg = cfg["model"]
    t = model_cfg.get("type", "xgboost")
    if t == "ensemble":
        return list(model_cfg["ensemble"]["members"])
    if t in DISPATCH:
        return [t]
    raise ValueError(f"unknown model.type: {t}")


def rolling_train_predict_ensemble(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[dict[str, pd.Series], pd.Series, dict[str, Any], pd.DataFrame]:
    """Train all ensemble members on rolling windows and combine predictions.

    Returns:
        per_model_predictions: {member: Series of out-of-sample predictions}
        ensemble_predictions: Series — combine_predictions over per_model
        last_models: {member: most-recent fitted model object} (for SHAP)
        last_test_X: DataFrame fed to the most recent prediction call
    """
    if cfg is None:
        cfg = load_config()

    model_cfg = cfg["model"]
    train_weeks = model_cfg["train_weeks"]
    val_weeks = model_cfg["val_weeks"]
    retrain_freq = model_cfg["retrain_freq"]
    members = _resolve_members(cfg)
    combine_method = model_cfg.get("ensemble", {}).get("combine", "rank_mean")
    # Persisted retrain bundles must be namespaced by the run's signal day so a
    # THU/FRI backtest cannot overwrite the live WED production bundles (and vice
    # versa). Falls back to WED for back-compat with configs lacking the key.
    bundle_signal_day = cfg.get("backtest", {}).get("signal_day", "WED").upper()

    log.info(f"Training members: {members} (combine={combine_method})")

    dates = features.index.get_level_values("date").unique().sort_values()
    min_start = train_weeks + val_weeks
    if len(dates) <= min_start:
        raise ValueError(
            f"Not enough dates: have {len(dates)}, need at least {min_start + 1}"
        )

    per_model_series, val_ic_by_week, current_models, last_test_X = (
        _rolling_train_collect(features, target, cfg, members)
    )

    if len(members) > 1:
        ensemble_preds = combine_predictions(
            per_model_series,
            method=combine_method,
            val_ic=val_ic_by_week,
            target=target,
        )
    else:
        ensemble_preds = per_model_series[members[0]].copy()
        ensemble_preds.name = "prediction"

    return per_model_series, ensemble_preds, current_models, last_test_X


def _rolling_train_collect(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict,
    members: list[str],
) -> tuple[dict[str, pd.Series], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Walk-forward train/predict the members WITHOUT combining. Returns the
    per-member OOS prediction Series, a (date × member) frame of the validation
    rank-IC that produced each OOS week's prediction (used by IR-weighted /
    stacked combination), the most-recent fitted models, and the last test X.

    Splitting collection from combination lets the combiner be swapped (or
    several compared) on a single training run instead of retraining per method.
    """
    model_cfg = cfg["model"]
    train_weeks = model_cfg["train_weeks"]
    val_weeks = model_cfg["val_weeks"]
    retrain_freq = model_cfg["retrain_freq"]
    window = model_cfg.get("window", "rolling")
    # Optional cap on the expanding window's lookback (weeks). None = unbounded.
    max_train_weeks = model_cfg.get("max_train_weeks")
    combine_method = model_cfg.get("ensemble", {}).get("combine", "rank_mean")
    bundle_signal_day = cfg.get("backtest", {}).get("signal_day", "WED").upper()

    dates = features.index.get_level_values("date").unique().sort_values()
    # min_start = warmup before the first OOS week. For both modes this is
    # train_weeks + val_weeks; in expanding mode train_weeks is just the MINIMUM
    # initial train (the train slice then grows back to the start of history).
    min_start = train_weeks + val_weeks
    log.info(
        f"Walk-forward window={window} (train_weeks={train_weeks}"
        f"{'+ (expanding)' if window == 'expanding' else ''}, "
        f"val_weeks={val_weeks}, retrain_freq={retrain_freq})"
    )

    # Per-retrain checkpointing: each saved bundle doubles as a resume point. On
    # restart, a retrain step whose bundle already exists on disk and matches
    # this run's fingerprint is reloaded instead of refit (the dominant cost) —
    # only the cheap weekly scoring is replayed. Disable via
    # model.reuse_saved_retrains: false to force every step to refit.
    persist_retrains = model_cfg.get("persist_retrains", True)
    reuse_saved = model_cfg.get("reuse_saved_retrains", True)
    # One-time manual resume escape hatch (RESUME_RESTAMP=1): reuse a bundle whose
    # fingerprint mismatches but whose *structure* (feature order, members,
    # train/val/retrain windows) matches, then re-save it under the current
    # fingerprint. Use ONLY when you have verified out-of-band that the input data
    # is unchanged and the mismatch is a benign config delta that does not alter
    # this bundle's training window (e.g. adding a max_train_weeks cap that does
    # not yet bind on older expanding retrains). Off by default; the fingerprint
    # stays strict for normal runs.
    resume_restamp = os.environ.get("RESUME_RESTAMP") == "1"
    fingerprint = _run_fingerprint(
        list(features.columns), members, train_weeks, val_weeks,
        retrain_freq, window, bundle_signal_day, dates, max_train_weeks,
        member_params={m: model_cfg.get(m) for m in members},
    )
    from src.model.persistence import load_matching_bundle, save_bundle

    # Per-member running state and prediction buckets
    current_models: dict[str, Any] = {m: None for m in members}
    current_val_ic: dict[str, float] = {m: float("nan") for m in members}
    per_model_preds: dict[str, list[pd.Series]] = {m: [] for m in members}
    val_ic_rows: list[pd.Series] = []  # one row per OOS week: member → val rank-IC
    last_test_X: pd.DataFrame | None = None

    for i in range(min_start, len(dates)):
        t = dates[i]

        # Retrain all members periodically on the same train/val slice
        if current_models[members[0]] is None or (i - min_start) % retrain_freq == 0:
            # Expanding: train on ALL history up to the validation block (start of
            # data → val_start-1). Rolling: fixed train_weeks ending at val_start-1.
            # Capped-expanding (max_train_weeks set): grow from dates[0] until the
            # lookback reaches max_train_weeks, then hold it fixed and roll forward —
            # bounds memory without delaying the first OOS week (min_start is still
            # the small train_weeks+val_weeks minimum, so early OOS is preserved).
            if window == "expanding":
                if max_train_weeks:
                    train_start = dates[max(0, i - val_weeks - max_train_weeks)]
                else:
                    train_start = dates[0]
            else:
                train_start = dates[i - train_weeks - val_weeks]
            train_end = dates[i - val_weeks - 1]
            val_start = dates[i - val_weeks]
            val_end = dates[i - 1]

            val_mask = features.index.get_level_values("date").to_series().between(
                val_start, val_end
            ).values
            # Validation slice is needed whether we fit fresh or reuse a saved
            # bundle (it produces the val rank-IC used as the member skill weight).
            X_val = features.loc[val_mask].fillna(0.0)
            y_val = target["target"].reindex(X_val.index)
            valid_val = y_val.notna()
            X_val = X_val[valid_val]
            y_val = y_val[valid_val]

            # Resume: reload this step's models from a matching saved bundle.
            reused = False
            restamped = False
            if reuse_saved:
                cached = load_matching_bundle(
                    t, bundle_signal_day,
                    feature_columns=list(features.columns),
                    members=members,
                    train_weeks=train_weeks,
                    val_weeks=val_weeks,
                    retrain_freq=retrain_freq,
                    fingerprint=fingerprint,
                    allow_structural_restamp=resume_restamp,
                )
                if cached is not None:
                    current_models = dict(cached["models"])
                    reused = True
                    # Reused on a structural match despite a fingerprint delta →
                    # re-save below so the bundle carries the current fingerprint
                    # and future resumes match cleanly.
                    if (cached.get("extra") or {}).get("fingerprint") != fingerprint:
                        restamped = True

            if not reused:
                train_mask = features.index.get_level_values("date").to_series().between(
                    train_start, train_end
                ).values
                # Paper convention (page 13): "missing z-scores are replaced with
                # zero" after tickers with >10 missing factors are dropped.
                # neutralize_stacked handles the >10 drop AND fills the rest to 0
                # for non-passthrough columns, so this fillna(0) is a no-op for
                # current configs (no passthroughs). Kept explicit so that any
                # future passthrough column (e.g., re-enabled macro) cannot
                # silently gut the universe via row-level dropna — the only
                # legitimate row filter is on the target (y), enforced below.
                X_train = features.loc[train_mask].fillna(0.0)
                y_train = target["target"].reindex(X_train.index)
                valid_train = y_train.notna()
                X_train = X_train[valid_train]
                y_train = y_train[valid_train]
                log.info(
                    f"  retrain @ {t.date()}: train={len(X_train):,} rows "
                    f"({train_start.date()}→{train_end.date()}), "
                    f"val={len(X_val):,} rows ({val_start.date()}→{val_end.date()})"
                )
                for m in members:
                    fit_fn, _ = DISPATCH[m]
                    current_models[m] = fit_fn(model_cfg[m], X_train, y_train, X_val, y_val)
            else:
                log.info(
                    f"  retrain @ {t.date()}: reused saved bundle"
                    + (" (restamped to current fingerprint)" if restamped else "")
                    + f" (val={len(X_val):,} rows {val_start.date()}→{val_end.date()})"
                )

            # Validation rank-IC on the held-out val window — the same
            # cross-sectional objective the strategy trades on. Used as the
            # member's skill weight for IR-weighted combination. Computed only
            # from val rows (no leakage into the OOS test week). Recomputed even
            # on reuse so the IR weighting is identical to a fresh fit.
            y_val_arr = y_val.to_numpy()
            val_dates = X_val.index.get_level_values("date")
            for m in members:
                _, pred_fn = DISPATCH[m]
                # A member that trained on the val rows (random_forest) stashes
                # out-of-bag predictions for them; prefer those so its weight is
                # not inflated by an in-sample score. Everyone else is genuinely
                # held out here, so .predict() is already honest.
                val_pred = getattr(current_models[m], "_oob_val_pred", None)
                if val_pred is None or len(val_pred) != len(X_val):
                    val_pred = pred_fn(current_models[m], X_val)
                current_val_ic[m] = _rank_ic(val_pred, y_val_arr, val_dates)
            log.info(
                "    val rank-IC: "
                + ", ".join(f"{m}={current_val_ic[m]:.4f}" for m in members)
            )
            log.info(
                f"    train window {train_start.date()}→{train_end.date()} "
                f"({(train_end - train_start).days // 7}w), RSS={_rss_gb():.1f} GB"
            )

            if persist_retrains and (not reused or restamped):
                # Persist under the run's configured signal day (WED/THU/FRI) so
                # backtest retrains land in their own namespace and never clobber
                # another day's live production bundles written by src/predict.py.
                # The fingerprint lets a later resume verify config + data match.
                save_bundle(
                    retrain_date=t,
                    models=current_models,
                    members=members,
                    combine_method=combine_method,
                    feature_columns=list(features.columns),
                    train_weeks=train_weeks,
                    val_weeks=val_weeks,
                    retrain_freq=retrain_freq,
                    signal_day=bundle_signal_day,
                    extra={"fingerprint": fingerprint, "window": window},
                )

        # Predict for this week with each member. Match the training-time
        # convention: fill NaN with 0 (paper page 13) instead of dropping rows,
        # so that a single sparse passthrough column or a one-off NaN feature
        # cannot mask out the entire week's universe.
        week_data = features.loc[features.index.get_level_values("date") == t].fillna(0.0)
        if len(week_data) == 0:
            continue
        last_test_X = week_data
        for m in members:
            _, pred_fn = DISPATCH[m]
            preds = pd.Series(
                pred_fn(current_models[m], week_data),
                index=week_data.index,
                name="prediction",
            )
            per_model_preds[m].append(preds)
        val_ic_rows.append(pd.Series(dict(current_val_ic), name=t))

    per_model_series: dict[str, pd.Series] = {
        m: pd.concat(parts) for m, parts in per_model_preds.items()
    }
    val_ic_by_week = pd.DataFrame(val_ic_rows)
    val_ic_by_week.index.name = "date"
    return per_model_series, val_ic_by_week, current_models, last_test_X


def rolling_train_predict(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict | None = None,
    return_artifacts: bool = False,
):
    """Back-compat shim — returns the XGBoost member's predictions.

    Existing callers (diagnostics weekday_effect_analysis, tests) get the
    XGBoost-only predictions matching the legacy single-model API.
    """
    if cfg is None:
        cfg = load_config()

    # Force a single-member run regardless of cfg.model.type — callers of this
    # function want one model, not an ensemble.
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["type"] = "xgboost"

    per_model, _, last_models, last_test_X = rolling_train_predict_ensemble(
        features, target, cfg=cfg
    )
    predictions = per_model["xgboost"]

    if return_artifacts:
        return predictions, last_models["xgboost"], last_test_X
    return predictions


if __name__ == "__main__":
    print("Model training module — run via src.main or import directly.")
