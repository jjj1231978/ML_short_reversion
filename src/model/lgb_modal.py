"""Modal CPU fan-out for the LightGBM random-hyperparameter ensemble.

Standalone on purpose (like ``src/data/finbert_modal.py``): the only top-level
import is ``modal``, so a worker container reconstructing this module does NOT
drag in the repo's ``src.*`` package (absent from the image). lightgbm / numpy /
scipy are imported inside the remote function; the local driver
(``src.model.train.fit_lgb_bag_modal``) builds the payloads and collects results.

Each worker trains a chunk of candidate hyperparameter sets on the same
train/val window — broadcast once via ``.map(..., kwargs=...)`` — scores each by
validation rank-IC, and **writes the trained boosters to a Modal Volume**,
returning only ``{idx, val_ic}`` (tiny). The driver reads back just the global
top-N booster files from the volume.

Why the volume instead of returning the boosters directly: a worker's ~20
boosters serialize to tens of MB, which Modal pushes through result-blob storage
— whose download RPC fails for this client (``BlobGet not implemented``). Staging
through a volume keeps every ``.map`` return small and sidesteps that path.

CPU, not GPU: the pip ``lightgbm`` wheel is CPU-only and GPU offers no speedup at
this data size; the win is horizontal fan-out across containers.
"""

import modal

image = modal.Image.debian_slim().pip_install(
    "lightgbm>=4.0",
    "numpy>=1.26",
    "scipy>=1.11",
)
app = modal.App("lgb-random-ensemble", image=image)

# Staging volume for trained boosters (driver writes nothing here; workers write,
# driver reads the selected files back locally via vol.read_file).
vol = modal.Volume.from_name("lgb-ensemble-stage", create_if_missing=True)
STAGE = "/stage"


def _rank_ic(pred, y, codes) -> float:
    """Mean cross-sectional Spearman rank-IC, grouped by ``codes`` (date ids).
    Weeks with <5 names or no rank dispersion are skipped. NaN if none qualify.
    Self-contained (no src.* import) so it runs inside the worker."""
    import numpy as np
    from scipy.stats import rankdata

    ics = []
    for g in np.unique(codes):
        m = codes == g
        if m.sum() < 5:
            continue
        rp, ry = rankdata(pred[m]), rankdata(y[m])
        if rp.std() == 0 or ry.std() == 0:
            continue
        ics.append(float(np.corrcoef(rp, ry)[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


# cpu=8 → lightgbm uses ~8 threads per fit; max_containers=10 caps the fan-out at
# 10 concurrent workers (chunks queue if the autoscaler is slower to warm).
@app.function(cpu=8.0, timeout=60 * 60, max_containers=10, volumes={STAGE: vol})
def train_chunk(
    task: dict,
    *,
    Xtr,
    ytr,
    Xval,
    yval,
    val_codes,
    n_estimators: int,
    early_stop: int,
) -> list:
    """Train this worker's candidates on the shared window, save each booster
    (truncated to best_iteration) to the volume under ``{run_id}/{idx}.txt``, and
    return ``{idx, val_ic}`` per candidate."""
    import os

    import lightgbm as lgb
    import numpy as np

    run_id = task["run_id"]
    items = task["items"]  # list of [idx, params]
    out_dir = f"{STAGE}/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    ds_params = {"feature_pre_filter": False}
    train_set = lgb.Dataset(Xtr, label=ytr, params=ds_params)
    val_set = lgb.Dataset(Xval, label=yval, reference=train_set, params=ds_params)
    yval = np.asarray(yval)
    val_codes = np.asarray(val_codes)

    out = []
    for idx, params in items:
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        best = booster.best_iteration or n_estimators
        booster.save_model(f"{out_dir}/{idx}.txt", num_iteration=best)
        pred = np.asarray(booster.predict(Xval, num_iteration=best))
        out.append({"idx": int(idx), "val_ic": _rank_ic(pred, yval, val_codes)})

    vol.commit()  # publish this worker's booster files for the driver to read
    return out
