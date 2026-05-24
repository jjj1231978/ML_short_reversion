"""PyTorch MLP regressor for the rolling ensemble.

Slots into `src.model.train` via the DISPATCH dict alongside XGBoost, LightGBM,
and RandomForest. Per-fold fit:
    1. Fit a StandardScaler on X_train (only) — no leakage from val/test.
    2. Build an MLP with `hidden_dims` layers, dropout, batchnorm.
    3. Train with Adam + MSE, early-stop on val MSE.
    4. Return a `MLPWrapper` whose `.predict(X)` applies the scaler and runs
       the net in eval mode.

The wrapper exposes the .scaler attribute so the SHAP integration (which
operates in standardized feature space) can apply the same transform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    batchnorm: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 15
    device: str | None = None        # "cuda", "cpu", or None for auto
    seed: int = 42
    grad_clip: float = 1.0
    activation: str = "relu"         # "relu" | "gelu" | "tanh"

    @classmethod
    def from_dict(cls, d: dict) -> "MLPConfig":
        # Defensive: only carry fields the dataclass knows about; ignore
        # extras so configs/default.yaml can document fields not yet wired.
        known = {f.name for f in fields_of(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def fields_of(cls):
    """Tiny shim around dataclasses.fields so callers don't need the import."""
    import dataclasses
    return dataclasses.fields(cls)


class MLPWrapper:
    """Holds the fitted scaler + module + device. Exposes .predict(X)."""

    def __init__(self, scaler, module, device, feature_names: list[str]):
        self.scaler = scaler
        self.module = module
        self.device = device
        self.feature_names = list(feature_names)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch
        arr = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # Match StandardScaler's NaN handling — neutralization should have
        # zero-filled, but defensive against caller skipping it.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(arr).astype(np.float32)
        self.module.eval()
        with torch.no_grad():
            t = torch.from_numpy(Xs).to(self.device)
            yhat = self.module(t).cpu().numpy().ravel()
        return yhat


def _build_module(input_dim: int, cfg: MLPConfig):
    """Construct an `nn.Sequential` per `cfg`. Imported lazily."""
    import torch.nn as nn

    act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    act_cls = act_map.get(cfg.activation, nn.ReLU)

    layers: list[Any] = []
    prev = input_dim
    for h in cfg.hidden_dims:
        layers.append(nn.Linear(prev, h))
        if cfg.batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act_cls())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _resolve_device(spec: str | None):
    import torch
    if spec in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def fit_mlp(
    cfg: MLPConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> MLPWrapper:
    """Train an MLP on (X_train, y_train), early-stop on (X_val, y_val).

    Standardization is fit on training rows only (not val) so the val MSE
    we early-stop on is honest.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _resolve_device(cfg.device)
    log.info(
        f"    MLP fit: in_dim={X_train.shape[1]}, train={len(X_train):,}, "
        f"val={len(X_val):,}, device={device}, hidden={cfg.hidden_dims}"
    )

    scaler = StandardScaler()
    Xt = scaler.fit_transform(np.nan_to_num(X_train.values)).astype(np.float32)
    Xv = scaler.transform(np.nan_to_num(X_val.values)).astype(np.float32)
    yt = y_train.values.astype(np.float32)
    yv = y_val.values.astype(np.float32)

    module = _build_module(X_train.shape[1], cfg).to(device)
    optim = torch.optim.AdamW(module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # Convert once, keep on device for speed.
    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(yt).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(yv).to(device)

    n_train = Xt_t.shape[0]
    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(cfg.max_epochs):
        module.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            # Skip the final partial batch when it would have <2 samples —
            # BatchNorm requires >1 sample per channel during training, so
            # any single-sample tail batch otherwise crashes the epoch.
            if idx.numel() < 2:
                continue
            xb, yb = Xt_t[idx], yt_t[idx]
            optim.zero_grad()
            yhat = module(xb).squeeze(-1)
            loss = loss_fn(yhat, yb)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            optim.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n_train

        module.eval()
        with torch.no_grad():
            val_pred = module(Xv_t).squeeze(-1)
            val_loss = loss_fn(val_pred, yv_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                log.info(f"      early-stop at epoch {epoch} (val={best_val:.6f})")
                break

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()

    return MLPWrapper(scaler, module, device, list(X_train.columns))


def predict_mlp(model: MLPWrapper, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def fit_from_member_cfg(member_cfg: dict, X_train, y_train, X_val, y_val) -> MLPWrapper:
    """Adapter matching the DISPATCH dict's (member_cfg, X_train, ...) signature."""
    raw = dict(member_cfg.get("params", {}))
    cfg = MLPConfig.from_dict(raw)
    return fit_mlp(cfg, X_train, y_train, X_val, y_val)
