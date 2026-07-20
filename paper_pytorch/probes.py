"""Reusable Decoders and Per-timestep centering for hidden-state probes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = _PACKAGE_DIR / "data" / "kf_dataset.pt"
DEFAULT_CHECKPOINT = _PACKAGE_DIR / "kalman_checkpoints" / "kf_allgains.pt"


def trial_train_validation_indices(
    n_trials: int,
    validation_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Trials into training / validation index sets before pooling."""
    generator = torch.Generator().manual_seed(seed)
    trial_order = torch.randperm(n_trials, generator=generator)
    n_validation = round(validation_fraction * n_trials)
    return trial_order[n_validation:], trial_order[:n_validation]


def pool_timesteps(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten Trial × timestep into probe samples.

    (N, T, D) -> (N*T, D); (N, T) -> (N*T, 1).
    """
    if tensor.ndim == 2:
        return tensor.reshape(-1, 1).float()
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1]).float()
    raise ValueError(f"Expected rank 2 or 3 tensor, got shape {tuple(tensor.shape)}")


class PerTimestepCenterer:
    """Across-Trial means at each timestep, fit on train Trials only."""

    def __init__(self) -> None:
        self._means: dict[str, torch.Tensor] = {}

    def fit(self, **fields: torch.Tensor) -> PerTimestepCenterer:
        for name, value in fields.items():
            if value.ndim < 2:
                raise ValueError(f"{name}: expected (N, T, ...), got {tuple(value.shape)}")
            self._means[name] = value.mean(dim=0)
        return self

    def transform(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self._means:
            raise KeyError(f"No fitted mean for {name!r}")
        mean = self._means[name]
        if value.shape[1:] != mean.shape:
            raise ValueError(
                f"{name}: value trailing shape {tuple(value.shape[1:])} "
                f"!= mean shape {tuple(mean.shape)}"
            )
        return value - mean

    def mean(self, name: str) -> torch.Tensor:
        return self._means[name]


def select_trials(
    split: Mapping[str, torch.Tensor],
    trial_indices: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Optionally subset a split to the given Trial indices."""
    if trial_indices is None:
        return {key: value for key, value in split.items()}
    return {key: value[trial_indices] for key, value in split.items()}


def opt_precision(opt_variance: torch.Tensor) -> torch.Tensor:
    """Reciprocal of Opt variance; rejects non-finite values."""
    precision = opt_variance.reciprocal()
    if not torch.isfinite(precision).all():
        raise ValueError("Opt precision contains a non-finite value")
    return precision


class LinearDecoder:
    """Ridge regression probe onto a scalar target."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> LinearDecoder:
        self.model.fit(X.numpy(), y.reshape(-1).numpy())
        return self

    def predict(self, X: torch.Tensor) -> np.ndarray:
        return self.model.predict(X.numpy())

    def r2(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float(r2_score(y.reshape(-1).numpy(), self.predict(X)))

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_.reshape(-1)


class NonlinearDecoder:
    """One-hidden-layer ReLU MLP probe onto a scalar target."""

    def __init__(
        self,
        n_in: int,
        width: int = 64,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(n_in, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        ).to(self.device)
        self.best_epoch = 0

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_validation: torch.Tensor,
        y_validation: torch.Tensor,
        *,
        batch_size: int = 1024,
        max_epochs: int = 100,
        patience: int = 10,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> NonlinearDecoder:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        y_train = y_train.reshape(-1, 1).float()
        y_validation = y_validation.reshape(-1, 1).float()
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        best_validation_r2 = -np.inf
        best_state: Optional[dict[str, torch.Tensor]] = None
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            validation_r2 = self.r2(X_validation, y_validation)
            if validation_r2 > best_validation_r2 + 1e-5:
                best_validation_r2 = validation_r2
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is None:
            raise RuntimeError("Nonlinear decoder training produced no checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device).eval()
        return self

    def predict(self, X: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = X[start : start + batch_size].to(self.device)
                predictions.append(self.model(batch).cpu())
        return torch.cat(predictions).squeeze(1).numpy()

    def r2(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return float(r2_score(y.reshape(-1).numpy(), self.predict(X)))
