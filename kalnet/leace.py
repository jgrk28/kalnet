"""Least-squares concept erasure (LEACE) closed form.

Belrose et al., 2023: the affine edit r(x) = P x + b that linearly guards Z
while minimizing mean squared change to X.

Written in covariances (equivalent to the paper's whitening form, Eq. 1):

    P = I - Σ_XZ (Σ_XZᵀ Σ_XX⁺ Σ_XZ)⁺ Σ_XZᵀ Σ_XX⁺
    b = μ_X - P μ_X

This is *not* the OLS residual X - Σ_XZ Σ_ZZ⁻¹ (Z - μ_Z), which needs Z at
transform time. LEACE is a function of X alone.
"""

from dataclasses import dataclass

import numpy as np
import torch


def _to_numpy(a: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(a, dtype=np.float64)


@dataclass
class LeaceEraser:
    """Fitted LEACE transform: r(x) = P x + b (column-vector convention)."""

    projection: np.ndarray  # (d, d)
    bias: np.ndarray  # (d,)

    @classmethod
    def fit(
        cls,
        X: torch.Tensor | np.ndarray,
        Z: torch.Tensor | np.ndarray,
    ) -> LeaceEraser:
        """Fit the LEACE eraser on probe samples (X, Z)."""
        X = _to_numpy(X)
        Z = _to_numpy(Z)
        if Z.ndim == 1:
            Z = Z[:, None]

        n, d = X.shape
        mean_x = X.mean(axis=0)
        Xc = X - mean_x
        Zc = Z - Z.mean(axis=0)

        sigma_xx = (Xc.T @ Xc) / n  # (d, d)
        sigma_xz = (Xc.T @ Zc) / n  # (d, k)

        # P = I - Σ_XZ (Σ_XZᵀ Σ_XX⁺ Σ_XZ)⁺ Σ_XZᵀ Σ_XX⁺
        xx_pinv = np.linalg.pinv(sigma_xx)
        inner = sigma_xz.T @ xx_pinv @ sigma_xz  # (k, k)
        projection = np.eye(d) - sigma_xz @ np.linalg.pinv(inner) @ sigma_xz.T @ xx_pinv
        bias = mean_x - projection @ mean_x
        return cls(projection=projection, bias=bias)

    def transform(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply the fitted eraser; returns a float32 tensor."""
        was_tensor = isinstance(X, torch.Tensor)
        device = X.device if was_tensor else torch.device("cpu")
        erased = _to_numpy(X) @ self.projection.T + self.bias
        out = torch.from_numpy(erased.astype(np.float32))
        return out.to(device) if was_tensor else out
