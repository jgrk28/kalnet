"""Statistic correlation: Mean activity and Kurtosis vs Opt variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, pearsonr
import torch


@dataclass(frozen=True)
class StatisticCorrelationResult:
    """Pooled summary stats and their Pearson correlations with Opt variance."""

    mean_activity: np.ndarray
    kurtosis: np.ndarray
    opt_variance: np.ndarray
    r_mean: float
    p_mean: float
    r_kurtosis: float
    p_kurtosis: float


def pool_hidden_and_opt_variance(
    r_hid: torch.Tensor | np.ndarray,
    opt_variance: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool Trial × timestep samples.

    Accepts (N, T, n_hid) / (N, T) or already-flat (n_samples, n_hid) / (n_samples,).
    """
    if isinstance(r_hid, torch.Tensor):
        r_hid = r_hid.detach().cpu().numpy()
    if isinstance(opt_variance, torch.Tensor):
        opt_variance = opt_variance.detach().cpu().numpy()

    if r_hid.ndim == 3:
        r_hid = r_hid.reshape(-1, r_hid.shape[-1])
    elif r_hid.ndim != 2:
        raise ValueError(f"r_hid expected rank 2 or 3, got shape {r_hid.shape}")

    if opt_variance.ndim == 2:
        opt_variance = opt_variance.reshape(-1)
    elif opt_variance.ndim != 1:
        raise ValueError(
            f"opt_variance expected rank 1 or 2, got shape {opt_variance.shape}"
        )

    if r_hid.shape[0] != opt_variance.shape[0]:
        raise ValueError(
            f"sample count mismatch: r_hid has {r_hid.shape[0]}, "
            f"opt_variance has {opt_variance.shape[0]}"
        )
    return r_hid.astype(np.float64, copy=False), opt_variance.astype(
        np.float64, copy=False
    )


def hidden_layer_stats(r_hid_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean activity and Fisher kurtosis per sample, across the unit dimension."""
    mean_activity = r_hid_flat.mean(axis=1)
    # fisher=True: 0 for Gaussian; positive = more peaked/sparse than Gaussian
    kurt = kurtosis(r_hid_flat, axis=1, fisher=True)
    return mean_activity, kurt


def correlate_hidden_stats(
    r_hid: torch.Tensor | np.ndarray,
    opt_variance: torch.Tensor | np.ndarray,
) -> StatisticCorrelationResult:
    """Correlate Mean activity and Kurtosis with Opt variance (no Decoder)."""
    r_hid_flat, sigma_sq = pool_hidden_and_opt_variance(r_hid, opt_variance)
    mean_activity, kurt = hidden_layer_stats(r_hid_flat)
    r_mean, p_mean = pearsonr(mean_activity, sigma_sq)
    r_kurt, p_kurt = pearsonr(kurt, sigma_sq)
    return StatisticCorrelationResult(
        mean_activity=mean_activity,
        kurtosis=kurt,
        opt_variance=sigma_sq,
        r_mean=float(r_mean),
        p_mean=float(p_mean),
        r_kurtosis=float(r_kurt),
        p_kurtosis=float(p_kurt),
    )
