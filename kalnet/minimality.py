"""Minimality / input-decodability analysis.

Sufficiency (the existing decoder analysis) asks: is the posterior mean/
precision linearly extractable from r_hid? Minimality asks the complementary
question, inspired by Kalburge & Lengyel (2025)'s functional information
bottleneck (fIB): has the network actually compressed away everything except
what's needed to represent the posterior, or does r_hid still linearly carry
the raw input around (a "heuristic recoding" rather than a genuine
sufficient-statistic code)?

We test this by training the same kind of probe used for opt-precision, but
targeting the raw input signal x_{t-lag} instead. Good posterior decodability
alongside good input decodability (especially at short lags) indicates a
non-minimal, input-preserving code; good posterior decodability with poor
input decodability indicates compression.

This module only imports from kalnet.decoders -- it does not modify it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kalnet.decoders import LinearDecoder, NonlinearDecoder, pool_timesteps


def build_lagged_dataset(
    r_hid: torch.Tensor,
    x: torch.Tensor,
    lag: int,
    burn_in: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair hidden state at time t with a (raw/summary) input signal at time t-lag.

    r_hid: (N, T, n_hid)
    x: (N, T) or (N, T, 1) -- e.g. mean input-population response per timestep
    burn_in: first timestep to evaluate at; must be >= lag so t-lag is valid.

    Returns pooled (X, y) with trial and timestep dims flattened together,
    using the same convention as `pool_timesteps` in kalnet.decoders.
    """
    if burn_in < lag:
        raise ValueError(f"burn_in ({burn_in}) must be >= lag ({lag})")
    if x.ndim == 3:
        x = x.squeeze(-1)
    T = r_hid.shape[1]
    if burn_in >= T:
        raise ValueError(f"burn_in ({burn_in}) must be < T ({T})")

    r_hid_valid = r_hid[:, burn_in:, :]
    x_lagged = x[:, burn_in - lag : T - lag]
    return pool_timesteps(r_hid_valid), pool_timesteps(x_lagged)


def center_per_timestep(x: torch.Tensor, train_mean: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the across-trial mean at each timestep.

    If train_mean is given, uses it (for val/test splits); otherwise computes
    and returns it (fit on this split, intended for the train split only).
    x: (N, T) or (N, T, 1).
    """
    squeeze = x.ndim == 3
    if squeeze:
        x = x.squeeze(-1)
    mean = train_mean if train_mean is not None else x.mean(dim=0)
    centered = x - mean
    return (centered.unsqueeze(-1) if squeeze else centered), mean


@dataclass(frozen=True)
class LagResult:
    lag: int
    linear_r2: float
    nonlinear_r2: float


def run_lag_sweep(
    r_hid_train: torch.Tensor,
    x_train: torch.Tensor,
    r_hid_validation: torch.Tensor,
    x_validation: torch.Tensor,
    r_hid_test: torch.Tensor,
    x_test: torch.Tensor,
    lags: list[int],
    burn_in: int,
    fit_nonlinear: bool = True,
) -> list[LagResult]:
    """Fit a decoder r_hid,t -> x_{t-lag} for each lag; report test R^2.

    All x_* should already be per-timestep centered (see `center_per_timestep`),
    using means fit on the train split only, mirroring the sufficiency
    analysis's PerTimestepCenterer convention.
    """
    results = []
    for lag in lags:
        X_train, y_train = build_lagged_dataset(r_hid_train, x_train, lag, burn_in)
        X_validation, y_validation = build_lagged_dataset(
            r_hid_validation, x_validation, lag, burn_in
        )
        X_test, y_test = build_lagged_dataset(r_hid_test, x_test, lag, burn_in)

        linear = LinearDecoder().fit(X_train, y_train)
        linear_r2 = linear.r2(X_test, y_test)

        nonlinear_r2 = float("nan")
        if fit_nonlinear:
            nonlinear = NonlinearDecoder(n_in=X_train.shape[1], seed=lag).fit(
                X_train, y_train, X_validation, y_validation
            )
            nonlinear_r2 = nonlinear.r2(X_test, y_test)

        results.append(LagResult(lag=lag, linear_r2=linear_r2, nonlinear_r2=nonlinear_r2))
    return results
