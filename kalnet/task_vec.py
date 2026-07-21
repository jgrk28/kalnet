from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple

import torch

GainCondition = Literal["all_gains", "two_gains"]


@dataclass
class BatchInternals:
    """Optional generative and filter quantities for a Batch. All (B, T, 1) except
    expected_rate, which is (B, T, n_in)."""

    gain: torch.Tensor          # hidden gain, (B, T, 1)
    expected_rate: torch.Tensor  # Poisson means, (B, T, n_in)
    opt_var: torch.Tensor       # Kalman posterior variance, (B, T, 1)


@dataclass
class Batch:
    """One minibatch of Kalman filtering trials. Shapes are (batch, time, features)."""

    input: torch.Tensor                       # Poisson population responses, (B, T, n_in)
    target: torch.Tensor                      # true latent s_t, (B, T, 1)
    opt_mean: torch.Tensor                    # Kalman posterior mean, (B, T, 1)
    internals: Optional[BatchInternals] = None


class KalmanFilteringTask:
    """Online generator for the 1-D Kalman filtering psychophysics task.

    Latent dynamics:
        s_t = (1 - gamma) * s_{t-1} + eta_t,   eta ~ N(0, signu_sq)

    Observations are a population of Poisson neurons with Gaussian tuning
    (preferred stimuli `phi`), scaled by a per-timestep gain `g`.

    The network is supervised with the true stimulus `s`, not the optimal
    posterior mean (which is returned only for evaluation).
    """

    def __init__(
        self,
        max_iter: Optional[int] = None,
        batch_size: int = 10,
        n_in: int = 50,
        stim_dur: int = 25,
        sigtc_sq: float = 4.0,
        signu_sq: float = 1.0,
        gamma: float = 0.1,
        tr_cond: GainCondition = "all_gains",
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_in = n_in
        self.stim_dur = stim_dur
        self.sigtc_sq = sigtc_sq
        self.signu_sq = signu_sq
        self.gamma = gamma
        self.tr_cond = tr_cond
        self.device = device

        # Kept as in the file this was ported from: a single, evenly-spaced
        # linspace, NOT this project's own task.py's two-piece replication.
        self.phi = torch.linspace(-9.0, 9.0, n_in, device=device)

        self._num_iter = 0
        if seed is not None:
            self._generator = torch.Generator(device=device).manual_seed(seed)
        else:
            self._generator = None

    def __iter__(self) -> Iterator[Tuple[int, Batch]]:
        self._num_iter = 0
        return self

    def __next__(self) -> Tuple[int, Batch]:
        if self.max_iter is not None and self._num_iter >= self.max_iter:
            raise StopIteration
        idx = self._num_iter
        self._num_iter += 1
        return idx, self.sample()

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self._generator)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self._generator)

    def _randint(self, low, high, shape):
        return torch.randint(low, high, shape, device=self.device, generator=self._generator)

    def sample(self, *, include_internals: bool = False) -> Batch:
        B, T, N = self.batch_size, self.stim_dur, self.n_in
        device = self.device

        # ---- Latent state trajectory: vectorized across the batch, looped only
        # over time (T iterations total, not B*T) ----
        s = torch.zeros(B, T, device=device)
        s[:, 0] = self._randn(B) * (self.signu_sq ** 0.5)
        eta = self._randn(B, T) * (self.signu_sq ** 0.5)
        for t in range(1, T):
            s[:, t] = (1.0 - self.gamma) * s[:, t - 1] + eta[:, t]

        # ---- Gain: vectorized, independent per (trial, timestep) ----
        if self.tr_cond == "all_gains":
            g = (3.0 - 0.3) * self._rand(B, T) + 0.3
        elif self.tr_cond == "two_gains":
            choices = torch.tensor([0.3, 3.0], device=device)
            idx = self._randint(0, 2, (B, T))
            g = choices[idx]
        else:
            raise ValueError(f"Unknown tr_cond: {self.tr_cond}")

        # ---- Poisson input population responses: one vectorized call per
        # timestep, all B trials computed simultaneously ----
        diff = s.unsqueeze(-1) - self.phi                      # (B, T, N)
        rate = g.unsqueeze(-1) * torch.exp(-(diff ** 2) / (2.0 * self.sigtc_sq))
        r = torch.poisson(rate)                                # (B, T, N)

        # ---- Ground-truth Kalman recursion (Eqs. 7-8): vectorized across the
        # batch, looped only over time ----
        sum_r = r.sum(dim=-1)                     # (B, T)  = dot(a_in, r)
        weighted_sum_r = (r * self.phi).sum(dim=-1)  # (B, T)  = dot(b_in, r)

        m = torch.zeros(B, T, device=device)
        sig_sq = torch.zeros(B, T, device=device)

        m[:, 0] = weighted_sum_r[:, 0] / (sum_r[:, 0] + self.sigtc_sq / self.signu_sq)
        sig_sq[:, 0] = 1.0 / (sum_r[:, 0] / self.sigtc_sq + 1.0 / self.signu_sq)

        for t in range(1, T):
            k = self.signu_sq + (1.0 - self.gamma) ** 2 * sig_sq[:, t - 1]
            m[:, t] = (
                weighted_sum_r[:, t] * k + (1.0 - self.gamma) * m[:, t - 1] * self.sigtc_sq
            ) / (sum_r[:, t] * k + self.sigtc_sq)
            sig_sq[:, t] = (self.sigtc_sq * k) / (sum_r[:, t] * k + self.sigtc_sq)

        internals = None
        if include_internals:
            internals = BatchInternals(
                gain=g.unsqueeze(-1),
                expected_rate=rate,
                opt_var=sig_sq.unsqueeze(-1),
            )

        return Batch(
            input=r,
            target=s.unsqueeze(-1),
            opt_mean=m.unsqueeze(-1),
            internals=internals,
        )


if __name__ == "__main__":
    # Quick check: results should look statistically identical to the original
    # nested-loop version, just computed via vectorized tensor ops.
    task = KalmanFilteringTask(batch_size=500, seed=0, tr_cond="all_gains")
    batch = task.sample(include_internals=True)

    print("input shape:      ", tuple(batch.input.shape))
    print("target shape:      ", tuple(batch.target.shape))
    print("opt_mean shape:    ", tuple(batch.opt_mean.shape))
    print("opt_var shape:     ", tuple(batch.internals.opt_var.shape))
    print("expected_rate shape:", tuple(batch.internals.expected_rate.shape))

    rmse_opt = torch.sqrt(torch.mean((batch.opt_mean - batch.target) ** 2)).item()
    print(f"\nOptimal filter RMSE vs true s_t: {rmse_opt:.3f}  (sanity check: should be small, nonzero)")
    print(f"Mean opt_var at t=0:  {batch.internals.opt_var[:, 0, 0].mean().item():.3f}")
    print(f"Mean opt_var at t=24: {batch.internals.opt_var[:, -1, 0].mean().item():.3f}")
