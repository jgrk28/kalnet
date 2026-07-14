"""Kalman filtering task from Orhan & Ma (2017).

Ported from ``ffwd/generators.py`` (``KalmanFilteringTaskFFWD``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple

import numpy as np

GainCondition = Literal["all_gains", "two_gains"]


@dataclass
class Batch:
    """One minibatch of Kalman filtering trials.

    Shapes are ``(batch, time, features)``.
    """

    input: np.ndarray  # Poisson population responses, (B, T, n_in)
    target: np.ndarray  # true latent s_t, (B, T, 1)
    opt_mean: np.ndarray  # Kalman posterior mean, (B, T, 1)


class KalmanFilteringTask:
    """Online generator for the 1-D Kalman filtering psychophysics task.

    Latent dynamics:
        s_t = (1 - gamma) * s_{t-1} + eta_t,  eta ~ N(0, signu_sq)

    Observations are a population of Poisson neurons with Gaussian tuning
    (preferred stimuli ``phi``), scaled by a per-timestep gain ``g``.
    The network is supervised with the true stimulus ``s``, not the optimal
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
    ) -> None:
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_in = n_in
        self.stim_dur = stim_dur
        self.sigtc_sq = sigtc_sq
        self.signu_sq = signu_sq
        self.gamma = gamma
        self.tr_cond = tr_cond
        self.phi = np.linspace(-9.0, 9.0, n_in)
        self._num_iter = 0
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Tuple[int, Batch]]:
        self._num_iter = 0
        return self

    def __next__(self) -> Tuple[int, Batch]:
        if self.max_iter is not None and self._num_iter >= self.max_iter:
            raise StopIteration
        idx = self._num_iter
        self._num_iter += 1
        return idx, self.sample()

    def sample(self) -> Batch:
        B, T, N = self.batch_size, self.stim_dur, self.n_in
        rng = self._rng

        eta = np.sqrt(self.signu_sq) * rng.standard_normal((T, B))
        if self.tr_cond == "all_gains":
            g = (3.0 - 0.3) * rng.random((T, B)) + 0.3
        else:
            g = rng.choice([0.3, 3.0], size=(T, B))

        r = np.zeros((N, T, B))
        s = np.zeros((1, T, B))
        m = np.zeros((1, T, B))
        sig_sq = np.zeros((1, T, B))

        a_in = np.ones(N)
        b_in = self.phi
        scale = np.sqrt(2.0 * self.sigtc_sq)

        for ii in range(B):
            s[0, 0, ii] = np.sqrt(self.signu_sq) * rng.standard_normal()
            rate = g[0, ii] * np.exp(-(((s[0, 0, ii] - self.phi) / scale) ** 2))
            r[:, 0, ii] = rng.poisson(rate)

            a_dot = float(np.dot(a_in, r[:, 0, ii]))
            b_dot = float(np.dot(b_in, r[:, 0, ii]))
            m[0, 0, ii] = b_dot / (a_dot + (self.sigtc_sq / self.signu_sq))
            sig_sq[0, 0, ii] = 1.0 / (a_dot / self.sigtc_sq + (1.0 / self.signu_sq))

            for tt in range(1, T):
                s[0, tt, ii] = (1.0 - self.gamma) * s[0, tt - 1, ii] + eta[tt, ii]
                rate = g[tt, ii] * np.exp(
                    -(((s[0, tt, ii] - self.phi) / scale) ** 2)
                )
                r[:, tt, ii] = rng.poisson(rate)

                a_dot = float(np.dot(a_in, r[:, tt, ii]))
                b_dot = float(np.dot(b_in, r[:, tt, ii]))
                k = self.signu_sq + (1.0 - self.gamma) ** 2 * sig_sq[0, tt - 1, ii]
                m[0, tt, ii] = (
                    b_dot * k + (1.0 - self.gamma) * m[0, tt - 1, ii] * self.sigtc_sq
                ) / (a_dot * k + self.sigtc_sq)
                sig_sq[0, tt, ii] = (self.sigtc_sq * k) / (a_dot * k + self.sigtc_sq)

        # (n_in, T, B) / (1, T, B) -> (B, T, *)
        return Batch(
            input=np.swapaxes(r, 0, 2).astype(np.float32),
            target=np.swapaxes(s, 0, 2).astype(np.float32),
            opt_mean=np.swapaxes(m, 0, 2).astype(np.float32),
        )
