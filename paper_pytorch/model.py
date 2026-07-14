"""Vanilla ReLU RNN for the Kalman filtering task (Orhan & Ma 2017).

Ported from ``ffwd/ffwd_kalman_filtering_expt.py``, which builds a Lasagne
``CustomRecurrentLayer`` equivalent to:

    h_t = ReLU(W_in @ r_t + W_rec @ h_{t-1} + b)
    y_t = w @ h_t + b_out
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanRNN(nn.Module):
    """Generic recurrent net used for Kalman filtering in the paper.

    Parameters
    ----------
    n_in:
        Input population size (default 50).
    n_hid:
        Number of ReLU recurrent units (default 200).
    n_out:
        Readout dimension (default 1).
    """

    def __init__(self, n_in: int = 50, n_hid: int = 200, n_out: int = 1) -> None:
        super().__init__()
        # Match Lasagne: input→hid has no bias; hid→hid carries the bias.
        self.W_in = nn.Linear(n_in, n_hid, bias=False)
        self.W_rec = nn.Linear(n_hid, n_hid, bias=True)
        self.readout = nn.Linear(n_hid, n_out, bias=True)
        self.n_hid = n_hid
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Lasagne DenseLayer default is Glorot uniform.
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        nn.init.zeros_(self.W_rec.bias)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Run the RNN over a batch of sequences.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, time, n_in)``.
        return_hidden:
            If True, also return hidden states ``(batch, time, n_hid)``.

        Returns
        -------
        y:
            Readouts of shape ``(batch, time, n_out)``.
        h_seq (optional):
            Hidden trajectory if ``return_hidden``.
        """
        batch, time, _ = x.shape
        h = x.new_zeros(batch, self.n_hid)
        outputs = []
        hiddens = [] if return_hidden else None

        for t in range(time):
            h = F.relu(self.W_in(x[:, t]) + self.W_rec(h))
            outputs.append(self.readout(h))
            if return_hidden:
                hiddens.append(h)

        y = torch.stack(outputs, dim=1)
        if return_hidden:
            return y, torch.stack(hiddens, dim=1)
        return y
