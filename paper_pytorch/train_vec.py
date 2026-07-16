from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import KalmanRNN
from task import Batch, GainCondition, KalmanFilteringTask

DEFAULT_CHECKPOINT = Path("kalman_checkpoints/kf_allgains.pt")


def fractional_rmse(targets, preds, opt_means):
    """Return (frac_rmse, rmse_opt, rmse_net), pooled over EVERY timestep in the
    provided arrays (CHANGED: previously final-timestep-only)."""
    rmse_opt = float(np.sqrt(np.nanmean((targets - opt_means) ** 2)))
    rmse_net = float(np.sqrt(np.nanmean((targets - preds) ** 2)))
    frac = (rmse_net - rmse_opt) / rmse_opt if rmse_opt > 0 else float("nan")
    return frac, rmse_opt, rmse_net


def batch_to_tensors(batch: Batch, device: torch.device):
    # already torch.Tensors from this project's task.py -- no torch.from_numpy()
    x = batch.input.to(device)
    y = batch.target.to(device)
    opt = batch.opt_mean.to(device)
    return x, y, opt


@torch.no_grad()
def evaluate(model: KalmanRNN, task: KalmanFilteringTask, device: torch.device):
    model.eval()
    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    opts: List[np.ndarray] = []

    for _, batch in task:
        x, y, opt = batch_to_tensors(batch, device)
        out = model(x)
        # CHANGED: keep every timestep (was y[:, -1, 0]) -- flatten (batch, T) -> 1D
        targets.append(y[:, :, 0].cpu().numpy().reshape(-1))
        preds.append(out[:, :, 0].cpu().numpy().reshape(-1))
        opts.append(opt[:, :, 0].cpu().numpy().reshape(-1))

    return fractional_rmse(
        np.concatenate(targets), np.concatenate(preds), np.concatenate(opts)
    )


def train(
    *,
    n_in: int = 50,
    n_hid: int = 200,
    stim_dur: int = 25,
    batch_size: int = 100,              # CHANGED (was 10)
    max_iter: int = 20_000,             # CHANGED (was 50_001) -- see note below
    test_iter: int = 500,               # CHANGED (was 2_501) -- fewer batches needed at batch_size=100
    lr: float = 3e-4,                   # CHANGED (was 2e-4)
    l2_penalty: float = 1e-6,           # NEW
    tr_cond: GainCondition = "all_gains",
    test_cond: GainCondition = "all_gains",
    log_every: int = 500,
    seed: int = 0,
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_t = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = KalmanRNN(n_in=n_in, n_hid=n_hid, n_out=1).to(device_t)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)  # CHANGED: weight_decay added

    train_task = KalmanFilteringTask(
        max_iter=max_iter, batch_size=batch_size, n_in=n_in,
        stim_dur=stim_dur, tr_cond=tr_cond, seed=seed,
    )
    test_task = KalmanFilteringTask(
        max_iter=test_iter, batch_size=batch_size, n_in=n_in,
        stim_dur=stim_dur, tr_cond=test_cond, seed=seed + 1,
    )

    frac_rmse_vec: List[float] = []
    s_buf: List[np.ndarray] = []
    opt_buf: List[np.ndarray] = []
    pred_buf: List[np.ndarray] = []

    model.train()
    for i, batch in train_task:
        x, y, opt = batch_to_tensors(batch, device_t)
        pred = model(x)

        # CHANGED: full-sequence loss (was pred[:, -1, :] vs y[:, -1, :])
        loss = F.mse_loss(pred, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        with torch.no_grad():
            # CHANGED: accumulate every timestep, not just the last
            s_buf.append(y[:, :, 0].cpu().numpy().reshape(-1))
            opt_buf.append(opt[:, :, 0].cpu().numpy().reshape(-1))
            pred_buf.append(pred[:, :, 0].cpu().numpy().reshape(-1))

        if i % log_every == 0:
            frac, rmse_opt, rmse_net = fractional_rmse(
                np.concatenate(s_buf), np.concatenate(pred_buf), np.concatenate(opt_buf)
            )
            frac_rmse_vec.append(frac)
            print(f"Batch #{i}; Frac. RMSE: {frac:.6f}; Opt. RMSE: {rmse_opt:.6f}; Net. RMSE: {rmse_net:.6f}")
            s_buf, opt_buf, pred_buf = [], [], []

    frac_test, rmse_opt_t, rmse_net_t = evaluate(model, test_task, device_t)
    print(f"Test data; Frac. RMSE: {frac_test:.6f}; Opt. RMSE: {rmse_opt_t:.6f}; Net. RMSE: {rmse_net_t:.6f}")

    result = {
        "frac_rmse_vec": np.asarray(frac_rmse_vec),
        "frac_rmse_test": frac_test,
        "rmse_opt_test": rmse_opt_t,
        "rmse_net_test": rmse_net_t,
        "state_dict": model.state_dict(),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "frac_rmse_vec": result["frac_rmse_vec"],
            "frac_rmse_test": frac_test,
            "config": {
                "n_in": n_in, "n_hid": n_hid, "stim_dur": stim_dur,
                "batch_size": batch_size, "lr": lr, "l2_penalty": l2_penalty,
                "tr_cond": tr_cond, "test_cond": test_cond, "seed": seed,
                "log_every": log_every, "max_iter": max_iter,
            },
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    return result


if __name__ == "__main__":
    # test: short run, no save, to verify everything is wired correctly.
    train(max_iter=1000, test_iter=20, log_every=200, save_path=None)
