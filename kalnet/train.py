"""Train / evaluate the Kalman filtering RNN (Orhan & Ma 2017).

Matches the defaults in ``ffwd/ffwd_kalman_filtering_expt.py``:
  - Adam, lr=2e-4
  - MSE on the final timestep only
  - 50k training batches, batch size 10
  - fractional RMSE vs the optimal Kalman mean for logging

Example
-------
    python -m kalnet.train --max-iter 50001 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import KalmanRNN
from .task import Batch, GainCondition, KalmanFilteringTask

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "kf_default.pt"


def fractional_rmse(
    targets: np.ndarray,
    preds: np.ndarray,
    opt_means: np.ndarray,
) -> tuple[float, float, float]:
    """Return (frac_rmse, rmse_opt, rmse_net) on final-timestep scalars."""
    rmse_opt = float(np.sqrt(np.nanmean((targets - opt_means) ** 2)))
    rmse_net = float(np.sqrt(np.nanmean((targets - preds) ** 2)))
    frac = (rmse_net - rmse_opt) / rmse_opt if rmse_opt > 0 else float("nan")
    return frac, rmse_opt, rmse_net


def batch_to_tensors(
    batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(batch.input).to(device)
    y = torch.from_numpy(batch.target).to(device)
    opt = torch.from_numpy(batch.opt_mean).to(device)
    return x, y, opt


@torch.no_grad()
def evaluate(
    model: KalmanRNN,
    task: KalmanFilteringTask,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    targets: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    opts: List[np.ndarray] = []
    for _, batch in task:
        x, y, opt = batch_to_tensors(batch, device)
        out = model(x)
        targets.append(y[:, -1, 0].cpu().numpy())
        preds.append(out[:, -1, 0].cpu().numpy())
        opts.append(opt[:, -1, 0].cpu().numpy())
    return fractional_rmse(
        np.concatenate(targets),
        np.concatenate(preds),
        np.concatenate(opts),
    )


def train(
    *,
    n_in: int = 50,
    n_hid: int = 200,
    stim_dur: int = 25,
    batch_size: int = 10,
    max_iter: int = 50_001,
    test_iter: int = 2_501,
    lr: float = 2e-4,
    tr_cond: GainCondition = "all_gains",
    test_cond: GainCondition = "all_gains",
    log_every: int = 500,
    seed: int = 0,
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> dict:
    """Train the Kalman RNN and return metrics + state dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    model = KalmanRNN(n_in=n_in, n_hid=n_hid, n_out=1).to(device_t)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_task = KalmanFilteringTask(
        max_iter=max_iter,
        batch_size=batch_size,
        n_in=n_in,
        stim_dur=stim_dur,
        tr_cond=tr_cond,
        seed=seed,
    )
    test_task = KalmanFilteringTask(
        max_iter=test_iter,
        batch_size=batch_size,
        n_in=n_in,
        stim_dur=stim_dur,
        tr_cond=test_cond,
        seed=seed + 1,
    )

    frac_rmse_vec: List[float] = []
    s_buf: List[np.ndarray] = []
    opt_buf: List[np.ndarray] = []
    pred_buf: List[np.ndarray] = []

    model.train()
    for i, batch in train_task:
        x, y, opt = batch_to_tensors(batch, device_t)
        pred = model(x)
        # Original: MSE on the final timestep only.
        loss = F.mse_loss(pred[:, -1, :], y[:, -1, :])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        with torch.no_grad():
            s_buf.append(y[:, -1, 0].cpu().numpy())
            opt_buf.append(opt[:, -1, 0].cpu().numpy())
            pred_buf.append(pred[:, -1, 0].cpu().numpy())

        if i % log_every == 0:
            frac, rmse_opt, rmse_net = fractional_rmse(
                np.concatenate(s_buf),
                np.concatenate(pred_buf),
                np.concatenate(opt_buf),
            )
            frac_rmse_vec.append(frac)
            print(
                f"Batch #{i}; Frac. RMSE: {frac:.6f}; "
                f"Opt. RMSE: {rmse_opt:.6f}; Net. RMSE: {rmse_net:.6f}"
            )
            s_buf, opt_buf, pred_buf = [], [], []

    frac_test, rmse_opt_t, rmse_net_t = evaluate(model, test_task, device_t)
    print(
        f"Test data; Frac. RMSE: {frac_test:.6f}; "
        f"Opt. RMSE: {rmse_opt_t:.6f}; Net. RMSE: {rmse_net_t:.6f}"
    )

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
        torch.save(
            {
                "state_dict": model.state_dict(),
                "frac_rmse_vec": result["frac_rmse_vec"],
                "frac_rmse_test": frac_test,
                "config": {
                    "n_in": n_in,
                    "n_hid": n_hid,
                    "stim_dur": stim_dur,
                    "batch_size": batch_size,
                    "lr": lr,
                    "tr_cond": tr_cond,
                    "test_cond": test_cond,
                    "seed": seed,
                    "log_every": log_every,
                    "max_iter": max_iter,
                },
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

    return result


def load_checkpoint(
    path: Path | str = DEFAULT_CHECKPOINT,
    *,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`train`.

    Returns a dict with ``model`` (eval mode), ``frac_rmse_vec``, ``frac_rmse_test``,
    ``config``, and the raw ``checkpoint`` payload.
    """
    path = Path(path)
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    checkpoint = torch.load(path, map_location=device_t, weights_only=False)
    cfg = checkpoint["config"]
    model = KalmanRNN(
        n_in=cfg["n_in"],
        n_hid=cfg["n_hid"],
        n_out=1,
    ).to(device_t)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "device": device_t,
        "frac_rmse_vec": np.asarray(checkpoint["frac_rmse_vec"]),
        "frac_rmse_test": float(checkpoint["frac_rmse_test"]),
        "config": cfg,
        "checkpoint": checkpoint,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-in", type=int, default=50)
    p.add_argument("--n-hid", type=int, default=200)
    p.add_argument("--stim-dur", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=50_001)
    p.add_argument("--test-iter", type=int, default=2_501)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--tr-cond",
        choices=["all_gains", "two_gains"],
        default="all_gains",
    )
    p.add_argument(
        "--test-cond",
        choices=["all_gains", "two_gains"],
        default="all_gains",
    )
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    # Smoke-test helper: tiny run without saving.
    p.add_argument("--smoke", action="store_true", help="Short run for sanity check")
    args = p.parse_args()

    if args.smoke:
        train(
            max_iter=201,
            test_iter=51,
            log_every=100,
            save_path=None,
            seed=args.seed,
            device=args.device,
            n_in=args.n_in,
            n_hid=args.n_hid,
            stim_dur=args.stim_dur,
            batch_size=args.batch_size,
            lr=args.lr,
            tr_cond=args.tr_cond,
            test_cond=args.test_cond,
        )
        return

    train(
        n_in=args.n_in,
        n_hid=args.n_hid,
        stim_dur=args.stim_dur,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        test_iter=args.test_iter,
        lr=args.lr,
        tr_cond=args.tr_cond,
        test_cond=args.test_cond,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
