import torch
from task import KalmanFilteringTask
from model import KalmanRNN


def collect(task: KalmanFilteringTask, net: KalmanRNN, n_trials: int, device: str = "cpu"):
    """Collect hidden states + ground truth for n_trials, in chunks of task.batch_size."""
    net.eval()
    all_r_hid, all_s, all_g, all_mu, all_sigma_sq, all_s_hat = [], [], [], [], [], []

    n_collected = 0
    with torch.no_grad():
        while n_collected < n_trials:
            batch = task.sample(include_internals=True)

            # Explicit NumPy -> torch conversion, since batch.* are np.ndarray here
            x = torch.from_numpy(batch.input).to(device)
            s = torch.from_numpy(batch.target[:, :, 0]).to(device)
            g = torch.from_numpy(batch.internals.gain[:, :, 0]).to(device)
            mu = torch.from_numpy(batch.opt_mean[:, :, 0]).to(device)
            sigma_sq = torch.from_numpy(batch.internals.opt_var[:, :, 0]).to(device)

            y, hidden = net(x, return_hidden=True)
            s_hat = y[:, :, 0]  # already a torch.Tensor -- net's own output

            all_r_hid.append(hidden.cpu())
            all_s.append(s.cpu())
            all_g.append(g.cpu())
            all_mu.append(mu.cpu())
            all_sigma_sq.append(sigma_sq.cpu())
            all_s_hat.append(s_hat.cpu())

            n_collected += batch.input.shape[0]  # = task.batch_size, each call

    return {
        "r_hid": torch.cat(all_r_hid, dim=0)[:n_trials],
        "s": torch.cat(all_s, dim=0)[:n_trials],
        "g": torch.cat(all_g, dim=0)[:n_trials],
        "mu": torch.cat(all_mu, dim=0)[:n_trials],
        "sigma_sq": torch.cat(all_sigma_sq, dim=0)[:n_trials],
        "s_hat": torch.cat(all_s_hat, dim=0)[:n_trials],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="path to the trained checkpoint .pt file")
    parser.add_argument("--output", required=True,
                        help="path to save the collected dataset .pt file")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--tr_cond", default="all_gains")
    args = parser.parse_args()

    device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = KalmanRNN(
        n_in=checkpoint["config"]["n_in"],
        n_hid=checkpoint["config"]["n_hid"],
        n_out=1,
    ).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded trained network from {args.checkpoint}")

    # NOTE: no device= argument here -- the original KalmanFilteringTask doesn't accept one.
    train_task = KalmanFilteringTask(batch_size=500, tr_cond=args.tr_cond, seed=1000)
    test_task = KalmanFilteringTask(batch_size=500, tr_cond=args.tr_cond, seed=2000)

    print("Collecting training set (normal trials)...")
    train_data = collect(train_task, net, n_trials=args.n_train, device=device)

    print("Collecting held-out test set (normal trials)...")
    test_data = collect(test_task, net, n_trials=args.n_test, device=device)

    torch.save(
        {"train": train_data, "test": test_data, "checkpoint_path": args.checkpoint},
        args.output,
    )
    print(f"Saved {args.output}")
    print(f"  train:    r_hid {train_data['r_hid'].shape}")
    print(f"  test:     r_hid {test_data['r_hid'].shape}")
