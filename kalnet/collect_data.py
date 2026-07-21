import torch
from task_vec import KalmanFilteringTask
from model import KalmanRNN


def collect(task: KalmanFilteringTask, net: KalmanRNN, n_trials: int, device: str = "cpu"):
    """Collect hidden states + ground truth for n_trials, in chunks of task.batch_size.
    task should already be constructed with the batch_size/tr_cond/seed """
    
    net.eval()
    all_r_hid, all_s, all_g, all_mu, all_sigma_sq, all_s_hat = [], [], [], [], [], []

    n_collected = 0
    with torch.no_grad():
        while n_collected < n_trials:
            batch = task.sample(include_internals=True)

            y, hidden = net(batch.input.to(device), return_hidden=True)
            # squeeze away the trailing singleton feature dim to match this
            # project's (n_trials, T) convention throughout
            s_hat = y[:, :, 0]
            s = batch.target[:, :, 0].to(device)
            g = batch.internals.gain[:, :, 0].to(device)
            mu = batch.opt_mean[:, :, 0].to(device)
            sigma_sq = batch.internals.opt_var[:, :, 0].to(device)

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


def run(
    checkpoint_path: str = "checkpoints/kf_default.pt",
    output_path: str = "saved_data/kf_dataset.pt",
    n_train: int = 5000,
    n_test: int = 2000,
    batch_size: int = 500,
    tr_cond: str = "all_gains",
    device: str = "cpu",
):
    
    # Load the checkpoint saved by train.py 
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = KalmanRNN(
        n_in=checkpoint["config"]["n_in"],
        n_hid=checkpoint["config"]["n_hid"],
        n_out=1,
    ).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded trained network from {checkpoint_path}")

    # One KalmanFilteringTask instance per split, each with its own widely-separated
    # seed (guaranteeing non-overlapping draws), fixed to a convenient batch_size
    # for chunked collection.
    train_task = KalmanFilteringTask(batch_size=batch_size, tr_cond=tr_cond, seed=1000, device=device)
    test_task = KalmanFilteringTask(batch_size=batch_size, tr_cond=tr_cond, seed=2000, device=device)

    print("Collecting training set (normal trials)...")
    train_data = collect(train_task, net, n_trials=n_train, device=device)

    print("Collecting held-out test set (normal trials)...")
    test_data = collect(test_task, net, n_trials=n_test, device=device)

    torch.save({"train": train_data, "test": test_data}, output_path)
    print(f"Saved {output_path}")
    print(f"  train:    r_hid {train_data['r_hid'].shape}")
    print(f"  test:     r_hid {test_data['r_hid'].shape}")

    return train_data, test_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/kf_default.pt",
                         help="path to the trained checkpoint .pt file")
    parser.add_argument("--output", default="saved_data/kf_dataset.pt",
                         help="path to save the collected dataset .pt file")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--tr_cond", default="all_gains")
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        n_train=args.n_train,
        n_test=args.n_test,
        tr_cond=args.tr_cond,
    )
