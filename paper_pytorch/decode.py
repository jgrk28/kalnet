"""
1. Linear decoding: r_hid -> sigma_sq, and separately r_hid -> 1/sigma_sq (precision),
   using pooled (trial, timestep) samples from the "all_gains" train/test splits.
   Compared against a shuffled-label control (decoder fit on permuted targets).
2. Specific-statistic correlations: mean hidden-layer activity and kurtosis
   (sparsity), each correlated against sigma_sq.

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def flatten(split):
    """(n_trials, T, n_hid) / (n_trials, T) -> pooled (n_trials*T, ...) samples."""
    r_hid = split["r_hid"].reshape(-1, split["r_hid"].shape[-1]).numpy()
    sigma_sq = split["sigma_sq"].reshape(-1).numpy()
    return r_hid, sigma_sq


def fit_and_eval_decoder(X_train, y_train, X_test, y_test, alpha=1.0, label=""):
    """Ridge regression (small L2 for stability given ~200 correlated ReLU features).
    Returns R^2 on held-out test set."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"  {label:30s} R^2 = {r2:.4f}")
    return r2, model


def shuffled_control(X_train, y_train, X_test, y_test, alpha=1.0, n_repeats=5, seed=0):
    """Refit the decoder several times on label-permuted training data; report the
    resulting test R^2 distribution as the floor a real effect must clear."""
    rng = np.random.default_rng(seed)
    r2s = []
    for i in range(n_repeats):
        y_shuffled = rng.permutation(y_train)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_shuffled)
        y_pred = model.predict(X_test)
        r2s.append(r2_score(y_test, y_pred))
    r2s = np.array(r2s)
    print(f"  {'shuffled-label control':30s} R^2 = {r2s.mean():.4f} +/- {r2s.std():.4f}  "
          f"(range: [{r2s.min():.4f}, {r2s.max():.4f}])")
    return r2s


def hidden_layer_stats(r_hid_flat):
    """mean activity and kurtosis (sparsity) per sample, across the unit dimension.
    r_hid_flat: (n_samples, n_hid) -> returns (mean: (n_samples,), kurt: (n_samples,))"""
    mean_act = r_hid_flat.mean(axis=1)
    # fisher=True kurtosis: 0 for Gaussian, positive = more peaked/sparse than Gaussian
    kurt = kurtosis(r_hid_flat, axis=1, fisher=True)
    return mean_act, kurt


def lin_decoder(path="kf_dataset.pt"):
    data = torch.load(path, weights_only=False)
    train, test = data["train"], data["test"]

    X_train, sigma_sq_train = flatten(train)
    X_test, sigma_sq_test = flatten(test)
    precision_train = 1.0 / sigma_sq_train
    precision_test = 1.0 / sigma_sq_test

    print(f"Pooled samples: train={X_train.shape[0]}, test={X_test.shape[0]}\n")

    print("=" * 60)
    print("1. LINEAR DECODING: sigma_sq vs. precision (1/sigma_sq)")
    print("=" * 60)
    r2_sigma, model_sigma = fit_and_eval_decoder(X_train, sigma_sq_train, X_test, sigma_sq_test,
                                                  label="decode sigma_sq")
    r2_precision, model_precision = fit_and_eval_decoder(X_train, precision_train, X_test, precision_test,
                                                           label="decode precision (1/sigma_sq)")
    shuffled_r2s = shuffled_control(X_train, sigma_sq_train, X_test, sigma_sq_test)
    print(f"\n  --> {'precision' if r2_precision > r2_sigma else 'sigma_sq'} decodes better "
          f"({max(r2_precision, r2_sigma):.4f} vs {min(r2_precision, r2_sigma):.4f})")

    print("\n" + "=" * 60)
    print("2. SPECIFIC-STATISTIC CORRELATIONS (mean activity, sparsity/kurtosis)")
    print("=" * 60)
    mean_act_test, kurt_test = hidden_layer_stats(X_test)

    r_mean, p_mean = pearsonr(mean_act_test, sigma_sq_test)
    r_kurt, p_kurt = pearsonr(kurt_test, sigma_sq_test)
    print(f"  corr(mean activity, sigma_sq):  r={r_mean:+.4f}  p={p_mean:.2e}")
    print(f"  corr(kurtosis,      sigma_sq):  r={r_kurt:+.4f}  p={p_kurt:.2e}")

    if abs(r_mean) > abs(r_kurt) * 1.5:
        verdict = "MEAN-ACTIVITY code looks dominant"
    elif abs(r_kurt) > abs(r_mean) * 1.5:
        verdict = "SPARSITY code looks dominant"
    else:
        verdict = "BOTH statistics show comparable correlation -- inconclusive from this alone"
    print(f"\n  --> {verdict}")

    # ============================================================
    # PLOTS
    # ============================================================
    # Subsample for scatter plots -- 50,000 points renders as an unreadable blob
    # and is slow; a random subset shows the same relationship clearly.
    rng = np.random.default_rng(0)
    n_plot = 3000
    idx = rng.choice(len(sigma_sq_test), size=n_plot, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Panel 1: decoded vs true sigma_sq ---
    pred_sigma = model_sigma.predict(X_test)
    axes[0, 0].scatter(sigma_sq_test[idx], pred_sigma[idx], s=4, alpha=0.3)
    lims = [0, max(sigma_sq_test.max(), pred_sigma.max())]
    axes[0, 0].plot(lims, lims, "r--", linewidth=1, label="perfect decoding")
    axes[0, 0].set_xlabel("true sigma_t^2")
    axes[0, 0].set_ylabel("decoded sigma_t^2")
    axes[0, 0].set_title(f"Decoding sigma_sq (R^2={r2_sigma:.3f})")
    axes[0, 0].legend()

    # --- Panel 2: decoded vs true precision ---
    pred_prec = model_precision.predict(X_test)
    axes[0, 1].scatter(precision_test[idx], pred_prec[idx], s=4, alpha=0.3, color="tab:orange")
    lims = [0, max(precision_test.max(), pred_prec.max())]
    axes[0, 1].plot(lims, lims, "r--", linewidth=1, label="perfect decoding")
    axes[0, 1].set_xlabel("true precision (1/sigma_t^2)")
    axes[0, 1].set_ylabel("decoded precision")
    axes[0, 1].set_title(f"Decoding precision (R^2={r2_precision:.3f})")
    axes[0, 1].legend()

    # --- Panel 3: R^2 bar comparison, with shuffled control as error bar ---
    labels = ["sigma_sq", "precision", "shuffled\ncontrol"]
    values = [r2_sigma, r2_precision, shuffled_r2s.mean()]
    errors = [0, 0, shuffled_r2s.std()]
    colors = ["tab:blue", "tab:orange", "gray"]
    axes[0, 2].bar(labels, values, yerr=errors, color=colors, capsize=5)
    axes[0, 2].axhline(0, color="black", linewidth=0.8)
    axes[0, 2].set_ylabel("test R^2")
    axes[0, 2].set_title("Decoding performance vs. shuffled-label floor")

    # --- Panel 4: mean activity vs sigma_sq ---
    axes[1, 0].scatter(mean_act_test[idx], sigma_sq_test[idx], s=4, alpha=0.3, color="tab:green")
    axes[1, 0].set_xlabel("mean hidden-layer activity")
    axes[1, 0].set_ylabel("sigma_t^2")
    axes[1, 0].set_title(f"Mean activity vs. sigma_sq (r={r_mean:+.3f})")

    # --- Panel 5: kurtosis vs sigma_sq ---
    axes[1, 1].scatter(kurt_test[idx], sigma_sq_test[idx], s=4, alpha=0.3, color="tab:purple")
    axes[1, 1].set_xlabel("kurtosis (sparsity)")
    axes[1, 1].set_ylabel("sigma_t^2")
    axes[1, 1].set_title(f"Kurtosis vs. sigma_sq (r={r_kurt:+.3f})")

    # --- Panel 6: binned means, both statistics on one plot for direct comparison ---
    n_bins = 15
    bin_edges = np.quantile(sigma_sq_test, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(sigma_sq_test, bin_edges[1:-1])
    bin_centers, mean_act_binned, kurt_binned = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_centers.append(sigma_sq_test[mask].mean())
            mean_act_binned.append(mean_act_test[mask].mean())
            kurt_binned.append(kurt_test[mask].mean())
    ax6 = axes[1, 2]
    ax6.plot(bin_centers, mean_act_binned, "o-", color="tab:green", label="mean activity")
    ax6.set_xlabel("sigma_t^2 (binned)")
    ax6.set_ylabel("mean activity", color="tab:green")
    ax6.tick_params(axis="y", labelcolor="tab:green")
    ax6b = ax6.twinx()
    ax6b.plot(bin_centers, kurt_binned, "s-", color="tab:purple", label="kurtosis")
    ax6b.set_ylabel("kurtosis", color="tab:purple")
    ax6b.tick_params(axis="y", labelcolor="tab:purple")
    ax6.set_title("Binned trend: both statistics vs. sigma_sq")

    plt.tight_layout()
    plt.savefig("decode_results.png", dpi=120)
    plt.show()
    print("\nSaved decode_results.png")

    return data


if __name__ == "__main__":
    lin_decoder()
