from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(loss_history: List[Dict[str, float]], save_path: str):
    epochs = [h["epoch"] for h in loss_history]

    def series(key):
        return [h.get(key, 0) for h in loss_history]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    axes[0, 0].plot(epochs, series("total_loss"), label="Total Loss", linewidth=2)
    axes[0, 0].plot(epochs, series("vae_recon_loss"), label="VAE Recon Loss")
    axes[0, 0].set_title("Main Losses")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, series("raw_kl"), label="Raw KL")
    axes[0, 1].plot(epochs, series("kl_term"), label="KL Term (with warmup)")
    axes[0, 1].set_title("KL Divergence")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("KL Value")

    axes[0, 2].plot(epochs, series("gate_mean"), label="Gate Mean", linewidth=2)
    axes[0, 2].plot(epochs, series("gate_std"), label="Gate Std")
    axes[0, 2].axhline(y=0.5, color="r", linestyle="--", label="0.5 Threshold")
    axes[0, 2].set_title("Gate Statistics")
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("Epoch")

    axes[1, 0].plot(epochs, series("mu_cells_norm_mean"), label="Mu Cells Norm Mean")
    axes[1, 0].plot(epochs, series("mu_genes_norm_mean"), label="Mu Genes Norm Mean")
    axes[1, 0].set_title("Latent Mu Statistics")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, series("logvar_cells_mean"), label="Logvar Cells Mean")
    axes[1, 1].set_title("Logvar Statistics")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Epoch")

    axes[1, 2].plot(epochs, series("phase_sep_loss"), label="Phase Sep Loss")
    axes[1, 2].set_title("Phase Separation Loss")
    axes[1, 2].legend()
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Loss")

    axes[2, 0].plot(epochs, series("energy_balance_loss"), label="Energy Balance")
    axes[2, 0].plot(epochs, series("program_var_floor_loss"), label="Program Var Floor")
    axes[2, 0].set_title("Program Balance & Variance Floor")
    axes[2, 0].legend()
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Loss")

    axes[2, 1].plot(epochs, series("gate_entropy_loss"), label="Gate Entropy")
    axes[2, 1].plot(epochs, series("gate_variance_loss"), label="Gate Variance")
    axes[2, 1].plot(epochs, series("gate_smooth_loss"), label="Gate Smoothness")
    axes[2, 1].set_title("Gate Regularization Losses")
    axes[2, 1].legend()
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Loss")

    if "dec_loss" in loss_history[0]:
        axes[2, 2].plot(epochs, series("dec_loss"), label="DEC Loss")
        axes[2, 2].set_title("DEC Clustering Loss")
        axes[2, 2].legend()
        axes[2, 2].set_xlabel("Epoch")
    else:
        axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
