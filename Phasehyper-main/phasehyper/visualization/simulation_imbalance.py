"""Chromosome and genome-wide allelic imbalance visualizations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import INK, INK_2, NEG, POS, save_figure, style_axis
from .simulation_diagnostics import EPS, SimulationBundle


def signed_imbalance(first, second, eps: float = EPS):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return (first - second) / (np.abs(first) + np.abs(second) + eps)


def _bundle_imbalances(bundle: SimulationBundle, genes: list[str]):
    truth = signed_imbalance(
        bundle.maternal[genes].to_numpy(), bundle.paternal[genes].to_numpy()
    )
    predicted = signed_imbalance(
        bundle.pred_maternal[genes].to_numpy(),
        bundle.pred_paternal[genes].to_numpy(),
    )
    return truth, predicted


def write_gene_level_imbalance(bundle: SimulationBundle, path: Path) -> Path:
    genes = bundle.ordered_genes()
    truth, predicted = _bundle_imbalances(bundle, genes)
    frame = pd.DataFrame(
        {
            "gene_id": genes,
            "chromosome": bundle.annotation.reindex(genes)["chromosome"].to_numpy(),
            "truth_mean_signed_imbalance": np.mean(truth, axis=0),
            "truth_mean_absolute_imbalance": np.mean(np.abs(truth), axis=0),
            "pred_mean_signed_imbalance": np.mean(predicted, axis=0),
            "pred_mean_absolute_imbalance": np.mean(np.abs(predicted), axis=0),
            "mean_residual": np.mean(predicted - truth, axis=0),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def plot_chromosome_imbalance_track(
    bundle: SimulationBundle, chromosome: str, path: Path, dpi: int
) -> Path:
    genes = bundle.genes_for_chromosome(chromosome)
    truth, predicted = _bundle_imbalances(bundle, genes)
    truth_mean = truth.mean(axis=0)
    pred_mean = predicted.mean(axis=0)
    residual = pred_mean - truth_mean
    window = max(1, min(9, len(genes) // 2 * 2 + 1))
    x = np.arange(len(genes))
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, values, label, color in (
        (axes[0], truth_mean, "truth signed", INK_2),
        (axes[1], pred_mean, "predicted signed", POS),
    ):
        ax.plot(x, values, color=color, linewidth=1)
        rolling = pd.Series(values).rolling(window, center=True, min_periods=1).median()
        ax.plot(x, rolling, color=INK, linewidth=1.6, label=f"rolling median ({window})")
        ax.legend(frameon=False)
        ax.set_ylabel(label)
        style_axis(ax)
    axes[2].plot(x, truth_mean, color=INK_2, label="truth")
    axes[2].plot(x, pred_mean, color=POS, label="prediction")
    axes[2].legend(frameon=False)
    axes[2].set_ylabel("comparison")
    style_axis(axes[2])
    axes[3].plot(x, residual, color=NEG)
    axes[3].axhline(0, color=INK_2, linewidth=0.7)
    axes[3].set_ylabel("prediction − truth")
    axes[3].set_xlabel("genes in genomic order")
    style_axis(axes[3])
    fig.suptitle(f"{chromosome} expression imbalance", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def _pseudobulk_imbalance(bundle: SimulationBundle, genes: list[str]):
    labels = bundle.metadata["cell_type"]
    contexts = list(dict.fromkeys(labels.tolist()))
    truth_rows, predicted_rows = [], []
    for context in contexts:
        mask = labels.to_numpy() == context
        truth_rows.append(
            signed_imbalance(
                bundle.maternal.loc[mask, genes].mean(axis=0),
                bundle.paternal.loc[mask, genes].mean(axis=0),
            )
        )
        predicted_rows.append(
            signed_imbalance(
                bundle.pred_maternal.loc[mask, genes].mean(axis=0),
                bundle.pred_paternal.loc[mask, genes].mean(axis=0),
            )
        )
    return contexts, np.vstack(truth_rows), np.vstack(predicted_rows)


def _plot_imbalance_heatmap(
    bundle: SimulationBundle, genes: list[str], title: str, path: Path, dpi: int
) -> Path:
    contexts, truth, predicted = _pseudobulk_imbalance(bundle, genes)
    residual = predicted - truth
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, values, label in zip(
        axes, (truth, predicted, residual), ("Truth", "Prediction", "Residual")
    ):
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(np.arange(len(contexts)))
        ax.set_yticklabels(contexts)
        ax.set_ylabel(label)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.015)
    axes[-1].set_xlabel("genes in genomic order")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_chromosome_imbalance_heatmap(
    bundle: SimulationBundle, chromosome: str, path: Path, dpi: int
) -> Path:
    return _plot_imbalance_heatmap(
        bundle,
        bundle.genes_for_chromosome(chromosome),
        f"{chromosome} cell-type pseudobulk imbalance",
        path,
        dpi,
    )


def plot_genome_imbalance_heatmap(
    bundle: SimulationBundle, path: Path, dpi: int
) -> Path:
    genes = bundle.ordered_genes()
    contexts, truth, predicted = _pseudobulk_imbalance(bundle, genes)
    residual = predicted - truth
    boundaries = []
    offset = 0
    for chromosome in bundle.chromosomes:
        offset += len(bundle.genes_for_chromosome(chromosome))
        boundaries.append(offset - 0.5)
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    for ax, values, label in zip(
        axes, (truth, predicted, residual), ("Truth", "Prediction", "Residual")
    ):
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        for boundary in boundaries[:-1]:
            ax.axvline(boundary, color=INK, linewidth=0.6)
        ax.set_yticks(np.arange(len(contexts)))
        ax.set_yticklabels(contexts)
        ax.set_ylabel(label)
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("genes ordered by chromosome and genomic position")
    fig.suptitle("Genome-wide cell-type pseudobulk imbalance", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)
