"""Cell-group and cell-level phase gene heatmaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phasehyper.visualization.plot_style import apply_plot_style


def build_cellgroup_phase_matrices(bundle, selected_genes: list[str]) -> pd.DataFrame:
    pos = {gene: i for i, gene in enumerate(bundle.genes)}
    genes = [gene for gene in selected_genes if gene in pos]
    rows = []
    labels = list(dict.fromkeys(bundle.label_names))
    for label in labels:
        mask = np.asarray(bundle.label_names) == label
        for gene in genes:
            index = pos[gene]
            mean_a = float(bundle.phase_a[mask, index].mean())
            mean_b = float(bundle.phase_b[mask, index].mean())
            rows.append({
                "label_name": label, "gene": gene, "mean_A": mean_a, "mean_B": mean_b,
                "normalized_contrast": (mean_b - mean_a) / (
                    np.abs(bundle.phase_a[mask, index]).mean()
                    + np.abs(bundle.phase_b[mask, index]).mean() + 1e-12
                ),
            })
    return pd.DataFrame(rows)


def build_cell_gene_contrast(bundle, selected_genes: list[str], allocation: pd.DataFrame) -> pd.DataFrame:
    pos = {gene: i for i, gene in enumerate(bundle.genes)}
    genes = [g for g in selected_genes if g in pos]
    indices = [pos[g] for g in genes]
    contrast = (bundle.phase_b[:, indices] - bundle.phase_a[:, indices]) / (
        np.abs(bundle.phase_a[:, indices]) + np.abs(bundle.phase_b[:, indices]) + 1e-12
    )
    order_frame = allocation.copy()
    order_frame["label_order"] = pd.Categorical(
        order_frame["label_name"], categories=list(dict.fromkeys(order_frame["label_name"])), ordered=True
    )
    order = order_frame.sort_values(["label_order", "allocation_score"]).index.to_numpy()
    frame = pd.DataFrame(contrast[order], columns=genes)
    frame.insert(0, "label_name", np.asarray(bundle.label_names)[order])
    frame.insert(0, "cell_id", np.asarray(bundle.cell_ids)[order])
    frame.insert(2, "cell_order", np.arange(1, len(frame) + 1))
    return frame


def plot_cellgroup_phase_triptych(data: pd.DataFrame):
    apply_plot_style()
    genes = list(dict.fromkeys(data["gene"]))
    labels = list(dict.fromkeys(data["label_name"]))
    matrices = [
        data.pivot(index="gene", columns="label_name", values=column).reindex(index=genes, columns=labels)
        for column in ("mean_A", "mean_B", "normalized_contrast")
    ]
    ab_max = max(float(np.nanquantile(np.abs(matrices[0]), 0.98)), float(np.nanquantile(np.abs(matrices[1]), 0.98)), 1e-8)
    contrast_max = max(float(np.nanquantile(np.abs(matrices[2]), 0.98)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(7, 0.18 * len(genes))), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, ("Phase A", "Phase B", "Normalized B − A")):
        if title == "Normalized B − A":
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-contrast_max, vmax=contrast_max)
        else:
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-ab_max, vmax=ab_max)
        ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
        ax.set_yticks(range(len(genes)), genes if ax is axes[0] else [], fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.035)
    fig.suptitle("Cell-group × gene phase triptych")
    return fig


def plot_cell_gene_contrast(data: pd.DataFrame):
    apply_plot_style()
    matrix = data.drop(columns=["cell_id", "label_name", "cell_order"]).to_numpy()
    vmax = max(float(np.nanquantile(np.abs(matrix), 0.99)), 1e-8)
    fig, ax = plt.subplots(figsize=(12, max(6, min(14, len(data) * 0.025))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
    ax.set_xticks(range(matrix.shape[1]), data.columns[3:], rotation=70, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel("Cells ordered by label and allocation")
    ax.set_title("Cell × top-gene normalized contrast")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig
