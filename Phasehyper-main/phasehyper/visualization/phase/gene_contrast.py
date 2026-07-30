"""Cell-group gene contrast calculation and heatmap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage

from phasehyper.visualization.plot_style import apply_plot_style


def compute_gene_contrast(bundle, *, top_genes: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_order = list(dict.fromkeys(bundle.label_names))
    rows = []
    normalized = []
    for label in label_order:
        mask = np.asarray(bundle.label_names) == label
        mean_a = bundle.phase_a[mask].mean(axis=0)
        mean_b = bundle.phase_b[mask].mean(axis=0)
        denom = np.abs(bundle.phase_a[mask]).mean(axis=0) + np.abs(bundle.phase_b[mask]).mean(axis=0) + 1e-12
        norm = (mean_b - mean_a) / denom
        normalized.append(norm)
        rows.extend({
            "gene": gene, "label_name": label, "mean_A": a, "mean_B": b,
            "raw_contrast": b - a, "normalized_contrast": n,
        } for gene, a, b, n in zip(bundle.genes, mean_a, mean_b, norm))
    matrix = np.vstack(normalized)
    mean_abs = np.mean(np.abs(matrix), axis=0)
    contrast_std = np.std(matrix, axis=0)
    score = mean_abs + contrast_std
    eligible = (np.var(bundle.raw_rna, axis=0) > 1e-12) & ((bundle.raw_rna != 0).mean(axis=0) > 0.01)
    ranking = np.argsort(np.where(eligible, score, -np.inf))[::-1]
    ranking = ranking[np.isfinite(score[ranking])][:min(top_genes, int(eligible.sum()))]
    selected = pd.DataFrame({
        "gene": np.asarray(bundle.genes)[ranking],
        "selection_score": score[ranking],
        "mean_abs_contrast": mean_abs[ranking],
        "contrast_std": contrast_std[ranking],
        "dominant_phase": np.where(matrix[:, ranking].mean(axis=0) >= 0, "Phase_B", "Phase_A"),
        "selection_rank": np.arange(1, len(ranking) + 1),
    })
    return pd.DataFrame(rows), selected


def plot_gene_contrast_heatmap(contrast: pd.DataFrame, selected: pd.DataFrame):
    apply_plot_style()
    if selected.empty:
        raise ValueError("no eligible genes for contrast heatmap")
    pivot = contrast.pivot(index="gene", columns="label_name", values="normalized_contrast")
    pivot = pivot.reindex(index=selected["gene"])
    if len(pivot) > 2:
        order = leaves_list(linkage(pivot.to_numpy(), method="average", metric="euclidean"))
        pivot = pivot.iloc[order]
    vmax = max(float(np.nanquantile(np.abs(pivot.to_numpy()), 0.98)), 1e-6)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * pivot.shape[1]), max(6, 0.18 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=7)
    ax.set_title("Normalized Phase B − Phase A gene contrast")
    fig.colorbar(image, ax=ax, label="Normalized contrast")
    fig.tight_layout()
    return fig
