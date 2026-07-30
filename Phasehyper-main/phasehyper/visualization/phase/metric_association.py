"""Spearman associations among gene-level diagnostic metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from phasehyper.visualization.plot_style import apply_plot_style


ASSOCIATION_METRICS = (
    "mean_expression", "detection_rate", "rna_variance", "separation_magnitude",
    "direction_consistency", "context_effect", "phase_correlation", "variance_balance",
    "outlier_dependence", "pathway_coherence", "ppi_coherence", "grn_in_degree",
    "grn_out_degree", "prior_coverage", "weighted_gate_exposure", "local_gene_density",
    "resolution_score",
)


def compute_gene_metric_associations(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available = [name for name in ASSOCIATION_METRICS if name in metrics]
    for left in available:
        for right in available:
            pair = pd.DataFrame({
                "left": pd.to_numeric(metrics[left], errors="coerce"),
                "right": pd.to_numeric(metrics[right], errors="coerce"),
            }).replace([np.inf, -np.inf], np.nan).dropna()
            if left == right and len(pair) >= 1:
                rho, p_value = 1.0, 0.0
            elif len(pair) >= 3 and pair["left"].nunique() > 1 and pair["right"].nunique() > 1:
                rho, p_value = spearmanr(pair["left"], pair["right"])
            else:
                rho, p_value = np.nan, np.nan
            rows.append({
                "metric_x": left, "metric_y": right, "spearman_rho": rho,
                "p_value": p_value, "n_genes": len(pair),
            })
    return pd.DataFrame(rows)


def plot_metric_association_heatmap(data: pd.DataFrame):
    apply_plot_style()
    matrix = data.pivot(index="metric_y", columns="metric_x", values="spearman_rho")
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [x.replace("_", " ") for x in matrix.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    if len(labels) <= 18:
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5)
    ax.set_title("Gene metric associations (Spearman; association is not causation)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig
