"""Representative gene selection and detailed diagnostic panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..plot_style import NEG, POS, apply_plot_style


DETAIL_CLASSES = ("well_resolved", "ambiguous", "low_support", "potential_collapse")


def select_representative_genes(metrics: pd.DataFrame, clusters: pd.DataFrame, *, per_class: int = 4) -> pd.DataFrame:
    data = metrics.merge(clusters[["gene", "gene_cluster", "atlas_order"]], on="gene")
    rows = []
    for quality in DETAIL_CLASSES:
        subset = data[data["resolution_class"] == quality].copy()
        if subset.empty:
            continue
        ascending = quality != "well_resolved"
        subset = subset.sort_values("resolution_score", ascending=ascending)
        chosen = []
        for _, row in subset.iterrows():
            if row["gene_cluster"] not in {x["gene_cluster"] for x in chosen} or len(chosen) >= subset["gene_cluster"].nunique():
                chosen.append(row.to_dict())
            if len(chosen) >= per_class:
                break
        for rank, row in enumerate(chosen, 1):
            rows.append({
                "gene": row["gene"], "resolution_class": quality,
                "selection_rank": rank, "gene_cluster": row["gene_cluster"],
                "resolution_score": row["resolution_score"],
            })
    return pd.DataFrame(rows)


def plot_gene_detail_panels(bundle, metrics: pd.DataFrame, selected: pd.DataFrame):
    apply_plot_style()
    pos = {gene: i for i, gene in enumerate(bundle.genes)}
    metric_table = metrics.set_index("gene")
    figures = {}
    label_order = list(dict.fromkeys(bundle.label_names))
    label_array = np.asarray(bundle.label_names)
    for quality, frame in selected.groupby("resolution_class"):
        genes = frame["gene"].tolist()
        fig, axes = plt.subplots(len(genes), 4, figsize=(15, max(3.5, 3.0 * len(genes))), squeeze=False)
        for row_index, gene in enumerate(genes):
            index = pos[gene]
            ax_dist, ax_bulk, ax_strip, ax_metric = axes[row_index]
            positions = np.arange(len(label_order))
            a_groups = [bundle.phase_a[label_array == label, index] for label in label_order]
            b_groups = [bundle.phase_b[label_array == label, index] for label in label_order]
            ax_dist.boxplot(a_groups, positions=positions - 0.16, widths=0.25, patch_artist=True,
                            boxprops={"facecolor": NEG, "alpha": 0.5})
            ax_dist.boxplot(b_groups, positions=positions + 0.16, widths=0.25, patch_artist=True,
                            boxprops={"facecolor": POS, "alpha": 0.5})
            ax_dist.set_xticks(positions, label_order, rotation=45, ha="right", fontsize=7)
            ax_dist.set_title(f"{gene}: group distributions")
            bulk = np.asarray([
                [bundle.raw_rna[label_array == label, index].mean(),
                 bundle.phase_a[label_array == label, index].mean(),
                 bundle.phase_b[label_array == label, index].mean()]
                for label in label_order
            ]).T
            ax_bulk.imshow(bulk, aspect="auto", cmap="viridis")
            ax_bulk.set_yticks(range(3), ["Raw", "A", "B"])
            ax_bulk.set_xticks(range(len(label_order)), label_order, rotation=45, ha="right", fontsize=7)
            contrast = (bundle.phase_b[:, index] - bundle.phase_a[:, index]) / (
                np.abs(bundle.phase_a[:, index]) + np.abs(bundle.phase_b[:, index]) + 1e-12
            )
            order = np.argsort(label_array.astype(str), kind="stable")
            ax_strip.imshow(contrast[order][None, :], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax_strip.set_yticks([0], ["B − A"])
            ax_strip.set_xticks([])
            ax_strip.set_title("Cell-level contrast")
            names = ["separation_magnitude", "direction_consistency", "context_effect",
                     "variance_balance", "detection_rate", "prior_coverage"]
            values = metric_table.loc[gene, names].astype(float).to_numpy()
            ax_metric.barh(range(len(names)), values, color=POS)
            ax_metric.set_yticks(range(len(names)), [x.replace("_", " ") for x in names], fontsize=7)
            ax_metric.set_title(f"Internal metrics ({quality})")
        fig.suptitle(f"{quality.replace('_', ' ').title()} representative genes")
        fig.tight_layout()
        figures[f"14_{quality}_gene_details"] = fig
    return figures
