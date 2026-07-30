"""Aligned Raw RNA / Phase A / Phase B / delta correlation blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from phasehyper.visualization.plot_style import apply_plot_style


def _select_genes(bundle, resolution: pd.DataFrame, limit: int) -> list[str]:
    resolution = resolution.set_index("gene")
    variance = pd.Series(np.var(bundle.raw_rna, axis=0), index=bundle.genes)
    selected: list[str] = []
    pools = [
        variance.sort_values(ascending=False).index,
        resolution["resolution_score"].sort_values(ascending=False).index,
        resolution.loc[resolution["resolution_class"] == "ambiguous", "resolution_score"].sort_values().index,
    ]
    for pool in pools:
        for gene in pool[:max(1, limit // 3)]:
            if gene not in selected and variance.get(gene, 0) > 1e-12:
                selected.append(gene)
    return selected[:limit]


def compute_phase_correlation_blocks(bundle, resolution: pd.DataFrame, *, limit: int = 200):
    genes = _select_genes(bundle, resolution, limit)
    pos = {gene: i for i, gene in enumerate(bundle.genes)}
    indices = [pos[g] for g in genes]
    common_keep = (
        (np.std(bundle.raw_rna[:, indices], axis=0) > 1e-12)
        & (np.std(bundle.phase_a[:, indices], axis=0) > 1e-12)
        & (np.std(bundle.phase_b[:, indices], axis=0) > 1e-12)
    )
    genes = [gene for gene, keep in zip(genes, common_keep) if keep]
    indices = [pos[g] for g in genes]
    if len(genes) < 2:
        raise ValueError("fewer than two common non-constant genes for correlation")
    matrices = {}
    for name, values in (
        ("raw", bundle.raw_rna[:, indices]),
        ("phase_a", bundle.phase_a[:, indices]),
        ("phase_b", bundle.phase_b[:, indices]),
    ):
        matrices[name] = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
    matrices["delta"] = matrices["phase_a"] - matrices["phase_b"]

    delta_distance = 1 - np.clip(np.abs(matrices["delta"]), 0, 1)
    np.fill_diagonal(delta_distance, 0)
    clustered = leaves_list(linkage(squareform(delta_distance, checks=False), method="average"))
    orders = {"clustered": [genes[i] for i in clustered]}

    annotation = bundle.gene_annotation.rename(columns={"gene_id": "gene"})
    if not annotation.empty and {"gene", "chromosome", "TSS"}.issubset(annotation):
        ann = annotation.set_index("gene").reindex(genes)
        if ann["chromosome"].notna().sum() >= 2:
            orders["chromosome"] = ann.sort_values(["chromosome", "TSS"], na_position="last").index.tolist()
    if not bundle.pathway_membership.empty:
        membership = bundle.pathway_membership[bundle.pathway_membership["gene"].isin(genes)]
        primary = membership.drop_duplicates("gene").set_index("gene")["module"]
        score = resolution.set_index("gene")["resolution_score"]
        orders["pathway"] = sorted(genes, key=lambda g: (str(primary.get(g, "~")), -float(score.get(g, 0))))
    return genes, matrices, orders


def correlation_source_frames(genes: list[str], matrices: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.DataFrame(values, index=pd.Index(genes, name="gene"), columns=genes)
        for name, values in matrices.items()
    }


def plot_correlation_block_heatmaps(
    genes: list[str], matrices: dict[str, np.ndarray], order: list[str], title: str
):
    apply_plot_style()
    pos = {gene: i for i, gene in enumerate(genes)}
    indices = [pos[g] for g in order if g in pos]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    for ax, name, label in zip(
        axes, ("raw", "phase_a", "phase_b", "delta"),
        ("Raw RNA", "Phase A", "Phase B", "Phase A − Phase B"),
    ):
        matrix = matrices[name][np.ix_(indices, indices)]
        vmax = 1 if name != "delta" else max(float(np.quantile(np.abs(matrix), 0.99)), 0.1)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
        fig.colorbar(image, ax=ax, fraction=0.045)
    fig.suptitle(title)
    return fig
