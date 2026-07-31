"""Gene-level internal resolution metrics, classes, clustering, and atlas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import AgglomerativeClustering

from ..plot_style import apply_plot_style
from .validation import robust_zscore


def _context_effect(values: np.ndarray, label_names: list[str]) -> np.ndarray:
    labels = np.asarray(label_names)
    total = np.var(values, axis=0)
    means = np.vstack([values[labels == label].mean(axis=0) for label in dict.fromkeys(labels)])
    return np.var(means, axis=0) / (total + 1e-12)


def _column_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    denom = np.sqrt(np.square(a0).sum(axis=0) * np.square(b0).sum(axis=0))
    return np.divide((a0 * b0).sum(axis=0), denom, out=np.zeros(a.shape[1]), where=denom > 1e-12)


def _module_coherence(
    contrast: np.ndarray,
    genes: list[str],
    membership: pd.DataFrame,
) -> np.ndarray:
    result: dict[str, list[float]] = {gene: [] for gene in genes}
    if membership.empty or not {"module", "gene"}.issubset(membership):
        return np.zeros(len(genes))
    pos = {gene: i for i, gene in enumerate(genes)}
    for _, frame in membership.groupby("module"):
        members = [pos[g] for g in frame["gene"].astype(str).unique() if g in pos]
        members = [
            index for index in members
            if np.std(contrast[:, index]) > 1e-12
        ]
        if len(members) < 3:
            continue
        corr = np.nan_to_num(np.corrcoef(contrast[:, members], rowvar=False), nan=0.0)
        for j, index in enumerate(members):
            result[genes[index]].append(float((corr[j].sum() - 1.0) / (len(members) - 1)))
    return np.asarray([np.mean(result[gene]) if result[gene] else 0.0 for gene in genes])


def compute_gene_resolution_metrics(bundle, exposure: pd.DataFrame, config) -> pd.DataFrame:
    raw = bundle.raw_rna
    a = bundle.phase_a
    b = bundle.phase_b
    contrast = (b - a) / (np.abs(a) + np.abs(b) + 1e-12)
    abs_contrast = np.abs(contrast)
    top_n = max(1, int(np.ceil(0.05 * len(raw))))
    sorted_abs = np.sort(abs_contrast, axis=0)
    outlier = sorted_abs[-top_n:].sum(axis=0) / (abs_contrast.sum(axis=0) + 1e-12)
    var_a, var_b = np.var(a, axis=0), np.var(b, axis=0)
    variance_balance = np.minimum(var_a, var_b) / (np.maximum(var_a, var_b) + 1e-12)
    pathway_coherence = _module_coherence(contrast, bundle.genes, bundle.pathway_membership)
    ppi_coherence = _module_coherence(contrast, bundle.genes, bundle.ppi_membership)
    metrics = pd.DataFrame({
        "gene": bundle.genes,
        "mean_expression": raw.mean(axis=0),
        "detection_rate": (raw != 0).mean(axis=0),
        "rna_variance": np.var(raw, axis=0),
        "mean_signed_contrast": contrast.mean(axis=0),
        "separation_magnitude": np.median(abs_contrast, axis=0),
        "direction_consistency": np.abs(contrast.mean(axis=0)) / (abs_contrast.mean(axis=0) + 1e-12),
        "context_effect": _context_effect(contrast, bundle.label_names),
        "phase_correlation": _column_corr(a, b),
        "variance_balance": variance_balance,
        "outlier_dependence": outlier,
        "pathway_coherence": pathway_coherence,
        "ppi_coherence": ppi_coherence,
    })
    if exposure.empty:
        metrics["prior_coverage"] = 0.0
        metrics["weighted_gate_exposure"] = 0.0
        metrics["grn_in_degree"] = 0.0
        metrics["grn_out_degree"] = 0.0
    else:
        coverage = exposure.groupby("gene")["incident_edge_count"].sum()
        weighted = exposure.groupby("gene")["structural_exposure"].sum()
        grn = exposure[exposure["edge_type"].astype(str).str.contains("grn|reg_", case=False, regex=True)]
        grn_degree = grn.groupby("gene")["incident_edge_count"].sum()
        metrics["prior_coverage"] = metrics["gene"].map(coverage).fillna(0)
        metrics["weighted_gate_exposure"] = metrics["gene"].map(weighted).fillna(0)
        metrics["grn_in_degree"] = metrics["gene"].map(grn_degree).fillna(0)
        metrics["grn_out_degree"] = metrics["grn_in_degree"]
    if not bundle.gene_annotation.empty:
        annotation = bundle.gene_annotation.rename(columns={"gene_id": "gene"})
        keep = [c for c in ("gene", "chromosome", "TSS", "local_gene_density", "is_TF") if c in annotation]
        metrics = metrics.merge(annotation[keep].drop_duplicates("gene"), on="gene", how="left")
    for column, default in (("local_gene_density", 0.0), ("is_TF", 0)):
        if column not in metrics:
            metrics[column] = default

    score = (
        robust_zscore(metrics["separation_magnitude"])
        + robust_zscore(metrics["direction_consistency"])
        + robust_zscore(metrics["context_effect"])
        + robust_zscore((metrics["pathway_coherence"] + metrics["ppi_coherence"]) / 2)
        - robust_zscore(metrics["outlier_dependence"])
    )
    metrics["resolution_score"] = score
    low = (metrics["detection_rate"] <= config.low_detection) | (metrics["rna_variance"] <= config.low_variance)
    collapse = (~low) & (metrics["variance_balance"] < config.collapse_balance)
    high_cut = float(score.quantile(0.75))
    low_cut = float(score.quantile(0.25))
    well = (
        (~low) & (~collapse) & (score >= high_cut)
        & (metrics["detection_rate"] > config.low_detection)
        & (metrics["variance_balance"] >= config.well_resolved_balance)
    )
    ambiguous = (~low) & (~collapse) & (~well) & (score <= low_cut)
    metrics["resolution_class"] = "intermediate"
    metrics.loc[ambiguous, "resolution_class"] = "ambiguous"
    metrics.loc[well, "resolution_class"] = "well_resolved"
    metrics.loc[collapse, "resolution_class"] = "potential_collapse"
    metrics.loc[low, "resolution_class"] = "low_support"
    metrics["dominant_phase"] = np.where(metrics["mean_signed_contrast"] >= 0, "Phase_B", "Phase_A")
    return metrics


def cluster_gene_resolution(metrics: pd.DataFrame, config) -> pd.DataFrame:
    columns = [
        "mean_expression", "detection_rate", "rna_variance", "separation_magnitude",
        "direction_consistency", "context_effect", "phase_correlation", "variance_balance",
        "outlier_dependence", "pathway_coherence", "ppi_coherence", "prior_coverage",
        "weighted_gate_exposure",
    ]
    matrix = np.column_stack([robust_zscore(metrics[c]).to_numpy() for c in columns])
    matrix = np.nan_to_num(matrix)
    n_clusters = min(config.gene_clusters, max(1, len(metrics)))
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(matrix) + 1
    out = metrics[["gene", "resolution_class", "resolution_score", "dominant_phase"]].copy()
    out["gene_cluster"] = labels
    if len(matrix) > 2:
        order = leaves_list(linkage(matrix, method="ward"))
    else:
        order = np.arange(len(matrix))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    out["atlas_order"] = rank
    return out.sort_values("atlas_order")


def plot_gene_resolution_atlas(metrics: pd.DataFrame, clusters: pd.DataFrame):
    apply_plot_style()
    columns = [
        "mean_expression", "detection_rate", "rna_variance", "separation_magnitude",
        "direction_consistency", "context_effect", "phase_correlation", "variance_balance",
        "outlier_dependence", "pathway_coherence", "ppi_coherence", "prior_coverage",
        "weighted_gate_exposure", "resolution_score",
    ]
    ordered = clusters.sort_values("atlas_order")["gene"]
    table = metrics.set_index("gene").reindex(ordered)
    matrix = np.column_stack([robust_zscore(table[c]).clip(-3, 3) for c in columns])
    fig, ax = plt.subplots(figsize=(14, max(8, 0.09 * len(table))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(columns)), [c.replace("_", " ") for c in columns], rotation=55, ha="right", fontsize=8)
    step = max(1, len(table) // 40)
    ax.set_yticks(np.arange(0, len(table), step), table.index[::step], fontsize=6)
    ax.set_title("Gene resolution atlas (internal diagnostics)")
    fig.colorbar(image, ax=ax, label="Robust z-score")
    fig.tight_layout()
    return fig
