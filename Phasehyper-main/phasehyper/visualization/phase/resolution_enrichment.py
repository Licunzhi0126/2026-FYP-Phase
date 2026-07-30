"""Resolution-cluster enrichment against model-input annotations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from phasehyper.visualization.plot_style import apply_plot_style


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    selected = values[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[valid] = restored
    return result


def _annotations(bundle, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, membership in (("pathway", bundle.pathway_membership), ("ppi", bundle.ppi_membership)):
        if not membership.empty:
            rows.extend({
                "gene": str(row.gene), "annotation_family": family, "annotation": str(row.module)
            } for row in membership.itertuples())
    if "chromosome" in metrics:
        rows.extend({
            "gene": str(row.gene), "annotation_family": "chromosome",
            "annotation": str(row.chromosome),
        } for row in metrics.dropna(subset=["chromosome"]).itertuples())
    if "is_TF" in metrics:
        rows.extend({
            "gene": str(row.gene), "annotation_family": "structural",
            "annotation": "TF" if bool(row.is_TF) else "non_TF",
        } for row in metrics.itertuples())
    rows.extend({
        "gene": str(row.gene), "annotation_family": "support",
        "annotation": (
            "low_expression_support"
            if float(row.detection_rate) <= 0.10 or float(row.rna_variance) <= 1e-8
            else "adequate_expression_support"
        ),
    } for row in metrics.itertuples())
    return pd.DataFrame(rows)


def compute_resolution_cluster_enrichment(bundle, metrics: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = _annotations(bundle, metrics)
    members = clusters[["gene", "gene_cluster"]].copy()
    if membership.empty:
        return pd.DataFrame(), members
    universe = set(metrics["gene"].astype(str))
    rows = []
    for family, family_frame in membership.groupby("annotation_family"):
        for annotation, ann_frame in family_frame.groupby("annotation"):
            annotated = set(ann_frame["gene"]) & universe
            if len(annotated) < 2 or len(annotated) >= len(universe):
                continue
            for cluster, cluster_frame in members.groupby("gene_cluster"):
                clustered = set(cluster_frame["gene"]) & universe
                overlap = clustered & annotated
                table = [
                    [len(overlap), len(clustered - annotated)],
                    [len(annotated - clustered), len(universe - clustered - annotated)],
                ]
                odds, p_value = fisher_exact(table)
                rows.append({
                    "annotation_family": family, "annotation": annotation,
                    "gene_cluster": cluster, "odds_ratio": odds, "p_value": p_value,
                    "overlap_count": len(overlap), "cluster_size": len(clustered),
                    "annotation_size": len(annotated), "overlap_genes": ";".join(sorted(overlap)),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = result.groupby("annotation_family")["p_value"].transform(
            lambda s: _bh_fdr(s.to_numpy())
        )
        log_or = np.log2(result["odds_ratio"].replace({0: np.nan}).clip(1e-12, 1e12))
        result["signed_score"] = np.sign(log_or) * -np.log10(result["fdr"] + 1e-12)
    return result, members


def plot_resolution_cluster_enrichment(data: pd.DataFrame, *, fdr_alpha: float, min_overlap: int):
    apply_plot_style()
    shown = data[(data["fdr"] <= fdr_alpha) & (data["overlap_count"] >= min_overlap)]
    if shown.empty:
        raise ValueError("no enrichment passes FDR and overlap thresholds")
    shown = shown.sort_values(["annotation_family", "fdr"]).head(60)
    pivot = shown.pivot_table(index="gene_cluster", columns=["annotation_family", "annotation"],
                              values="signed_score", aggfunc="max", fill_value=0)
    counts = shown.pivot_table(index="gene_cluster", columns=["annotation_family", "annotation"],
                               values="overlap_count", aggfunc="max", fill_value=0).reindex_like(pivot)
    vmax = max(float(np.nanquantile(np.abs(pivot), 0.98)), 1)
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * pivot.shape[1]), max(4, 0.6 * pivot.shape[0])))
    image = ax.imshow(pivot, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    labels = [f"{a}\n{b}" for a, b in pivot.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(pivot)), [f"Cluster {x}" for x in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(counts.iloc[i, j])), ha="center", va="center", fontsize=6)
    ax.set_title("Enrichment relative to model-input annotations")
    fig.colorbar(image, ax=ax, label="Signed −log10(FDR)")
    fig.tight_layout()
    return fig
