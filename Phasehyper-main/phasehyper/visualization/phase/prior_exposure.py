"""Gene-level exposure to directed gates and undirected structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phasehyper.visualization.plot_style import apply_plot_style


def compute_gene_prior_exposure(bundle) -> pd.DataFrame:
    membership = bundle.hyperedge_membership.copy()
    columns = [
        "gene", "channel", "edge_type", "incident_edge_count", "total_edge_weight",
        "final_gate", "weighted_gate_exposure", "structural_exposure",
    ]
    if membership.empty:
        return pd.DataFrame(columns=columns)
    required = {"gene", "channel", "edge_type", "edge_id", "edge_weight"}
    missing = required - set(membership)
    if missing:
        raise ValueError(f"hyperedge_membership.csv missing {sorted(missing)}")
    membership["gene"] = membership["gene"].astype(str)
    membership["edge_weight"] = pd.to_numeric(membership["edge_weight"], errors="coerce").fillna(0).abs()
    grouped = membership.groupby(["gene", "channel", "edge_type"], as_index=False).agg(
        incident_edge_count=("edge_id", "nunique"),
        total_edge_weight=("edge_weight", "sum"),
    )
    gates = bundle.edge_gates[["channel", "edge_type", "gate"]].copy()
    grouped = grouped.merge(gates, on=["channel", "edge_type"], how="left")
    grouped = grouped.rename(columns={"gate": "final_gate"})
    directed = grouped["channel"].eq("directed")
    grouped["weighted_gate_exposure"] = np.where(
        directed, grouped["total_edge_weight"] * grouped["final_gate"], np.nan
    )
    grouped["structural_exposure"] = np.where(
        directed, grouped["weighted_gate_exposure"], grouped["total_edge_weight"]
    )
    missing_genes = sorted(set(bundle.genes) - set(grouped["gene"]))
    if missing_genes:
        grouped = pd.concat([
            grouped,
            pd.DataFrame({
                "gene": missing_genes,
                "channel": "none",
                "edge_type": "no_edge",
                "incident_edge_count": 0,
                "total_edge_weight": 0.0,
                "final_gate": np.nan,
                "weighted_gate_exposure": np.nan,
                "structural_exposure": 0.0,
            }),
        ], ignore_index=True)
    return grouped[columns]


def plot_gene_prior_exposure(exposure: pd.DataFrame, gene_order: list[str], classes: pd.DataFrame):
    apply_plot_style()
    if exposure.empty:
        raise ValueError("gene prior exposure is unavailable")
    channels = [x for x in ("directed", "undirected") if x in set(exposure["channel"])]
    fig, axes = plt.subplots(
        1, len(channels), figsize=(max(8, 0.5 * exposure["edge_type"].nunique()), max(7, 0.12 * len(gene_order))),
        squeeze=False, constrained_layout=True,
    )
    for ax, channel in zip(axes.flat, channels):
        subset = exposure[exposure["channel"] == channel]
        value = "weighted_gate_exposure" if channel == "directed" else "structural_exposure"
        pivot = subset.pivot_table(index="gene", columns="edge_type", values=value, aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(gene_order, fill_value=0)
        shown = np.log1p(pivot.to_numpy())
        image = ax.imshow(shown, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=70, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_title(f"{channel.title()} exposure\nlog1p({value})")
        fig.colorbar(image, ax=ax, fraction=0.025)
    fig.suptitle("Gene prior exposure ordered by resolution atlas")
    return fig
