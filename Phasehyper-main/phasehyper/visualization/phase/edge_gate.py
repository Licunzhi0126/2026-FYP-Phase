"""Directed edge-type gate diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phasehyper.visualization.plot_style import apply_plot_style, POS


def merge_edge_gate_data(edge_gates: pd.DataFrame, edge_summary: pd.DataFrame) -> pd.DataFrame:
    if edge_gates.empty:
        raise ValueError("edge_gates.csv is empty")
    gates = edge_gates.rename(columns={"gate": "gate_final"}).copy()
    summary = edge_summary.copy()
    if "channel" in summary:
        summary = summary[summary["channel"] == "directed"]
    merged = gates.merge(summary, on=["channel", "edge_type"], how="left")
    merged["gate_initial"] = 0.9
    merged["gate_delta"] = merged["gate_final"] - merged["gate_initial"]
    candidates = pd.to_numeric(merged.get("candidate_count", 0), errors="coerce").fillna(0)
    n_edges = pd.to_numeric(merged.get("n_edges", 0), errors="coerce").fillna(0)
    merged["retention_rate"] = n_edges / np.maximum(candidates, 1)
    merged["sort_rank"] = merged["gate_final"].rank(method="first", ascending=False).astype(int)
    return merged.sort_values(["gate_final", "edge_type"], ascending=[True, True])


def plot_edge_gates(data: pd.DataFrame):
    apply_plot_style()
    y = np.arange(len(data))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, max(5, 0.34 * len(data))), sharey=True)
    left.hlines(y, 0.9, data["gate_final"], color="#aaa", linewidth=1.5)
    left.scatter(data["gate_final"], y, color=POS, zorder=3)
    left.axvline(0.9, linestyle="--", color="#555", linewidth=0.9, label="Initial prior = 0.9")
    left.set_xlim(0, 1.02)
    left.set_yticks(y, data["edge_type"])
    left.set_xlabel("Final gate")
    left.legend(loc="lower left", fontsize=8)
    n_edges = pd.to_numeric(data.get("n_edges", 0), errors="coerce").fillna(0)
    coverage = pd.to_numeric(data.get("node_coverage", 0), errors="coerce").fillna(0)
    right.scatter(np.log10(n_edges + 1), y, s=25 + 100 * np.clip(coverage, 0, 1), color=POS, alpha=0.7)
    right.set_xlabel("log10(n_edges + 1)")
    right.set_title("Structural support (size = node coverage)")
    fig.suptitle("Directed edge-type gates")
    fig.tight_layout()
    return fig
