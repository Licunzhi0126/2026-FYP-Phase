"""Per-cell phase allocation calculation and violin plot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phasehyper.visualization.plot_style import apply_plot_style, POS


def compute_phase_allocation(bundle) -> pd.DataFrame:
    energy_a = np.square(bundle.phase_a).sum(axis=1)
    energy_b = np.square(bundle.phase_b).sum(axis=1)
    total = energy_a + energy_b + 1e-12
    score = np.clip((energy_b - energy_a) / total, -1.0, 1.0)
    return pd.DataFrame({
        "cell_id": bundle.cell_ids,
        "label_id": bundle.labels,
        "label_name": bundle.label_names,
        "energy_A": energy_a,
        "energy_B": energy_b,
        "allocation_score": score,
        "phase_B_share": energy_b / total,
    })


def plot_phase_allocation(data: pd.DataFrame):
    apply_plot_style()
    labels = list(dict.fromkeys(data["label_name"].astype(str)))
    groups = [data.loc[data["label_name"].astype(str) == label, "allocation_score"] for label in labels]
    fig, (ax, hist_ax) = plt.subplots(
        1, 2, figsize=(max(9, len(labels) * 1.2), 5),
        gridspec_kw={"width_ratios": [4, 1]}, constrained_layout=True,
    )
    parts = ax.violinplot(groups, positions=range(len(labels)), showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(POS)
        body.set_alpha(0.45)
    ax.boxplot(groups, positions=range(len(labels)), widths=0.16, showfliers=False)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(range(len(labels)), [f"{x}\n(n={len(g)})" for x, g in zip(labels, groups)], rotation=30, ha="right")
    ax.set_ylabel("Phase allocation score (B − A)")
    ax.set_title("Phase allocation by cell group")
    values = data["allocation_score"].to_numpy()
    hist_ax.hist(values, bins=25, orientation="horizontal", color=POS, alpha=0.65)
    for value, style in (
        (np.median(values), "-"),
        (np.quantile(values, 0.05), "--"),
        (np.quantile(values, 0.95), "--"),
    ):
        hist_ax.axhline(value, color="#555", linestyle=style, linewidth=0.8)
    hist_ax.set_ylim(-1.05, 1.05)
    hist_ax.set_title("All cells")
    return fig
