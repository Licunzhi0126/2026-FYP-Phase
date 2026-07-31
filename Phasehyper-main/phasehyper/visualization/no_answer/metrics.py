"""Representation metric table and point plot."""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from ..plot_style import POS, apply_plot_style


def build_metrics_table(metrics: dict) -> pd.DataFrame:
    order = ("Raw_RNA", "cell_h", "Phase_A", "Phase_B")
    rows = []
    for representation in order:
        values = metrics[representation]
        rows.append({"representation": representation, **{
            key: values[key] for key in ("NMI", "FMI", "ARI", "ASW", "PredClusters")
        }})
    return pd.DataFrame(rows)


def plot_representation_metrics(table: pd.DataFrame):
    apply_plot_style()
    metrics = ("NMI", "FMI", "ARI", "ASW")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    labels = table["representation"].str.replace("_", " ").tolist()
    y = list(range(len(table)))
    for ax, metric in zip(axes.flat, metrics):
        values = table[metric].astype(float).to_numpy()
        ax.hlines(y, 0, values, color="#c9c8c4", linewidth=1.5)
        ax.scatter(values, y, color=POS, s=45, zorder=3)
        ax.axvline(0, color="#777", linewidth=0.8)
        for yi, value in zip(y, values):
            ax.text(value, yi, f"  {value:.3f}", va="center", fontsize=8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(metric)
        if metric in {"NMI", "FMI"}:
            ax.set_xlim(-0.03, 1.05)
        elif metric == "ASW":
            ax.set_xlim(-1.05, 1.05)
    predicted = ", ".join(
        f"{row.representation.replace('_', ' ')}={int(row.PredClusters)}"
        for row in table.itertuples()
    )
    fig.suptitle(f"Representation metrics\nPredicted clusters: {predicted}", fontsize=11)
    return fig
