"""Four-representation PCA computation and plotting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..plot_style import apply_plot_style


def _coordinates(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float))
    keep = np.std(values, axis=0) > 1e-12
    if keep.sum() < 2:
        raise ValueError("PCA input has fewer than two non-constant features")
    scaled = StandardScaler().fit_transform(values[:, keep])
    return PCA(n_components=2, random_state=seed).fit_transform(scaled)


def compute_pca_data(bundle, *, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, values in {
        "Raw_RNA": bundle.raw_rna,
        "cell_h": bundle.cell_h,
        "Phase_A": bundle.phase_a,
        "Phase_B": bundle.phase_b,
    }.items():
        coords = _coordinates(values, seed)
        rows.append(pd.DataFrame({
            "representation": name,
            "cell_id": bundle.cell_ids,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "label_id": bundle.labels,
            "label_name": bundle.label_names,
        }))
    return pd.concat(rows, ignore_index=True)


def plot_four_representation_pca(data: pd.DataFrame, metrics: dict):
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    order = ("Raw_RNA", "cell_h", "Phase_A", "Phase_B")
    labels = list(dict.fromkeys(data["label_name"].astype(str)))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    colors = {label: cmap(i) for i, label in enumerate(labels)}
    for ax, name in zip(axes.flat, order):
        subset = data[data["representation"] == name]
        for label in labels:
            group = subset[subset["label_name"].astype(str) == label]
            ax.scatter(
                group["PC1"], group["PC2"], s=max(6, min(28, 2500 / len(subset))),
                alpha=0.8, color=colors[label], label=label, linewidths=0,
            )
        values = metrics[name]
        ax.set_title(
            f"{name.replace('_', ' ')}\nARI={values['ARI']:.3f}  ASW={values['ASW']:.3f}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Four representations (independently fitted PCAs)")
    return fig
