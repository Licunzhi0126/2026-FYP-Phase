from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None


def embedding_2d(embedding: np.ndarray, seed: int = 42) -> np.ndarray:
    x = np.nan_to_num(np.asarray(embedding, dtype=np.float32))
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.pad(x[:, :1], ((0, 0), (0, 1)))
    if umap is not None:
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(x).astype(np.float32)
    return PCA(n_components=2, random_state=seed).fit_transform(x).astype(np.float32)


def save_umap(
    embedding: np.ndarray,
    *,
    cell_names: List[str],
    labels: List[str],
    title: str,
    figure_path: Path,
    data_path: Path,
    seed: int,
) -> pd.DataFrame:
    coords = embedding_2d(embedding, seed=seed)
    frame = pd.DataFrame({"cell_id": cell_names, "UMAP1": coords[:, 0], "UMAP2": coords[:, 1], "label": labels})
    frame.to_csv(data_path, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, group in frame.groupby("label", sort=False):
        ax.scatter(group["UMAP1"], group["UMAP2"], s=18, alpha=0.8, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if frame["label"].nunique() <= 15:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return frame


def save_umap_overview(frames: Dict[str, pd.DataFrame], path: Path) -> None:
    if not frames:
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(6 * len(frames), 5), squeeze=False)
    for ax, (name, frame) in zip(axes[0], frames.items()):
        for label, group in frame.groupby("label", sort=False):
            ax.scatter(group["UMAP1"], group["UMAP2"], s=12, alpha=0.8, label=str(label))
        ax.set_title(name)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(8, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(frame: pd.DataFrame, path: Path, *, source_rows: bool = False) -> None:
    if frame.empty:
        return
    row_groups = list(frame.groupby("source", sort=False)) if source_rows and "source" in frame else [("metrics", frame)]
    fig, axes = plt.subplots(len(row_groups), 3, figsize=(16, 5 * len(row_groups)), squeeze=False)
    for row_idx, (source, group) in enumerate(row_groups):
        x = np.arange(len(group))
        for col_idx, metric in enumerate(["ARI", "NMI", "FMI"]):
            ax = axes[row_idx, col_idx]
            ax.bar(x, group[metric].to_numpy(dtype=float))
            ax.set_xticks(x)
            ax.set_xticklabels(group["representation"], rotation=25, ha="right")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{source}: {metric}" if source_rows else metric)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

