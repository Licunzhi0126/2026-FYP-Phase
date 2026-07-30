from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
except Exception:
    umap = None


def build_similarity_from_embedding(Z: np.ndarray) -> np.ndarray:
    Z = np.nan_to_num(Z, nan=0.0).astype(np.float32)
    if Z.shape[0] < 2:
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float32)
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Z_norm = np.clip((Z - mean) / std, -10, 10)
    dist2 = (
        np.sum(Z_norm ** 2, axis=1, keepdims=True)
        + np.sum(Z_norm ** 2, axis=1, keepdims=True).T
        - 2 * Z_norm @ Z_norm.T
    )
    dist2 = np.clip(dist2, 0, None)
    sigma = np.sqrt(np.median(dist2[np.triu_indices_from(dist2, k=1)])) + 1e-8
    sim = np.exp(-dist2 / (2 * sigma ** 2))
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def _try_umap_or_pca(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if umap is not None and n >= 3:
        try:
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
            )
            return reducer.fit_transform(embedding).astype(np.float32)
        except Exception:
            pass
    n_components = 2 if embedding.shape[1] >= 2 else 1
    coords = (
        PCA(n_components=n_components, random_state=42)
        .fit_transform(embedding)
        .astype(np.float32)
    )
    if coords.shape[1] == 1:
        coords = np.concatenate(
            [coords, np.zeros((coords.shape[0], 1), dtype=np.float32)], axis=1
        )
    return coords


def _encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    unique = sorted({str(x) for x in labels})
    mapping = {name: idx for idx, name in enumerate(unique)}
    encoded = np.array([mapping[str(x)] for x in labels], dtype=np.int64)
    return encoded, mapping


def _save_embedding_umap(
    path: Path, embedding: np.ndarray, label_names: List[str], title: str
) -> None:
    coords = _try_umap_or_pca(np.asarray(embedding, dtype=np.float32))
    enc, mapping = _encode_labels(label_names)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=enc, cmap="viridis", s=40)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=scatter.cmap(scatter.norm(i)))
        for i in mapping.values()
    ]
    plt.legend(
        handles, mapping.keys(), title="Cell Stage", bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def visualize_phase_results(
    z_genes: np.ndarray,
    g_genes: np.ndarray,
    gene_names: List[str],
    expression_df: pd.DataFrame,
    label_ids: np.ndarray,
    label_map: Dict[int, str],
    sample_names: List[str],
    output_path: Path,
    *,
    xm: Optional[np.ndarray] = None,
    xp: Optional[np.ndarray] = None,
    Cm: Optional[np.ndarray] = None,
    Cp: Optional[np.ndarray] = None,
    shared: Optional[np.ndarray] = None,
    phase_a: Optional[np.ndarray] = None,
    phase_b: Optional[np.ndarray] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_names = [str(name) for name in sample_names]
    label_names = [
        label_map.get(int(label), str(label)) for label in np.asarray(label_ids, dtype=np.int64)
    ]
    sample_embedding = (
        expression_df.loc[sample_names].values.astype(np.float32)
        if sample_names
        else np.zeros((0, 2), dtype=np.float32)
    )
    gene_coords = _try_umap_or_pca(np.asarray(z_genes, dtype=np.float32))
    sample_coords = _try_umap_or_pca(sample_embedding)
    xm_coords = _try_umap_or_pca(np.asarray(xm, dtype=np.float32)) if xm is not None and len(xm) > 0 else None
    xp_coords = _try_umap_or_pca(np.asarray(xp, dtype=np.float32)) if xp is not None and len(xp) > 0 else None
    shared_coords = _try_umap_or_pca(np.asarray(shared, dtype=np.float32)) if shared is not None and len(shared) > 0 else None
    phase_a_coords = _try_umap_or_pca(np.asarray(phase_a, dtype=np.float32)) if phase_a is not None and len(phase_a) > 0 else None
    phase_b_coords = _try_umap_or_pca(np.asarray(phase_b, dtype=np.float32)) if phase_b is not None and len(phase_b) > 0 else None

    gene_scores = np.asarray(g_genes, dtype=np.float32)
    maternal_idx = np.argsort(gene_scores)[: min(10, len(gene_scores))]
    paternal_idx = np.argsort(-gene_scores)[: min(10, len(gene_scores))]

    has_shared_components = (
        shared_coords is not None and phase_a_coords is not None and phase_b_coords is not None
    )
    n_cols = 3 if has_shared_components else 3
    n_rows = 3 if has_shared_components else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 2:
        axes = axes.reshape(2, -1)

    ax = axes[0, 0]
    scatter = ax.scatter(gene_coords[:, 0], gene_coords[:, 1], c=gene_scores, cmap="coolwarm", s=30)
    ax.set_title("Gene Embedding")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Paternal Gate")

    ax = axes[0, 1]
    enc, mapping = _encode_labels(label_names)
    sample_scatter = ax.scatter(sample_coords[:, 0], sample_coords[:, 1], c=enc, cmap="viridis", s=25)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=sample_scatter.cmap(sample_scatter.norm(i)))
        for i in mapping.values()
    ]
    ax.legend(handles, mapping.keys(), title="Cell Stage", fontsize=8, loc="best")
    ax.set_title("Sample Expression")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    ax = axes[0, 2]
    ax.hist(gene_scores, bins=min(20, max(5, len(gene_scores) // 3)), color="#4C72B0", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_title("Gene Gate Distribution")
    ax.set_xlabel("Paternal Gate")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    maternal_names = [gene_names[idx] for idx in maternal_idx][::-1]
    maternal_scores = gene_scores[maternal_idx][::-1]
    ax.barh(maternal_names, maternal_scores, color="#55A868")
    ax.set_title("Top Maternal Genes")
    ax.set_xlabel("Gate")

    ax = axes[1, 1]
    paternal_names = [gene_names[idx] for idx in paternal_idx][::-1]
    paternal_scores = gene_scores[paternal_idx][::-1]
    ax.barh(paternal_names, paternal_scores, color="#C44E52")
    ax.set_title("Top Paternal Genes")
    ax.set_xlabel("Gate")

    ax = axes[1, 2]
    if Cm is not None and Cp is not None and Cm.size > 0 and Cp.size > 0:
        diff = np.asarray(Cp, dtype=np.float32) - np.asarray(Cm, dtype=np.float32)
        im = ax.imshow(diff, aspect="auto", cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Cp - Cm")
    else:
        ax.axis("off")

    if has_shared_components:
        ax = axes[2, 0]
        ax.scatter(shared_coords[:, 0], shared_coords[:, 1], c="green", s=30, alpha=0.7)
        ax.set_title("Shared Component")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        ax = axes[2, 1]
        ax.scatter(phase_a_coords[:, 0], phase_a_coords[:, 1], c="blue", s=30, alpha=0.7)
        ax.set_title("Phase A (Maternal-specific)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        ax = axes[2, 2]
        ax.scatter(phase_b_coords[:, 0], phase_b_coords[:, 1], c="red", s=30, alpha=0.7)
        ax.set_title("Phase B (Paternal-specific)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
