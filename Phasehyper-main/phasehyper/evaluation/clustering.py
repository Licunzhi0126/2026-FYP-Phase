from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import igraph as ig
import leidenalg
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from phasehyper.config import DEFAULT_LEIDEN_RESOLUTION


def _build_knn_graph(embedding: np.ndarray, k: int = 10) -> ig.Graph:
    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    if n <= 1:
        return ig.Graph(n=max(1, n))
    k = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(embedding)
    indices = nbrs.kneighbors(embedding, return_distance=False)
    edges = set()
    for src, row in enumerate(indices):
        for dst in row[1:]:
            if int(dst) == src:
                continue
            a, b = sorted((src, int(dst)))
            edges.add((a, b))
    graph = ig.Graph(n=n, edges=sorted(edges), directed=False)
    graph.simplify()
    return graph


def _cluster_embedding(
    embedding: np.ndarray,
    *,
    method: str,
    dataset_type: str,
    expected_clusters: Optional[int] = None,
    resolution: Optional[float] = None,
    seed: int = 42,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)
    expected_clusters = int(expected_clusters or min(3, n))
    if not 2 <= expected_clusters <= n:
        raise ValueError(
            f"expected_clusters must be between 2 and {n}, got {expected_clusters}"
        )

    if method == "kmeans":
        return (
            KMeans(n_clusters=expected_clusters, random_state=seed, n_init=10)
            .fit_predict(embedding)
            .astype(np.int64)
        )

    graph = _build_knn_graph(embedding, k=min(10, n - 1))
    if graph.ecount() == 0:
        return np.zeros(n, dtype=np.int64)

    if method == "leiden":
        resolution = float(
            resolution
            if resolution is not None
            else DEFAULT_LEIDEN_RESOLUTION.get(dataset_type, 1.0)
        )
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )
        return np.asarray(partition.membership, dtype=np.int64)

    if method == "louvain":
        return np.asarray(graph.community_multilevel().membership, dtype=np.int64)

    raise ValueError(f"Unsupported cluster method: {method}")


def prepare_embedding(
    embedding: np.ndarray,
    *,
    use_pca: bool = False,
    pca_dim: int = 30,
    pca_seed: int = 0,
) -> np.ndarray:
    """Validate and normalise a cell-by-feature matrix for post-hoc scoring."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"embedding must be a 2D array, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embedding must contain at least one sample and one feature")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not use_pca:
        return values

    max_dim = min(values.shape[0] - 1, values.shape[1])
    if max_dim < 1:
        return np.zeros((values.shape[0], 1), dtype=np.float64)
    target_dim = min(max(1, int(pca_dim)), max_dim)
    if float(values.std()) < 1e-9:
        return np.zeros((values.shape[0], target_dim), dtype=np.float64)
    return PCA(n_components=target_dim, random_state=pca_seed).fit_transform(values)


def _validated_labels(true_labels: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(true_labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.shape[0] != n_samples:
        raise ValueError(
            f"label length {labels.shape[0]} does not match sample count {n_samples}"
        )
    return labels


def evaluate_clustering(
    embedding: np.ndarray,
    true_labels: np.ndarray,
    *,
    method: str = "kmeans",
    expected_clusters: int | None = None,
    dataset_type: str = "default",
    resolution: float | None = None,
    seed: int = 42,
    use_pca: bool = False,
    pca_dim: int = 30,
) -> dict[str, float | int]:
    """Cluster one representation and return the common post-hoc metrics."""
    values = prepare_embedding(
        embedding, use_pca=use_pca, pca_dim=pca_dim, pca_seed=seed
    )
    labels = _validated_labels(true_labels, values.shape[0])
    n_samples = values.shape[0]
    if n_samples < 2:
        raise ValueError("clustering evaluation requires at least two samples")
    if expected_clusters is None:
        expected_clusters = int(np.unique(labels).size)
    expected_clusters = int(expected_clusters)
    if not 2 <= expected_clusters <= n_samples:
        raise ValueError(
            f"expected_clusters must be between 2 and {n_samples}, got {expected_clusters}"
        )

    predicted = _cluster_embedding(
        values,
        method=method,
        dataset_type=dataset_type,
        expected_clusters=expected_clusters,
        resolution=resolution,
        seed=seed,
    )
    unique_true = np.unique(labels)
    asw = (
        float(silhouette_score(values, labels))
        if 1 < unique_true.size < n_samples
        else float("nan")
    )
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
        "fmi": float(fowlkes_mallows_score(labels, predicted)),
        "asw": asw,
        "expected_clusters": expected_clusters,
        "pred_clusters": int(np.unique(predicted).size),
    }


def evaluate_clustering_stability(
    embedding: np.ndarray,
    true_labels: np.ndarray,
    *,
    expected_clusters: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    use_pca: bool = True,
    pca_dim: int = 30,
) -> dict:
    """Return the mean and spread of KMeans ARI over fixed random seeds."""
    if not seeds:
        raise ValueError("seeds must contain at least one value")
    values = prepare_embedding(
        embedding, use_pca=use_pca, pca_dim=pca_dim, pca_seed=0
    )
    labels = _validated_labels(true_labels, values.shape[0])
    n_samples = values.shape[0]
    clusters = int(expected_clusters)
    if not 2 <= clusters <= n_samples:
        raise ValueError(
            f"expected_clusters must be between 2 and {n_samples}, got {clusters}"
        )
    runs = [
        float(
            adjusted_rand_score(
                labels,
                KMeans(
                    n_clusters=clusters, n_init=5, random_state=int(seed)
                ).fit_predict(values),
            )
        )
        for seed in seeds
    ]
    return {
        "ari_mean": float(np.mean(runs)),
        "ari_std": float(np.std(runs)),
        "ari_runs": runs,
    }


def _emb(X, d=30):
    """Compatibility wrapper for the former root-level helper."""
    values = np.nan_to_num(np.asarray(X, dtype=float))
    if values.ndim != 2:
        raise ValueError(f"embedding must be a 2D array, got shape {values.shape}")
    if float(values.std()) < 1e-9:
        return np.zeros((values.shape[0], 2))
    return prepare_embedding(values, use_pca=True, pca_dim=d, pca_seed=0)


def _ari(Z, k, y):
    """Compatibility wrapper preserving the former five-seed mean ARI."""
    return evaluate_clustering_stability(
        Z,
        y,
        expected_clusters=k,
        seeds=(0, 1, 2, 3, 4),
        use_pca=False,
    )["ari_mean"]


def _evaluate_embedding_metrics(
    embedding,
    true_ids,
    *,
    dataset_type,
    cluster_method,
    cluster_resolution,
) -> Dict[str, float]:
    embedding = np.asarray(embedding, dtype=np.float32)
    true_ids = np.asarray(true_ids, dtype=np.int64)
    n = embedding.shape[0]
    unique_true = np.unique(true_ids)
    if n < 2 or unique_true.size < 2:
        return {"fmi": 0.0, "nmi": 0.0, "ari": 0.0, "pred_clusters": 1}
    pred = _cluster_embedding(
        embedding,
        method=cluster_method,
        dataset_type=dataset_type,
        expected_clusters=len(unique_true),
        resolution=cluster_resolution,
        seed=42,
    )
    unique_pred = np.unique(pred)
    return {
        "fmi": float(fowlkes_mallows_score(true_ids, pred)),
        "nmi": float(normalized_mutual_info_score(true_ids, pred)),
        "ari": float(adjusted_rand_score(true_ids, pred)),
        "pred_clusters": int(unique_pred.size),
    }


def _align_labels_to_cells(dataset, sample_names) -> Tuple[np.ndarray, List[str]]:
    if sample_names is None:
        return (
            np.asarray(dataset.labels, dtype=np.int64),
            [str(name) for name in dataset.label_names],
        )
    sample_names = [str(name) for name in sample_names]
    label_by_cell = {
        str(cell): int(dataset.labels[idx]) for idx, cell in enumerate(dataset.common_cells)
    }
    label_name_by_cell = {
        str(cell): str(dataset.label_names[idx])
        for idx, cell in enumerate(dataset.common_cells)
    }
    missing = [name for name in sample_names if name not in label_by_cell]
    if missing:
        raise KeyError(f"Missing sample names during metric alignment: {missing[:5]}")
    true_ids = np.asarray([label_by_cell[name] for name in sample_names], dtype=np.int64)
    aligned_label_names = [label_name_by_cell[name] for name in sample_names]
    return true_ids, aligned_label_names
