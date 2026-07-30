from __future__ import annotations

from typing import Any

import numpy as np

from phasehyper.evaluation.clustering import evaluate_clustering


def evaluate_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    *,
    method: str = "kmeans",
    seed: int = 0,
    use_pca: bool = False,
    pca_dim: int = 30,
) -> dict[str, float | int]:
    """Evaluate one real-data representation with the common metric fields."""
    derived_clusters = int(np.unique(labels).size)
    if int(n_clusters) != derived_clusters:
        raise ValueError(
            "n_clusters does not match unique label count: "
            f"provided={n_clusters}, derived={derived_clusters}"
        )
    metrics = evaluate_clustering(
        embedding,
        labels,
        method=method,
        expected_clusters=n_clusters,
        seed=seed,
        use_pca=use_pca,
        pca_dim=pca_dim,
    )
    return {
        "NMI": metrics["nmi"],
        "FMI": metrics["fmi"],
        "ARI": metrics["ari"],
        "ASW": metrics["asw"],
        "ExpectedClusters": metrics["expected_clusters"],
        "PredClusters": metrics["pred_clusters"],
    }


def evaluate_phase_embeddings(
    *,
    raw_rna: np.ndarray,
    cell_embedding: np.ndarray,
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Evaluate the four representations reported by ``run_phase.py``."""
    representations = {
        "Raw_RNA": raw_rna,
        "cell_h": cell_embedding,
        "Phase_A": phase_a,
        "Phase_B": phase_b,
    }
    return {
        name: evaluate_embedding(values, labels, n_clusters, seed=seed)
        for name, values in representations.items()
    }


def _phase_arrays(
    raw_rna: np.ndarray, phase_a: np.ndarray, phase_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_rna)
    a = np.asarray(phase_a)
    b = np.asarray(phase_b)
    if raw.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("raw_rna, phase_a and phase_b must all be 2D arrays")
    if not (raw.shape == a.shape == b.shape):
        raise ValueError(
            f"phase shapes must match raw RNA: raw={raw.shape}, A={a.shape}, B={b.shape}"
        )
    return raw, a, b


def evaluate_phase_quality(
    raw_rna: np.ndarray,
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute unsupervised numeric quality checks for the two phase outputs."""
    raw, a, b = _phase_arrays(raw_rna, phase_a, phase_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    row_denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return {
        "reconstruction_relative_error": float(
            np.linalg.norm(a + b - raw) / max(float(np.linalg.norm(raw)), 1e-12)
        ),
        "phase_cosine_similarity": float(
            np.mean(np.sum(a * b, axis=1) / row_denom)
        ),
        "phase_energy_ratio": float(
            norm_a**2 / max(norm_a**2 + norm_b**2, eps)
        ),
        "phase_imbalance": float(
            np.mean(np.abs(a - b)) / (np.mean(np.abs(a + b)) + eps)
        ),
    }


def evaluate_phase_model(
    *,
    raw_rna: np.ndarray,
    cell_embedding: np.ndarray,
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Return the complete JSON-ready real-data evaluation payload."""
    result: dict[str, Any] = evaluate_phase_embeddings(
        raw_rna=raw_rna,
        cell_embedding=cell_embedding,
        phase_a=phase_a,
        phase_b=phase_b,
        labels=labels,
        n_clusters=n_clusters,
        seed=seed,
    )
    result.update(evaluate_phase_quality(raw_rna, phase_a, phase_b))
    return result
