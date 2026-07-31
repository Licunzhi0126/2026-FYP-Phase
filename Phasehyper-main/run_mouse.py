"""Unsupervised allele phasing for the GSE80810 WT mouse embryo data.

The CountTable is the model input and RPRT is the only source used to build
the biological hypergraph.  AllelicRatio is deliberately kept out of the
training path and is read only for post-training evaluation.

Examples
--------
Full run (the project environment is ``conda activate new_bi``)::

    python run_mouse.py

Small end-to-end smoke run::

    python run_mouse.py --max-genes 100 --epochs 2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from phasehyper.evaluation.metrics_io import (
    print_final,
    print_headline,
    print_orientation_audit,
    print_saber_table,
    save_saber_evaluation,
)
from phasehyper.evaluation.simulation import (
    evaluate_embedding_quality,
    evaluate_simulation_clustering,
    evaluate_simulation_expression,
)
from phasehyper.model import build_criterion, build_model, build_optimizer


DEFAULT_DATA_DIR = Path("mouse_data") / "GSE80810"
DEFAULT_OUTPUT_DIR = Path("result_mouse")
STAGE_ORDER = ("Oo", "2C", "4C", "8C", "16C", "32C", "64C")
EPS = 1e-8


def _read_gene_table(path: Path, *, has_chromosomes: bool) -> pd.DataFrame:
    """Read one gene-by-cell CSV while preserving NA values and axis order."""
    if not path.is_file():
        raise FileNotFoundError(f"Mouse input file does not exist: {path}")
    frame = pd.read_csv(path, na_values=["NA", "NaN", ""])
    required = ["Genes"] + (["Chromosomes"] if has_chromosomes else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame["Genes"].isna().any():
        raise ValueError(f"{path.name} contains an empty gene identifier")
    frame["Genes"] = frame["Genes"].astype(str)
    duplicates = frame.loc[frame["Genes"].duplicated(), "Genes"].unique()
    if len(duplicates):
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"{path.name} contains duplicate genes: {preview}")
    metadata = {"Genes", "Chromosomes"}
    cell_columns = [column for column in frame.columns if column not in metadata]
    if not cell_columns:
        raise ValueError(f"{path.name} contains no cell columns")
    frame[cell_columns] = frame[cell_columns].apply(pd.to_numeric, errors="coerce")
    return frame


def _stage_from_cell(cell_id: str) -> str:
    return str(cell_id).split("_", 1)[0]


def load_mouse_data(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    max_genes: int | None = None,
) -> dict[str, Any]:
    """Load, validate and align the three WT tables.

    Axes in the returned numerical matrices are cell x gene.  Gene ordering
    follows the AllelicRatio table and cell ordering follows CountTable.
    """
    data_dir = Path(data_dir)
    count_frame = _read_gene_table(
        data_dir / "GSE80810_CountTable_WT.csv", has_chromosomes=False
    )
    rprt_frame = _read_gene_table(
        data_dir / "GSE80810_RPRT_WT.csv", has_chromosomes=True
    )
    ratio_frame = _read_gene_table(
        data_dir / "GSE80810_AllelicRatio_WT.csv", has_chromosomes=True
    )

    count_cells = [column for column in count_frame.columns if column != "Genes"]
    rprt_cells = [
        column
        for column in rprt_frame.columns
        if column not in {"Genes", "Chromosomes"}
    ]
    ratio_cells = [
        column
        for column in ratio_frame.columns
        if column not in {"Genes", "Chromosomes"}
    ]
    if set(count_cells) != set(rprt_cells) or set(count_cells) != set(ratio_cells):
        raise ValueError(
            "CountTable, RPRT and AllelicRatio do not contain the same cells"
        )

    count_genes = set(count_frame["Genes"])
    rprt_genes = set(rprt_frame["Genes"])
    common_genes = [
        gene
        for gene in ratio_frame["Genes"]
        if gene in count_genes and gene in rprt_genes
    ]
    if not common_genes:
        raise ValueError("The three WT tables have no genes in common")

    count_indexed = count_frame.set_index("Genes")
    rprt_indexed = rprt_frame.set_index("Genes")
    ratio_indexed = ratio_frame.set_index("Genes")

    if max_genes is not None:
        if max_genes < 2:
            raise ValueError(f"max_genes must be at least 2, got {max_genes}")
        if max_genes < len(common_genes):
            count_candidates = (
                count_indexed.loc[common_genes, count_cells]
                .to_numpy(dtype=np.float64)
                .T
            )
            variability = np.var(np.log1p(np.clip(count_candidates, 0, None)), axis=0)
            order = np.argsort(-variability, kind="stable")[:max_genes]
            selected = set(np.asarray(common_genes, dtype=object)[order].tolist())
            common_genes = [gene for gene in common_genes if gene in selected]

    chromosomes = (
        ratio_indexed.loc[common_genes, "Chromosomes"].fillna("").astype(str).tolist()
    )
    count = (
        count_indexed.loc[common_genes, count_cells]
        .to_numpy(dtype=np.float32)
        .T
    )
    rprt = (
        rprt_indexed.loc[common_genes, ["Chromosomes", *count_cells]]
        .drop(columns=["Chromosomes"])
        .to_numpy(dtype=np.float32)
        .T
    )
    ratio = (
        ratio_indexed.loc[common_genes, ["Chromosomes", *count_cells]]
        .drop(columns=["Chromosomes"])
        .to_numpy(dtype=np.float32)
        .T
    )
    if not np.isfinite(count).all() or not np.isfinite(rprt).all():
        raise ValueError("CountTable and RPRT must not contain missing values")
    if (count < 0).any() or (rprt < 0).any():
        raise ValueError("CountTable and RPRT must contain non-negative values")
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size and ((finite_ratio < 0).any() or (finite_ratio > 1).any()):
        raise ValueError("Observed AllelicRatio values must be within [0, 1]")

    stages = [_stage_from_cell(cell) for cell in count_cells]
    unknown_stages = sorted(set(stages).difference(STAGE_ORDER))
    if unknown_stages:
        raise ValueError(f"Unsupported developmental stages: {unknown_stages}")
    return {
        "count": count,
        "rprt": rprt,
        "ratio": ratio,
        "label_mask": np.isfinite(ratio),
        "genes": common_genes,
        "chromosomes": chromosomes,
        "cells": count_cells,
        "stages": stages,
        "data_dir": data_dir,
    }


def preprocess_count_table(
    count: np.ndarray, *, target_sum: float = 1e4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CP10K + log1p, followed by per-gene standardisation."""
    values = np.asarray(count, dtype=np.float64)
    library_size = values.sum(axis=1, keepdims=True)
    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size),
        where=library_size > 0,
    )
    log_count = np.log1p(values * scale)
    mu = log_count.mean(axis=0)
    sigma = log_count.std(axis=0)
    safe_sigma = np.where(sigma > EPS, sigma, 1.0)
    standardized = (log_count - mu) / safe_sigma
    return (
        standardized.astype(np.float32),
        log_count.astype(np.float32),
        mu.astype(np.float32),
        safe_sigma.astype(np.float32),
    )


def standardize_gene_profiles(rprt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return log1p RPRT and zero-mean/unit-norm gene profiles."""
    log_rprt = np.log1p(np.clip(np.asarray(rprt, dtype=np.float64), 0, None))
    profiles = log_rprt.T
    profiles -= profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = np.divide(
        profiles, norms, out=np.zeros_like(profiles), where=norms > EPS
    )
    return log_rprt.astype(np.float32), profiles.astype(np.float32)


def _top_correlation_edges(
    profiles: np.ndarray,
    *,
    top_k: int,
    block_size: int,
) -> list[tuple[list[int], float, str]]:
    """Build exact Pearson-neighbour edges without materialising G x G."""
    n_genes = profiles.shape[0]
    if n_genes < 2:
        return []
    resolved_k = min(max(1, int(top_k)), n_genes - 1)
    block_size = max(1, int(block_size))
    edges: list[tuple[list[int], float, str]] = []
    profiles_t = profiles.T
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        similarity = profiles[start:stop] @ profiles_t
        local_rows = np.arange(stop - start)
        similarity[local_rows, np.arange(start, stop)] = -np.inf
        candidates = np.argpartition(
            similarity, kth=n_genes - resolved_k, axis=1
        )[:, -resolved_k:]
        for local_index, candidate_indices in enumerate(candidates):
            gene_index = start + local_index
            values = similarity[local_index, candidate_indices]
            order = np.argsort(-values, kind="stable")
            neighbours = candidate_indices[order]
            correlations = values[order]
            positive = neighbours[np.isfinite(correlations) & (correlations > 0)]
            if not len(positive):
                continue
            nodes = [gene_index, *positive.astype(int).tolist()]
            weight = max(float(np.mean(correlations[: len(positive)])), 0.05)
            edges.append((nodes, weight, "expression_similarity"))
    return edges


def _stage_edges(
    log_rprt: np.ndarray,
    stages: list[str],
    *,
    stage_gene_count: int,
) -> list[tuple[list[int], list[int], float, str]]:
    """Create cell-to-gene stage edges from RPRT stage specificity."""
    n_cells, n_genes = log_rprt.shape
    edges: list[tuple[list[int], list[int], float, str]] = []
    all_rows = np.arange(n_cells)
    for stage in STAGE_ORDER:
        stage_rows = np.array([i for i, value in enumerate(stages) if value == stage])
        if not len(stage_rows):
            continue
        other_rows = np.setdiff1d(all_rows, stage_rows, assume_unique=True)
        stage_mean = log_rprt[stage_rows].mean(axis=0)
        baseline = (
            log_rprt[other_rows].mean(axis=0)
            if len(other_rows)
            else np.zeros(n_genes, dtype=np.float32)
        )
        specificity = stage_mean - baseline
        positive = np.where(specificity > 0)[0]
        if not len(positive):
            continue
        limit = min(max(2, int(stage_gene_count)), len(positive))
        selected = positive[
            np.argsort(-specificity[positive], kind="stable")[:limit]
        ]
        if len(selected) < 2:
            continue
        weight = max(float(np.mean(specificity[selected])), 0.05)
        edges.append(
            (
                stage_rows.astype(int).tolist(),
                selected.astype(int).tolist(),
                weight,
                f"stage_{stage}",
            )
        )
    return edges


def _module_edges(
    profiles: np.ndarray,
    *,
    n_modules: int,
    seed: int,
) -> list[tuple[list[int], float, str]]:
    """Cluster RPRT gene profiles and turn each non-trivial module into an edge."""
    n_genes = profiles.shape[0]
    resolved_modules = min(max(1, int(n_modules)), n_genes)
    labels = KMeans(
        n_clusters=resolved_modules,
        random_state=seed,
        n_init=10,
    ).fit_predict(profiles)
    edges: list[tuple[list[int], float, str]] = []
    for module_id in range(resolved_modules):
        members = np.where(labels == module_id)[0].astype(int).tolist()
        if len(members) >= 2:
            edges.append(
                (members, math.sqrt(len(members)), f"expression_module_{module_id}")
            )
    return edges


def build_rprt_hypergraph(
    rprt: np.ndarray,
    stages: Iterable[str],
    *,
    top_k: int = 15,
    n_modules: int = 8,
    stage_gene_count: int = 200,
    correlation_block_size: int = 256,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build both shared-model graph channels using RPRT and stage names only."""
    rprt = np.asarray(rprt, dtype=np.float32)
    stage_list = list(stages)
    n_cells, n_genes = rprt.shape
    if len(stage_list) != n_cells:
        raise ValueError("stages length must match the number of RPRT cells")
    log_rprt, profiles = standardize_gene_profiles(rprt)
    similarity_edges = _top_correlation_edges(
        profiles, top_k=top_k, block_size=correlation_block_size
    )
    module_edges = _module_edges(profiles, n_modules=n_modules, seed=seed)
    stage_edges = _stage_edges(
        log_rprt, stage_list, stage_gene_count=stage_gene_count
    )

    n_nodes = n_cells + n_genes
    undir_rows: list[int] = []
    undir_cols: list[int] = []
    undir_values: list[float] = []
    undir_weights: list[float] = []
    directed_tail_rows: list[int] = []
    directed_tail_cols: list[int] = []
    directed_head_rows: list[int] = []
    directed_head_cols: list[int] = []
    directed_weights: list[float] = []
    directed_types: list[int] = []
    edge_type_names: list[str] = []
    edge_type_to_id: dict[str, int] = {}
    source_counts: Counter[str] = Counter()
    undir_index = 0
    directed_index = 0

    def type_id(name: str) -> int:
        if name not in edge_type_to_id:
            edge_type_to_id[name] = len(edge_type_names)
            edge_type_names.append(name)
        return edge_type_to_id[name]

    def add_undirected(nodes: list[int], weight: float, source: str) -> None:
        nonlocal undir_index
        unique_nodes = list(dict.fromkeys(nodes))
        if len(unique_nodes) < 2:
            return
        undir_rows.extend(unique_nodes)
        undir_cols.extend([undir_index] * len(unique_nodes))
        undir_values.extend([1.0] * len(unique_nodes))
        undir_weights.append(weight)
        source_counts[source] += 1
        undir_index += 1

    def add_directed(
        tail: list[int], head: list[int], weight: float, source: str
    ) -> None:
        nonlocal directed_index
        tail = list(dict.fromkeys(tail))
        head = list(dict.fromkeys(head))
        if not tail or not head:
            return
        directed_tail_rows.extend(tail)
        directed_tail_cols.extend([directed_index] * len(tail))
        directed_head_rows.extend(head)
        directed_head_cols.extend([directed_index] * len(head))
        directed_weights.append(weight)
        directed_types.append(type_id(source))
        directed_index += 1

    for gene_nodes, weight, source in similarity_edges + module_edges:
        nodes = [n_cells + gene for gene in gene_nodes]
        add_undirected(nodes, weight, source)
        add_directed(nodes, nodes, weight, source)
    for cell_nodes, gene_nodes, weight, source in stage_edges:
        shifted_genes = [n_cells + gene for gene in gene_nodes]
        add_undirected([*cell_nodes, *shifted_genes], weight, source)
        add_directed(cell_nodes, shifted_genes, weight, f"{source}_inject")
        add_directed(shifted_genes, cell_nodes, weight, f"{source}_readout")

    if undir_index == 0 or directed_index == 0:
        raise ValueError("RPRT did not produce a usable hypergraph")
    h_undirected = sp.csr_matrix(
        (undir_values, (undir_rows, undir_cols)),
        shape=(n_nodes, undir_index),
        dtype=np.float32,
    )
    h_tail = sp.csr_matrix(
        (
            np.ones(len(directed_tail_rows), dtype=np.float32),
            (directed_tail_rows, directed_tail_cols),
        ),
        shape=(n_nodes, directed_index),
    )
    h_head = sp.csr_matrix(
        (
            np.ones(len(directed_head_rows), dtype=np.float32),
            (directed_head_rows, directed_head_cols),
        ),
        shape=(n_nodes, directed_index),
    )
    directed = {
        "H_tail": h_tail,
        "H_head": h_head,
        "W": np.asarray(directed_weights, dtype=np.float32),
        "etype": np.asarray(directed_types, dtype=np.int64),
        "n_types": len(edge_type_names),
        "et_names": edge_type_names,
        "cnt": dict(Counter(edge_type_names[index] for index in directed_types)),
    }
    undirected = {
        "H": h_undirected,
        "W": np.asarray(undir_weights, dtype=np.float32),
    }
    graph_stats = {
        "nodes": n_nodes,
        "cell_nodes": n_cells,
        "gene_nodes": n_genes,
        "directed_edges": directed_index,
        "undirected_edges": undir_index,
        "source_counts": dict(source_counts),
        "source_proportions": {
            source: count / undir_index for source, count in source_counts.items()
        },
        "directed_type_counts": directed["cnt"],
    }
    return directed, undirected, graph_stats


def build_node_features(
    standardized_count: np.ndarray,
    rprt: np.ndarray,
    *,
    dc: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse CountTable and RPRT PCA loadings into gene node features."""
    count_pca = PCA(n_components=dc, random_state=seed).fit(standardized_count)
    log_rprt = np.log1p(np.clip(rprt.astype(np.float64), 0, None))
    rprt_mean = log_rprt.mean(axis=0)
    rprt_std = log_rprt.std(axis=0)
    rprt_standardized = (log_rprt - rprt_mean) / np.where(
        rprt_std > EPS, rprt_std, 1.0
    )
    rprt_pca = PCA(n_components=dc, random_state=seed).fit(rprt_standardized)
    count_loading = count_pca.components_.T.astype(np.float32)
    rprt_loading = rprt_pca.components_.T.astype(np.float32)
    for component in range(dc):
        if np.dot(count_loading[:, component], rprt_loading[:, component]) < 0:
            rprt_loading[:, component] *= -1
    gene_features = 0.5 * (count_loading + rprt_loading)
    target_std = float(np.std(count_loading))
    current_std = float(np.std(gene_features))
    if current_std > EPS:
        gene_features *= target_std / current_std
    return gene_features.astype(np.float32), count_pca.components_.astype(np.float32)


def phases_to_ratio(
    phase_a: np.ndarray, phase_b: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Canonicalise two exchangeable phases and return a bounded A fraction."""
    phase_a = np.asarray(phase_a, dtype=np.float64)
    phase_b = np.asarray(phase_b, dtype=np.float64)
    strength_a = float(np.clip(phase_a, 0, None).sum())
    strength_b = float(np.clip(phase_b, 0, None).sum())
    swapped = strength_a > strength_b
    canonical_a, canonical_b = (
        (phase_b, phase_a) if swapped else (phase_a, phase_b)
    )
    positive_a = np.clip(canonical_a, 0, None)
    positive_b = np.clip(canonical_b, 0, None)
    total = positive_a + positive_b
    ratio = np.divide(
        positive_a,
        total,
        out=np.full_like(total, 0.5, dtype=np.float64),
        where=total > EPS,
    )
    info = {
        "rule": "lower_global_nonnegative_intensity_is_phase_A",
        "swapped": bool(swapped),
        "original_phase_a_strength": strength_a,
        "original_phase_b_strength": strength_b,
        "zero_total_entries": int(np.count_nonzero(total <= EPS)),
    }
    return ratio.astype(np.float32), info


def _safe_pearson(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if prediction.size < 2:
        return None
    if np.std(prediction) <= EPS or np.std(target) <= EPS:
        return None
    value = float(np.corrcoef(prediction, target)[0, 1])
    return value if np.isfinite(value) else None


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    pred = np.asarray(prediction, dtype=np.float64)[mask]
    true = np.asarray(target, dtype=np.float64)[mask]
    if not len(true):
        return {
            "n_observed": 0,
            "mse": None,
            "mae": None,
            "pearson": None,
            "classification_n": 0,
            "classification_accuracy": None,
        }
    class_mask = true != 0.5
    accuracy = (
        float(np.mean((pred[class_mask] > 0.5) == (true[class_mask] > 0.5)))
        if class_mask.any()
        else None
    )
    return {
        "n_observed": int(len(true)),
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
        "pearson": _safe_pearson(pred, true),
        "classification_n": int(class_mask.sum()),
        "classification_accuracy": accuracy,
    }


def evaluate_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    stages: Iterable[str],
    genes: Iterable[str],
    min_gene_observations: int = 10,
) -> dict[str, Any]:
    """Evaluate canonical and globally flipped ratios without changing output."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    stage_list = list(stages)
    gene_list = list(genes)
    canonical = regression_metrics(prediction, target, mask)
    flipped = regression_metrics(1.0 - prediction, target, mask)
    canonical_mse = math.inf if canonical["mse"] is None else canonical["mse"]
    flipped_mse = math.inf if flipped["mse"] is None else flipped["mse"]
    best_orientation = "canonical" if canonical_mse <= flipped_mse else "flipped"
    evaluation_prediction = (
        prediction if best_orientation == "canonical" else 1.0 - prediction
    )

    by_stage: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        rows = np.array([value == stage for value in stage_list])
        if rows.any():
            by_stage[stage] = regression_metrics(
                evaluation_prediction[rows], target[rows], mask[rows]
            )

    gene_rows = []
    for gene_index, gene in enumerate(gene_list):
        gene_mask = mask[:, gene_index]
        support = int(gene_mask.sum())
        if support < min_gene_observations:
            continue
        correlation = _safe_pearson(
            evaluation_prediction[gene_mask, gene_index],
            target[gene_mask, gene_index],
        )
        if correlation is not None:
            gene_rows.append(
                {"gene": gene, "n_observed": support, "pearson": correlation}
            )
    correlations = np.asarray(
        [row["pearson"] for row in gene_rows], dtype=np.float64
    )
    gene_summary = {
        "minimum_observations": int(min_gene_observations),
        "n_eligible_genes": len(gene_rows),
        "mean_pearson": float(correlations.mean()) if len(correlations) else None,
        "median_pearson": (
            float(np.median(correlations)) if len(correlations) else None
        ),
        "per_gene": gene_rows,
    }
    return {
        "best_evaluation_orientation": best_orientation,
        "canonical": canonical,
        "flipped": flipped,
        "reported_overall": (
            canonical if best_orientation == "canonical" else flipped
        ),
        "by_stage": by_stage,
        "gene_level": gene_summary,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def _write_json_yaml(path: Path, config: dict[str, Any]) -> None:
    """JSON is a valid YAML 1.2 document and avoids adding a PyYAML dependency."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    epochs: int = 1200,
    seed: int = 0,
    top_k: int = 15,
    n_modules: int = 8,
    stage_gene_count: int = 200,
    correlation_block_size: int = 256,
    max_genes: int | None = None,
    device: str = "auto",
    log_interval: int = 100,
    visualize: bool = True,
    visualization_dpi: int = 100,
    genes_to_plot: list[str] | None = None,
) -> dict[str, Any]:
    """Train the unsupervised mouse pipeline and persist predictions/metrics."""
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    torch_device = torch.device(resolved_device)
    output_dir = Path(output_dir)
    expression_dir = output_dir / "expression"
    grn_dir = output_dir / "grn"
    evaluation_dir = output_dir / "evaluation_data"
    evaluation_input_dir = evaluation_dir / "input"
    evaluation_truth_dir = evaluation_dir / "groundtruth"
    for directory in (
        expression_dir,
        grn_dir,
        evaluation_input_dir,
        evaluation_truth_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_mouse_data(data_dir, max_genes=max_genes)
    count = data["count"]
    rprt = data["rprt"]
    n_cells, n_genes = count.shape
    observed = int(data["label_mask"].sum())
    total_labels = int(data["label_mask"].size)
    print(
        f"data: cells={n_cells} genes={n_genes} "
        f"observed_labels={observed}/{total_labels} "
        f"missing={1.0 - observed / total_labels:.3%}"
    )
    print(f"stages: {dict(Counter(data['stages']))}")

    standardized, log_count, mu, sigma = preprocess_count_table(count)
    dc = max(2, min(64, n_cells - 1, n_genes))
    gene_features, pca_components = build_node_features(
        standardized, rprt, dc=dc, seed=seed
    )
    directed, undirected, graph_stats = build_rprt_hypergraph(
        rprt,
        data["stages"],
        top_k=top_k,
        n_modules=n_modules,
        stage_gene_count=stage_gene_count,
        correlation_block_size=correlation_block_size,
        seed=seed,
    )
    print(
        f"graph: nodes={graph_stats['nodes']} "
        f"directed={graph_stats['directed_edges']} "
        f"undirected={graph_stats['undirected_edges']}"
    )
    for source, count_value in graph_stats["source_counts"].items():
        proportion = graph_stats["source_proportions"][source]
        print(f"  {source:<30} {count_value:>7} ({proportion:.2%})")

    model = build_model(
        directed_data=directed,
        undirected_data=undirected,
        n_cells=n_cells,
        n_genes=n_genes,
        dc=dc,
        pca_init=pca_components,
        hidden=256,
        latent=dc,
        use_asym=True,
        device=torch_device,
    )
    criterion = build_criterion()
    optimizer = build_optimizer(model)
    count_tensor = torch.from_numpy(standardized).to(torch_device)
    gene_tensor = torch.from_numpy(gene_features).to(torch_device)
    projection_tensor = torch.from_numpy(pca_components).to(torch_device)
    target_tensor = count_tensor @ projection_tensor.T

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    print(f"training: epochs={epochs} device={torch_device}")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        model_output = model(count_tensor, gene_tensor, target_tensor)
        loss, terms = criterion(
            model=model,
            model_output=model_output,
            gene_projection=projection_tensor,
            compartment_indicator=None,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        record = {
            "epoch": epoch,
            **{
                name: float(value.detach().cpu())
                for name, value in terms.items()
            },
        }
        history.append(record)
        if epoch == 1 or epoch == epochs or epoch % max(1, log_interval) == 0:
            print(
                f"  epoch={epoch:4d} loss={record['total']:.6f} "
                f"cyc={record['cyc_comp']:.6f} "
                f"ortho={record['orthogonality']:.6f}"
            )

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, phase_a_dc, phase_b_dc = model(
            count_tensor, gene_tensor, target_tensor
        )
    phase_a_dc_np = phase_a_dc.detach().cpu().numpy()
    phase_b_dc_np = phase_b_dc.detach().cpu().numpy()
    phase_a_gene = (
        phase_a_dc_np @ pca_components
    ) * sigma[None, :] + mu[None, :] / 2.0
    phase_b_gene = (
        phase_b_dc_np @ pca_components
    ) * sigma[None, :] + mu[None, :] / 2.0
    correction = 0.5 * (log_count - phase_a_gene - phase_b_gene)
    phase_a_gene = (phase_a_gene + correction).astype(np.float32)
    phase_b_gene = (phase_b_gene + correction).astype(np.float32)
    norm_a = float(np.linalg.norm(phase_a_gene, axis=1).mean())
    norm_b = float(np.linalg.norm(phase_b_gene, axis=1).mean())
    canonical_swapped = norm_a > norm_b
    if canonical_swapped:
        phase_a_gene, phase_b_gene = phase_b_gene, phase_a_gene
    phase_sum_error = float(
        np.linalg.norm(phase_a_gene + phase_b_gene - log_count)
        / max(np.linalg.norm(log_count), 1e-12)
    )
    if phase_sum_error >= 1e-6:
        raise RuntimeError(
            f"phase-sum relative error {phase_sum_error} >= 1e-6"
        )
    canonical_info = {
        "rule": "lower_mean_cell_l2_norm_is_phase_A",
        "swapped": canonical_swapped,
        "original_phase_a_mean_norm": norm_a,
        "original_phase_b_mean_norm": norm_b,
        "phase_sum_relative_error": phase_sum_error,
    }

    positive_a = np.clip(phase_a_gene, 0, None)
    positive_b = np.clip(phase_b_gene, 0, None)
    positive_total = positive_a + positive_b
    predicted_ratio = np.divide(
        positive_a,
        positive_total,
        out=np.full_like(positive_total, 0.5),
        where=positive_total > EPS,
    ).astype(np.float32)

    axes = {
        "index": pd.Index(data["cells"], name="cell_id"),
        "columns": data["genes"],
    }
    pd.DataFrame(phase_a_gene, **axes).to_csv(
        expression_dir / "phase_A.csv", float_format="%.8g"
    )
    pd.DataFrame(phase_b_gene, **axes).to_csv(
        expression_dir / "phase_B.csv", float_format="%.8g"
    )
    prediction_frame = pd.DataFrame(
        predicted_ratio.T, index=data["genes"], columns=data["cells"]
    )
    prediction_frame.index.name = "Genes"
    prediction_frame.insert(0, "Chromosomes", data["chromosomes"])
    prediction_path = expression_dir / "predicted_AllelicRatio_WT.csv"
    prediction_frame.to_csv(prediction_path, float_format="%.8g")

    observed_mask = data["label_mask"]
    truth_a = np.where(
        observed_mask, log_count * data["ratio"], np.nan
    ).astype(np.float32)
    truth_b = np.where(
        observed_mask, log_count * (1.0 - data["ratio"]), np.nan
    ).astype(np.float32)
    expression_evaluation = evaluate_simulation_expression(
        phase_a_pred=phase_a_gene,
        phase_b_pred=phase_b_gene,
        maternal_true=truth_a,
        paternal_true=truth_b,
        combined=log_count,
        seed=seed,
        projection=None,
        to_dc=None,
        method_name="phasehyper",
        observed_mask=observed_mask,
    )
    save_saber_evaluation(
        output_dir=expression_dir,
        headline_rows=expression_evaluation["headline_rows"],
        saber_rows=expression_evaluation["saber_rows"],
        orientation_rows=expression_evaluation["orientation_audit"],
        metadata={
            "seed": seed,
            "epochs": epochs,
            "dc": dc,
            "n_observed": observed,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        },
    )
    print_headline(expression_evaluation["headline_rows"])
    print_saber_table(expression_evaluation["saber_rows"])
    print_orientation_audit(expression_evaluation["orientation_audit"])
    print_final(expression_evaluation["headline_rows"])

    oriented_a = expression_evaluation["phase_a_oriented"]
    oriented_b = expression_evaluation["phase_b_oriented"]
    oriented_pos_a = np.clip(oriented_a, 0, None)
    oriented_pos_b = np.clip(oriented_b, 0, None)
    oriented_total = oriented_pos_a + oriented_pos_b
    oriented_ratio = np.divide(
        oriented_pos_a,
        oriented_total,
        out=np.full_like(oriented_total, 0.5),
        where=oriented_total > EPS,
    )
    ratio_evaluation = evaluate_predictions(
        predicted_ratio,
        data["ratio"],
        observed_mask,
        stages=data["stages"],
        genes=data["genes"],
    )
    stage_rows = []
    for stage in STAGE_ORDER:
        row_mask = np.asarray(data["stages"]) == stage
        if not row_mask.any():
            continue
        stage_rows.append({
            "stage": stage,
            **regression_metrics(
                oriented_ratio[row_mask],
                data["ratio"][row_mask],
                observed_mask[row_mask],
            ),
        })
    pd.DataFrame(stage_rows).to_csv(
        expression_dir / "stage_metrics.csv", index=False
    )
    gene_rows = []
    for gene_index, gene in enumerate(data["genes"]):
        gene_mask = observed_mask[:, gene_index]
        if int(gene_mask.sum()) < 10:
            continue
        gene_rows.append({
            "gene": gene,
            **regression_metrics(
                oriented_ratio[:, gene_index],
                data["ratio"][:, gene_index],
                gene_mask,
            ),
        })
    pd.DataFrame(
        gene_rows,
        columns=[
            "gene",
            "n_observed",
            "mse",
            "mae",
            "pearson",
            "classification_n",
            "classification_accuracy",
        ],
    ).to_csv(expression_dir / "gene_metrics.csv", index=False)

    stage_ids = np.asarray(
        [STAGE_ORDER.index(stage) for stage in data["stages"]], dtype=int
    )
    clustering = evaluate_simulation_clustering(
        raw_rna=standardized,
        cell_embedding=target_tensor.detach().cpu().numpy(),
        phase_a_embedding=phase_a_dc_np,
        phase_b_embedding=phase_b_dc_np,
        labels=stage_ids,
        n_clusters=len(set(data["stages"])),
        seed=seed,
    )
    embedding_quality = evaluate_embedding_quality(
        cell_embedding=target_tensor.detach().cpu().numpy(),
        phase_a_embedding=phase_a_dc_np,
        phase_b_embedding=phase_b_dc_np,
    )
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )

    checkpoint = {
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "dc": dc,
        "genes": data["genes"],
        "cells": data["cells"],
        "graph_parameters": {
            "top_k": top_k,
            "n_modules": n_modules,
            "stage_gene_count": stage_gene_count,
            "correlation_block_size": correlation_block_size,
        },
        "canonical_orientation": canonical_info,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")

    config = {
        "data": {
            "data_dir": Path(data_dir),
            "dataset": "GSE80810_WT",
            "n_cells": n_cells,
            "n_genes": n_genes,
            "label_missing_fraction": 1.0 - observed / total_labels,
            "labels_used_for_training": False,
        },
        "preprocessing": {
            "count_library_target_sum": 1e4,
            "count_transform": "library_normalize_log1p_gene_zscore",
            "rprt_transform": "log1p_gene_profile_standardization",
        },
        "hypergraph": {
            "source": "RPRT_only",
            "top_k": top_k,
            "n_modules": n_modules,
            "stage_gene_count": stage_gene_count,
            "correlation_block_size": correlation_block_size,
            "stages": list(STAGE_ORDER),
        },
        "training": {
            "epochs": epochs,
            "seed": seed,
            "device": str(torch_device),
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "objective": "shared_unsupervised_six_term_loss",
            "scheduler": None,
            "phase_sum_relative_error": phase_sum_error,
        },
        "visualization": {
            "enabled": visualize,
            "dpi": visualization_dpi,
            "pipeline": "have_answer",
            "max_correlation_genes": 300,
            "max_genome_correlation_genes": 500,
            "genes": list(genes_to_plot or []),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _write_json_yaml(output_dir / "config.yaml", config)

    pd.DataFrame(log_count, **axes).to_csv(
        evaluation_input_dir / "combined_true_expression.csv",
        float_format="%.8g",
    )
    pd.DataFrame({
        "gene_id": data["genes"],
        "chromosome": data["chromosomes"],
        "input_order": np.arange(n_genes),
    }).to_csv(evaluation_input_dir / "gene_info.csv", index=False)
    pd.DataFrame({
        "cell_id": data["cells"],
        "cell_type": data["stages"],
    }).to_csv(evaluation_input_dir / "cell_metadata.csv", index=False)
    pd.DataFrame(truth_a, **axes).to_csv(
        evaluation_truth_dir / "phase_A_true.csv", float_format="%.8g"
    )
    pd.DataFrame(truth_b, **axes).to_csv(
        evaluation_truth_dir / "phase_B_true.csv", float_format="%.8g"
    )
    pd.DataFrame(
        observed_mask.astype(np.uint8), **axes
    ).to_csv(evaluation_dir / "observed_mask.csv")
    with (grn_dir / "skipped.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "skipped",
                "reason": "GSE80810 WT inputs contain no reference allele GRN",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    visualization_outputs = []
    if visualize:
        try:
            from phasehyper.visualization.have_answer import run_visualization

            visualization_outputs = run_visualization(
                sim_dir=evaluation_dir,
                result_dir=output_dir,
                dpi=visualization_dpi,
                genes_to_plot=genes_to_plot,
                max_correlation_genes=300,
                max_genome_correlation_genes=500,
            )
        except Exception as exc:
            print(
                f"  [visualization warning] "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        print("  [visualization] skipped")

    print(f"outputs written to {output_dir}")
    return {
        "model": model,
        "predicted_ratio": predicted_ratio,
        "expression_evaluation": expression_evaluation,
        "ratio_evaluation": ratio_evaluation,
        "clustering": clustering,
        "embedding_quality": embedding_quality,
        "visualization_outputs": visualization_outputs,
        "output_dir": output_dir,
        "prediction_path": prediction_path,
        "fit": {
            "phase_A": phase_a_gene,
            "phase_B": phase_b_gene,
            "truth_A": truth_a,
            "truth_B": truth_b,
            "observed_mask": observed_mask,
            "cell_h": target_tensor.detach().cpu().numpy(),
            "phase_sum_relative_error": phase_sum_error,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--modules", type=int, default=8, dest="n_modules")
    parser.add_argument("--stage-gene-count", type=int, default=200)
    parser.add_argument("--correlation-block-size", type=int, default=256)
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="select the most variable common genes for a smoke/overfit run",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device (auto, cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="skip the final have-answer visualization pipeline",
    )
    parser.add_argument(
        "--visualization-dpi",
        type=int,
        default=100,
        help="DPI for generated mouse benchmark figures",
    )
    parser.add_argument(
        "--gene",
        action="append",
        dest="genes_to_plot",
        help="gene ID for a detail figure; may be supplied more than once",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    values = vars(args)
    values["visualize"] = not values.pop("no_visualization")
    return run(**values)


if __name__ == "__main__":
    main()
