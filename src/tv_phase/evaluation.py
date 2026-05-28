from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import leidenalg
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    umap = None

from .config import *
from .visualization import *

def _match_embedding_dim(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
    embedding = embedding.float()
    emb_dim = int(embedding.shape[1])
    if emb_dim == target_dim:
        return embedding
    if emb_dim > target_dim:
        return embedding[:, :target_dim]
    return F.pad(embedding, (0, target_dim - emb_dim))

# DEPRECATED: _build_node_features_two_types - replaced by _initialize_node_features_prior_only

# DEPRECATED: _infer_hyperedge_type - no longer needed in prior-only mode


# DEPRECATED: build_hypergraph_dict - replaced by build_prior_only_hypergraph_dict

def _build_knn_graph(embedding: np.ndarray, k: int = 10) -> ig.Graph:
    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    if n <= 1:
        return ig.Graph(n=max(1, n))
    k = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(embedding)
    indices = nbrs.kneighbors(return_distance=False)
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

def _cluster_embedding(embedding: np.ndarray, *, method: str, dataset_type: str,
    expected_clusters: Optional[int] = None, resolution: Optional[float] = None) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)
    expected_clusters = int(expected_clusters or min(3, n))
    expected_clusters = max(2, min(expected_clusters, n - 1))

    if method == "kmeans":
        return KMeans(n_clusters=expected_clusters, random_state=42, n_init=10).fit_predict(embedding).astype(np.int64)

    graph = _build_knn_graph(embedding, k=min(10, n - 1))
    if graph.ecount() == 0:
        return np.zeros(n, dtype=np.int64)

    if method == "leiden":
        resolution = float(resolution if resolution is not None else DEFAULT_LEIDEN_RESOLUTION.get(dataset_type, 1.0))
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )
        return np.asarray(partition.membership, dtype=np.int64)

    if method == "louvain":
        return np.asarray(graph.community_multilevel().membership, dtype=np.int64)

    raise ValueError(f"Unsupported cluster method: {method}")

def _align_embedding_to_names(source_names: List[str], embedding: np.ndarray, target_names: List[str]) -> np.ndarray:
    source_names = [str(name) for name in source_names]
    target_names = [str(name) for name in target_names]
    embedding = np.asarray(embedding, dtype=np.float32)
    source_map = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_map]
    if missing:
        raise KeyError(f"Missing cell names during embedding alignment: {missing[:5]}")
    return np.stack([embedding[source_map[name]] for name in target_names], axis=0).astype(np.float32, copy=False)



def _expression_only_original_embedding(expression_df: pd.DataFrame, sample_names: List[str]) -> np.ndarray:
    sample_names = [str(name) for name in sample_names]
    missing = [name for name in sample_names if name not in expression_df.index]
    if missing:
        raise KeyError(f"Missing sample names during original embedding export: {missing[:5]}")
    return expression_df.loc[sample_names].values.astype(np.float32)


def build_expression_space_phase_embeddings(
    df_cell: pd.DataFrame,
    expression_df: pd.DataFrame,
    sample_names: List[str],
    common_genes: List[str],
) -> Dict[str, np.ndarray]:
    sample_names = [str(name) for name in sample_names]
    common_genes = [str(gene) for gene in common_genes]

    original_expression_embedding = expression_df.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)

    maternal_pivot = df_cell.pivot(index="cell", columns="gene", values="maternal_expression")
    paternal_pivot = df_cell.pivot(index="cell", columns="gene", values="paternal_expression")

    maternal_expression_embedding = (
        maternal_pivot.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)
    )
    paternal_expression_embedding = (
        paternal_pivot.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)
    )

    shapes = {
        "original_expression_embedding": original_expression_embedding.shape,
        "maternal_expression_embedding": maternal_expression_embedding.shape,
        "paternal_expression_embedding": paternal_expression_embedding.shape,
    }

    if not all(shape == shapes["original_expression_embedding"] for shape in shapes.values()):
        shape_msg = "\n".join([f"  {name}: {shape}" for name, shape in shapes.items()])
        raise ValueError(
            f"Embedding shape mismatch after alignment:\n{shape_msg}\n"
            f"sample_names count: {len(sample_names)}\n"
            f"common_genes count: {len(common_genes)}\n"
            f"maternal_pivot shape: {maternal_pivot.shape}, index: {list(maternal_pivot.index[:5])}, columns: {list(maternal_pivot.columns[:5])}\n"
            f"paternal_pivot shape: {paternal_pivot.shape}, index: {list(paternal_pivot.index[:5])}, columns: {list(paternal_pivot.columns[:5])}"
        )

    return {
        "original_expression_embedding": original_expression_embedding,
        "maternal_expression_embedding": maternal_expression_embedding,
        "paternal_expression_embedding": paternal_expression_embedding,
    }


def _save_embedding_npz(path: Path, embedding: np.ndarray, cell_names: List[str]) -> None:
    np.savez(
        path,
        embedding=np.asarray(embedding, dtype=np.float32),
        cell_names=np.asarray([str(name) for name in cell_names], dtype=object),
    )

def _save_embedding_umap(path: Path, embedding: np.ndarray, label_names: List[str], title: str) -> None:
    coords = _try_umap_or_pca(np.asarray(embedding, dtype=np.float32))
    enc, mapping = _encode_labels(label_names)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=enc, cmap="viridis", s=40)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=scatter.cmap(scatter.norm(i)))
        for i in mapping.values()
    ]
    plt.legend(handles, mapping.keys(), title="Cell Stage", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _evaluate_embedding_metrics(embedding, true_ids, *, dataset_type, cluster_method,
    cluster_resolution ) -> Dict[str, float]:
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
    )
    unique_pred = np.unique(pred)
    return {
        "fmi": float(fowlkes_mallows_score(true_ids, pred)),
        "nmi": float(normalized_mutual_info_score(true_ids, pred)),
        "ari": float(adjusted_rand_score(true_ids, pred)),
        "pred_clusters": int(unique_pred.size),
    }

def _align_labels_to_cells( dataset, sample_names) -> Tuple[np.ndarray, List[str]]:
    if sample_names is None:
        return np.asarray(dataset.labels, dtype=np.int64), [str(name) for name in dataset.label_names]
    sample_names = [str(name) for name in sample_names]
    label_by_cell = {str(cell): int(dataset.labels[idx]) for idx, cell in enumerate(dataset.common_cells)}
    label_name_by_cell = {str(cell): str(dataset.label_names[idx]) for idx, cell in enumerate(dataset.common_cells)}
    missing = [name for name in sample_names if name not in label_by_cell]
    if missing:
        raise KeyError(f"Missing sample names during metric alignment: {missing[:5]}")
    true_ids = np.asarray([label_by_cell[name] for name in sample_names], dtype=np.int64)
    aligned_label_names = [label_name_by_cell[name] for name in sample_names]
    return true_ids, aligned_label_names

def _save_embedding_metrics(
    outputs: Dict[str, np.ndarray],
    *,
    dataset: DatasetBundle,
    cluster_method: str,
    cluster_resolution: Optional[float],
    out_dir: Path,
    report_title: str,
    sample_names: Optional[List[str]] = None,
    gate_g: Optional[np.ndarray] = None,
    edge_type_weights: Optional[Dict[str, float]] = None,
    phase_recon_mse: Optional[float] = None,
    phase_cosine_sim: Optional[float] = None,
    phase_l2_dist: Optional[float] = None,
) -> pd.DataFrame:
    true_ids, aligned_label_names = _align_labels_to_cells(dataset, sample_names)
    assessment_mode = "aligned_by_exported_sample_names" if sample_names is not None else "dataset_order"
    rows = []
    for name, emb in outputs.items():
        metrics = _evaluate_embedding_metrics(
            emb,
            true_ids,
            dataset_type=dataset.dataset_type,
            cluster_method=cluster_method,
            cluster_resolution=cluster_resolution,
        )
        rows.append(
            {
                "embedding": name,
                "cluster_method": cluster_method,
                "pred_clusters": metrics["pred_clusters"],
                "fmi": metrics["fmi"],
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
            }
        )
        _save_embedding_umap(out_dir / f"{name}_umap.png", emb, aligned_label_names, f"{name} UMAP")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "embedding_metrics_summary.csv", index=False, encoding="utf-8-sig")
    
    with open(out_dir / "metric_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(report_title + "\n")
        f.write("=" * 60 + "\n")
        f.write(f"Assessment mode: {assessment_mode}\n")
        f.write(f"Aligned samples : {len(true_ids)}\n\n")

        f.write("Assessment groups:\n\n")
        f.write("1. Expression-space embeddings:\n")
        f.write("   - original_expression_embedding: total/original cell x gene expression matrix\n")
        f.write("   - phase_A_expression_embedding: phase A-specific cell x gene expression matrix\n")
        f.write("   - phase_B_expression_embedding: phase B-specific cell x gene expression matrix\n\n")
        f.write("2. HGNN-VAE intermediate embeddings (inference pipeline):\n")
        f.write("   - cell_embedding_input_x: Original input node features (learnable embedding initialized)\n")
        f.write("   - cell_embedding_hgnn_h: HGNN output after hypergraph propagation (before VAE)\n")
        f.write("   - cell_embedding_vae_mu: VAE encoder mean output (deterministic latent)\n")
        f.write("   - cell_embedding_vae_z: VAE sampled latent (with reparameterization)\n\n")

        f.write("Important note:\n")
        f.write("Phase A and Phase B are learned unsupervised labels.\n")
        f.write("They should not be directly interpreted as maternal/paternal without biological validation.\n\n")

        # Gate statistics
        if gate_g is not None:
            f.write("-" * 60 + "\n")
            f.write("Gate Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  gate_mean: {float(gate_g.mean()):.4f}\n")
            f.write(f"  gate_std: {float(gate_g.std()):.4f}\n")
            f.write(f"  gate_min: {float(gate_g.min()):.4f}\n")
            f.write(f"  gate_max: {float(gate_g.max()):.4f}\n")
            f.write(f"  percent_gate_lt_0.3: {float((gate_g < 0.3).mean() * 100):.2f}%\n")
            f.write(f"  percent_gate_gt_0.7: {float((gate_g > 0.7).mean() * 100):.2f}%\n")
            f.write(f"  percent_gate_between_0.4_0.6: {float(((gate_g >= 0.4) & (gate_g <= 0.6)).mean() * 100):.2f}%\n")
            f.write(f"  phase_A_genes (gate < 0.5): {int((gate_g < 0.5).sum())}\n")
            f.write(f"  phase_B_genes (gate >= 0.5): {int((gate_g >= 0.5).sum())}\n\n")

        # Phase reconstruction metrics
        if phase_recon_mse is not None:
            f.write("-" * 60 + "\n")
            f.write("Phase Reconstruction Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  phase_recon_mse: {phase_recon_mse:.6f}\n")
            if phase_cosine_sim is not None:
                f.write(f"  phase_A_phase_B_cosine_similarity: {phase_cosine_sim:.4f}\n")
            if phase_l2_dist is not None:
                f.write(f"  phase_A_phase_B_l2_distance: {phase_l2_dist:.4f}\n")
            f.write("\n")

        # Edge type weights
        if edge_type_weights is not None:
            f.write("-" * 60 + "\n")
            f.write("Learned Edge Type Weights:\n")
            f.write("-" * 60 + "\n")
            for et, weight in edge_type_weights.items():
                f.write(f"  {et}: {weight:.4f}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("Clustering Metrics:\n")
        f.write("-" * 60 + "\n")
        for row in rows:
            f.write(f"[{row['embedding']}]\n")
            f.write(f"  Cluster Method: {row['cluster_method']}\n")
            f.write(f"  Pred Clusters : {row['pred_clusters']}\n")
            f.write(f"  FMI: {row['fmi']:.4f}\n")
            f.write(f"  NMI: {row['nmi']:.4f}\n")
            f.write(f"  ARI: {row['ari']:.4f}\n\n")
    return df


def _save_run_metadata_phase(
    out_dir: Path,
    *,
    version_name: str,
    dataset: DatasetBundle,
    config: PhaseTrainingConfig,
    edge_type_weights: Dict[str, float],
) -> None:
    """Save run metadata with learned edge type weights for end-to-end phase training."""
    payload = {
        "version": version_name,
        "dataset_type": dataset.dataset_type,
        "view1_name": dataset.view1_name,
        "n_cells": len(dataset.common_cells),
        "n_genes": len(dataset.common_genes),
        "label_names": sorted(set(dataset.label_names)),
        "training_config": {
            "feature_dim": config.feature_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "prior_dim": config.prior_dim,
            "prior_name": config.prior_name,
            "prior_top_k": config.prior_top_k,
            "prior_max_features": config.prior_max_features,
            "denoise_candidate_top_k": config.denoise_candidate_top_k,
            "denoise_epochs": config.denoise_epochs,
            "denoise_top_percent": config.denoise_top_percent,
            "seed": config.seed,
            "train_epochs": config.train_epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "vae_recon_weight": config.vae_recon_weight,
            "vae_kl_weight": config.vae_kl_weight,
            "vae_kl_warmup_epochs": config.vae_kl_warmup_epochs,
            "cell_gene_recon_weight": config.cell_gene_recon_weight,
            "phase_sep_weight": config.phase_sep_weight,
            "gate_balance_weight": config.gate_balance_weight,
            "gate_entropy_weight": config.gate_entropy_weight,
            "gene_gate_smoothness_weight": config.gene_gate_smoothness_weight,
        },
        "learned_edge_type_weights": edge_type_weights,
        "training_mode": "end_to_end_unsupervised",
        "note": "Phase A and Phase B are learned unsupervised labels and should not be directly interpreted as maternal/paternal without biological validation.",
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _default_output_dir(version_name: str, dataset_type: str, cluster_method: str) -> Path:
    return Path(r"output") / version_name / dataset_type / cluster_method

def _run_evaluation_visualization(
    out_dir: Path,
    dataset,
    dataset_config: dict,
    phase_a_expression: np.ndarray,
    phase_b_expression: np.ndarray,
    gene_names: List[str],
    sample_names: List[str],
    gate_g: np.ndarray,
):
    """
    Run evaluation visualization for datasets with ground truth (have_answer=True).
    This includes:
    1. Gene correlation heatmap ordered by chromosome position
    2. Gene correlation heatmap ordered by KEGG pathway
    3. Phase separation evaluation and visualization
    """
    # Import seaborn if not already imported
    import seaborn as sns
    
    # Create output directory
    eval_dir = out_dir / "evaluation_visualization"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Get paths from dataset config
    root_dir = dataset_config["root"]
    gene_pos_file = root_dir / dataset_config["files"]["poswin_prior"]
    kegg_file = root_dir / dataset_config["files"]["kegg_prior"]
    e_m_file = root_dir / "E_M.csv"
    e_p_file = root_dir / "E_P.csv"
    
    # Load ground truth expression data early for 4-panel correlation comparison
    e_m_truth = None
    e_p_truth = None
    if e_m_file.exists() and e_p_file.exists():
        e_m_truth = pd.read_csv(e_m_file, index_col=0)
        e_p_truth = pd.read_csv(e_p_file, index_col=0)
    
    print(f"  Creating evaluation visualizations in: {eval_dir}")
    
    # ============================================
    # 1. Gene Correlation by Chromosome Position (4-panel comparison)
    # ============================================
    print("  [1/3] Gene correlation by chromosome position...")
    if gene_pos_file.exists():
        # Load gene positions
        gene_pos = pd.read_csv(
            gene_pos_file,
            sep="\t",
            header=None,
            names=["Gene", "Chr", "Start", "End", "Strand"]
        )
        gene_order = gene_pos["Gene"].tolist()
        
        # Create DataFrames for all available expression matrices
        expr_dfs = {
            "Phase_A": pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names),
            "Phase_B": pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names),
        }
        
        if e_m_truth is not None and e_p_truth is not None:
            expr_dfs["E_M (Truth)"] = e_m_truth
            expr_dfs["E_P (Truth)"] = e_p_truth
        
        # Calculate correlation matrices
        corr_matrices = {}
        for name, expr_df in expr_dfs.items():
            common_genes = [g for g in gene_order if g in expr_df.columns]
            expr_filtered = expr_df[common_genes]
            corr_matrix = expr_filtered.corr(method="pearson")
            corr_matrices[name] = corr_matrix.loc[common_genes, common_genes]
        
        # Determine subplot layout
        n_plots = len(corr_matrices)
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)"]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)", "E_M (Ground Truth)", "E_P (Ground Truth)"]
        
        for i, (name, corr_sorted) in enumerate(corr_matrices.items()):
            ax = axes[i]
            sns.heatmap(
                corr_sorted,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0,
                xticklabels=False,
                yticklabels=False,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            ax.set_title(titles[i], fontsize=14, fontweight="bold")
        
        # Hide unused subplots if only 2 plots
        if n_plots == 2:
            axes[1].axis('off')
        
        plt.suptitle("Gene Correlation Heatmaps\nOrdered by Chromosome Position", fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(eval_dir / "correlation_chromosome.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save individual correlation matrices
        for name, corr_sorted in corr_matrices.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            corr_sorted.to_csv(eval_dir / f"gene_correlation_{safe_name}_chromosome.csv")
        
        print("    Saved: correlation_chromosome.png")
    else:
        print(f"    Skipped: gene position file not found ({gene_pos_file})")
    
    # ============================================
    # 2. Gene Correlation by KEGG Pathway (4-panel comparison)
    # ============================================
    print("  [2/3] Gene correlation by KEGG pathway...")
    if kegg_file.exists():
        # Load KEGG annotation
        anno = pd.read_csv(
            kegg_file,
            sep="\t",
            header=None,
            names=["Gene", "KEGG", "Pathway"]
        )
        anno_sorted = anno.sort_values(by=["KEGG", "Gene"])
        gene_order = anno_sorted["Gene"].tolist()
        
        # Create DataFrames for all available expression matrices
        expr_dfs = {
            "Phase_A": pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names),
            "Phase_B": pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names),
        }
        
        if e_m_truth is not None and e_p_truth is not None:
            expr_dfs["E_M (Truth)"] = e_m_truth
            expr_dfs["E_P (Truth)"] = e_p_truth
        
        # Calculate correlation matrices
        corr_matrices = {}
        for name, expr_df in expr_dfs.items():
            common_genes = [g for g in gene_order if g in expr_df.columns]
            expr_filtered = expr_df[common_genes]
            corr_matrix = expr_filtered.corr(method="pearson")
            corr_matrices[name] = corr_matrix.loc[common_genes, common_genes]
        
        # Determine subplot layout
        n_plots = len(corr_matrices)
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)"]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)", "E_M (Ground Truth)", "E_P (Ground Truth)"]
        
        for i, (name, corr_sorted) in enumerate(corr_matrices.items()):
            ax = axes[i]
            sns.heatmap(
                corr_sorted,
                cmap="coolwarm",
                center=0,
                square=True,
                vmin=-1,
                vmax=1,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
                ax=ax
            )
            ax.set_title(titles[i], fontsize=14, fontweight="bold")
        
        # Hide unused subplots if only 2 plots
        if n_plots == 2:
            axes[1].axis('off')
        
        plt.suptitle("Gene Correlation Heatmaps\nOrdered by KEGG Pathway", fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(eval_dir / "correlation_kegg.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save individual correlation matrices
        for name, corr_sorted in corr_matrices.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            corr_sorted.to_csv(eval_dir / f"gene_correlation_{safe_name}_kegg.csv")
        
        print("    Saved: correlation_kegg.png (4-panel comparison)")
    else:
        print(f"    Skipped: KEGG file not found ({kegg_file})")
    
    # ============================================
    # 3. Phase Separation Evaluation
    # ============================================
    print("  [3/3] Phase separation evaluation...")
    if e_m_file.exists() and e_p_file.exists():
        # Load ground truth
        e_m_truth = pd.read_csv(e_m_file, index_col=0)
        e_p_truth = pd.read_csv(e_p_file, index_col=0)
        
        # Create predicted DataFrames
        phase_a_pred = pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names)
        phase_b_pred = pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names)
        
        # Get common cells and genes
        common_cells = list(set(phase_a_pred.index) & set(e_m_truth.index))
        common_genes = list(set(phase_a_pred.columns) & set(e_m_truth.columns))
        common_cells = sorted(common_cells)
        common_genes = sorted(common_genes)
        
        # Align data
        phase_a_pred = phase_a_pred.loc[common_cells, common_genes]
        phase_b_pred = phase_b_pred.loc[common_cells, common_genes]
        e_m_truth = e_m_truth.loc[common_cells, common_genes]
        e_p_truth = e_p_truth.loc[common_cells, common_genes]
        
        # Plot individual heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        data_list = [
            (e_m_truth, "E_M (Ground Truth - Maternal)", axes[0, 0]),
            (e_p_truth, "E_P (Ground Truth - Paternal)", axes[0, 1]),
            (phase_a_pred, "Phase A (Predicted)", axes[1, 0]),
            (phase_b_pred, "Phase B (Predicted)", axes[1, 1]),
        ]
        for df, title, ax in data_list:
            sns.heatmap(df, ax=ax, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Genes")
            ax.set_ylabel("Cells")
        plt.tight_layout()
        plt.savefig(eval_dir / "individual_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot difference heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        diff_m = (e_m_truth - phase_a_pred).abs()
        diff_p = (e_p_truth - phase_b_pred).abs()
        diff_list = [
            (diff_m, "|E_M - Phase A|", axes[0]),
            (diff_p, "|E_P - Phase B|", axes[1]),
        ]
        for df, title, ax in diff_list:
            sns.heatmap(df, ax=ax, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Genes")
            ax.set_ylabel("Cells")
        plt.tight_layout()
        plt.savefig(eval_dir / "difference_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot chromosome phase distribution
        if gene_pos_file.exists():
            pos_df = pd.read_csv(gene_pos_file, sep="\t", header=None, 
                                 names=["gene", "chromosome", "start", "end", "strand"])
            pos_df["mid_mb"] = (pos_df["start"] + pos_df["end"]) / 2_000_000
            
            # Create predicted phase_df from gate_g
            phase_df_pred = pd.DataFrame({
                "gene": gene_names,
                "gate_g": gate_g,
                "phase": ["Phase_B" if g > 0.5 else "Phase_A" for g in gate_g]
            })
            
            # Create ground truth phase_df from E_M and E_P
            e_m_mean = e_m_truth.mean(axis=0)
            e_p_mean = e_p_truth.mean(axis=0)
            total = e_m_mean + e_p_mean
            gate_g_true = e_p_mean / total
            gate_g_true = gate_g_true.fillna(0.5)
            
            phase_df_true = pd.DataFrame({
                "gene": gate_g_true.index.tolist(),
                "gate_g": gate_g_true.values,
                "phase": ["Phase_B" if g > 0.5 else "Phase_A" for g in gate_g_true.values]
            })
            
            # ============================================
            # Chromosome Phase Distribution - Predicted
            # ============================================
            merged_pred = phase_df_pred.merge(pos_df, on="gene")
            counts_pred = merged_pred.groupby(["chromosome", "phase"]).size().unstack(fill_value=0)
            for phase in ["Phase_A", "Phase_B"]:
                if phase not in counts_pred.columns:
                    counts_pred[phase] = 0
            counts_pred["total"] = counts_pred.sum(axis=1)
            counts_pred = counts_pred.sort_values("total", ascending=False)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            counts_pred[["Phase_A", "Phase_B"]].plot(kind="bar", stacked=True, ax=ax, 
                                                     color=["#1f77b4", "#ff7f0e"])
            ax.set_title("Predicted Chromosome Phase Distribution", fontsize=16, fontweight="bold")
            ax.set_xlabel("Chromosome", fontsize=12)
            ax.set_ylabel("Number of Genes", fontsize=12)
            ax.legend(title="Phase", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(eval_dir / "chromosome_phase_distribution_predicted.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # ============================================
            # Chromosome Phase Distribution - Ground Truth
            # ============================================
            merged_true = phase_df_true.merge(pos_df, on="gene")
            counts_true = merged_true.groupby(["chromosome", "phase"]).size().unstack(fill_value=0)
            for phase in ["Phase_A", "Phase_B"]:
                if phase not in counts_true.columns:
                    counts_true[phase] = 0
            counts_true["total"] = counts_true.sum(axis=1)
            counts_true = counts_true.sort_values("total", ascending=False)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            counts_true[["Phase_A", "Phase_B"]].plot(kind="bar", stacked=True, ax=ax, 
                                                     color=["#1f77b4", "#ff7f0e"])
            ax.set_title("Ground Truth Chromosome Phase Distribution", fontsize=16, fontweight="bold")
            ax.set_xlabel("Chromosome", fontsize=12)
            ax.set_ylabel("Number of Genes", fontsize=12)
            ax.legend(title="Phase", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(eval_dir / "chromosome_phase_distribution_truth.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # ============================================
            # Gene Position vs Gate Value - Predicted
            # ============================================
            fig, ax = plt.subplots(figsize=(16, 6))
            chrom_order = [str(i) for i in range(1, 23)]
            chrom_colors = plt.cm.tab20(np.linspace(0, 1, 22))
            chrom_color_map = {chrom: chrom_colors[i] for i, chrom in enumerate(chrom_order)}
            
            for chrom in chrom_order:
                subset = merged_pred[merged_pred["chromosome"] == int(chrom)]
                if len(subset) > 0:
                    ax.scatter(subset["mid_mb"], subset["gate_g"], 
                               color=chrom_color_map[chrom], label=f"chr{chrom}", s=50, alpha=0.7)
            
            from scipy.ndimage import gaussian_filter1d
            sorted_df = merged_pred.sort_values("mid_mb")
            smoothed_gate = gaussian_filter1d(sorted_df["gate_g"], sigma=2)
            ax.plot(sorted_df["mid_mb"], smoothed_gate, color="red", linewidth=2, 
                    label="Smoothed Gate Trend", alpha=0.8)
            ax.set_title("Predicted Gene Position vs Gate Value", fontsize=16, fontweight="bold")
            ax.set_xlabel("Genomic Position (Mb)", fontsize=12)
            ax.set_ylabel("Gate Value (0 = Phase A, 1 = Phase B)", fontsize=12)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=10)
            plt.tight_layout()
            plt.savefig(eval_dir / "chromosome_position_scatter_predicted.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            # ============================================
            # Gene Position vs Gate Value - Ground Truth
            # ============================================
            fig, ax = plt.subplots(figsize=(16, 6))
            
            for chrom in chrom_order:
                subset = merged_true[merged_true["chromosome"] == int(chrom)]
                if len(subset) > 0:
                    ax.scatter(subset["mid_mb"], subset["gate_g"], 
                               color=chrom_color_map[chrom], label=f"chr{chrom}", s=50, alpha=0.7)
            
            sorted_df = merged_true.sort_values("mid_mb")
            smoothed_gate = gaussian_filter1d(sorted_df["gate_g"], sigma=2)
            ax.plot(sorted_df["mid_mb"], smoothed_gate, color="red", linewidth=2, 
                    label="Smoothed Gate Trend", alpha=0.8)
            
            ax.set_title("Ground Truth Gene Position vs Gate Value", fontsize=16, fontweight="bold")
            ax.set_xlabel("Genomic Position (Mb)", fontsize=12)
            ax.set_ylabel("Gate Value (0 = Phase A, 1 = Phase B)", fontsize=12)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=10)
            plt.tight_layout()
            plt.savefig(eval_dir / "chromosome_position_scatter_truth.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        # Calculate evaluation metrics
        phase_a_np = phase_a_pred.values.flatten()
        phase_b_np = phase_b_pred.values.flatten()
        e_m_np = e_m_truth.values.flatten()
        e_p_np = e_p_truth.values.flatten()
        
        mse_a = np.mean((phase_a_np - e_m_np) ** 2)
        mse_b = np.mean((phase_b_np - e_p_np) ** 2)
        mae_a = np.mean(np.abs(phase_a_np - e_m_np))
        mae_b = np.mean(np.abs(phase_b_np - e_p_np))
        
        metrics = {
            "mse_phase_a": float(mse_a),
            "mse_phase_b": float(mse_b),
            "mse_mean": float(np.mean([mse_a, mse_b])),
            "mae_phase_a": float(mae_a),
            "mae_phase_b": float(mae_b),
            "mae_mean": float((mae_a + mae_b) / 2),
            "rmse_mean": float(np.sqrt(np.mean([mse_a, mse_b]))),
            "pearson_corr_mean": float((np.corrcoef(phase_a_np, e_m_np)[0, 1] + 
                                       np.corrcoef(phase_b_np, e_p_np)[0, 1]) / 2)
        }
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(eval_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
        
        # Generate report
        report_lines = [
            "=" * 60,
            "PHASE SEPARATION EVALUATION METRICS",
            "=" * 60,
            "",
            "# 误差指标",
            "----------------------------------------",
            f"MSE (Phase A):             {metrics['mse_phase_a']:.4f}",
            f"MSE (Phase B):             {metrics['mse_phase_b']:.4f}",
            f"MSE Mean:                  {metrics['mse_mean']:.4f}",
            "",
            f"MAE (Phase A):             {metrics['mae_phase_a']:.4f}",
            f"MAE (Phase B):             {metrics['mae_phase_b']:.4f}",
            f"MAE Mean:                  {metrics['mae_mean']:.4f}",
            "",
            f"RMSE Mean:                 {metrics['rmse_mean']:.4f}",
            "",
            "# 相关性指标",
            "----------------------------------------",
            f"Pearson Correlation Mean:  {metrics['pearson_corr_mean']:.4f}",
            ""
        ]
        with open(eval_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print("    Saved: individual_heatmaps.png, difference_heatmaps.png")
        print("    Saved: chromosome_phase_distribution_predicted.png, chromosome_phase_distribution_truth.png")
        print("    Saved: chromosome_position_scatter_predicted.png, chromosome_position_scatter_truth.png")
        print("    Saved: evaluation_metrics.csv, evaluation_report.txt")
    else:
        print(f"    Skipped: ground truth files not found ({e_m_file}, {e_p_file})")
    
    print(f"  Evaluation visualization completed")

__all__ = [name for name in globals() if not name.startswith("__")]
