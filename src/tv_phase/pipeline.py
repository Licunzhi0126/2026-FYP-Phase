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
from .data_loading import *
from .priors import *
from .hypergraph import *
from .training import *
from .visualization import *
from .evaluation import *

def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_data_contract(out_dir: Path, dataset: DatasetBundle, dataset_config: Dict[str, Any]) -> None:
    label_counts = pd.Series(dataset.label_names, dtype=str).value_counts().to_dict()
    files = dataset_config.get("files", {})
    root = Path(dataset_config.get("root", ""))
    contract = {
        "dataset_type": dataset.dataset_type,
        "description": dataset_config.get("description", ""),
        "root": str(root.resolve()) if root else "",
        "expression_file": str((root / files.get("expression", "")).resolve()) if files.get("expression") else "",
        "view_files": [str((root / name).resolve()) for name in files.get("view", [])],
        "stage_file": str((root / files.get("stage", "")).resolve()) if files.get("stage") else "",
        "view1_name": dataset.view1_name,
        "n_cells": len(dataset.common_cells),
        "n_genes": len(dataset.common_genes),
        "n_view1": 0 if dataset.view1_dfs is None else len(dataset.view1_dfs),
        "expression_shape": list(dataset.expression_df.shape),
        "view_shapes": [] if dataset.view1_dfs is None else [list(df.shape) for df in dataset.view1_dfs],
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_map": {str(k): str(v) for k, v in (dataset.label_map or {}).items()},
        "cell_order_policy": "modalities aligned by cell_id index after loading",
        "feature_policy": "common genes/features are used across expression and view1 modalities",
        "missing_value_policy": "NaN/inf values are filled with 0 during CSV loading",
        "runtime_dependency_policy": "does not import code from the old project directory",
    }
    if dataset.metadata:
        contract.update(dataset.metadata)
    _save_json(out_dir / "data" / "data_contract.json", contract)


def _save_prior_artifacts(out_dir: Path, prior: PriorBundle) -> None:
    prior_dir = out_dir / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    edge_table = prior.edge_table if prior.edge_table is not None else pd.DataFrame()
    edge_table.to_csv(prior_dir / "prior_edges.csv", index=False, encoding="utf-8-sig")
    metadata = prior.metadata or {}
    metadata = dict(metadata)
    metadata.setdefault("n_kegg_groups", len(prior.kegg_groups))
    metadata.setdefault("n_poswin_groups", len(prior.poswin_groups))
    metadata.setdefault("n_ppi_groups", 0 if prior.ppi_groups is None else len(prior.ppi_groups))
    metadata.setdefault("n_data_groups", 0 if prior.data_groups is None else len(prior.data_groups))
    _save_json(prior_dir / "prior_metadata.json", metadata)


def run_hgnn_vae_phase_end2end(config: PhaseTrainingConfig, *, version_name: str = "HGNN-VAE-Phase") -> Dict[str, Any]:
    data_name = config.data_name

    view1_dfs, expression_df, stage_name_by_cell, view1_name = load_dataset(
        Path("."), data_name
    )

    dataset_type = _resolve_dataset_type(Path("."), data_name)

    if view1_dfs is not None and len(view1_dfs) > 0:
        common_cells = view1_dfs[0].index.tolist()
        common_genes = list(set(view1_dfs[0].columns) & set(expression_df.columns))
        common_genes = [gene for gene in common_genes if gene in view1_dfs[0].columns and gene in expression_df.columns]

        for i in range(len(view1_dfs)):
            view1_dfs[i] = view1_dfs[i][common_genes]
    else:
        common_cells = expression_df.index.tolist()
        common_genes = expression_df.columns.tolist()

    expression_df = expression_df[common_genes]

    label_names = [stage_name_by_cell.get(cell, "Unknown") for cell in common_cells]
    labels, label_map = _labels_to_ids(label_names)

    dataset = DatasetBundle(
        dataset_type=dataset_type,
        view1_name=view1_name,
        view1_dfs=view1_dfs,
        expression_df=expression_df,
        common_cells=common_cells,
        common_genes=common_genes,
        labels=labels,
        label_names=label_names,
        label_map=label_map,
        metadata={
            "prior_name_requested": config.prior_name,
        },
    )

    out_dir = Path(config.output_dir) if config.output_dir else _default_output_dir(version_name, dataset.dataset_type, config.cluster_method)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = DATASET_CONFIG[dataset_type]
    _save_data_contract(out_dir, dataset, dataset_config)

    print("=" * 60)
    print(f"{version_name} | End-to-End Unsupervised Phase Decomposition")
    print("=" * 60)
    print(f"  Dataset type: {dataset.dataset_type}")
    print(f"  View-1 modality: {dataset.view1_name}")
    print(f"  Training epochs: {config.train_epochs}")
    print(f"  Feature dim: {config.feature_dim}, Hidden dim: {config.hidden_dim}, Latent dim: {config.latent_dim}")
    print(f"  Device: {config.device}")
    print(f"  Prior: {config.prior_name}")

    print("\n[Step 1] Loading aligned modalities...")
    if dataset.view1_dfs is not None and len(dataset.view1_dfs) > 0:
        print(f"  {dataset.view1_name.title():<11}: {len(dataset.view1_dfs)} view(s)")
        for i, view1_df in enumerate(dataset.view1_dfs):
            print(f"    View {i+1}: {view1_df.shape}")
    else:
        print(f"  {dataset.view1_name.title():<11}: No view1 data")
    print(f"  Expression : {dataset.expression_df.shape}")
    print(f"  Labels     : {len(set(dataset.label_names))} classes -> {sorted(set(dataset.label_names))}")

    print("\n[Step 2] Building dataset-specific priors...")
    prior = build_prior_bundle(
        Path("."),
        dataset,
        d_prior=config.prior_dim,
        allow_position_file_fallback=False,
        genomic_window_bp=200000,
        include_window_groups=True,
        prior_name=config.prior_name,
        prior_top_k=config.prior_top_k,
        prior_max_features=config.prior_max_features,
        denoise_candidate_top_k=config.denoise_candidate_top_k,
        denoise_node_feature_dim=config.denoise_node_feature_dim,
        denoise_hidden_dim=config.denoise_hidden_dim,
        denoise_epochs=config.denoise_epochs,
        denoise_lr=config.denoise_lr,
        denoise_top_percent=config.denoise_top_percent,
        device=config.device,
        seed=config.seed,
    )

    print(f"  KEGG groups: {len(prior.kegg_groups)}")
    print(f"  Position window groups: {len(prior.poswin_groups)}")
    if prior.ppi_groups:
        print(f"  PPI groups: {len(prior.ppi_groups)}")
    if prior.data_groups:
        print(f"  Data-driven prior groups: {len(prior.data_groups)}")
    _save_position_prior_audit(out_dir, dataset.dataset_type)
    _save_prior_artifacts(out_dir, prior)

    print("\n[Step 3] Building prior-only hypergraph...")
    hg_data = build_prior_only_hypergraph_dict(
        dataset, prior, 
        feature_dim=config.feature_dim,
        cell_hyperedge_top_k=config.cell_hyperedge_top_k,
        cell_hyperedge_top_fraction=config.cell_hyperedge_top_fraction,
        min_cell_hyperedge_size=config.min_cell_hyperedge_size,
    )
    
    # Add edge type information to hg_data
    edge_type_to_id = {
        "pathway": 0,
        "poswin": 1,
        "ppi": 2,
        "data_prior": 3,
        "cell": 4,
    }
    hyperedge_type_ids = torch.zeros(hg_data["H"].shape[1], dtype=torch.int64)
    if "pathway_edge_mask" in hg_data:
        hyperedge_type_ids[hg_data["pathway_edge_mask"]] = edge_type_to_id["pathway"]
    if "poswin_edge_mask" in hg_data:
        hyperedge_type_ids[hg_data["poswin_edge_mask"]] = edge_type_to_id["poswin"]
    if "ppi_edge_mask" in hg_data:
        hyperedge_type_ids[hg_data["ppi_edge_mask"]] = edge_type_to_id["ppi"]
    if "data_prior_edge_mask" in hg_data:
        hyperedge_type_ids[hg_data["data_prior_edge_mask"]] = edge_type_to_id["data_prior"]
    if "cell_edge_mask" in hg_data:
        hyperedge_type_ids[hg_data["cell_edge_mask"]] = edge_type_to_id["cell"]
    
    hg_data["hyperedge_type_ids"] = hyperedge_type_ids
    hg_data["edge_type_to_id"] = edge_type_to_id
    
    sample_indices = torch.where(hg_data["sample_mask"])[0]
    gene_indices = torch.where(hg_data["gene_mask"])[0]
    print(f"  Nodes: {hg_data['n_nodes']}, Edges: {hg_data['n_edges']}")
    print(f"  Samples: {len(sample_indices)}, Genes: {len(gene_indices)}")
    print(f"  Pathway edges: {int(hg_data['pathway_edge_mask'].sum())}")
    print(f"  Position edges: {int(hg_data['poswin_edge_mask'].sum())}")
    if prior.ppi_groups:
        print(f"  PPI edges: {int(hg_data['ppi_edge_mask'].sum())}")
    if "data_prior_edge_mask" in hg_data:
        print(f"  Data prior edges: {int(hg_data['data_prior_edge_mask'].sum())}")
    print(f"  Cell-gene observation edges: {int(hg_data['cell_edge_mask'].sum())}")

    print(f"\n[Step 4] End-to-end HGNN-VAE-Phase training ({config.train_epochs} epochs)...")
    result = train_hgnn_vae_phase(hg_data, config, device=config.device)

    gene_names = list(result["gene_names"])
    sample_names = [str(name) for name in result["sample_names"]]
    gate_g = np.asarray(result["gate_g"], dtype=np.float32)

    cell_embed = np.asarray(result["cell_embed"], dtype=np.float32)
    mu_cells = np.asarray(result["mu_cells"], dtype=np.float32)
    h_cells = np.asarray(result["h_cells"], dtype=np.float32)
    x_cells = np.asarray(result["x_cells"], dtype=np.float32)
    mu_genes = np.asarray(result["mu_genes"], dtype=np.float32)

    z_genes = mu_genes
    z_cells = cell_embed

    print(f"  Gene embedding shape: {z_genes.shape}")
    print(f"  Cell embedding shape (cell_embed): {z_cells.shape}")
    print(f"  Cell embedding shape (mu): {mu_cells.shape}")
    print(f"  Cell embedding shape (h): {h_cells.shape}")
    print(f"  Cell embedding shape (x): {x_cells.shape}")
    
    # Gate statistics
    gate_mean = float(gate_g.mean())
    gate_std = float(gate_g.std())
    gate_min = float(gate_g.min())
    gate_max = float(gate_g.max())
    pct_lt_03 = float((gate_g < 0.3).mean() * 100)
    pct_gt_07 = float((gate_g > 0.7).mean() * 100)
    pct_between = float(((gate_g >= 0.4) & (gate_g <= 0.6)).mean() * 100)
    
    print(f"\n  Gate statistics:")
    print(f"    mean={gate_mean:.4f}, std={gate_std:.4f}, min={gate_min:.4f}, max={gate_max:.4f}")
    print(f"    <0.3: {pct_lt_03:.2f}%, >0.7: {pct_gt_07:.2f}%, 0.4-0.6: {pct_between:.2f}%")
    print(f"    Phase A genes (gate < 0.5): {int((gate_g < 0.5).sum())}")
    print(f"    Phase B genes (gate >= 0.5): {int((gate_g >= 0.5).sum())}")

    print("\n[Step 5] Exporting tables and embeddings...")
    
    # Save phase separation table
    df_phase = pd.DataFrame({
        "gene": gene_names,
        "gate_g": gate_g,
        "phase": ["Phase_A" if gi < 0.5 else "Phase_B" for gi in gate_g],
    })
    df_phase.to_csv(out_dir / "hgnn_vae_phase_separation.csv", index=False, encoding="utf-8-sig")
    
    # Save gene expression table
    df_gene = pd.DataFrame({
        "gene": gene_names,
        "gate_g": gate_g,
        "confidence": np.abs(gate_g - 0.5) * 2.0,
    })
    df_gene.to_csv(out_dir / "gene_phase_expression.csv", index=False, encoding="utf-8-sig")

    # Save intermediate cell embeddings
    _save_embedding_npz(out_dir / "cell_embedding_input_x.npz", x_cells.astype(np.float32), sample_names)
    _save_embedding_npz(out_dir / "cell_embedding_hgnn_h.npz", h_cells.astype(np.float32), sample_names)
    _save_embedding_npz(out_dir / "cell_embedding_vae_mu.npz", mu_cells.astype(np.float32), sample_names)
    _save_embedding_npz(out_dir / "cell_embedding_phase_aware.npz", cell_embed.astype(np.float32), sample_names)
    print(f"  Saved intermediate embeddings: input_x, hgnn_h, vae_mu, phase_aware")

    print("\n[Step 6] Saving training loss log...")
    # Save training loss log
    loss_df = pd.DataFrame(result["loss_history"])
    loss_df.to_csv(out_dir / "training_loss_log.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved training_loss_log.csv with {len(loss_df)} epochs")
    
    # Plot training curves
    print("\n[Step 6.5] Plotting training curves...")
    plot_training_curves(result["loss_history"], str(out_dir / "training_curves.png"))
    print(f"  Saved training_curves.png")

    print("\n[Step 7] Rendering figures and reports...")
    
    # Build expression space embeddings
    # For unsupervised mode, we use gate to split expression
    expr_matrix = dataset.expression_df.loc[sample_names, gene_names].fillna(0).values.astype(np.float32)
    original_expression_embedding = expr_matrix
    
    # Phase-specific expression matrices
    gate_matrix = gate_g[np.newaxis, :]
    phase_a_expression = expr_matrix * (1 - gate_matrix)  # gate < 0.5 -> Phase A
    phase_b_expression = expr_matrix * gate_matrix        # gate >= 0.5 -> Phase B

    # Save phase-separated expression matrices as CSV (simulation E_P/E_M separation)
    df_phase_a = pd.DataFrame(
        phase_a_expression,
        index=sample_names,
        columns=gene_names
    )
    df_phase_a.to_csv(out_dir / "phase_A_expression.csv", index=True, encoding="utf-8-sig")

    df_phase_b = pd.DataFrame(
        phase_b_expression,
        index=sample_names,
        columns=gene_names
    )
    df_phase_b.to_csv(out_dir / "phase_B_expression.csv", index=True, encoding="utf-8-sig")
    print(f"  Saved phase-separated expression: phase_A_expression.csv, phase_B_expression.csv")
    
    # Compute phase reconstruction metrics
    phase_recon_mse = float(((phase_a_expression + phase_b_expression) - expr_matrix).mean() ** 2)
    phase_a_norm = phase_a_expression / (np.linalg.norm(phase_a_expression, axis=1, keepdims=True) + 1e-8)
    phase_b_norm = phase_b_expression / (np.linalg.norm(phase_b_expression, axis=1, keepdims=True) + 1e-8)
    cosine_sim = float(np.mean(np.sum(phase_a_norm * phase_b_norm, axis=1)))
    l2_dist = float(np.mean(np.linalg.norm(phase_a_expression - phase_b_expression, axis=1)))
    
    print(f"  Phase reconstruction MSE: {phase_recon_mse:.6f}")
    print(f"  Phase A-B cosine similarity: {cosine_sim:.4f}")
    print(f"  Phase A-B L2 distance: {l2_dist:.4f}")
    
    # Collect all embeddings for metrics
    all_cell_embeddings = {
        "original_expression_embedding": original_expression_embedding,
        "phase_A_expression_embedding": phase_a_expression,
        "phase_B_expression_embedding": phase_b_expression,
        "cell_embedding_input_x": x_cells,
        "cell_embedding_hgnn_h": h_cells,
        "cell_embedding_vae_mu": mu_cells,
        "cell_embedding_phase_aware": cell_embed,
    }
    
    # Compute clustering metrics using true labels (only for evaluation)
    # Use configured cluster method, defaulting to kmeans
    cluster_method = config.cluster_method
    cluster_resolution = config.cluster_resolution
    
    metric_df = _save_embedding_metrics(
        all_cell_embeddings,
        dataset=dataset,
        cluster_method=cluster_method,
        cluster_resolution=cluster_resolution,
        out_dir=out_dir,
        report_title=f"{version_name} Embedding Clustering Report",
        sample_names=sample_names,
        gate_g=gate_g,
        edge_type_weights=result["edge_type_weights"],
        phase_recon_mse=phase_recon_mse,
        phase_cosine_sim=cosine_sim,
        phase_l2_dist=l2_dist,
    )
    
    # Save run metadata with edge type weights
    _save_run_metadata_phase(
        out_dir,
        version_name=version_name,
        dataset=dataset,
        config=config,
        edge_type_weights=result["edge_type_weights"],
    )

    # ============================================
    # Step 8: Evaluation Visualization (only for datasets with ground truth)
    # ============================================
    if dataset_config.get("have_answer", False):
        print("\n[Step 8] Running evaluation visualization...")
        _run_evaluation_visualization(
            out_dir=out_dir,
            dataset=dataset,
            dataset_config=dataset_config,
            phase_a_expression=phase_a_expression,
            phase_b_expression=phase_b_expression,
            gene_names=gene_names,
            sample_names=sample_names,
            gate_g=gate_g,
        )

    print("\n[Done] Outputs saved to:")
    print(f"  {out_dir.resolve()}")
    return {
        "dataset": dataset,
        "gene_expressions": df_gene,
        "gene_names": gene_names,
        "sample_names": sample_names,
        "gate_g": gate_g,
        "original_expression_embedding": original_expression_embedding,
        "phase_A_expression_embedding": phase_a_expression,
        "phase_B_expression_embedding": phase_b_expression,
        "cell_embedding_input_x": x_cells,
        "cell_embedding_hgnn_h": h_cells,
        "cell_embedding_vae_mu": mu_cells,
        "cell_embedding_phase_aware": cell_embed,
        "edge_type_weights": result["edge_type_weights"],
        "metric_df": metric_df.copy(),
        "out_dir": out_dir,
        "version_name": version_name,
    }

def main(*, version_name: str, config: Optional[PhaseTrainingConfig] = None, **kwargs) -> None:
    """Main function for end-to-end HGNN-VAE-Phase unsupervised training."""
    if config is None:
        config = PhaseTrainingConfig(**kwargs)
    run_hgnn_vae_phase_end2end(config, version_name=version_name)



def _run_default_sequence() -> None:
    configs = [
        PhaseTrainingConfig(
            data_name="sim_gene100_alpha_1_beta_1",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
        PhaseTrainingConfig(
            data_name="sim_gene100_alpha_2_beta_1",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
        PhaseTrainingConfig(
            data_name="sim_gene100_alpha_1_beta_2",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
        PhaseTrainingConfig(
            data_name="sim_gene100_alpha_2_beta_2",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
        PhaseTrainingConfig(
            data_name="PEA_STA",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
        PhaseTrainingConfig(
            data_name="sc_GEM",
            device="cpu",
            feature_dim=64,
            hidden_dim=64,
            latent_dim=16,
            prior_dim=16,
            train_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            cluster_method="kmeans",
        ),
    ]
    for cfg in configs:
        main(version_name="TV-PHASE_v11", config=cfg)


if __name__ == "__main__":
    _run_default_sequence()

__all__ = [name for name in globals() if not name.startswith("__")]
