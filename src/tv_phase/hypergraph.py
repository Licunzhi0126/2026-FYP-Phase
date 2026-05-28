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
from .utils import *
from .priors import *

def _groups_to_gene_maps(common_genes, primary_groups, secondary_groups):
    gene_set = set(common_genes)
    gene_to_chrom = {}
    gene_to_pathways: Dict[str, List[str]] = {gene: [] for gene in common_genes}

    for group_name, genes in primary_groups.items():
        for gene in genes:
            gene = str(gene)
            if gene in gene_set:
                gene_to_pathways[gene].append(str(group_name))
    
    for group_name, genes in secondary_groups.items():
        group_name = str(group_name)
        if group_name.startswith("chrom::"):
            chrom_name = group_name.split("::", 1)[1]
            for gene in genes:
                gene = str(gene)
                if gene in gene_set and gene not in gene_to_chrom:
                    gene_to_chrom[gene] = chrom_name
            continue
        for gene in genes:
            gene = str(gene)
            if gene in gene_set:
                gene_to_pathways[gene].append(group_name)

    gene_to_pathways = {
        gene: list(dict.fromkeys(groups))
        for gene, groups in gene_to_pathways.items()
        if groups
    }
    return gene_to_chrom, gene_to_pathways

# DEPRECATED: train_multiview_embeddings - removed in prior-only hypergraph mode
# Cell embeddings are now learned directly via HGNN on the heterogeneous hypergraph

def _select_top_genes_for_cell(row, top_k):
    values = row.astype(float).abs()
    if top_k is None or top_k >= len(values):
        top_k = min(10, len(values))
    return values.sort_values(ascending=False).head(top_k).index.tolist()


def _build_cell_hyperedges(
    view1_dfs: Optional[List[pd.DataFrame]] = None,
    expression_df: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    top_fraction: Optional[float] = None,
    min_size: int = 2,
    merge_strategy: str = "separate"
) -> List[Dict[str, List[str]]]:
    """
    Build cell hyperedges using top-k or top-fraction strategy.
    
    Args:
        view1_dfs: list of cell x gene dataframes (optional)
        expression_df: cell x gene dataframe (fallback if no view1)
        top_k: number of top genes to select per cell
        top_fraction: fraction of genes to select (overrides top_k if not None)
        min_size: minimum hyperedge size
        merge_strategy: how to handle multiple view1s ("separate")
    
    Returns:
        List of dictionaries, each mapping cell names to gene lists for one view1
    """
    if view1_dfs is None or len(view1_dfs) == 0:
        if expression_df is None:
            raise ValueError("Either view1_dfs or expression_df must be provided")
        view1_dfs = [expression_df]
    
    # separate 策略：每个 view1 独立处理，返回超边列表
    cell_hyperedges_list = []
    for view1_df in view1_dfs:
        edges = _build_hyperedges_from_df(view1_df, top_k, top_fraction, min_size)
        cell_hyperedges_list.append(edges)
    
    return cell_hyperedges_list


def _build_hyperedges_from_df(
    df: pd.DataFrame,
    top_k: int = 10,
    top_fraction: Optional[float] = None,
    min_size: int = 2
) -> Dict[str, List[str]]:
    """Helper function to build hyperedges from a single dataframe."""
    cell_hyperedges: Dict[str, List[str]] = {}
    n_genes = df.shape[1]
    
    if top_fraction is not None:
        top_k = max(min_size, int(np.ceil(n_genes * top_fraction)))
    
    for cell in df.index:
        row = df.loc[cell]
        top_genes = _select_top_genes_for_cell(row, top_k)
        members = list(dict.fromkeys(top_genes + [str(cell)]))
        if len(members) > 1:
            cell_hyperedges[str(cell)] = members
    return cell_hyperedges


def _initialize_node_features_prior_only(
    node_list: List[str],
    node_types: List[str],
    common_cells: List[str],
    common_genes: List[str],
    expression_df: pd.DataFrame,
    view1_dfs: Optional[List[pd.DataFrame]] = None,
    feature_dim: int = 64,
) -> torch.Tensor:
    """
    Initialize data-driven node features for prior-only hypergraph.
    - cell node feature: expression_df.loc[common_cells, common_genes] as primary
    - gene node feature: expression_df.loc[common_cells, common_genes].T
    - If view1_dfs exists, concatenate all view1s with expression
    - Cell and gene features are standardized and reduced separately
    """
    expr = expression_df.reindex(index=common_cells, columns=common_genes).fillna(0.0).values.astype(np.float32)

    cell_views = [expr]
    gene_views = [expr.T]

    if view1_dfs is not None and len(view1_dfs) > 0:
        for view1_df in view1_dfs:
            view1 = view1_df.reindex(index=common_cells, columns=common_genes).fillna(0.0).values.astype(np.float32)
            cell_views.append(view1)
            gene_views.append(view1.T)

    cell_raw = np.concatenate([_safe_standardize(v) for v in cell_views], axis=1)
    gene_raw = np.concatenate([_safe_standardize(v) for v in gene_views], axis=1)

    cell_features = _reduce_to_fixed_dim(cell_raw, feature_dim)
    gene_features = _reduce_to_fixed_dim(gene_raw, feature_dim)

    cell_features = _safe_standardize(cell_features)
    gene_features = _safe_standardize(gene_features)

    X = np.zeros((len(node_list), feature_dim), dtype=np.float32)

    cell_index = {str(c): i for i, c in enumerate(common_cells)}
    gene_index = {str(g): i for i, g in enumerate(common_genes)}

    for i, node_name in enumerate(node_list):
        node_name = str(node_name)
        if node_types[i] == "sample":
            X[i] = cell_features[cell_index[node_name]]
        elif node_types[i] == "gene":
            X[i] = gene_features[gene_index[node_name]]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(X.astype(np.float32))


def build_prior_only_hypergraph_dict(
    dataset, prior, feature_dim: int = 64,
    cell_hyperedge_top_k: int = 10,
    cell_hyperedge_top_fraction: Optional[float] = None,
    min_cell_hyperedge_size: int = 2,
    merge_strategy: str = "average"
):
    """
    Build hypergraph dictionary for prior-only hypergraph mode:
    - Nodes: gene nodes + cell/sample nodes only
    - Hyperedges:
      1. KEGG prior hyperedges (from kegg_groups)
      2. Gene position/window prior hyperedges (from poswin_groups)
      3. Cell-gene observation hyperedges (each cell connects to top-k genes)
    - Cell nodes have no X_c features - using learnable embedding
    """
    gene_set = set(dataset.common_genes)
    cell_set = set(dataset.common_cells)

    # Build cell-gene observation hyperedges using top-k strategy
    cell_hyperedges = _build_cell_hyperedges(
        view1_dfs=dataset.view1_dfs,
        expression_df=dataset.expression_df,
        top_k=cell_hyperedge_top_k,
        top_fraction=cell_hyperedge_top_fraction,
        min_size=min_cell_hyperedge_size,
        merge_strategy=merge_strategy
    )
    
    hyperedge_list: List[List[str]] = []
    hyperedge_names: List[str] = []
    hyperedge_types: List[str] = []
    hyperedge_weights: List[float] = []

    # 1) KEGG prior hyperedges
    for edge_name, genes in prior.kegg_groups.items():
        members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
        if len(members) > 1:
            hyperedge_list.append(members)
            hyperedge_names.append(str(edge_name))
            hyperedge_types.append("pathway")
            hyperedge_weights.append(1.0)

    # 2) Position window prior hyperedges
    for edge_name, genes in prior.poswin_groups.items():
        members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
        if len(members) > 1:
            hyperedge_list.append(members)
            hyperedge_names.append(str(edge_name))
            hyperedge_types.append("poswin")
            hyperedge_weights.append(1.0)

    # 3) PPI hyperedges (for PPI dataset)
    if prior.ppi_groups:
        for edge_name, genes in prior.ppi_groups.items():
            members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
            if len(members) > 1:
                hyperedge_list.append(members)
                hyperedge_names.append(str(edge_name))
                hyperedge_types.append("ppi")
                hyperedge_weights.append(1.0)

    # 4) Data-driven prior hyperedges (GLUE / denoise style pseudo-priors)
    if getattr(prior, "data_groups", None):
        data_weights = getattr(prior, "data_group_weights", None) or {}
        for edge_name, genes in prior.data_groups.items():
            members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
            if len(members) > 1:
                hyperedge_list.append(members)
                hyperedge_names.append(str(edge_name))
                hyperedge_types.append("data_prior")
                hyperedge_weights.append(float(data_weights.get(edge_name, 1.0)))

    # 5) Cell-gene observation hyperedges (NOT prior edges)
    # separate strategy: each view1 has its own cell hyperedges
    for view_idx, view_hyperedges in enumerate(cell_hyperedges):
        for cell_name, members in view_hyperedges.items():
            valid = list(dict.fromkeys([
                node for node in members
                if node in gene_set or node in cell_set
            ]))
            if len(valid) > 1:
                hyperedge_list.append(valid)
                hyperedge_names.append(f"obs::{cell_name}::view{view_idx+1}")
                hyperedge_types.append(f"cell_view{view_idx+1}")
                hyperedge_weights.append(1.0)

    # Nodes: only gene + sample
    all_nodes = set(dataset.common_genes) | set(dataset.common_cells)
    node_list = sorted(list(all_nodes))
    node_id_map = {node: idx for idx, node in enumerate(node_list)}

    # Build incidence matrix H: [n_nodes, n_edges]
    H = torch.zeros(len(node_list), len(hyperedge_list), dtype=torch.float32)
    edge_weights = torch.tensor(hyperedge_weights, dtype=torch.float32)

    for edge_idx, hyperedge in enumerate(hyperedge_list):
        for node in hyperedge:
            if node not in node_id_map:
                raise KeyError(f"Node {node} in hyperedge not found in node_id_map")
            H[node_id_map[node], edge_idx] = 1.0

    # Node types: gene / sample only
    node_types: List[str] = []
    for node in node_list:
        if node in gene_set:
            node_types.append("gene")
        elif node in cell_set:
            node_types.append("sample")
        else:
            raise ValueError(f"Unexpected node outside gene/cell sets: {node}")

    gene_mask = torch.tensor([t == "gene" for t in node_types], dtype=torch.bool)
    sample_mask = torch.tensor([t == "sample" for t in node_types], dtype=torch.bool)

    X = _initialize_node_features_prior_only(
        node_list=node_list,
        node_types=node_types,
        common_cells=dataset.common_cells,
        common_genes=dataset.common_genes,
        expression_df=dataset.expression_df,
        view1_dfs=dataset.view1_dfs,
        feature_dim=feature_dim,
    )

    # True labels only for sample nodes
    true_sample_labels = torch.zeros(len(node_list), dtype=torch.long)
    label_by_cell = {
        cell: int(dataset.labels[idx])
        for idx, cell in enumerate(dataset.common_cells)
    }
    for idx, node_name in enumerate(node_list):
        if node_types[idx] == "sample":
            true_sample_labels[idx] = label_by_cell.get(node_name, 0)

    sample_node_names = {
        idx: node_list[idx]
        for idx, node_type in enumerate(node_types)
        if node_type == "sample"
    }
    gene_node_names = {
        idx: node_list[idx]
        for idx, node_type in enumerate(node_types)
        if node_type == "gene"
    }

    # Hyperedge type encoding
    edge_type_to_id = {
        "pathway": 0,
        "ppi": 1,
        "poswin": 2,
        "data_prior": 3,
        "cell": 4,
    }
    hyperedge_type_ids = torch.tensor(
        [edge_type_to_id.get(t, 3) for t in hyperedge_types],
        dtype=torch.long
    )

    pathway_edge_mask = torch.tensor([t == "pathway" for t in hyperedge_types], dtype=torch.bool)
    ppi_edge_mask = torch.tensor([t == "ppi" for t in hyperedge_types], dtype=torch.bool)
    poswin_edge_mask = torch.tensor([t == "poswin" for t in hyperedge_types], dtype=torch.bool)
    data_prior_edge_mask = torch.tensor([t == "data_prior" for t in hyperedge_types], dtype=torch.bool)
    cell_edge_mask = torch.tensor([t == "cell" or t.startswith("cell_view") for t in hyperedge_types], dtype=torch.bool)

    return {
        "H": H,
        "X": X,
        "W": edge_weights,
        "sample_mask": sample_mask,
        "gene_mask": gene_mask,
        "sample_labels": torch.zeros(len(node_list), dtype=torch.long),
        "true_sample_labels": true_sample_labels,
        "sample_node_names": sample_node_names,
        "gene_node_names": gene_node_names,
        "n_nodes": len(node_list),
        "n_edges": len(hyperedge_list),
        "hyperedge_names": hyperedge_names,
        "hyperedge_types": hyperedge_types,
        "hyperedge_type_ids": hyperedge_type_ids,
        "edge_type_to_id": edge_type_to_id,
        "pathway_edge_mask": pathway_edge_mask,
        "ppi_edge_mask": ppi_edge_mask,
        "poswin_edge_mask": poswin_edge_mask,
        "data_prior_edge_mask": data_prior_edge_mask,
        "cell_edge_mask": cell_edge_mask,
        # For backward compatibility - empty placeholders
        "raw_cell_embedding": np.zeros((len(dataset.common_cells), feature_dim), dtype=np.float32),
        "aftermultiview_cell_embedding": np.zeros((len(dataset.common_cells), feature_dim), dtype=np.float32),
        "multiview_common_cells": list(dataset.common_cells),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
