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

def build_gene_prior_features(common_genes, kegg_txt_path, d_prior: int = 16):
    gene_set = set(common_genes)
    gene_to_pathways: Dict[str, List[str]] = {gene: [] for gene in common_genes}
    kegg_hyperedge: Dict[str, List[str]] = {}

    kegg_df = pd.read_csv(kegg_txt_path, sep="\t", header=None, na_values=["NA", ""])
    for _, row in kegg_df.iterrows():
        gene = str(row.iloc[0]).strip()
        pathway = str(row.iloc[2]).strip() if len(row) > 2 and pd.notna(row.iloc[2]) else ""

        if gene in gene_set and pathway and pathway.lower() != "nan":
            if pathway not in gene_to_pathways[gene]:
                gene_to_pathways[gene].append(pathway)

            kegg_hyperedge.setdefault(pathway, [])
            if gene not in kegg_hyperedge[pathway]:
                kegg_hyperedge[pathway].append(gene)

    pathway_values = sorted({pathway for pathways in gene_to_pathways.values() for pathway in pathways})
    pathway_index = {pathway: idx for idx, pathway in enumerate(pathway_values)}

    pathway_multihot = np.zeros((len(common_genes), len(pathway_values)), dtype=np.float32)
    for gene_idx, gene in enumerate(common_genes):
        for pathway in gene_to_pathways.get(gene, []):
            pathway_multihot[gene_idx, pathway_index[pathway]] = 1.0
    prior_matrix = pathway_multihot
    if prior_matrix.shape[1] == 0:
        gene_prior_matrix = np.zeros((len(common_genes), d_prior), dtype=np.float32)

    elif prior_matrix.shape[1] == 1 or len(common_genes) <= 1:
        gene_prior_matrix = np.concatenate(
            [
                prior_matrix.astype(np.float32),
                np.zeros((len(common_genes), max(0, d_prior - prior_matrix.shape[1])), dtype=np.float32),
            ],
            axis=1,
        )[:, :d_prior]

    else:
        n_components = min(d_prior, prior_matrix.shape[1] - 1, len(common_genes) - 1)
        n_components = max(1, n_components)

        gene_prior_matrix = TruncatedSVD(
            n_components=n_components,
            random_state=42
        ).fit_transform(prior_matrix)

        if gene_prior_matrix.shape[1] < d_prior:
            pad = np.zeros((len(common_genes), d_prior - gene_prior_matrix.shape[1]), dtype=np.float32)
            gene_prior_matrix = np.concatenate(
                [gene_prior_matrix.astype(np.float32), pad],
                axis=1
            )
        else:
            gene_prior_matrix = gene_prior_matrix[:, :d_prior].astype(np.float32)

    gene_prior_matrix = np.nan_to_num(
        gene_prior_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    print(f"  gene prior feature shape: {gene_prior_matrix.shape}")
    return gene_prior_matrix, gene_to_pathways, kegg_hyperedge


def _safe_standardize(x: np.ndarray) -> np.ndarray:
    return _standardize_matrix(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

def _match_embedding_to_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    current_dim = int(array.shape[1]) if array.ndim == 2 else 0
    if current_dim == target_dim:
        return array.astype(np.float32, copy=False)
    if current_dim > target_dim:
        return array[:, :target_dim].astype(np.float32, copy=False)
    pad = np.zeros((array.shape[0], target_dim - current_dim), dtype=np.float32)
    return np.concatenate([array, pad], axis=1).astype(np.float32, copy=False)

def _position_file_for_dataset(base_dir: Path, dataset_type: str) -> Path:
    if dataset_type == "simulation":
        sim_path = base_dir / "gene_positions_sim.txt"
        if sim_path.exists():
            return sim_path
    if dataset_type == "sc_GEM":
        expected_name = "gene_positions_sc.txt"
    else:
        expected_name = "gene_positions_pea.txt"
    expected_path = Path(base_dir) / expected_name
    return expected_path

def _position_candidate_files(base_dir: Path, dataset_type: str, allow_fallback: bool = False) -> List[Path]:
    primary_path = _position_file_for_dataset(base_dir, dataset_type)
    candidates = [primary_path]
    if allow_fallback:
        input_root = Path(base_dir).parent
        if input_root.exists():
            candidates.extend(sorted(input_root.glob("*/gene_positions*.txt")))

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped

def _chromosome_code(chromosome: str) -> float:
    token = str(chromosome).strip().lower().replace("chr", "")
    if token == "x":
        return 23.0
    if token == "y":
        return 24.0
    try:
        return float(token)
    except ValueError:
        return 0.0

def _load_position_prior(
    base_dir: Path,
    dataset_type: str,
    common_genes: List[str],
    *,
    allow_fallback: bool = False,
) -> Dict[str, object]:
    paths = _position_candidate_files(base_dir, dataset_type, allow_fallback=allow_fallback)
    primary_path = paths[0]
    gene_index = {str(gene).strip(): idx for idx, gene in enumerate(common_genes)}
    candidate_to_gene: Dict[str, str] = {}
    for gene in common_genes:
        gene_key = str(gene).strip()
        candidates = [gene_key] + GENE_POSITION_ALIASES.get(gene_key.upper(), [])
        for candidate in candidates:
            candidate_key = str(candidate).strip().upper()
            if candidate_key and candidate_key not in candidate_to_gene:
                candidate_to_gene[candidate_key] = gene_key

    rows: Dict[str, Dict[str, object]] = {}
    alias_matches: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                fields = raw_line.strip().split("\t")
                if len(fields) < 5:
                    continue
                raw_gene, chrom, start, end, strand = fields[:5]
                raw_gene = str(raw_gene).strip()
                gene = candidate_to_gene.get(raw_gene.upper())
                if gene is None or gene not in gene_index or gene in rows:
                    continue
                try:
                    start_i = int(float(start))
                    end_i = int(float(end))
                except ValueError:
                    continue
                if end_i < start_i:
                    start_i, end_i = end_i, start_i
                rows[gene] = {
                    "chrom": str(chrom).strip(),
                    "start": start_i,
                    "end": end_i,
                    "strand": str(strand).strip(),
                    "source_file": str(path.resolve()),
                }
                if raw_gene.upper() != str(gene).strip().upper():
                    alias_matches.append(
                        {
                            "dataset_type": dataset_type,
                            "common_gene": gene,
                            "matched_position_gene": raw_gene,
                            "position_file": str(path.resolve()),
                        }
                    )

    n_genes = len(common_genes)
    raw_features = np.zeros((n_genes, 7), dtype=np.float32)
    raw_features[:, 6] = 1.0
    max_midpoint = 1.0
    max_length = 1.0
    for info in rows.values():
        midpoint = (int(info["start"]) + int(info["end"])) / 2.0
        length = max(1.0, float(int(info["end"]) - int(info["start"]) + 1))
        max_midpoint = max(max_midpoint, midpoint)
        max_length = max(max_length, length)

    chrom_groups: Dict[str, List[str]] = {}
    chrom_position_rows: Dict[str, List[Tuple[float, str]]] = {}
    for gene_idx, gene in enumerate(common_genes):
        info = rows.get(gene)
        if info is None:
            continue
        chrom = str(info["chrom"])
        start_i = int(info["start"])
        end_i = int(info["end"])
        midpoint = (start_i + end_i) / 2.0
        length = max(1.0, float(end_i - start_i + 1))
        strand = str(info["strand"])
        raw_features[gene_idx, 0] = 1.0
        raw_features[gene_idx, 1] = _chromosome_code(chrom) / 24.0
        raw_features[gene_idx, 2] = np.log1p(midpoint) / np.log1p(max_midpoint)
        raw_features[gene_idx, 3] = np.log1p(length) / np.log1p(max_length)
        raw_features[gene_idx, 4] = 1.0 if strand == "+" else 0.0
        raw_features[gene_idx, 5] = 1.0 if strand == "-" else 0.0
        raw_features[gene_idx, 6] = 0.0
        chrom_groups.setdefault(chrom, []).append(gene)
        chrom_position_rows.setdefault(chrom, []).append((midpoint, gene))

    nearby_groups: Dict[str, List[str]] = {}
    window = 2
    for chrom, chrom_rows in chrom_position_rows.items():
        ordered = [gene for _, gene in sorted(chrom_rows)]
        for idx, gene in enumerate(ordered):
            members = ordered[max(0, idx - window): min(len(ordered), idx + window + 1)]
            if len(members) >= 2:
                nearby_groups[f"{chrom}:{gene}"] = members

    position_features = _safe_standardize(raw_features)
    matched_genes = sorted(rows.keys())
    missing_genes = [gene for gene in common_genes if gene not in rows]
    audit = {
        "dataset_type": dataset_type,
        "position_file": str(primary_path.resolve()) if primary_path.exists() else str(primary_path),
        "position_files_scanned": ";".join(str(path.resolve()) for path in paths if path.exists()),
        "allow_position_file_fallback": int(bool(allow_fallback)),
        "n_common_genes": len(common_genes),
        "n_position_matched": len(matched_genes),
        "n_position_missing": len(missing_genes),
        "missing_genes": ";".join(missing_genes),
        "n_alias_matched": len(alias_matches),
        "n_chrom_groups": sum(1 for genes in chrom_groups.values() if len(genes) > 1),
        "n_nearby_groups": len(nearby_groups),
        "position_feature_columns": "has_position;chromosome_code;midpoint;length;strand_plus;strand_minus;missing_position",
    }
    _POSITION_ALIAS_AUDIT_BY_DATASET[dataset_type] = alias_matches
    return {
        "features": position_features.astype(np.float32),
        "chrom_groups": chrom_groups,
        "nearby_groups": nearby_groups,
        "audit": audit,
        "rows": rows,
    }

def _save_position_prior_audit(out_dir: Path, dataset_type: str) -> None:
    audit = _POSITION_PRIOR_AUDIT.get(dataset_type)
    if audit:
        pd.DataFrame([audit]).to_csv(Path(out_dir) / "position_prior_audit.csv", index=False, encoding="utf-8-sig")
    alias_rows = _POSITION_ALIAS_AUDIT_BY_DATASET.get(dataset_type, [])
    if alias_rows:
        pd.DataFrame(alias_rows).drop_duplicates().to_csv(
            Path(out_dir) / "position_alias_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

def _build_incidence_from_memberships(num_nodes: int, memberships: List[List[int]]) -> torch.Tensor:
    valid: List[List[int]] = []
    for mem in memberships:
        uniq = sorted(set(int(i) for i in mem if 0 <= int(i) < num_nodes))
        if len(uniq) >= 2:
            valid.append(uniq)
    if not valid:
        return torch.eye(num_nodes, dtype=torch.float32)
    H = torch.zeros((num_nodes, len(valid)), dtype=torch.float32)
    for edge_idx, nodes in enumerate(valid):
        H[nodes, edge_idx] = 1.0
    return H

# 筛选Top Fraction的成员，让他可以筛选最高的
def _top_fraction_memberships(array_2d: np.ndarray, fraction: float = 0.2, min_size: int = 2) -> List[List[int]]:
    n_cells, n_cols = array_2d.shape
    memberships: List[List[int]] = []
    k = max(min_size, int(np.ceil(n_cells * fraction)))
    for j in range(n_cols):
        col = np.asarray(array_2d[:, j], dtype=np.float32)
        if np.allclose(col, col[0]):
            continue
        idx = np.argsort(col)[-k:]
        memberships.append(idx.tolist())
    return memberships

# 筛选非零成员
def _nonzero_memberships(array_2d: np.ndarray, min_size: int = 2) -> List[List[int]]:
    memberships: List[List[int]] = []
    for j in range(array_2d.shape[1]):
        idx = np.where(np.abs(array_2d[:, j]) > 1e-8)[0]
        if idx.size >= min_size:
            memberships.append(idx.tolist())
    return memberships

# DEPRECATED: build_cell_hypergraphs_for_three_views - removed in prior-only hypergraph mode
# Cell-gene connections are now handled via observation hyperedges

def build_gene_hypergraph(common_genes, gene_to_chrom, gene_to_pathways):
    gene_index = {str(gene): idx for idx, gene in enumerate(common_genes)}
    memberships: List[List[int]] = []

    chrom_groups: Dict[str, List[int]] = {}
    for gene in common_genes:
        chrom = gene_to_chrom.get(gene)
        if chrom:
            chrom_groups.setdefault(str(chrom), []).append(gene_index[str(gene)])
    memberships.extend(list(chrom_groups.values()))

    pathway_groups: Dict[str, List[int]] = {}
    for gene, pathways in gene_to_pathways.items():
        gene = str(gene)
        if gene not in gene_index:
            continue
        for pathway in pathways:
            pathway_groups.setdefault(str(pathway), []).append(gene_index[gene])
    memberships.extend(list(pathway_groups.values()))
    return _build_incidence_from_memberships(len(common_genes), memberships)

def build_unique_gene_inputs(view1_dfs, expression_df, gene_prior_matrix):
    expr_array = np.nan_to_num(expression_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    gene_prior_matrix = np.nan_to_num(np.asarray(gene_prior_matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    
    if view1_dfs is not None and len(view1_dfs) > 0:
        gene_mean_views = []
        for view1_df in view1_dfs:
            view1_array = np.nan_to_num(view1_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            gene_mean_views.append(view1_array.mean(axis=0, keepdims=True).T)
        gene_mean_view1 = np.concatenate(gene_mean_views, axis=1)
    else:
        gene_mean_view1 = np.zeros((expression_df.shape[1], 0), dtype=np.float32)
    
    gene_mean_expr = expr_array.mean(axis=0, keepdims=True).T
    x_g_unique = np.concatenate([gene_mean_view1, gene_mean_expr, gene_prior_matrix], axis=1).astype(np.float32)
    return torch.from_numpy(x_g_unique)

def _build_prefixed_groups(prefix: str, raw_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name, genes in raw_groups.items():
        unique_genes = [str(g).strip() for g in dict.fromkeys(genes) if str(g).strip()]
        if len(unique_genes) > 1:
            groups[f"{prefix}{name}"] = unique_genes
    return groups


def _select_top_variable_columns(df: pd.DataFrame, max_features: Optional[int]) -> Tuple[pd.DataFrame, List[str]]:
    if max_features is None or int(max_features) <= 0 or df.shape[1] <= int(max_features):
        selected = [str(col) for col in df.columns]
        return df.loc[:, selected], selected
    variances = df.var(axis=0).sort_values(ascending=False)
    selected = [str(col) for col in variances.index[: int(max_features)]]
    return df.loc[:, selected], selected


def _standardize_columns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / np.clip(x.std(axis=0, keepdims=True), 1e-8, None)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _top_corr_edge_table(
    matrix_a: np.ndarray,
    names_a: List[str],
    matrix_b: np.ndarray,
    names_b: List[str],
    top_k: int,
    evidence: str,
    *,
    include_self: bool = False,
) -> pd.DataFrame:
    xa = _standardize_columns(matrix_a)
    xb = _standardize_columns(matrix_b)
    corr = (xa.T @ xb) / max(1, xa.shape[0] - 1)
    rows = []
    for i, source_name in enumerate(names_a):
        vals = corr[i].copy()
        if not include_self and names_a is names_b and i < vals.shape[0]:
            vals[i] = 0.0
        k = min(int(top_k), vals.shape[0])
        if k <= 0:
            continue
        idx = np.argpartition(np.abs(vals), -k)[-k:]
        for j in idx:
            value = float(vals[int(j)])
            if abs(value) <= 1e-8:
                continue
            rows.append(
                {
                    "source": str(source_name),
                    "target": str(names_b[int(j)]),
                    "weight": abs(value),
                    "sign": 1 if value >= 0 else -1,
                    "evidence": evidence,
                    "raw_score": value,
                }
            )
    return pd.DataFrame(rows)


def _collapse_prior_edge_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "sign", "evidence", "raw_score"])
    table = table.copy()
    table["source"] = table["source"].astype(str).str.strip()
    table["target"] = table["target"].astype(str).str.strip()
    table = table[(table["source"] != "") & (table["target"] != "") & (table["source"] != table["target"])]
    if table.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "sign", "evidence", "raw_score"])
    table["signed_weight"] = table["weight"].astype(float) * table["sign"].astype(float)
    collapsed = (
        table.groupby(["source", "target"], as_index=False)
        .agg(
            weight=("weight", "mean"),
            signed_weight=("signed_weight", "mean"),
            evidence=("evidence", lambda vals: ";".join(sorted(set(map(str, vals))))),
            raw_score=("raw_score", "mean"),
        )
    )
    collapsed["sign"] = np.where(collapsed["signed_weight"] >= 0, 1, -1)
    collapsed["weight"] = np.clip(np.abs(collapsed["signed_weight"]), 0.0, 1.0)
    return collapsed[["source", "target", "weight", "sign", "evidence", "raw_score"]]


def _build_correlation_candidate_table(
    dataset: DatasetBundle,
    *,
    top_k: int,
    max_features: Optional[int],
    evidence_prefix: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if dataset.view1_dfs is None or len(dataset.view1_dfs) == 0:
        view_df = dataset.expression_df
        view_label = "view"
    else:
        view_df = dataset.view1_dfs[0]
        view_label = str(dataset.view1_name or "view")

    common_genes = [str(g) for g in dataset.common_genes if g in dataset.expression_df.columns and g in view_df.columns]
    expr_df = dataset.expression_df.loc[dataset.common_cells, common_genes].fillna(0.0)
    view_df = view_df.loc[dataset.common_cells, common_genes].fillna(0.0)
    expr_df, selected_expr = _select_top_variable_columns(expr_df, max_features)
    view_df, selected_view = _select_top_variable_columns(view_df, max_features)
    selected_nodes = sorted(set(selected_expr) | set(selected_view))

    tables = [
        _top_corr_edge_table(
            expr_df.values,
            selected_expr,
            expr_df.values,
            selected_expr,
            top_k,
            f"{evidence_prefix}_expression_corr",
        ),
        _top_corr_edge_table(
            view_df.values,
            selected_view,
            view_df.values,
            selected_view,
            top_k,
            f"{evidence_prefix}_{view_label}_corr",
        ),
        _top_corr_edge_table(
            expr_df.values,
            selected_expr,
            view_df.values,
            selected_view,
            top_k,
            f"{evidence_prefix}_cross_corr",
        ),
        _top_corr_edge_table(
            view_df.values,
            selected_view,
            expr_df.values,
            selected_expr,
            top_k,
            f"{evidence_prefix}_cross_corr",
        ),
    ]
    table = _collapse_prior_edge_table(pd.concat(tables, ignore_index=True))
    return table, selected_nodes


class _EdgeConfidenceNet(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.node_proj(node_features))
        pair = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_mlp(pair).squeeze(1)


def _feature_node_matrix(dataset: DatasetBundle, nodes: List[str], dim: int, seed: int) -> np.ndarray:
    if dataset.view1_dfs is None or len(dataset.view1_dfs) == 0:
        view_df = dataset.expression_df
    else:
        view_df = dataset.view1_dfs[0]
    expr = dataset.expression_df.loc[dataset.common_cells, nodes].fillna(0.0).values.astype(np.float32).T
    view = view_df.loc[dataset.common_cells, nodes].fillna(0.0).values.astype(np.float32).T
    raw = np.concatenate([_safe_standardize(expr), _safe_standardize(view)], axis=1)
    return _reduce_to_fixed_dim(raw, int(dim)).astype(np.float32)


def _train_edge_confidence(
    nodes: List[str],
    table: pd.DataFrame,
    node_features: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> pd.DataFrame:
    if table.empty:
        return table
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rng = np.random.default_rng(int(seed))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    rows = table[table["source"].isin(node_index) & table["target"].isin(node_index)].copy()
    if rows.empty:
        return rows

    src_np = np.asarray([node_index[s] for s in rows["source"]], dtype=np.int64)
    dst_np = np.asarray([node_index[t] for t in rows["target"]], dtype=np.int64)
    raw_weight = rows["weight"].astype(float).to_numpy(dtype=np.float32)
    sign = rows["sign"].astype(float).to_numpy(dtype=np.float32)
    edge_attr_np = np.stack([raw_weight, np.abs(sign), np.maximum(sign, 0.0)], axis=1).astype(np.float32)

    x = torch.from_numpy(node_features.astype(np.float32)).to(device)
    edge_attr = torch.from_numpy(edge_attr_np).to(device)
    src = torch.from_numpy(src_np).long().to(device)
    dst = torch.from_numpy(dst_np).long().to(device)
    positives = set(zip(src_np.tolist(), dst_np.tolist()))

    model = _EdgeConfidenceNet(x.shape[1], edge_attr.shape[1], int(hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    epochs = int(epochs)
    for _ in range(max(0, epochs)):
        neg = []
        while len(neg) < len(rows):
            a = int(rng.integers(0, len(nodes)))
            b = int(rng.integers(0, len(nodes)))
            if a != b and (a, b) not in positives:
                neg.append((a, b))
        neg_np = np.asarray(neg, dtype=np.int64)
        neg_src = torch.from_numpy(neg_np[:, 0]).long().to(device)
        neg_dst = torch.from_numpy(neg_np[:, 1]).long().to(device)
        neg_attr = torch.zeros((len(neg), edge_attr.shape[1]), dtype=torch.float32, device=device)
        pos_logit = model(x, edge_attr, src, dst)
        neg_logit = model(x, neg_attr, neg_src, neg_dst)
        shuffled_x = x[torch.randperm(x.shape[0], device=device)]
        corrupt_logit = model(shuffled_x, edge_attr, src, dst)
        loss = (
            F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
            + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
            + 0.5 * F.binary_cross_entropy_with_logits(corrupt_logit, torch.zeros_like(corrupt_logit))
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        confidence = torch.sigmoid(model(x, edge_attr, src, dst)).detach().cpu().numpy()
    scored = rows.copy()
    scored["attention_score"] = confidence.astype(np.float32)
    scored["confidence"] = np.clip(0.7 * confidence + 0.3 * raw_weight, 0.0, 1.0)
    scored["weight"] = scored["confidence"].astype(float)
    return scored


def _edge_table_to_data_groups(table: pd.DataFrame, prefix: str) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    groups: Dict[str, List[str]] = {}
    weights: Dict[str, float] = {}
    if table.empty:
        return groups, weights
    for row_idx, row in table.reset_index(drop=True).iterrows():
        source = str(row["source"]).strip()
        target = str(row["target"]).strip()
        if not source or not target or source == target:
            continue
        evidence = str(row.get("evidence", "edge")).replace(";", "+")
        group_name = f"{prefix}::{row_idx:06d}::{evidence}"
        groups[group_name] = [source, target]
        weights[group_name] = float(row.get("weight", 1.0))
    return groups, weights


def _build_data_driven_prior_bundle(
    dataset: DatasetBundle,
    *,
    prior_name: str,
    top_k: int,
    max_features: Optional[int],
    denoise_candidate_top_k: int,
    denoise_node_feature_dim: int,
    denoise_hidden_dim: int,
    denoise_epochs: int,
    denoise_lr: float,
    denoise_top_percent: float,
    device: str,
    seed: int,
) -> PriorBundle:
    if prior_name == "none":
        metadata = {
            "prior_name": "none",
            "construction_method": "empty data-driven prior",
            "external_prior": False,
            "labels_used": False,
            "n_edges": 0,
        }
        return PriorBundle({}, {}, None, None, data_groups={}, data_group_weights={}, edge_table=pd.DataFrame(), metadata=metadata)

    if prior_name == "p_glue":
        table, nodes = _build_correlation_candidate_table(
            dataset,
            top_k=top_k,
            max_features=max_features,
            evidence_prefix="glue",
        )
        prefix = "p_glue"
        metadata = {
            "prior_name": "p_glue",
            "construction_method": "GLUE-inspired data-driven signed expression/protein feature graph",
            "top_k": int(top_k),
            "max_features": None if max_features is None else int(max_features),
            "external_prior": False,
            "labels_used": False,
            "selected_feature_count": len(nodes),
            "seed": int(seed),
        }
    elif prior_name == "p_denoise":
        table, nodes = _build_correlation_candidate_table(
            dataset,
            top_k=denoise_candidate_top_k,
            max_features=max_features,
            evidence_prefix="denoise_candidate",
        )
        node_features = _feature_node_matrix(dataset, nodes, denoise_node_feature_dim, seed)
        table = _train_edge_confidence(
            nodes,
            table,
            node_features,
            hidden_dim=denoise_hidden_dim,
            epochs=denoise_epochs,
            lr=denoise_lr,
            device=device,
            seed=seed,
        )
        top_percent = float(denoise_top_percent)
        if not table.empty and top_percent < 1.0:
            cutoff = table["confidence"].quantile(max(0.0, min(1.0, 1.0 - top_percent)))
            table = table[table["confidence"] >= cutoff].copy()
        prefix = "p_denoise"
        metadata = {
            "prior_name": "p_denoise",
            "construction_method": "data-driven candidate graph denoised by edge confidence model",
            "candidate_top_k": int(denoise_candidate_top_k),
            "max_features": None if max_features is None else int(max_features),
            "node_feature_dim": int(denoise_node_feature_dim),
            "hidden_dim": int(denoise_hidden_dim),
            "epochs": int(denoise_epochs),
            "top_percent": top_percent,
            "external_prior": False,
            "labels_used": False,
            "selected_feature_count": len(nodes),
            "seed": int(seed),
        }
    else:
        raise ValueError(f"Unknown prior_name={prior_name!r}. Valid values: dataset, none, p_glue, p_denoise")

    groups, weights = _edge_table_to_data_groups(table, prefix)
    metadata["n_edges"] = int(len(groups))
    metadata["density"] = float(len(groups) / max(1, len(dataset.common_genes) ** 2))
    return PriorBundle(
        kegg_groups={},
        poswin_groups={},
        ppi_groups=None,
        gene_prior_matrix=None,
        data_groups=groups,
        data_group_weights=weights,
        edge_table=table.reset_index(drop=True),
        metadata=metadata,
    )

def _build_base_prior_bundle(base_dir: Path, dataset: DatasetBundle, d_prior: int = 16) -> PriorBundle:
    base_dir = Path(base_dir)
    
    # 从 DATASET_CONFIG 获取配置
    config = DATASET_CONFIG.get(dataset.dataset_type)
    if config is None:
        raise ValueError(f"Unknown dataset type: {dataset.dataset_type}")
    
    files = config["files"]
    dataset_root = base_dir / config["root"]

    # 加载 KEGG prior - 从 DATASET_CONFIG 获取
    kegg_groups = {}
    kegg_name = str(files.get("kegg_prior", "") or "").strip()
    if kegg_name:
        kegg_file = dataset_root / kegg_name
        if kegg_file.exists():
            _, _, kegg_hypergraph = build_gene_prior_features(
                dataset.common_genes,
                kegg_file,
                d_prior,
            )
            kegg_groups = _build_prefixed_groups("path::", kegg_hypergraph)
    
    # 检查是否需要加载 PPI
    has_ppi = config.get("has_ppi", False)
    ppi_groups = None
    if has_ppi and files.get("ppi_prior"):
        ppi_file = dataset_root / files["ppi_prior"]
        ppi_df = pd.read_csv(ppi_file, header=0, index_col=0, sep=",").fillna(0)
        ppi_df.index = ppi_df.index.astype(str).str.strip()
        ppi_df.columns = [str(col).strip() for col in ppi_df.columns]

        ppi_groups_raw: Dict[str, List[str]] = {}
        for row_idx, gene_i in enumerate(dataset.common_genes):
            if gene_i not in ppi_df.index:
                continue
            neighbors: List[str] = []
            for col_idx, gene_j in enumerate(dataset.common_genes):
                if gene_i == gene_j or gene_j not in ppi_df.columns:
                    continue
                try:
                    value = float(ppi_df.loc[gene_i, gene_j])
                except Exception:
                    value = 0.0
                if value != 0.0:
                    neighbors.append(gene_j)
            members = sorted(list(dict.fromkeys([gene_i] + neighbors)))
            if len(members) > 1:
                ppi_groups_raw[gene_i] = members

        ppi_groups = _build_prefixed_groups("ppi::", ppi_groups_raw)
    
    return PriorBundle(
        kegg_groups=kegg_groups,
        poswin_groups={},
        ppi_groups=ppi_groups,
        gene_prior_matrix=None
    )

def build_prior_bundle(base_dir, dataset, d_prior=16, *,
            allow_position_file_fallback=False, genomic_window_bp=200000, include_window_groups=True,
            prior_name="dataset", prior_top_k=5, prior_max_features=800,
            denoise_candidate_top_k=5, denoise_node_feature_dim=64,
            denoise_hidden_dim=64, denoise_epochs=20, denoise_lr=1e-3,
            denoise_top_percent=0.7, device="cpu", seed=42):
    base_dir = Path(base_dir)
    config = DATASET_CONFIG.get(dataset.dataset_type)
    dataset_root = base_dir / config["root"] if config else base_dir

    prior_name = str(prior_name or "dataset")
    if prior_name != "dataset":
        return _build_data_driven_prior_bundle(
            dataset,
            prior_name=prior_name,
            top_k=prior_top_k,
            max_features=prior_max_features,
            denoise_candidate_top_k=denoise_candidate_top_k,
            denoise_node_feature_dim=denoise_node_feature_dim,
            denoise_hidden_dim=denoise_hidden_dim,
            denoise_epochs=denoise_epochs,
            denoise_lr=denoise_lr,
            denoise_top_percent=denoise_top_percent,
            device=device,
            seed=seed,
        )

    prior = _build_base_prior_bundle(base_dir, dataset, d_prior)

    position = _load_position_prior(
        dataset_root,
        dataset.dataset_type,
        dataset.common_genes,
        allow_fallback=allow_position_file_fallback
    )
    rows = position.get("rows", {}) or {}

    gene_positions = {}
    for gene in dataset.common_genes:
        info = rows.get(gene)
        if info is None:
            continue
        chrom = str(info.get("chrom", "")).strip()
        if not chrom:
            continue
        try:
            start_i = float(info.get("start"))
            end_i = float(info.get("end"))
        except Exception:
            continue
        if not np.isfinite(start_i) or not np.isfinite(end_i):
            continue
        gene_positions[str(gene).strip()] = {
            "chrom": chrom,
            "start": start_i,
            "end": end_i
        }
    
    poswin_groups: Dict[str, List[str]] = {}
    if include_window_groups:
        n_genes = len(dataset.common_genes)
        chroms = []
        mids = []
        for gene in dataset.common_genes:
            info = gene_positions.get(str(gene).strip())
            if info is None:
                chroms.append(None)
                mids.append(None)
            else:
                chroms.append(str(info["chrom"]).strip())
                mids.append((float(info["start"]) + float(info["end"])) / 2.0)
        
        for i in range(n_genes):
            if chroms[i] is None or mids[i] is None:
                continue
            members = []
            for j in range(n_genes):
                if chroms[j] is None or mids[j] is None:
                    continue
                if chroms[i] != chroms[j]:
                    continue
                dist = abs(mids[i] - mids[j])
                if dist < genomic_window_bp:
                    members.append(dataset.common_genes[j])
            members = [str(g).strip() for g in dict.fromkeys(members) if str(g).strip()]
            if len(members) > 1:
                poswin_groups[f"poswin::{chroms[i]}:{dataset.common_genes[i]}"] = members
    
    n_matched = sum(1 for gene in dataset.common_genes if str(gene).strip() in gene_positions)
    n_missing = len(dataset.common_genes) - n_matched
    n_window_groups = len(poswin_groups)

    audit = dict(position.get("audit", {}))
    audit["dataset_type"] = dataset.dataset_type
    audit["n_common_genes"] = len(dataset.common_genes)
    audit["n_position_matched"] = n_matched
    audit["n_position_missing"] = n_missing
    audit["uses_position_features"] = 0
    audit["uses_chrom_groups"] = 0
    audit["uses_nearby_groups"] = 0
    audit["uses_genomic_window"] = 1
    audit["genomic_window_bp"] = float(genomic_window_bp)
    audit["n_window_groups"] = n_window_groups
    audit["include_window_groups"] = int(bool(include_window_groups))

    _POSITION_PRIOR_AUDIT[dataset.dataset_type] = audit

    print(
        f"  Position prior (full/genomic_window): matched {n_matched}/{len(dataset.common_genes)} genes, "
        f"window_bp={float(genomic_window_bp):.0f}, window_groups={n_window_groups}, "
        f"fallback={bool(allow_position_file_fallback)}"
    )

    return PriorBundle(
        kegg_groups=prior.kegg_groups,
        poswin_groups=poswin_groups,
        ppi_groups=prior.ppi_groups,
        gene_prior_matrix=None,
        data_groups={},
        data_group_weights={},
        edge_table=pd.DataFrame(),
        metadata={
            "prior_name": "dataset",
            "construction_method": "dataset external KEGG/position/PPI prior",
            "external_prior": True,
            "labels_used": False,
            "n_kegg_groups": len(prior.kegg_groups),
            "n_poswin_groups": len(poswin_groups),
            "n_ppi_groups": 0 if prior.ppi_groups is None else len(prior.ppi_groups),
        },
    )

__all__ = [name for name in globals() if not name.startswith("__")]
