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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"

np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")


DATASET_CONFIG = {
    "PEA_STA": {
        "name": "PEA_STA",
        "description": "Protein Expression and STA data",
        "root": DATA_ROOT / "PEA_STA",
        "files": {
            "expression": "expression_data.csv",
            "view": ["protein_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_pea.txt",
            "ppi_prior": "human_ppi.csv"
        },
        "has_ppi": True,
        "have_answer": False
    },
    "SCoPE2": {
        "name": "SCoPE2",
        "description": "SCoPE2 protein marker and gene expression data",
        "root": DATA_ROOT / "SCoPE2",
        "files": {
            "expression": "expression_data.csv",
            "view": ["protein_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "",
            "poswin_prior": "",
            "ppi_prior": ""
        },
        "view_name": "protein_marker",
        "has_ppi": False,
        "have_answer": False
    },
    "sc_GEM": {
        "name": "sc_GEM",
        "description": "Single-cell Gene Expression Methylation data",
        "root": DATA_ROOT / "sc_GEM",
        "files": {
            "expression": "expression_data.csv",
            "view": ["methylation_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_sc.txt",
            "ppi_prior": ""
        },
        "has_ppi": False,
        "have_answer": False
    },
    "sim_gene100_alpha_1_beta_2":{
        "name": "sim_gene100_alpha_1_beta_2",
        "description": "Simulated Gene Expression data with alpha=0.1, beta=0.2",
        "root": DATA_ROOT / "sim_gene100" / "alpha_1_beta_2",
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "ppi_prior.csv"
        },
        "has_ppi": False,
        "have_answer": True
    },
    "sim_gene100_alpha_1_beta_1":{
        "name": "sim_gene100_alpha_1_beta_1",
        "description": "Simulated Gene Expression data with alpha=0.1, beta=0.1",
        "root": DATA_ROOT / "sim_gene100" / "alpha_1_beta_1",
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "ppi_prior.csv"
        },
        "has_ppi": False,
        "have_answer": True
    },
    "sim_gene100_alpha_2_beta_1":{
        "name": "sim_gene100_alpha_2_beta_1",
        "description": "Simulated Gene Expression data with alpha=0.2, beta=0.1",
        "root": DATA_ROOT / "sim_gene100" / "alpha_2_beta_1",
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "ppi_prior.csv"
        },
        "has_ppi": False,
        "have_answer": True
    },
    "sim_gene100_alpha_2_beta_2":{
        "name": "sim_gene100_alpha_2_beta_2",
        "description": "Simulated Gene Expression data with alpha=0.2, beta=0.2",
        "root": DATA_ROOT / "sim_gene100" / "alpha_2_beta_2",
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "ppi_prior.csv"
        },
        "has_ppi": False,
        "have_answer": True
    },
    "sim_gene100_alpha_2_beta_5":{
        "name": "sim_gene100_alpha_2_beta_5",
        "description": "Simulated Gene Expression data with alpha=0.2, beta=0.5",
        "root": DATA_ROOT / "sim_gene100" / "alpha_2_beta_5",
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "ppi_prior.csv"
        },
        "has_ppi": False,
        "have_answer": True
    }
}

CLUSTER_VERSION_NAMES = {
    "kmeans": "TV_PHASE_v11",
    "leiden": "TV_PHASE_v11",
    "louvain": "TV_PHASE_v11",
}
DEFAULT_LEIDEN_RESOLUTION = {
    "sc_GEM": 1.5,
    "PEA_STA": 0.5,
}

DEFAULT_CLUSTER_METHODS = tuple(CLUSTER_VERSION_NAMES.keys())
GENE_POSITION_MODES = ("none", "feature", "chrom", "near", "full")
DEFAULT_GENE_POSITION_MODE = "full"
DEFAULT_PHASE_RESIDUAL_SCALE_METHYLATION = 1.0
DEFAULT_PHASE_RESIDUAL_SCALE_PPI = 0.05

GENE_POSITION_ALIASES: Dict[str, List[str]] = {
    "OCT4": ["POU5F1"],
    "NESTIN": ["NES"],
    "CXCL8.IL.8": ["CXCL8", "IL8"],
    "IGFBP.2": ["IGFBP2"],
    "TENASCIN.C": ["TNC"],
    "SNAL1": ["SNAI1"],
    "CASPR1": ["CNTNAP1"],
    "HS1": ["HCLS1"],
}

_POSITION_PRIOR_AUDIT: Dict[str, Dict[str, object]] = {}
_POSITION_ALIAS_AUDIT_BY_DATASET: Dict[str, List[Dict[str, object]]] = {}

METH_PHASE_LABEL_MAP = {
    0: "BJ",
    1: "IPS",
    2: "ES",
    3: "d24T+",
    4: "d24T-",
    5: "d16T+",
    6: "d16T-",
    7: "d8",
    8: "Other",
}

PPI_PHASE_LABEL_MAP = {
    0: "0h_control",
    1: "6d_BMP4",
    2: "6d_control"
}

@dataclass
class DatasetBundle:
    dataset_type: str
    view1_name: str
    view1_dfs: Optional[List[pd.DataFrame]] = None
    expression_df: pd.DataFrame = None
    common_cells: List[str] = None
    common_genes: List[str] = None
    labels: np.ndarray = None
    label_names: List[str] = None
    label_map: Dict[int, str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PhaseTrainingConfig:

    data_name: str = "sc_GEM"
    output_dir: Optional[Path] = None
    device: str = "cpu"
    seed: int = 42

    prior_name: str = "dataset"
    prior_top_k: int = 5
    prior_max_features: Optional[int] = 800
    denoise_candidate_top_k: int = 5
    denoise_node_feature_dim: int = 64
    denoise_hidden_dim: int = 64
    denoise_epochs: int = 20
    denoise_lr: float = 1e-3
    denoise_top_percent: float = 0.7

    feature_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 32
    prior_dim: int = 16

    train_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    
    use_lr_scheduler: bool = True
    lr_min: float = 1e-6
    
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6

    learn_input_residual: bool = True
    input_residual_scale: float = 0.05

    vae_recon_weight: float = 1.0
    vae_kl_weight: float = 0.05
    vae_kl_warmup_epochs: int = 80
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    kl_capacity_max: float = 1.0

    cell_gene_recon_weight: float = 0.2
    cell_gene_temperature: float = 0.2
    cell_gene_num_neg: int = 1

    phase_sep_weight: float = 0.5
    gate_balance_weight: float = 0.1
    gate_entropy_weight: float = 0.02
    gene_gate_smoothness_weight: float = 0.02
    cell_structure_weight: float = 0.1
    gate_variance_weight: float = 0.1

    use_dec: bool = False
    num_clusters: Optional[int] = None
    dec_start_epoch: int = 100
    dec_weight: float = 0.1

    edge_type_init_weights: Dict[str, float] = None

    hyperedge_dropout_rate: float = 0.1
    feature_mask_rate: float = 0.1
    hgnn_dropout_rate: float = 0.05

    cluster_method: str = "kmeans"
    cluster_resolution: Optional[float] = None
    
    cell_structure_temp: float = 0.2
    cell_hyperedge_top_k: int = 10
    cell_hyperedge_top_fraction: Optional[float] = None
    min_cell_hyperedge_size: int = 2

    def __post_init__(self):
        if self.edge_type_init_weights is None:
            self.edge_type_init_weights = {
                "pathway": 1.0,
                "ppi": 1.0,
                "poswin": 1.0,
                "cell": 1.0,
            }


@dataclass
class PriorBundle:
    kegg_groups: Dict[str, List[str]]  # KEGG pathway gene groups
    poswin_groups: Dict[str, List[str]]  # gene position/window gene groups
    ppi_groups: Optional[Dict[str, List[str]]] = None  # PPI groups (for PPI dataset)
    gene_prior_matrix: Optional[np.ndarray] = None  # kept for backward compatibility
    data_groups: Optional[Dict[str, List[str]]] = None  # data-driven prior groups
    data_group_weights: Optional[Dict[str, float]] = None
    edge_table: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None

__all__ = [name for name in globals() if not name.startswith("__")]
