from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
    expression_mask: Optional[pd.DataFrame] = None
    view1_masks: Optional[List[pd.DataFrame]] = None


@dataclass
class PhaseTrainingConfig:
    data_name: str = "CITE_seq"
    output_dir: Optional[Path] = None
    device: str = "cpu"

    # ── 网络结构 ──
    hidden_dim: int = 128
    latent_dim: int = 64
    hgnn_num_layers: int = 2
    hgnn_dropout_rate: float = 0.05
    prior_dim: int = 16

    # ── Cross-Attention ──
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # ── VAE ──
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    vae_kl_weight: float = 0.05
    vae_kl_warmup_epochs: int = 80
    vae_recon_weight: float = 1.0

    # ── SlotJumpPhaseGate ──
    sae_expansion: int = 4
    jump_bandwidth: float = 0.1
    slot_compete_temp: float = 1.0
    slot_assign_init_std: float = 1.0

    # ── PhaseHead ──
    phase_head_rank: int = 32
    gene_bias_std: float = 1.0

    # ── 对比学习 ──
    contrast_tau: float = 0.2
    contrast_n_neg: int = 32

    # ── Loss 权重（4 项 uncertainty weighted）──
    recon_x_weight: float = 1.0

    # ── 训练 ──
    train_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    use_lr_scheduler: bool = True
    lr_min: float = 1e-6
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6

    # ── 聚类评估 ──
    cluster_method: str = "kmeans"
    cluster_resolution: Optional[float] = None

    # ── 旧版兼容（不再使用，保留避免其他脚本报错）──
    feature_dim: int = 64
    cell_hyperedge_top_k: int = 10
    cell_hyperedge_top_fraction: Optional[float] = None
    min_cell_hyperedge_size: int = 2
    edge_type_init_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.edge_type_init_weights is None:
            self.edge_type_init_weights = {
                "pathway": 1.0, "ppi": 1.0, "poswin": 1.0, "cell": 1.0,
            }


@dataclass
class PriorBundle:
    kegg_groups: Dict[str, List[str]]
    poswin_groups: Dict[str, List[str]]
    ppi_groups: Optional[Dict[str, List[str]]] = None
    gene_prior_matrix: Optional[np.ndarray] = None


@dataclass
class ModalitySpec:
    name: str
    node_type: str
    feature_table: pd.DataFrame
    mask: Optional[pd.DataFrame] = None
    bridge_to_gene: Optional[Dict[str, List[str]]] = None
    binary: bool = False


@dataclass
class HeteroBundle:
    cells: List[str]
    genes: List[str]
    modalities: List[ModalitySpec]
    dataset_type: str = "CITE_seq"
    labels: Optional[np.ndarray] = None
    label_names: Optional[List[str]] = None
    label_map: Optional[Dict[int, str]] = None
