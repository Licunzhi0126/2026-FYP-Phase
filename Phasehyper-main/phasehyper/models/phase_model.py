"""两层级 HGNN + Dual-RAE 分相模型。

  HGNN 两通道 → Gene Self-Attention → Cross-Attention → fused embedding
  → 两个独立 RAE (Enc_A+Dec_A / Enc_B+Dec_B) → X_A, X_B
  → X_A + X_B ≈ expr（学习目标，非恒等式）
  → X_A ≠ X_B（对比竞争，两相分化）
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from phasehyper.models.two_level_hgnn import HypergraphChannel, scipy_to_torch_sparse
from phasehyper.schemas import PhaseTrainingConfig
from phasehyper.utils import clean_nan


class ExprGuidedCrossAttention(nn.Module):
    """cell(Q) attend to gene(K,V)，expr z-score 做 attention bias。"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.bias_scale = nn.Parameter(torch.tensor(0.5))
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv, expr_bias):
        N, d = q.shape
        G = kv.shape[0]
        h, hd = self.num_heads, self.head_dim
        Q = self.W_q(q).view(N, h, hd).permute(1, 0, 2)
        K = self.W_k(kv).view(G, h, hd).permute(1, 0, 2)
        V = self.W_v(kv).view(G, h, hd).permute(1, 0, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn + self.bias_scale * expr_bias.unsqueeze(0)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).permute(1, 0, 2).contiguous().view(N, d)
        return self.norm(q + self.W_o(out))


class RAEEncoder(nn.Module):
    """共享编码器：fused → z。"""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RAEDecoder(nn.Module):
    """独立解码器：z → expression。"""

    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [latent_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class HyperedgeActivityPhaseDecoder(nn.Module):
    """Decode a shared latent state through competing hyperedge activities."""

    def __init__(
        self,
        latent_dim: int,
        H_gene,
        W_gene=None,
        output_projection=None,
        gene_names: Optional[Sequence[str]] = None,
        edge_names: Optional[Sequence[str]] = None,
        edge_types: Optional[Sequence[str]] = None,
        output_bias=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        H = sp.csr_matrix(H_gene, dtype=np.float32).copy()
        H.sum_duplicates()
        H.data[:] = 1.0
        H.eliminate_zeros()
        n_genes, n_prior_edges = H.shape
        if n_genes <= 0:
            raise ValueError("H_gene must contain at least one gene row")

        if W_gene is None:
            weights = np.ones(n_prior_edges, dtype=np.float32)
        else:
            weights = np.asarray(W_gene, dtype=np.float32).reshape(-1)
            if weights.size != n_prior_edges:
                raise ValueError(
                    f"W_gene has {weights.size} entries, expected {n_prior_edges}"
                )
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.abs(weights)

        names = list(gene_names) if gene_names is not None else [str(i) for i in range(n_genes)]
        if len(names) != n_genes:
            raise ValueError(f"gene_names has {len(names)} entries, expected {n_genes}")
        prior_names = list(edge_names) if edge_names is not None else [f"prior::{i}" for i in range(n_prior_edges)]
        prior_types = list(edge_types) if edge_types is not None else ["prior"] * n_prior_edges
        if len(prior_names) != n_prior_edges:
            raise ValueError(f"edge_names has {len(prior_names)} entries, expected {n_prior_edges}")
        if len(prior_types) != n_prior_edges:
            raise ValueError(f"edge_types has {len(prior_types)} entries, expected {n_prior_edges}")

        degree = np.asarray(H.multiply(weights[np.newaxis, :]).sum(axis=1)).reshape(-1)
        fallback_gene_index = np.flatnonzero(degree <= eps).astype(np.int64)
        if fallback_gene_index.size:
            fallback_cols = np.arange(fallback_gene_index.size, dtype=np.int64)
            fallback = sp.csr_matrix(
                (np.ones(fallback_gene_index.size, dtype=np.float32),
                 (fallback_gene_index, fallback_cols)),
                shape=(n_genes, fallback_gene_index.size),
            )
            H = sp.hstack([H, fallback], format="csr", dtype=np.float32)
            weights = np.concatenate([
                weights, np.ones(fallback_gene_index.size, dtype=np.float32)
            ])
            prior_names.extend(f"unannotated::{names[i]}" for i in fallback_gene_index)
            prior_types.extend(["fallback"] * fallback_gene_index.size)

        degree = np.asarray(H.multiply(weights[np.newaxis, :]).sum(axis=1)).reshape(-1)
        if np.any(degree <= eps):
            raise ValueError("Unable to construct a positive decoding degree for every gene")

        self.n_genes = int(n_genes)
        self.n_prior_edges = int(n_prior_edges)
        self.n_fallback_edges = int(fallback_gene_index.size)
        self.n_edges = int(H.shape[1])
        self.fallback_gene_index = fallback_gene_index.tolist()
        self.gene_names = names
        self.edge_names = prior_names
        self.edge_types = prior_types
        self.eps = float(eps)

        self.register_buffer("H_gene", scipy_to_torch_sparse(H))
        self.register_buffer("edge_weight", torch.from_numpy(weights))
        self.register_buffer("gene_degree", torch.from_numpy(degree.astype(np.float32)))

        if output_projection is None:
            projection = None
            output_dim = n_genes
        else:
            projection_np = np.asarray(output_projection, dtype=np.float32)
            if projection_np.ndim != 2 or projection_np.shape[0] != n_genes:
                raise ValueError(
                    "output_projection must have shape (n_genes, output_dim), "
                    f"received {projection_np.shape}"
                )
            projection = torch.from_numpy(projection_np)
            output_dim = int(projection_np.shape[1])
        self.register_buffer("output_projection", projection)

        if output_bias is None:
            bias = None
        else:
            bias_np = np.asarray(output_bias, dtype=np.float32).reshape(-1)
            if bias_np.size != output_dim:
                raise ValueError(f"output_bias has {bias_np.size} entries, expected {output_dim}")
            bias = torch.from_numpy(bias_np)
        self.register_buffer("output_bias", bias)

        self.activity_head = nn.Linear(latent_dim, self.n_edges)
        self.phase_gate_head = nn.Linear(latent_dim, self.n_edges)
        nn.init.constant_(self.phase_gate_head.bias, 0.0)
        initial_scale_raw = float(np.log(np.expm1(1.0)))
        self.gene_scale_raw = nn.Parameter(torch.full((n_genes,), initial_scale_raw))

    @classmethod
    def from_node_hypergraphs(
        cls,
        latent_dim: int,
        n_cells: int,
        n_genes: int,
        H_tail,
        H_head,
        W_directed,
        H_functional,
        W_functional,
        **kwargs,
    ):
        """Extract gene rows from existing node-level hypergraphs."""
        gene_rows = slice(n_cells, n_cells + n_genes)
        H_directed_gene = (
            sp.csr_matrix(H_tail)[gene_rows, :] + sp.csr_matrix(H_head)[gene_rows, :]
        )
        H_functional_gene = sp.csr_matrix(H_functional)[gene_rows, :]
        H_gene = sp.hstack([H_directed_gene, H_functional_gene], format="csr")
        W_gene = np.concatenate([
            np.asarray(W_directed).reshape(-1),
            np.asarray(W_functional).reshape(-1),
        ])
        return cls(latent_dim=latent_dim, H_gene=H_gene, W_gene=W_gene, **kwargs)

    def _edge_to_gene(self, edge_activity: torch.Tensor) -> torch.Tensor:
        weighted = edge_activity * self.edge_weight.unsqueeze(0)
        gene_signal = torch.sparse.mm(
            self.H_gene, weighted.transpose(0, 1)
        ).transpose(0, 1)
        gene_signal = gene_signal / self.gene_degree.unsqueeze(0)
        gene_scale = F.softplus(self.gene_scale_raw) + 1e-4
        return gene_signal * gene_scale.unsqueeze(0)

    def _project(self, gene_signal: torch.Tensor) -> torch.Tensor:
        if self.output_projection is None:
            return gene_signal
        return gene_signal @ self.output_projection

    def forward(
        self,
        z: torch.Tensor,
        target_total: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        total = F.softplus(self.activity_head(z))
        gate = torch.sigmoid(self.phase_gate_head(z))
        edge_a = total * gate
        edge_b = total * (1.0 - gate)

        gene_a = self._edge_to_gene(edge_a)
        gene_b = self._edge_to_gene(edge_b)
        out_a = self._project(gene_a)
        out_b = self._project(gene_b)
        if self.output_bias is not None:
            half_bias = 0.5 * self.output_bias.unsqueeze(0)
            out_a = out_a + half_bias
            out_b = out_b + half_bias
        raw_recon = out_a + out_b

        if target_total is not None:
            if target_total.shape != raw_recon.shape:
                raise ValueError(
                    f"target_total shape {tuple(target_total.shape)} does not match "
                    f"decoder output {tuple(raw_recon.shape)}"
                )
            correction = 0.5 * (target_total - raw_recon)
            out_a = out_a + correction
            out_b = out_b + correction

        return {
            "total_edge_activity": total,
            "phase_gate": gate,
            "phase_a_edge_activity": edge_a,
            "phase_b_edge_activity": edge_b,
            "phase_a_gene_signal": gene_a,
            "phase_b_gene_signal": gene_b,
            "raw_expr_recon": raw_recon,
            "phase_a_expr": out_a,
            "phase_b_expr": out_b,
            "expr_recon": out_a + out_b,
        }


class HGNN_DualRAE_Phase_Model(nn.Module):
    """两通道 HGNN + Gene SA + Cross-Attn + 共享Encoder双Decoder 分相。

    fused → SharedEncoder → z (N×latent)
    z → Decoder_A → X_A (N×G)
    z → Decoder_B → X_B (N×G)
    约束：X_A + X_B ≈ expr（学出来的，不是恒等式）
    竞争：X_A ≠ X_B（正交化损失驱动分化）

    共享 Encoder 保证两个 Decoder 都从同一个有结构的 latent 出发，
    防止一个学信号、一个学噪声的退化解。
    """

    def __init__(self, built: Dict, config: PhaseTrainingConfig):
        super().__init__()
        self.config = config
        n_cells = int(built["n_cells"])
        n_genes = int(built["n_genes"])
        self.num_cells = n_cells
        self.num_genes = n_genes

        hidden = config.hidden_dim
        latent = config.latent_dim
        dropout = config.hgnn_dropout_rate
        n_layers = config.hgnn_num_layers

        expr = np.nan_to_num(np.asarray(built["expr"], dtype=np.float32))
        self.register_buffer("expr", torch.from_numpy(expr))

        cell_features = np.nan_to_num(np.asarray(built["cell_features"], dtype=np.float32))
        self.register_buffer("cell_features", torch.from_numpy(cell_features))
        n_cell_features = int(built["n_cell_features"])
        # 基因节点状态（中心法则整合点）：gene_in 输入。flag 关或单模态时 == expr.T（n_cells 宽）
        gene_features = np.nan_to_num(np.asarray(built["gene_features"], dtype=np.float32))
        self.register_buffer("gene_features", torch.from_numpy(gene_features))
        n_gene_features = int(built["n_gene_features"])

        # ---- 输入投影 ----
        self.cell_in = nn.Linear(n_cell_features, hidden)
        self.gene_in = nn.Linear(n_gene_features, hidden)
        self.input_norm_cell = nn.LayerNorm(hidden)
        self.input_norm_gene = nn.LayerNorm(hidden)

        # ---- 两条并行超图卷积通道 ----
        self.cell_channel = HypergraphChannel(
            built["H_cell"], built["W_cell"],
            in_dim=hidden, hidden_dim=hidden, out_dim=hidden,
            num_layers=n_layers, dropout=dropout,
        )
        self.gene_channel = HypergraphChannel(
            built["H_gene"], built["W_gene"],
            in_dim=hidden, hidden_dim=hidden, out_dim=hidden,
            num_layers=n_layers, dropout=dropout,
        )

        # ---- 基因 Self-Attention ----
        self.gene_self_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout, batch_first=True,
        )
        self.gene_self_norm = nn.LayerNorm(hidden)

        # ---- Cross-Attention: cell(Q) × gene(K,V) → fused ----
        self.cross_attn = ExprGuidedCrossAttention(
            dim=hidden, num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )

        # ---- 共享 Encoder + 双独立 Decoder ----
        self.encoder = RAEEncoder(hidden, hidden, latent,
                                  num_layers=2, dropout=dropout)
        self.decoder_a = RAEDecoder(latent, hidden, n_genes,
                                    num_layers=2, dropout=dropout)
        self.decoder_b = RAEDecoder(latent, hidden, n_genes,
                                    num_layers=2, dropout=dropout)

    def forward(self) -> Dict[str, torch.Tensor]:
        expr = self.expr

        # ---- 两通道 HGNN ----
        cell_x = self.input_norm_cell(self.cell_in(self.cell_features))
        gene_x = self.input_norm_gene(self.gene_in(self.gene_features))
        cell_h = clean_nan(self.cell_channel(cell_x))
        gene_h = clean_nan(self.gene_channel(gene_x))

        # ---- 基因 Self-Attention ----
        gene_sa = gene_h.unsqueeze(0)
        gene_refined, _ = self.gene_self_attn(gene_sa, gene_sa, gene_sa)
        gene_refined = self.gene_self_norm(gene_h + gene_refined.squeeze(0))

        # ---- Cross-Attention → fused embedding ----
        fused = self.cross_attn(cell_h, gene_refined, expr)

        # ---- 共享 Encoder → 双 Decoder ----
        z = self.encoder(fused)                    # (N, latent) 共享 latent
        x_a = self.decoder_a(z)                    # (N, G) Phase A
        x_b = self.decoder_b(z)                    # (N, G) Phase B

        return {
            "fused": fused,
            "z": z,
            "z_a": z,     # 兼容 loss 接口
            "z_b": z,
            "phase_a_expr": x_a,
            "phase_b_expr": x_b,
            "expr_recon": x_a + x_b,
            "expr": expr,
        }
