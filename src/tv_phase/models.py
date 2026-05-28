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

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, hard_negative_weight=1.0, use_dynamic_temp=False):
        super().__init__()
        self.base_temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.use_dynamic_temp = use_dynamic_temp
        if use_dynamic_temp:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.temperature = float(temperature)

    def forward(self, features_1, features_2, hard_negatives=None):
        if features_1.shape != features_2.shape:
            raise ValueError(f"Feature dimensions do not match: {features_1.shape} vs {features_2.shape}")
        batch_size = features_1.shape[0]
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
        current_temp = self.temperature if self.use_dynamic_temp else self.base_temperature
        similarity_matrix = torch.matmul(features_1, features_2.T) / current_temp
        positive_samples = torch.arange(batch_size, device=features_1.device)
        loss_1 = F.cross_entropy(similarity_matrix, positive_samples)
        loss_2 = F.cross_entropy(similarity_matrix.T, positive_samples)
        base_loss = (loss_1 + loss_2) / 2.0

        hard_loss = 0.0
        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, p=2, dim=1)
            hard_sim_1 = torch.matmul(features_1, hard_negatives.T) / current_temp
            hard_sim_2 = torch.matmul(features_2, hard_negatives.T) / current_temp
            pos_sim_1 = torch.diag(similarity_matrix)
            pos_sim_2 = torch.diag(similarity_matrix.T)
            hard_loss_1 = F.margin_ranking_loss(
                pos_sim_1.unsqueeze(1),
                hard_sim_1,
                torch.ones_like(hard_sim_1),
                margin=0.1,
            )
            hard_loss_2 = F.margin_ranking_loss(
                pos_sim_2.unsqueeze(1),
                hard_sim_2,
                torch.ones_like(hard_sim_2),
                margin=0.1,
            )
            hard_loss = (hard_loss_1 + hard_loss_2) / 2.0 * self.hard_negative_weight
        return base_loss + hard_loss

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super().__init__()
        encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(np.log(10000.0) / embedding_dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.encoding[:, : x.size(0), :].to(x.device)


class AttentionalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, max_len=5000, dropout=0.2):
        super().__init__()
        self.num_heads = max(1, int(num_heads))
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.pos_encoding_dim = self.hidden_dim
        self.actual_input_dim = self.hidden_dim + self.pos_encoding_dim
        self.query_fc = nn.Linear(self.actual_input_dim, self.hidden_dim * self.num_heads)
        self.key_fc = nn.Linear(self.actual_input_dim, self.hidden_dim * self.num_heads)
        self.value_fc = nn.Linear(self.actual_input_dim, self.hidden_dim * self.num_heads)
        self.output_fc = nn.Linear(self.hidden_dim * self.num_heads, self.output_dim)
        self.residual_projection = nn.Linear(input_dim, self.output_dim)
        self.layer_norm1 = nn.LayerNorm(self.output_dim)
        self.layer_norm2 = nn.LayerNorm(self.output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim * 4, self.output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.position_encoding = PositionalEncoding(max_len, self.pos_encoding_dim)
        self.structure_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, Z, Z_g, Z_c, adjacency_matrix=None):
        num_nodes = Z.size(0)
        pos_encoding = self.position_encoding(Z).squeeze(0)[:num_nodes]
        Z_g_with_pos = torch.cat([Z_g, pos_encoding], dim=-1)
        Z_c_with_pos = torch.cat([Z_c, pos_encoding], dim=-1)
        Q = self.query_fc(Z_c_with_pos).view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        K = self.key_fc(Z_g_with_pos).view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        V = self.value_fc(Z_g_with_pos).view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        if adjacency_matrix is not None:
            structure_bias = adjacency_matrix.unsqueeze(0).repeat(self.num_heads, 1, 1)
            structure_mask = (structure_bias == 0).float() * (-1e9)
            attention_scores = attention_scores + self.structure_weight * structure_mask
        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))
        attended_values = torch.bmm(attention_weights, V)
        attended_values = attended_values.transpose(0, 1).contiguous().view(num_nodes, self.num_heads * self.hidden_dim)
        output = self.output_fc(attended_values)
        output = self.layer_norm1(output + self.residual_projection(Z))
        output = self.layer_norm2(output + self.ffn(output))
        return output, attention_weights


class AlignmentLoss(nn.Module):
    def __init__(self, z4_dim, xc_dim, lambda_center=0.99, lambda_structure=0.56):
        super().__init__()
        self.lambda_center = float(lambda_center)
        self.lambda_structure = float(lambda_structure)
        self.center_projection = nn.Sequential(
            nn.Linear(xc_dim, z4_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(z4_dim * 2, z4_dim),
        )
        self.hard_threshold = 0.7

    def _compute_robust_similarity(self, x1, x2, eps=1e-8):
        x1_norm = F.normalize(x1, p=2, dim=1, eps=eps)
        x2_norm = F.normalize(x2, p=2, dim=1, eps=eps)
        similarity = torch.matmul(x1_norm, x2_norm.T)
        return torch.clamp(similarity, -1.0 + eps, 1.0 - eps)

    def _hard_sample_mining(self, sim_pred, sim_target):
        diff = torch.abs(sim_pred - sim_target)
        threshold = torch.quantile(diff, self.hard_threshold)
        return 1.0 + (diff >= threshold).float()

    def forward(self, Z4, x_c):
        if Z4.shape[0] != x_c.shape[0]:
            raise ValueError("Z4 and x_c must have the same number of nodes")
        sim_Z4 = self._compute_robust_similarity(Z4, Z4)
        sim_xc = self._compute_robust_similarity(x_c, x_c)
        structure_loss = (((sim_Z4 - sim_xc) ** 2) * self._hard_sample_mining(sim_Z4, sim_xc)).mean()
        projected_center = self.center_projection(torch.mean(x_c, dim=0, keepdim=True)).to(Z4.device, dtype=Z4.dtype)
        center_loss = torch.norm(Z4 - projected_center, p=2, dim=1).mean()
        variance_loss = -torch.log(torch.var(Z4, dim=0) + 1e-8).mean()
        return self.lambda_center * center_loss + self.lambda_structure * structure_loss + 0.1 * variance_loss

class HGNNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, dropout: float = 0.0, use_residual: bool = True):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        
        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, X: torch.Tensor, H: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_nodes, n_edges = H.shape
        X = clean_nan(X)
        
        residual = X
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        if edge_weights is None:
            edge_weights = torch.ones(n_edges, device=H.device, dtype=H.dtype)
        edge_weights = torch.clamp(edge_weights, min=0.0, max=100.0)
        D_e = H.sum(dim=0) + 1e-8
        D_v = (H * edge_weights.unsqueeze(0)).sum(dim=1) + 1e-8
        D_e_inv = 1.0 / D_e
        D_v_inv_sqrt = 1.0 / torch.sqrt(D_v)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = X @ self.weight
        X = D_v_inv_sqrt.unsqueeze(1) * X
        X_edge = H.T @ X
        X_edge = (edge_weights * D_e_inv).unsqueeze(1) * X_edge
        X = H @ X_edge
        X = D_v_inv_sqrt.unsqueeze(1) * X
        if self.bias is not None:
            X = X + self.bias
        
        if self.use_residual:
            X = X + residual
        
        return clean_nan(X)

class CellHypergraphEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = HGNNConv(num_features, hidden_dim * 2, dropout=dropout)
        self.conv2 = HGNNConv(hidden_dim * 2, hidden_dim, dropout=dropout)
        self.conv3 = HGNNConv(hidden_dim, hidden_dim, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.residual_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        x1 = F.dropout(F.relu(self.bn1(self.conv1(x, H))), p=self.dropout_rate, training=self.training)
        x2 = self.bn2(self.conv2(x1, H)) + self.residual_projection(x1)
        x2 = F.dropout(F.relu(x2), p=self.dropout_rate, training=self.training)
        x3 = self.bn3(self.conv3(x2, H))
        return F.relu(x3 + x2)

class UniqueGeneHypergraphEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.conv1 = HGNNConv(num_features, hidden_dim, dropout=dropout)
        self.conv2 = HGNNConv(hidden_dim, hidden_dim, dropout=dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn_out = nn.BatchNorm1d(out_dim)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        z = F.dropout(F.gelu(self.bn1(self.conv1(x, H))), p=self.dropout_rate, training=self.training)
        z2 = self.bn2(self.conv2(z, H))
        z = F.dropout(F.gelu(z + z2), p=self.dropout_rate, training=self.training)
        return F.relu(self.bn_out(self.proj(z)))

class BroadcastGeneCellAggregator(nn.Module):
    def __init__(self, gene_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.instance_proj = nn.Linear(gene_dim + 2, hidden_dim)
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, gene_dim),
            nn.BatchNorm1d(gene_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, view1_array: torch.Tensor, expr_array: torch.Tensor, gene_unique_emb: torch.Tensor):
        nc, ng = view1_array.shape
        d = gene_unique_emb.shape[1]
        gene_expand = gene_unique_emb.unsqueeze(0).expand(nc, ng, d)
        inst = torch.cat([view1_array.unsqueeze(-1), expr_array.unsqueeze(-1), gene_expand], dim=-1)
        inst_h = self.instance_proj(inst)
        att = F.softmax(self.att_mlp(inst_h), dim=1)
        pooled = torch.sum(inst_h * att, dim=1)
        return self.out_proj(pooled), gene_expand

class CrossViewResidual(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, zc, zg):
        gate_input = torch.cat([zc, zg, zc - zg, zc * zg], dim=-1)
        alpha = self.gate(gate_input)
        fused = alpha * zc + (1.0 - alpha) * zg
        fused = self.out_norm(self.dropout(fused) + zc)
        return fused

class MultiviewEncoder(nn.Module):
    def __init__(self, gene_encoder, cell_encoder, aggregator, xc_dim, shared_dim=None, dropout=0.2,
                contrastive_temp=0.2,contractive_hard_weight=0.8, num_heads = 8, num_hard_negatives=8,
                lambda_center = 0.1, lambda_structure=0.1):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.cell_encoder = cell_encoder
        self.aggregator = aggregator
        
        self.cell_hidden_dim = cell_encoder.hidden_dim
        self.gene_hidden_dim = gene_encoder.out_dim
        self.shared_dim = shared_dim or self.cell_hidden_dim
        self.num_hard_negatives = num_hard_negatives

        if self.shared_dim % num_heads != 0:
            raise ValueError(
                f"shared_dim ({self.shared_dim}) must be divisible by num_heads ({num_heads})."
            )
        
        self.cell_proj = nn.Sequential(
            nn.Linear(self.cell_hidden_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(self.gene_hidden_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
        )

        self.res_fuse = CrossViewResidual(self.shared_dim, dropout = dropout)

        self.attention = AttentionalGNN(
            input_dim = self.shared_dim,
            hidden_dim = self.shared_dim,
            output_dim = self.shared_dim,
            dropout = dropout,
            num_heads = num_heads
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.shared_dim * 3, self.shared_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_dim * 3, self.shared_dim * 3)
        )

        self.final_projection = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.shared_dim, self.shared_dim),
        )

        self.alignment_loss = AlignmentLoss(
            z4_dim = self.shared_dim,
            xc_dim = xc_dim,
            lambda_center = lambda_center,
            lambda_structure = lambda_structure,
        )

        self.contrastive_loss = ContrastiveLoss(
            temperature = contrastive_temp,
            hard_negative_weight = contractive_hard_weight,
            use_dynamic_temp=False,
        )
    
    def _generate_hard_negatives(self, Z_q, Z_k, num_negatives = None):
        batch_size = Z_q.shape[0]
        if batch_size <= 1:
            return None
        if num_negatives is None:
            num_negatives = self.num_hard_negatives

        k = min(num_negatives, batch_size - 1)
        Z_q_norm = F.normalize(Z_q, p=2, dim=1)
        Z_k_norm = F.normalize(Z_k, p=2, dim=1)

        similarity = torch.matmul(Z_q_norm, Z_k_norm.T)
        similarity.fill_diagonal_(-float("inf"))

        hard_indices = torch.topk(similarity, k, dim=1).indices
        hardest_idx = hard_indices[:, 0]
        hard_negatives = Z_k[hardest_idx]

        return hard_negatives

    def _feature_fusion(self, Z1, Z2, Z3):
        stacked = torch.stack([Z1, Z2, Z3], dim=1)
        fusion_input = torch.cat([Z1, Z2, Z3], dim=1)

        fusion_logits = self.fusion_gate(fusion_input)
        fusion_logits = fusion_logits.view(-1, 3, self.shared_dim)
        fusion_weights = torch.softmax(fusion_logits, dim=1)

        Z4 = torch.sum(stacked * fusion_weights, dim=1)
        return Z4, fusion_weights


    def forward(self, x_c, x_g_unique, H_c, H_g, H_c1, H_c2, view1_array, expr_array):
        gene_unique = self.gene_encoder(x_g_unique, H_g)
        Z_g_raw, gene_broadcast = self.aggregator(view1_array, expr_array, gene_unique)
        Z_g = self.gene_proj(Z_g_raw)

        Z_c_raw = self.cell_encoder(x_c, H_c)
        Z_c1_raw = self.cell_encoder(x_c, H_c1)
        Z_c2_raw = self.cell_encoder(x_c, H_c2)

        Z_c = self.cell_proj(Z_c_raw)
        Z_c1 = self.cell_proj(Z_c1_raw)
        Z_c2 = self.cell_proj(Z_c2_raw)

        hard_neg_0 = self._generate_hard_negatives(Z_c, Z_g)
        hard_neg_1 = self._generate_hard_negatives(Z_c1, Z_g)
        hard_neg_2 = self._generate_hard_negatives(Z_c2, Z_g)

        contrast_loss = (
            self.contrastive_loss(Z_c, Z_g, hard_neg_0)
            + self.contrastive_loss(Z_c1, Z_g, hard_neg_1)
            + self.contrastive_loss(Z_c2, Z_g, hard_neg_2)
        ) / 3.0

        R1 = self.res_fuse(Z_c, Z_g)
        R2 = self.res_fuse(Z_c1, Z_g)
        R3 = self.res_fuse(Z_c2, Z_g)

        Z1, _ = self.attention(R1, Z_g, Z_c)
        Z2, _ = self.attention(R2, Z_g, Z_c1)
        Z3, _ = self.attention(R3, Z_g, Z_c2)

        Z4, fusion_weights = self._feature_fusion(Z1, Z2, Z3)
        Z4 = self.final_projection(Z4)

        alignment_loss = self.alignment_loss(Z4, x_c)

        return {
            "Z4": Z4,
            "Z_c": Z_c,
            "Z_g": Z_g,
            "gene_embedding_unique": gene_unique,
            "gene_embeddings_broadcast": gene_broadcast,
            "align_loss": alignment_loss,
            "contrast_loss": contrast_loss,
            "total_loss": alignment_loss + contrast_loss,
        }

class PhaseMultiviewPretrainer(nn.Module):
    def __init__(self, gene_encoder, cell_encoder, aggregator, xc_dim, output_dim=None, dropout=0.2):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.cell_encoder = cell_encoder
        self.aggregator = aggregator
        self.multiview_encoder = MultiviewEncoder(gene_encoder, cell_encoder, aggregator, xc_dim, dropout=dropout)
        hidden_dim = self.cell_encoder.hidden_dim
        self.output_dim = hidden_dim if output_dim is None else output_dim
        self.cell_projection = nn.Identity() if self.output_dim == hidden_dim else nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x_c, x_g_unique, H_c, H_g, H_c1, H_c2, view1_array, expr_array):
        out = self.multiview_encoder(
            x_c, x_g_unique, H_c, H_g, H_c1, H_c2, view1_array, expr_array
        )
        Z4_raw = out["Z4"]
        Z_c = out["Z_c"]
        Z_g = out["Z_g"]
        gene_unique = out["gene_embedding_unique"]
        gene_broadcast = out["gene_embeddings_broadcast"]
        alignment_loss = out["align_loss"]
        contrast_loss = out["contrast_loss"]
        Z4 = self.cell_projection(Z4_raw)
        total_loss = alignment_loss + contrast_loss
        return {
            "Z4": Z4,
            "Z_c": Z_c,
            "Z_g": Z_g,
            "gene_embedding_unique": gene_unique,
            "gene_embeddings_broadcast": gene_broadcast,
            "align_loss": alignment_loss,
            "contrast_loss": contrast_loss,
            "total_loss": total_loss,
        }

class HGNNAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads = 4, dropout = 0.2, negative_slope=0.2):
        super().__init__()
        heads = max(1, min(int(heads), int(out_channels)))
        while out_channels % heads != 0 and heads > 1:
            heads -= 1
        
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.head_dim = out_channels // self.heads

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, self.heads, 2 * self.head_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, X, H, edge_weights = None):
        n_nodes, n_edges = H.shape

        X = clean_nan(X)
        X = self.W(X).view(n_nodes, self.heads, self.head_dim)

        H_norm = H / (H.sum(dim=0, keepdim=True) + 1e-8)

        if edge_weights is not None:
            edge_weights = torch.clamp(edge_weights, min=0.0, max=100.0)
            H_norm = H_norm * edge_weights.view(1, -1)
        X_edge = torch.einsum("ne,nhd->ehd", H_norm, X)

        alpha = torch.zeros( n_nodes, n_edges, self.heads, device=X.device, dtype=X.dtype)
        node_idx, edge_idx = torch.where(H > 0)

        if len(node_idx) > 0:
            x_i = X[node_idx]
            x_e = X_edge[edge_idx]

            cat_feat = torch.cat([x_i, x_e], dim=-1)
            att_score = (cat_feat * self.att).sum(dim=-1)
            att_score = F.leaky_relu(att_score, negative_slope=self.negative_slope)
            alpha[node_idx, edge_idx] = att_score
        
        alpha = alpha.masked_fill(H.unsqueeze(-1) == 0, float("-inf"))
        alpha = torch.nan_to_num(F.softmax(alpha, dim=1), nan=0.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("neh,ehd->nhd", alpha, X_edge)
        return clean_nan(out.reshape(n_nodes, -1))

class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 2, dropout = 0.1, heads = 4):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            HGNNAttentionConv(in_channels = in_channels, out_channels = hidden_channels, heads = heads, dropout = dropout)
        )

        self.norms.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                HGNNAttentionConv(in_channels = hidden_channels, out_channels = hidden_channels, heads = heads, dropout = dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        if num_layers > 1:
            self.convs.append(
                HGNNAttentionConv(in_channels = hidden_channels, out_channels = out_channels, heads = heads, dropout = dropout)
            )
            self.norms.append(nn.LayerNorm(out_channels))

    def forward(self, X, H, edge_weights = None):
        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            X_new = norm(conv(X, H, edge_weights))

            if idx < len(self.convs) - 1:
                X_new = F.dropout(
                    F.gelu(X_new),
                    p=self.dropout,
                    training=self.training
                )

            X = X + X_new if X.shape == X_new.shape else X_new
        return clean_nan(X)

class HeterogeneousHGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 2, dropout = 0.1, heads = 4):
        super().__init__()
        self.gene_proj = nn.Linear(in_channels, hidden_channels)
        self.sample_proj = nn.Linear(in_channels, hidden_channels)

        self.hgnn = HGNN(in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            heads=4
        )

        self.gene_out = nn.Linear(out_channels, out_channels)
        self.sample_out = nn.Linear(out_channels, out_channels)

    def _check_masks(self, n_nodes, gene_mask, sample_mask):
        masks = [m for m in [gene_mask, sample_mask] if m is not None]
        if len(masks) == 0:
            return
        coverage = torch.zeros(n_nodes, device=masks[0].device, dtype=torch.long)
        for mask in masks:
            coverage += mask.long()
        if torch.any(coverage > 1):
            raise ValueError(
                "gene_mask and sample_mask must be mutually exclusive."
            )
        if torch.any(coverage == 0):
            raise ValueError(
                "Some nodes are not covered by any mask. "
                "Please make sure every node belongs to gene or sample."
            )
    
    def forward(self, X, H, edge_weights, gene_mask, sample_mask):
        n_nodes = X.shape[0]
        self._check_masks(n_nodes = n_nodes, gene_mask = gene_mask, sample_mask = sample_mask)
        X_proj = torch.zeros(n_nodes, self.gene_proj.out_features, device = X.device, dtype = X.dtype)

        if gene_mask is not None:
             X_proj[gene_mask] = self.gene_proj(X[gene_mask])
        if sample_mask is not None:
             X_proj[sample_mask] = self.sample_proj(X[sample_mask])
        Z = self.hgnn(X_proj, H, edge_weights)
        out = torch.zeros_like(Z)

        if gene_mask is not None:
            out[gene_mask] = self.gene_out(Z[gene_mask])
        if sample_mask is not None:
            out[sample_mask] = self.sample_out(Z[sample_mask])
        if gene_mask is None and sample_mask is None:
            out = Z
        return clean_nan(out)

class VAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers = 2, dropout = 0.1, logvar_min = -6.0, logvar_max = 2.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)

        for idx in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], hidden_dim))

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim, num_layers = 2, dropout = 0.1):
        super().__init__()
        layers = []
        dims = [latent_dim] + [hidden_dim] * num_layers

        for idx in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[idx], dims[idx + 1]),
                nn.LayerNorm(dims[idx + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], out_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers = 2, dropout = 0.1, beta = 1.0, logvar_min = -6.0, logvar_max = 2.0):
        super().__init__()
        self.encoder = VAEEncoder(in_dim, hidden_dim, latent_dim, num_layers, dropout, logvar_min, logvar_max)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, in_dim, num_layers, dropout)
        self.beta = beta
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

class HGNN_VAE_Phase_Model(nn.Module):
    def __init__(
        self,
        num_cells: int,
        num_genes: int,
        config: PhaseTrainingConfig,
        edge_type_to_id: Dict[str, int],
        initial_node_features: torch.Tensor,
        cell_gene_pairs: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.edge_type_to_id = edge_type_to_id
        self.num_edge_types = len(edge_type_to_id)
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.cell_gene_pairs = cell_gene_pairs

        self.register_buffer("initial_node_features", initial_node_features)

        if config.learn_input_residual:
            self.input_residual = nn.Parameter(torch.zeros_like(initial_node_features))
        else:
            self.input_residual = None

        self.input_layer_norm = nn.LayerNorm(config.feature_dim)

        def inverse_softplus(x):
            return x - torch.log(1 - torch.exp(-x)) if x > 0 else torch.log(torch.exp(x) - 1)
        
        edge_type_to_id_inv = {v: k for k, v in edge_type_to_id.items()}
        raw_weights = [
            inverse_softplus(torch.tensor(config.edge_type_init_weights.get(edge_type_to_id_inv[idx], 1.0)))
            for idx in range(len(edge_type_to_id))
        ]
        self.edge_type_weights = nn.Parameter(torch.tensor(raw_weights, dtype=torch.float32))

        self.hgnn = HeterogeneousHGNN(
            in_channels=config.feature_dim,
            hidden_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            num_layers=2,
            dropout=config.hgnn_dropout_rate
        )

        self.vae = VAE(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=2,
            dropout=config.hgnn_dropout_rate,
            beta=1.0,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max
        )

        self.gate_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        nn.init.normal_(self.gate_head[-1].weight, mean=0.0, std=0.5)
        nn.init.uniform_(self.gate_head[-1].bias, -2.0, 2.0)

        self.cell_phase_fuser = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        if config.use_dec and config.num_clusters is not None:
            self.cluster_centers = nn.Parameter(
                torch.randn(config.num_clusters, config.latent_dim)
            )
        else:
            self.cluster_centers = None

    def _compute_edge_weights(self, hyperedge_type_ids):
        base_weights = self.edge_type_weights[hyperedge_type_ids]
        weights = F.softplus(base_weights)
        return torch.clamp(weights, min=0.05, max=5.0)

    def _build_cell_gene_pairs_map(self, gene_mask, sample_mask):
        if self.cell_gene_pairs is None:
            return None, None

        cell_to_genes = {}

        for c_local, g_local in self.cell_gene_pairs:
            c_local = int(c_local)
            g_local = int(g_local)

            if 0 <= c_local < self.num_cells and 0 <= g_local < self.num_genes:
                cell_to_genes.setdefault(c_local, []).append(g_local)

        if not cell_to_genes:
            return None, None

        gene_indices = torch.where(gene_mask)[0]
        return cell_to_genes, gene_indices

    def _build_phase_aware_cell_embedding(
        self, mu_cells, mu_genes, gate_g, gene_mask, sample_mask, cell_to_genes
    ):
        n_cells = mu_cells.shape[0]
        n_genes = mu_genes.shape[0]

        phase_a_cells = torch.zeros_like(mu_cells)
        phase_b_cells = torch.zeros_like(mu_cells)

        if cell_to_genes is None:
            cell_embed = F.normalize(mu_cells, p=2, dim=1)
            return cell_embed

        for c_idx, gene_list in cell_to_genes.items():
            if len(gene_list) == 0:
                continue

            gene_indices = torch.tensor(gene_list, device=mu_genes.device)
            gene_z = mu_genes[gene_indices]
            gene_gate = gate_g[gene_indices]

            weight_a = (1 - gene_gate)
            weight_b = gene_gate

            weight_a = weight_a / (weight_a.sum() + 1e-8)
            weight_b = weight_b / (weight_b.sum() + 1e-8)

            phase_a_cells[c_idx] = (gene_z * weight_a.unsqueeze(1)).sum(dim=0)
            phase_b_cells[c_idx] = (gene_z * weight_b.unsqueeze(1)).sum(dim=0)

        phase_diff = phase_a_cells - phase_b_cells

        cell_phase_input = torch.cat([
            mu_cells,
            phase_a_cells,
            phase_b_cells,
            phase_diff
        ], dim=1)

        cell_embed = self.cell_phase_fuser(cell_phase_input)
        cell_embed = F.normalize(cell_embed, p=2, dim=1)

        return cell_embed

    def forward(self, H, hyperedge_type_ids, gene_mask, sample_mask, base_edge_weights=None):
        x0 = self.initial_node_features

        if self.input_residual is not None:
            x = x0 + self.config.input_residual_scale * self.input_residual
        else:
            x = x0

        x = self.input_layer_norm(x)

        edge_weights = self._compute_edge_weights(hyperedge_type_ids)
        if base_edge_weights is not None:
            edge_weights = edge_weights * torch.clamp(base_edge_weights, min=0.0, max=5.0)

        h = self.hgnn(x, H, edge_weights, gene_mask, sample_mask)
        h = clean_nan(h)

        recon, mu, logvar, z = self.vae(h)

        logvar = torch.clamp(logvar, min=self.config.logvar_min, max=self.config.logvar_max)

        mu = clean_nan(mu)
        logvar = clean_nan(logvar)
        recon = clean_nan(recon)
        z = clean_nan(z)

        gene_indices = torch.where(gene_mask)[0]
        sample_indices = torch.where(sample_mask)[0]

        mu_genes = mu[gene_indices]
        mu_cells = mu[sample_indices]

        gate_logits = self.gate_head(mu_genes)
        gate_g = torch.sigmoid(gate_logits).squeeze(-1)

        cell_to_genes, _ = self._build_cell_gene_pairs_map(gene_mask, sample_mask)
        cell_embed = self._build_phase_aware_cell_embedding(
            mu_cells, mu_genes, gate_g, gene_mask, sample_mask, cell_to_genes
        )

        if self.cluster_centers is not None:
            q = self._compute_soft_assignments(cell_embed)
        else:
            q = None

        return {
            "x0": x0,
            "x": x,
            "h": h,
            "z": z,
            "mu": mu,
            "mu_cells": mu_cells,
            "mu_genes": mu_genes,
            "logvar": logvar,
            "recon": recon,
            "gate_g": gate_g,
            "gate_logits": gate_logits,
            "edge_weights": edge_weights,
            "gene_indices": gene_indices,
            "sample_indices": sample_indices,
            "cell_embed": cell_embed,
            "q": q,
        }

    def _compute_soft_assignments(self, cell_embed):
        if self.cluster_centers is None:
            return None

        cell_embed_norm = F.normalize(cell_embed, p=2, dim=1)
        centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)

        similarity = torch.mm(cell_embed_norm, centers_norm.T)
        q = 1.0 / (1.0 + torch.pow(torch.cdist(cell_embed_norm, centers_norm, p=2), 2))
        q = q / q.sum(dim=1, keepdim=True)
        return q

__all__ = [name for name in globals() if not name.startswith("__")]
