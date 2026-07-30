"""两层级 HGNN-VAE 的可学习超图卷积通道（稀疏，对单通道单图固定 H）。

与 `hypergraph/two_level_s4.py` 的解析传播同一套归一化算子（Zhou 2007 / Feng 2019），
区别是每层多了一个可学习线性 Θ + LayerNorm + GELU + 残差，端到端训练。

    X' = Θ( Dv^{-1/2} H (De^{-1} W) Hᵀ Dv^{-1/2} X )

H / Hᵀ 全程以 torch 稀疏 COO 持有，**不物化 (n_nodes×n_edges) 稠密**——所以
7932 细胞 × 上万条 cell 边、2507 基因 × 数千 gene 边都跑得动（旧扁平版的稠密
注意力 alpha 会爆显存，这里换掉）。每个通道一张固定图，传播算子在 __init__ 预算好。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phasehyper.utils import clean_nan


def scipy_to_torch_sparse(mat, dtype=torch.float32) -> torch.Tensor:
    """scipy.sparse → 已 coalesce 的 torch 稀疏 COO（随模型 .to(device) 迁移）。"""
    coo = mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32)).to(dtype)
    return torch.sparse_coo_tensor(indices, values, size=tuple(coo.shape)).coalesce()


class HypergraphChannel(nn.Module):
    """固定稀疏 H 上的多层可学习 HGNN 卷积（一条通道 = 一张图）。

    H_scipy : scipy.sparse (n_nodes × n_edges)；W : (n_edges,) 超边权重。
    传播算子（dv_inv_sqrt / edge_scale / H / Hᵀ）按 H 预算并存为 buffer；
    层间用残差，输入维 != 隐藏维的第一层不残差。
    """

    def __init__(
        self,
        H_scipy,
        W,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        causal_prior=None,
        learnable_gate: bool = False,
    ):
        super().__init__()
        H = H_scipy.tocsr().astype(np.float64)
        n_nodes, n_edges = H.shape
        self.n_nodes, self.n_edges = int(n_nodes), int(n_edges)
        self.dropout = dropout

        w = np.asarray(W, dtype=np.float64).ravel()
        if w.size != n_edges:  # 容错：权重缺省一律 1
            w = np.ones(n_edges, dtype=np.float64)
        # 节点度用 |w|：支持带符号边权（GRN 抑制边 w<0）而不让 Dv 变负/开根号出 NaN；
        # 对全非负权(现状)abs 是 no-op。edge_scale 保留带符号 w → 抑制边传负向消息(反平滑)。
        dv = np.asarray(H.multiply(np.abs(w)[np.newaxis, :]).sum(axis=1)).ravel()  # 节点度 Σ_e |w_e| H[v,e]
        de = np.asarray(H.sum(axis=0)).ravel()                            # 超边度 Σ_v H[v,e]
        dv_inv_sqrt = np.where(dv > 0, 1.0 / np.sqrt(dv), 0.0)
        edge_scale = np.where(de > 0, 1.0 / de, 0.0) * w                  # De^{-1} 与 W(带符号) 合并

        # 训练中因果剪枝用：留存原始 H / De / 满权重，便于每次从全集重算（非累积）
        self._H_csr = H
        self._de = de
        self._w_full = w

        self.register_buffer("dv_inv_sqrt", torch.tensor(dv_inv_sqrt, dtype=torch.float32))
        self.register_buffer("edge_scale", torch.tensor(edge_scale, dtype=torch.float32))
        # 稀疏 H / Hᵀ 作为 buffer，随 model.to(device) 迁移
        H_sp = scipy_to_torch_sparse(H)
        self.register_buffer("H_sp", H_sp)
        self.register_buffer("Ht_sp", H_sp.transpose(0, 1).coalesce())

        # ── 可学边门（因果先验 c_e 当锚）：g_e=σ(edge_logit)，初始化到 c_e、正则拉回 c_e ──
        #   归一化算子(dv/de)仍按预处理的因果软权固定；门只对每条边的消息做可微调制，
        #   既稳（不破坏超图 Laplacian 归一）又让因果性成为网络可学的一部分。
        self.learnable_gate = bool(learnable_gate) and n_edges > 0
        if self.learnable_gate:
            prior = np.ones(n_edges, dtype=np.float64) if causal_prior is None \
                else np.asarray(causal_prior, dtype=np.float64).ravel()
            if prior.size != n_edges:
                prior = np.ones(n_edges, dtype=np.float64)
            self.register_buffer("causal_prior", torch.tensor(prior, dtype=torch.float32))
            p = np.clip(prior, 0.02, 0.98)
            self.edge_logit = nn.Parameter(torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32))

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.thetas = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dims[i + 1]) for i in range(num_layers)]
        )

    def _propagate(self, X: torch.Tensor) -> torch.Tensor:
        """Â X = Dv^{-1/2} H (De^{-1}W)·gate· Hᵀ Dv^{-1/2} X，全程稀疏 matmul。"""
        if self.n_edges == 0:
            return X
        y = X * self.dv_inv_sqrt[:, None]
        y = torch.sparse.mm(self.Ht_sp, y)        # (n_edges × d)
        y = y * self.edge_scale[:, None]
        if self.learnable_gate:                   # 可学因果门，逐边调制消息（可微）
            y = y * torch.sigmoid(self.edge_logit)[:, None]
        y = torch.sparse.mm(self.H_sp, y)         # (n_nodes × d)
        y = y * self.dv_inv_sqrt[:, None]
        return y

    def gate_reg(self) -> torch.Tensor:
        """边门向因果先验 c_e 的正则项（让因果当锚，数据只做有限自适应）。"""
        if not self.learnable_gate:
            return torch.zeros((), device=self.dv_inv_sqrt.device)
        return (torch.sigmoid(self.edge_logit) - self.causal_prior).pow(2).mean()

    @torch.no_grad()
    def set_edge_weights(self, w_new) -> None:
        """按新边权重建传播算子（H/De 不变，只改 Dv 与 De^{-1}W）。

        因果剪枝把冗余边权置 0 即可：edge_scale=0 该边不再传播，Dv 自动排除它。
        每次从满权重 _w_full 派生（如 w_new=keep_mask·_w_full），保证是从全集重算
        而非在已剪结果上累积。也可传 [0,1] 软权做软剪枝。
        """
        w = np.asarray(w_new, dtype=np.float64).ravel()
        if w.size != self.n_edges:
            raise ValueError(f"set_edge_weights 期望 {self.n_edges} 条边权，收到 {w.size}")
        dv = np.asarray(self._H_csr.multiply(np.abs(w)[np.newaxis, :]).sum(axis=1)).ravel()
        dv_inv_sqrt = np.where(dv > 0, 1.0 / np.sqrt(dv), 0.0)
        edge_scale = np.where(self._de > 0, 1.0 / self._de, 0.0) * w
        device = self.dv_inv_sqrt.device
        self.dv_inv_sqrt.copy_(torch.tensor(dv_inv_sqrt, dtype=torch.float32, device=device))
        self.edge_scale.copy_(torch.tensor(edge_scale, dtype=torch.float32, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = clean_nan(X)
        for idx, (theta, norm) in enumerate(zip(self.thetas, self.norms)):
            y = self._propagate(X)
            y = norm(theta(y))
            if idx < len(self.thetas) - 1:
                y = F.dropout(F.gelu(y), p=self.dropout, training=self.training)
            X = X + y if X.shape == y.shape else y
        return clean_nan(X)
