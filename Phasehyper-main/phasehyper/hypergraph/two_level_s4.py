"""S4 非 DL 阶段：两层级超图的解析传播 + 两通道 concat + 谱聚类评估。

无任何可学参数（不训练）。两通道都用超图归一化传播（Zhou 2007）把原始表达
在各自的图上平滑，再 PCA→128、z-score，concat 成 256 维细胞 embedding，KMeans 评估。

- 基因通道：在 H_gene（pathway/ppi/poswin）上沿基因轴平滑 → cell×128。
- 细胞通道：在 H_cell（rna_knn/adt_knn）上沿细胞轴平滑 → cell×128。
- 耦合 = concat（README §5）。M4/M5 消融见 demo_two_level_s4.py。

全程 float64（清洗后表达虽已 z-score，但传播/PCA 仍用 float64 防数值问题，
与 baseline_ari 的 float64 教训一致）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def zhou_propagate(H, W, X, steps: int = 2, alpha: float = 0.5) -> np.ndarray:
    """超图归一化传播 Â = Dv^{-1/2} H W De^{-1} Hᵀ Dv^{-1/2}（Zhou 2007），惰性混合。

    H : scipy.sparse (n_nodes × n_edges)；W : (n_edges,)；X : (n_nodes × d)。
    每步 X ← (1-alpha)·X + alpha·(Â X)。De^{-1} 自动拉平大小超边；
    孤立节点（度 0）经惰性混合保留自身信号，不被清零。返回 (n_nodes × d) float64。
    """
    H = H.tocsr().astype(np.float64)
    n_edges = H.shape[1]
    Xc = np.nan_to_num(np.asarray(X, dtype=np.float64)).copy()
    if n_edges == 0:
        return Xc

    w = np.asarray(W, dtype=np.float64).ravel()
    dv = np.asarray(H.multiply(w[np.newaxis, :]).sum(axis=1)).ravel()  # 节点度 Σ_e w_e H[v,e]
    de = np.asarray(H.sum(axis=0)).ravel()                            # 超边度 Σ_v H[v,e]
    dv_inv_sqrt = np.where(dv > 0, 1.0 / np.sqrt(dv), 0.0)
    de_inv = np.where(de > 0, 1.0 / de, 0.0)
    edge_scale = de_inv * w  # 把 De^{-1} 和 W 合并成每条边一个系数

    Ht = H.T.tocsr()
    for _ in range(max(1, int(steps))):
        y = Xc * dv_inv_sqrt[:, None]
        y = Ht @ y                      # (n_edges × d)
        y = y * edge_scale[:, None]
        y = H @ y                       # (n_nodes × d)
        y = y * dv_inv_sqrt[:, None]
        Xc = (1.0 - alpha) * Xc + alpha * y
    return Xc


def reduce_standardize(X: np.ndarray, out_dim: int) -> np.ndarray:
    """传播后的表示 → PCA(out_dim) → 列 z-score。PCA/标准化均无监督（不算训练）。"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    k = max(1, min(int(out_dim), X.shape[0] - 1, X.shape[1]))
    Z = PCA(n_components=k, random_state=42).fit_transform(X)
    Z = StandardScaler().fit_transform(Z)
    return Z.astype(np.float32)


def cell_channel(
    built: Dict,
    *,
    edge_mask: Optional[np.ndarray] = None,
    out_dim: Optional[int] = None,
    steps: Optional[int] = None,
) -> np.ndarray:
    """细胞通道：表达沿 H_cell（细胞-细胞）传播 → cell×out_dim。

    edge_mask（按 cell 边布尔/索引子集）用于消融（如只留 rna_knn）。
    """
    out_dim = int(out_dim or built["cell_channel_out"])
    steps = int(steps or built["smooth_steps"])
    H_cell, W_cell = built["H_cell"], built["W_cell"]
    if edge_mask is not None:
        cols = np.where(np.asarray(edge_mask))[0] if np.asarray(edge_mask).dtype == bool else np.asarray(edge_mask)
        H_cell = H_cell[:, cols]
        W_cell = np.asarray(W_cell)[cols]
    smoothed = zhou_propagate(H_cell, W_cell, built["expr"], steps=steps)  # (n_cells × n_genes)
    return reduce_standardize(smoothed, out_dim)


def gene_channel(
    built: Dict,
    *,
    expr: Optional[np.ndarray] = None,
    out_dim: Optional[int] = None,
    steps: Optional[int] = None,
) -> np.ndarray:
    """基因通道：表达沿 H_gene（基因-基因）传播 + 按细胞 pool → cell×out_dim。

    沿基因轴平滑（节点=基因，特征=该基因在各细胞的表达），再转回细胞×基因，PCA→out_dim。
    expr 可传桥接增广后的表达（M5）。
    """
    out_dim = int(out_dim or built["gene_channel_out"])
    steps = int(steps or built["smooth_steps"])
    E = built["expr"] if expr is None else expr
    E = np.nan_to_num(np.asarray(E, dtype=np.float64))
    # 基因为节点：X = Eᵀ (n_genes × n_cells)；Â_gene 对称 → 平滑后转回 (n_cells × n_genes)
    smoothed = zhou_propagate(built["H_gene"], built["W_gene"], E.T, steps=steps).T
    return reduce_standardize(smoothed, out_dim)


def bridge_augment_expr(built: Dict, beta: float = 0.5) -> np.ndarray:
    """M5：把 ADT 丰度（z-score）按中心法则桥接加到其编码基因的表达上（细胞内第二读出）。

    返回增广后的 expr（n_cells × n_genes），供 gene_channel(expr=...) 使用。
    """
    from sklearn.preprocessing import StandardScaler

    expr = np.nan_to_num(np.asarray(built["expr"], dtype=np.float64)).copy()
    protein = built.get("protein")
    bridge = built.get("bridge") or {}
    if protein is None or not bridge:
        return expr
    pz = StandardScaler().fit_transform(np.nan_to_num(np.asarray(protein, dtype=np.float64)))
    pidx = {n: i for i, n in enumerate(built["protein_names"])}
    gidx = {g: i for i, g in enumerate(built["genes"])}
    for adt, gene_list in bridge.items():
        if adt not in pidx:
            continue
        col = pz[:, pidx[adt]]
        for g in gene_list:
            if g in gidx:
                expr[:, gidx[g]] += beta * col
    return expr


def concat(*embeddings: np.ndarray) -> np.ndarray:
    """拼接多个 cell×d embedding → cell×Σd（耦合 = concat）。"""
    return np.concatenate([np.asarray(e, dtype=np.float32) for e in embeddings], axis=1)


def evaluate(emb: np.ndarray, true_labels, k: Optional[int] = None, *, n_init: int = 10) -> Dict:
    """KMeans(k=类数) 聚类 → ARI/NMI/FMI vs 真值。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        fowlkes_mallows_score,
        normalized_mutual_info_score,
    )

    y = np.asarray(true_labels)
    k = int(k or len(np.unique(y)))
    X = np.nan_to_num(np.asarray(emb, dtype=np.float64))
    pred = KMeans(n_clusters=k, random_state=42, n_init=n_init).fit_predict(X)
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "fmi": float(fowlkes_mallows_score(y, pred)),
        "k": k,
        "pred_clusters": int(np.unique(pred).size),
    }
