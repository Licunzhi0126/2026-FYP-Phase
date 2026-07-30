"""整边级因果剪枝（仿 CASCAT 的 CMI 思路，用在两层级细胞通道上）。

CASCAT 用条件互信息 CMI(A;B|C) 判断簇级节点对是否在条件节点下独立——
独立(CMI 低)的连接被当作"空间邻近但因果独立"的冗余边剪掉。这里把同一思路
搬到**超边**上：每条超边算一个"因果非冗余分"，分低=成员在全局背景条件下相互
独立=冗余边 → 训练中周期性剪掉，并重建传播算子。

为了能在训练循环里每 K 个 epoch 重算（CASCAT 的 KDE-CMI 太慢），这里用
**高斯/偏相关闭式 CMI 估计**（jointly-Gaussian 下 CMI = -1/2 log(1-ρ²)，ρ 为偏相关）：
把 d 维节点嵌入当"样本"（同 CASCAT 把基因维当样本），度量成员相对边质心、
在全局背景条件下的偏相关。闭式、无需采样、对上万条边可秒级重算。
"""
from __future__ import annotations

import numpy as np


def _zscore_rows(M: np.ndarray) -> np.ndarray:
    """按行（节点）去均值 + 单位方差，使行点积/d = Pearson 相关。"""
    M = M - M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return M / s


def gaussian_cmi_edge_scores(
    H_csr,
    node_emb: np.ndarray,
    background: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """每条超边的因果非冗余分（越高越该保留）。

    H_csr     : scipy.sparse (n_nodes × n_edges)，超边关联矩阵。
    node_emb  : (n_nodes × d) 当前节点嵌入（如 cell_h），把 d 维当作样本。
    background: 条件变量 z_bg（默认 = 全节点嵌入均值，类比 CASCAT 的 root/条件节点）。

    对超边 e（成员相对质心 c_e）：
        ρ_ic·bg = (ρ_ic − ρ_i,bg·ρ_c,bg) / sqrt((1−ρ_i,bg²)(1−ρ_c,bg²))
        CMI_i   = −½ log(1 − ρ_ic·bg²)
    边分 = mean_i CMI_i。分低 → 成员在背景条件下与边质心条件独立 → 冗余边。
    """
    H = H_csr.tocsc()
    n_nodes, n_edges = H.shape
    Z = np.asarray(node_emb, dtype=np.float64)
    d = Z.shape[1]
    Zc = _zscore_rows(Z)  # 行标准化（节点维度）

    bg = Z.mean(axis=0) if background is None else np.asarray(background, dtype=np.float64)
    bg = bg - bg.mean()
    bn = bg.std()
    bg = bg / (bn if bn > 1e-8 else 1.0)

    rho_node_bg = (Zc @ bg) / d  # 每个节点与背景的相关 (n_nodes,)

    scores = np.zeros(n_edges, dtype=np.float64)
    indptr, indices = H.indptr, H.indices
    for e in range(n_edges):
        members = indices[indptr[e]:indptr[e + 1]]
        if members.size < 2:
            continue
        Zm = Zc[members]               # (m × d)，已行标准化
        c = Zm.mean(axis=0)            # 边质心
        c = c - c.mean()
        cs = c.std()
        c = c / (cs if cs > 1e-8 else 1.0)
        rho_ic = (Zm @ c) / d          # 成员-质心相关 (m,)
        rho_cbg = float((c @ bg) / d)  # 质心-背景相关
        rho_ibg = rho_node_bg[members]  # 成员-背景相关 (m,)
        denom = np.sqrt(np.clip((1.0 - rho_ibg ** 2) * (1.0 - rho_cbg ** 2), eps, None))
        part = (rho_ic - rho_ibg * rho_cbg) / denom
        part = np.clip(part, -0.999, 0.999)
        cmi = -0.5 * np.log(1.0 - part ** 2 + eps)
        scores[e] = float(np.mean(cmi))
    return scores


def prune_incidence(
    H_csr,
    W,
    names,
    types,
    node_emb: np.ndarray,
    prunable_mask: np.ndarray | None = None,
    prune_frac: float = 0.3,
    min_keep_frac: float = 0.5,
):
    """构建期静态剪枝：用原始特征对关联矩阵 H 打 CMI 分，丢掉冗余边的列。

    node_emb : (n_nodes × d) 原始节点特征（细胞通道用 expr=细胞×基因，基因通道用
               expr.T=基因×细胞），把特征维当作样本——最贴 CASCAT 用 profile 当样本。
    返回剪枝后的 (H_csr, W, names, types, keep_mask)。
    """
    import numpy as _np

    scores = gaussian_cmi_edge_scores(H_csr, node_emb)
    keep = causal_keep_mask(scores, prunable_mask=prunable_mask,
                            prune_frac=prune_frac, min_keep_frac=min_keep_frac)
    keep_idx = _np.where(keep)[0]
    H2 = H_csr.tocsc()[:, keep_idx].tocsr()
    W2 = _np.asarray(W)[keep_idx]
    names2 = [names[i] for i in keep_idx]
    types2 = [types[i] for i in keep_idx]
    return H2, W2, names2, types2, keep


def causal_keep_mask(
    scores: np.ndarray,
    prunable_mask: np.ndarray | None = None,
    prune_frac: float = 0.3,
    min_keep_frac: float = 0.5,
) -> np.ndarray:
    """把可剪边里因果分最低的 prune_frac 剪掉，返回布尔保留掩码。

    prunable_mask : True=该边可被剪（默认全部可剪）；不可剪边一律保留。
    min_keep_frac : 可剪边至少保留这一比例，避免整层塌掉（cells 全孤立）。
    """
    n = len(scores)
    keep = np.ones(n, dtype=bool)
    if prunable_mask is None:
        prunable_mask = np.ones(n, dtype=bool)
    idx = np.where(prunable_mask)[0]
    if idx.size == 0:
        return keep
    n_prune = int(np.floor(idx.size * prune_frac))
    n_prune = min(n_prune, int(np.floor(idx.size * (1.0 - min_keep_frac))))
    if n_prune <= 0:
        return keep
    order = np.argsort(scores[idx])  # 升序：最低分先剪
    keep[idx[order[:n_prune]]] = False
    return keep
