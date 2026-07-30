"""PC 式（约束/条件独立检验）超边剪枝——真正的因果发现那一步。

和 causal_prune.py 里那个高斯偏相关"紧凑度"分数不同，这里照搬 PC 算法删边那一步：
对边内成员对 (i,j)，在其它成员里搜**分离集** S，若 ∃S 使

    CMI(i ; j | S) < ci_threshold        （条件独立）

则 (i,j) 的依赖被 S 解释掉、是冗余连接（PC 的 edge-removal）。整条超边按"不可分离
成员对占比" direct_fraction 打分：纯冗余边(全被分离)分→0，优先剪。

三处对齐 CASCAT 的因果本意（也正是之前高斯版缺的）：
  1. **非参 CMI**：分位分箱 + Miller-Madow 去偏估计互信息，抓非线性（高斯偏相关只抓线性）；
  2. **真分离集**：条件在边内其它成员 profile 上（逐阶 order-0/1/2 搜），不是全局均值；
  3. **CI 检验逻辑**：被某 S d-分离就删边，而非看"边内紧不紧凑"。

节点 profile：细胞通道用 expr（细胞×基因，基因维当样本），基因通道用 expr.T
（基因×细胞，细胞维当样本）——与 CASCAT 用簇表达 profile、把基因维当样本一脉相承。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


# ── 非参 (条件)互信息：分位分箱 + Miller-Madow 去偏（单位 nats）──────────────────

def _quantile_labels(v: np.ndarray, n_bins: int) -> np.ndarray:
    """分位分箱成 0..n_bins-1 整数标签（对偏态稳健）。"""
    v = np.asarray(v, dtype=np.float64)
    if n_bins <= 1:
        return np.zeros(v.shape[0], dtype=np.int64)
    edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(v, edges).astype(np.int64)


def _encode(labels_list) -> np.ndarray:
    """把若干整数标签数组编码成单一联合标签（单射）。"""
    code = labels_list[0].astype(np.int64).copy()
    for l in labels_list[1:]:
        K = int(l.max()) + 1
        code = code * K + l.astype(np.int64)
    return code


def _entropy(code: np.ndarray, n: int, mm: bool = True) -> float:
    """plugin 熵 + Miller-Madow 修正 (m-1)/(2n)，nats。"""
    c = np.bincount(code)
    nz = c[c > 0]
    p = nz / n
    H = float(-(p * np.log(p)).sum())
    if mm:
        H += (nz.shape[0] - 1) / (2.0 * n)
    return H


def cmi_binning(x, y, z_list=None, n_bins: int = 6, mm: bool = True) -> float:
    """CMI(x;y|Z)（Z=z_list 各变量），z_list=None/空 时退化为 MI(x;y)。

    MI  = H(x)+H(y)-H(x,y)
    CMI = H(x,z)+H(y,z)-H(z)-H(x,y,z)
    """
    n = len(x)
    bx = _quantile_labels(x, n_bins)
    by = _quantile_labels(y, n_bins)
    if not z_list:
        H_x = _entropy(bx, n, mm)
        H_y = _entropy(by, n, mm)
        H_xy = _entropy(_encode([bx, by]), n, mm)
        return max(0.0, H_x + H_y - H_xy)
    bz = [_quantile_labels(z, n_bins) for z in z_list]
    code_z = _encode(bz)
    H_xz = _entropy(_encode([bx] + bz), n, mm)
    H_yz = _entropy(_encode([by] + bz), n, mm)
    H_z = _entropy(code_z, n, mm)
    H_xyz = _entropy(_encode([bx, by] + bz), n, mm)
    return max(0.0, H_xz + H_yz - H_z - H_xyz)


# ── G 检验：2N·CMI ~ χ²_df（自由度+样本量自校准的条件独立检验）────────────────────

def cmi_gtest(x, y, z_list, n_bins: int, min_per_cell: float = 5.0):
    """条件 G 检验。H0(X⊥Y|Z) 下 G=2N·CMI_plugin ~ χ²_df，df=(rx-1)(ry-1)·rz。

    返回 (p_value, cmi_nats, df)；rx/ry/rz 用实际占用的箱/层数（自然处理零膨胀退化）。
    **Cochran 规则**：列联表格子数 rx·ry·rz 过大、平均每格样本 < min_per_cell 时
    χ² 近似失效 → 返回 p=NaN（不可检验），上层据此保守处理（不判独立、不剪）。
    """
    from scipy.stats import chi2
    n = len(x)
    bx = _quantile_labels(x, n_bins)
    by = _quantile_labels(y, n_bins)
    rx = int(np.unique(bx).size)
    ry = int(np.unique(by).size)
    if rx < 2 or ry < 2:                       # 某变量退化(如全零)→无信息
        return 1.0, 0.0, 0
    if not z_list:
        rz, cells = 1, rx * ry
        df = (rx - 1) * (ry - 1)
    else:
        code_z = _encode([_quantile_labels(z, n_bins) for z in z_list])
        rz = int(np.unique(code_z).size)
        cells = rx * ry * rz
        df = (rx - 1) * (ry - 1) * rz
    if n < min_per_cell * cells:               # Cochran：欠采样 → 该检验不可信
        return float("nan"), float("nan"), df
    cmi = cmi_binning(x, y, z_list, n_bins, mm=False)  # G 统计量用 plug-in CMI
    if df < 1:
        return 1.0, cmi, 0
    return float(chi2.sf(2.0 * n * cmi, df)), cmi, df


def _cond_indep(x, y, Z, *, ci_method, ci_floor, ci_alpha, n_bins, mm):
    """是否条件独立（可分离）。

    gtest（合理版）：真依赖 ⟺ 显著(p<α) 且 效应够大(CMI≥floor)；否则判独立。
                   —— 同时治 df 未校准 和 大 N 全显著两个病。
    threshold（旧）：CMI < floor 即判独立。
    """
    if ci_method == "gtest":
        p, cmi, _ = cmi_gtest(x, y, Z, n_bins)
        if not np.isfinite(p):           # Cochran 欠采样 → 不可检验 → 保守:不判独立(不剪)
            return False
        dependent = (p < ci_alpha) and (cmi >= ci_floor)
        return not dependent
    return cmi_binning(x, y, Z, n_bins, mm) < ci_floor


def _is_separable(pi, pj, cond_profiles, ci_floor, max_order, n_bins, mm,
                  *, ci_method="gtest", ci_alpha=0.05):
    """成员对 (i,j) 是否被某分离集 S（⊆cond_profiles）条件独立。

    order 0：边际就独立。order r：候选条件取 r 个组合，任一使其条件独立 → 可分离。
    判定走 _cond_indep（默认 G 检验 + 效应门槛）。返回 (separable, sepset_order)。
    """
    if _cond_indep(pi, pj, None, ci_method=ci_method, ci_floor=ci_floor,
                   ci_alpha=ci_alpha, n_bins=n_bins, mm=mm):
        return True, 0
    n_cond = len(cond_profiles)
    for order in range(1, max_order + 1):
        if n_cond < order:
            break
        for combo in combinations(range(n_cond), order):
            S = [cond_profiles[c] for c in combo]
            if _cond_indep(pi, pj, S, ci_method=ci_method, ci_floor=ci_floor,
                           ci_alpha=ci_alpha, n_bins=n_bins, mm=mm):
                return True, order
    return False, -1


def pc_edge_scores(
    H_csr,
    node_profiles: np.ndarray,
    prunable_mask: np.ndarray | None = None,
    *,
    ci_threshold: float = 0.02,
    max_order: int = 1,
    max_pairs: int = 6,
    max_cond: int = 4,
    max_members: int = 12,
    n_bins: int = 6,
    mm: bool = True,
    ci_method: str = "gtest",
    ci_alpha: float = 0.05,
):
    """每条(可剪)超边的 direct_fraction（不可分离成员对占比，越高越该保留）。

    H_csr        : scipy.sparse (n_nodes × n_edges)。
    node_profiles: (n_nodes × n_samples) 每个节点一条 profile（另一类维度当样本）。
    返回 (scores[n_edges], info)：不可剪边分=1.0（恒保留）；可剪边分∈[0,1]。
    """
    H = H_csr.tocsc()
    n_nodes, n_edges = H.shape
    P = np.asarray(node_profiles, dtype=np.float64)
    indptr, indices = H.indptr, H.indices

    scores = np.ones(n_edges, dtype=np.float64)
    if prunable_mask is None:
        prunable_mask = np.ones(n_edges, dtype=bool)

    n_tested_pairs_total = 0
    n_sep_pairs_total = 0
    for e in range(n_edges):
        if not prunable_mask[e]:
            continue
        members = indices[indptr[e]:indptr[e + 1]]
        if members.size < 2:
            scores[e] = 1.0
            continue
        mem = members[:max_members]
        # 成员对（封顶 max_pairs）
        all_pairs = list(combinations(range(mem.size), 2))
        if len(all_pairs) > max_pairs:
            step = len(all_pairs) / max_pairs
            all_pairs = [all_pairs[int(t * step)] for t in range(max_pairs)]
        direct = 0
        tested = 0
        for (a, b) in all_pairs:
            i, j = mem[a], mem[b]
            cond_idx = [mem[c] for c in range(mem.size) if c != a and c != b][:max_cond]
            cond_profiles = [P[c] for c in cond_idx]
            sep, _ = _is_separable(P[i], P[j], cond_profiles,
                                   ci_threshold, max_order, n_bins, mm,
                                   ci_method=ci_method, ci_alpha=ci_alpha)
            tested += 1
            if not sep:
                direct += 1
        scores[e] = (direct / tested) if tested > 0 else 1.0
        n_tested_pairs_total += tested
        n_sep_pairs_total += (tested - direct)

    info = {
        "tested_pairs": int(n_tested_pairs_total),
        "separable_pairs": int(n_sep_pairs_total),
        "n_prunable": int(prunable_mask.sum()),
    }
    return scores, info


def node_importance_from_edges(H_csr, edge_conf, eps: float = 1e-9) -> np.ndarray:
    """由边的因果置信度 c_e 聚合出每个节点的因果重要性（∈[0,1]）。

    importance_v = Σ_{e∋v} c_e，再归一。用于"因果性从基因流到细胞"：基因因果重要性
    给细胞相似度里的基因加权，因果中枢基因主导细胞超边、伪共表达基因被压低。
    """
    H = H_csr.tocsr()
    conf = np.asarray(edge_conf, dtype=np.float64).ravel()
    imp = np.asarray(H.multiply(conf[np.newaxis, :]).sum(axis=1)).ravel()
    mx = imp.max()
    return imp / mx if mx > eps else imp


def cell_edge_causal_cohesion(H_csr, raw_profiles, causal_profiles, eps: float = 1e-9):
    """每条细胞超边的因果置信度 = 因果加权基因空间内聚度 / 原始空间内聚度。

    内聚度 = 成员单位向量质心的模长平方(∈[0,1]，1=完全一致)。比值高=这条边只看因果
    验真过的基因时依然紧致(真边)；低=只在原始全基因相关里紧、一上因果权就散(假边)。
    不做病态的 cell-cell CI，只用基因因果重要性——因果性从基因层"量化地"流到细胞层。
    raw_profiles / causal_profiles : (n_nodes × n_genes)。返回归一到[0,1]的 c_e^cell。
    """
    def _coh(P, members):
        V = P[members]
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        nrm[nrm < eps] = 1.0
        U = V / nrm
        c = U.mean(axis=0)
        return float(c @ c)

    H = H_csr.tocsc()
    n_edges = H.shape[1]
    out = np.ones(n_edges, dtype=np.float64)
    indptr, indices = H.indptr, H.indices
    for e in range(n_edges):
        mem = indices[indptr[e]:indptr[e + 1]]
        if mem.size < 2:
            continue
        cc = _coh(causal_profiles, mem)
        cr = _coh(raw_profiles, mem)
        out[e] = cc / (cr + eps)
    out = np.clip(out, 0.0, None)
    mx = out.max()
    return out / mx if mx > eps else out


def metacell_ci_cell_confidence(
    H_csr,
    expr: np.ndarray,
    space: np.ndarray | None = None,
    *,
    q: int = 200,
    ci_threshold: float = 0.02,
    max_order: int = 1,
    max_cond: int = 4,
    n_bins: int = 6,
    knn_meta: int = 10,
    max_pairs: int = 8,
    seed: int = 0,
    ci_method: str = "gtest",
    ci_alpha: float = 0.05,
):
    """方案 A：metacell 辅助 CI 给细胞超边打因果置信度 c_e^cell（真 CI，数学良定）。

    p>n 硬约束：单细胞 CI 不可识别。这里辅助聚 q(<n_genes) 个 metacell，均值去噪 →
    metacell 当变量、基因当样本、p<n ⇒ Σ 可逆、CI 良定（=CASCAT 同级严谨度）。
    metacell 划分只当"测量仪器"：细胞节点/超边一律不动，只借良定的 CI 信号贴回细胞边。

    细胞边 e 的 c_e^cell = 成员对中"其所属 metacell 之间因果连通(不可分离)"的占比：
      同 metacell 对 → 1（真相关）；跨 metacell 对 → 该 metacell 对的 CI 结果(0/1)。
    返回 (c_cell[n_edges], labels[n_cells], q_eff)。
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    n_cells, n_genes = expr.shape
    q_eff = int(max(2, min(q, n_cells - 1, n_genes - 1)))  # 保证 q<n_genes（p<n 良定）
    X = expr if space is None else np.asarray(space, dtype=np.float64)
    labels = KMeans(n_clusters=q_eff, n_init=4, random_state=seed).fit_predict(X)

    P = np.zeros((q_eff, n_genes), dtype=np.float64)  # metacell profile = 成员均值(去噪)
    for m in range(q_eff):
        mask = labels == m
        if mask.any():
            P[m] = expr[mask].mean(axis=0)

    kk = max(1, min(knn_meta, q_eff - 1))
    midx = NearestNeighbors(n_neighbors=kk + 1).fit(P).kneighbors(return_distance=False)
    mneigh = {m: [int(j) for j in midx[m] if int(j) != m][:kk] for m in range(q_eff)}

    ci_cache: dict = {}

    def _meta_real(A, B):
        if A == B:
            return 1.0  # 同 metacell：真相关，无需检验
        key = (A, B) if A < B else (B, A)
        if key in ci_cache:
            return ci_cache[key]
        common = list((set(mneigh[A]) & set(mneigh[B])) - {A, B})[:max_cond]
        sep, _ = _is_separable(P[A], P[B], [P[c] for c in common],
                               ci_threshold, max_order, n_bins, True,
                               ci_method=ci_method, ci_alpha=ci_alpha)
        val = 0.0 if sep else 1.0
        ci_cache[key] = val
        return val

    H = H_csr.tocsc()
    n_edges = H.shape[1]
    indptr, indices = H.indptr, H.indices
    out = np.ones(n_edges, dtype=np.float64)
    for e in range(n_edges):
        mem = indices[indptr[e]:indptr[e + 1]][:16]
        if mem.size < 2:
            continue
        pairs = list(combinations(range(mem.size), 2))
        if len(pairs) > max_pairs:
            step = len(pairs) / max_pairs
            pairs = [pairs[int(t * step)] for t in range(max_pairs)]
        vals = [_meta_real(int(labels[mem[a]]), int(labels[mem[b]])) for a, b in pairs]
        out[e] = float(np.mean(vals)) if vals else 1.0
    return out, labels, q_eff


def causal_edge_confidence(H_csr, node_profiles, **pc_kwargs):
    """每条超边的因果置信度 c_e∈[0,1]（=不可分离成员对占比；高=真边，低=假边）。

    薄封装 pc_edge_scores；语义就是"这条相关性/先验超边里有多少成员对经得起 CI 验真"。
    """
    scores, info = pc_edge_scores(H_csr, node_profiles, **pc_kwargs)
    return scores, info


def prune_incidence_pc(
    H_csr,
    W,
    names,
    types,
    node_profiles: np.ndarray,
    prunable_mask: np.ndarray | None = None,
    *,
    prune_frac: float = 0.3,
    min_keep_frac: float = 0.5,
    drop_fully_redundant: bool = True,
    **pc_kwargs,
):
    """构建期 PC 式静态剪枝：CI 检验打分 → 丢冗余边的列。

    drop_fully_redundant：direct_fraction==0（全部成员对都被分离）的边一律剪掉，
                          再在剩余可剪边里按 prune_frac 补足（受 min_keep_frac 兜底）。
    返回剪枝后的 (H_csr, W, names, types, keep_mask, info)。
    """
    from phasehyper.hypergraph.causal_prune import causal_keep_mask

    scores, info = pc_edge_scores(H_csr, node_profiles, prunable_mask, **pc_kwargs)
    n_edges = H_csr.shape[1]
    if prunable_mask is None:
        prunable_mask = np.ones(n_edges, dtype=bool)

    keep = np.ones(n_edges, dtype=bool)
    # 1) 纯冗余边（分=0）直接剪
    if drop_fully_redundant:
        keep &= ~((scores <= 1e-9) & prunable_mask)
    # 2) 余下可剪边里按因果分排名补剪（复用 causal_keep_mask 的兜底逻辑）
    still = prunable_mask & keep
    extra = causal_keep_mask(scores, prunable_mask=still,
                             prune_frac=prune_frac, min_keep_frac=min_keep_frac)
    keep &= extra

    import numpy as _np
    keep_idx = _np.where(keep)[0]
    H2 = H_csr.tocsc()[:, keep_idx].tocsr()
    W2 = _np.asarray(W)[keep_idx]
    names2 = [names[i] for i in keep_idx]
    types2 = [types[i] for i in keep_idx]
    info["kept"] = int(keep.sum())
    info["pruned"] = int(n_edges - keep.sum())
    return H2, W2, names2, types2, keep, info
