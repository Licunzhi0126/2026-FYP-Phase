from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors

from phasehyper.utils import _reduce_to_fixed_dim, _safe_standardize
from phasehyper.hypergraph.causal_prune import prune_incidence


def _coverage_aware_standardize(mat, *, spike_frac_thresh=0.10, round_dec=6, eps=1e-8):
    """逐列(基因)按**观测值**标准化，处理均值填充造成的"无覆盖占位"问题。

    背景：scNMT 甲基化 63.5% 无 CpG 覆盖被列均值填充→z-score 后挤成一个≈0 的尖峰。
    裸拼接会让模型把"无覆盖"误读成"甲基化正好等于平均值"（强同质假信号），并稀释那
    36.5% 真测值的方差。RNA 稠密无此问题。本函数让两者走同一套：
      - 逐列检测最频值尖峰；占比 > 阈值 → 视为均值填充占位（非观测），置 0（中性）；
      - 观测值用其**自身**均值/方差 z-score（不被占位尖峰稀释）；
      - 无尖峰列（如稠密 RNA）退化为普通全列 z-score（near no-op）。
    mat:(n_samples × n_features)；返回同形 float32（占位/未观测=0）。"""
    X = np.asarray(mat, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    tol = 10.0 ** (-round_dec)
    for j in range(X.shape[1]):
        col = X[:, j]
        rcol = np.round(col, round_dec)
        vals, cnts = np.unique(rcol, return_counts=True)
        k = int(np.argmax(cnts))
        spike_v = vals[k]
        spike_f = cnts[k] / max(1, n)
        observed = (np.abs(rcol - spike_v) > tol) if spike_f > spike_frac_thresh \
            else np.ones(n, dtype=bool)
        obs = col[observed]
        if obs.size >= 2 and obs.std() > eps:
            out[observed, j] = ((obs - obs.mean()) / obs.std()).astype(np.float32)
    return out


def _pairwise_complete_abscorr(X):
    """X:(n_cells × n_genes) 可含 NaN。成对完整(pairwise-complete) Pearson：每对基因只用
    两者都观测的细胞估相关（缺测细胞不参与该对）。向量化矩阵积，n_genes 小(≤几千)很快。
    返回 (|corr| g×g, n_obs g×g)。无插补：缺测既不算 0 也不拉偏均值。"""
    Xn = np.asarray(X, dtype=np.float64)
    obs = np.isfinite(Xn)
    M = obs.astype(np.float64)
    X0 = np.where(obs, Xn, 0.0)
    n = M.T @ M                       # 共观测计数
    Sx = X0.T @ M                     # Σ x_i over cells where j 也观测
    Sxx = (X0 * X0).T @ M             # Σ x_i^2 over cells where j 也观测
    Sxy = X0.T @ X0                   # Σ x_i x_j over 共观测(任一缺则该项=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = Sx / n
        mj = Sx.T / n
        cov = Sxy / n - mi * mj
        vi = Sxx / n - mi * mi
        vj = Sxx.T / n - mj * mj
        corr = cov / np.sqrt(vi * vj)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return np.abs(corr), n


def _multiview_module_edges(views, genes, gene_index, *, corr_thresh=0.3, min_obs=20,
                            min_views=2, top_k=15, max_members=50, gmax=10**9, seen=None):
    """多视图一致共调控模块超边（数据驱动，无先验；scNMT 等无基因名数据的唯一基因边来源）。

    一组基因若在 ≥min_views 个模态视图里（且该对观测充分 n_obs≥min_obs）**同时**强相关，
    才算共识共调控；缺测/观测不足的视图对该对**弃权**（既不支持也不反对，不投 0 票）。
    每基因取其共识邻居 top_k 建一条模块超边（set），dedup + 封顶。强度=支持视图的均值 |corr|。
    单视图相关是噪声，多视图一致才是真模块——取代被删的裸相关 covar。

    views: [(n_cells×n_genes) 数组（可含 NaN），...]，按 genes 列对齐。
    返回 [(members:[(idx,1.0)], name, etype="mvmod"), ...]。
    """
    n_genes = len(genes)
    support = np.zeros((n_genes, n_genes), dtype=np.float64)
    strength = np.zeros((n_genes, n_genes), dtype=np.float64)
    for V in views:
        ac, nobs = _pairwise_complete_abscorr(V)
        sup = (nobs >= min_obs) & (ac >= corr_thresh)        # 该视图可用且强相关 → 支持
        support += sup.astype(np.float64)
        strength += np.where(sup, ac, 0.0)
    consensus = support >= min_views
    strength = np.where(support > 0, strength / np.maximum(support, 1.0), 0.0)
    seen = set() if seen is None else seen
    edges = []
    cap = min(int(max_members), int(gmax))
    for i in range(n_genes):
        nb = np.where(consensus[i])[0]
        if nb.size == 0:
            continue
        nb = nb[np.argsort(-strength[i, nb])][:top_k]
        members = sorted(set([i] + nb.tolist()))
        if not (2 <= len(members) <= cap):
            continue
        key = frozenset(members)
        if key in seen:
            continue
        seen.add(key)
        edges.append(([(g, 1.0) for g in members], f"mvmod::{genes[i]}", "mvmod"))
    return edges


def apply_multiomics_edge_weights(gene_edges_list, genes, gene_index, views, *, min_w=0.1):
    """多组学边构建：按"模态权威"给每条已建超边的**成员重新赋权**。

    成员权重 = 该成员与边**锚点**(首成员，如 GRN 的 TF / 模块的命名中心基因) 在**权威模态**里的
    成对完整 |Pearson|。权威分配（哪个分子层最能定义该类边）：
      ppi / mech_bridge → **蛋白**(物理复合物/亚基的共丰度)；pathway / grn / poswin → **RNA**(共表达)。
    缺测或权威模态无覆盖 → 回退 RNA；权重 floor 到 min_w，不把成员零化。
    把"边只挂 1.0 权"升级为"边的内部结构由多组学证据加权"——这是多组学真正进入超边构建，
    不只是进节点特征。covar_/mvmod 已是多视图构建，保留不动。

    views: {模态名: (n_cells × n_genes) 按 genes 列对齐，可含 NaN}；至少含 'rna'。
    返回新的 gene_edges_list（成员权重已多组学化）。
    """
    abscorr = {}
    for name, V in views.items():
        if V is None:
            continue
        ac, _ = _pairwise_complete_abscorr(V)
        abscorr[name] = ac
    rna = abscorr.get("rna")
    AUTH = {"ppi": "protein", "mech_bridge": "protein",
            "pathway": "rna", "poswin": "rna",
            "grn": "rna", "grn_stim": "rna", "grn_inhib": "rna"}
    out = []
    for members, name, et in gene_edges_list:
        if str(et).startswith("covar") or et == "mvmod":
            out.append((members, name, et)); continue
        C = abscorr.get(AUTH.get(et, "rna"), rna)
        idx = [m[0] for m in members]
        anchor = idx[0]
        new_members = []
        for j in idx:
            if j == anchor:
                w = 1.0
            else:
                w = float(C[anchor, j]) if C is not None else 0.0
                if w <= 0 and rna is not None:           # 权威模态无覆盖 → 回退 RNA 共表达
                    w = float(rna[anchor, j])
                w = max(min_w, w)
            new_members.append((j, w))
        out.append((new_members, name, et))
    return out


def _informative_mask_cols(mask, std_thresh=0.1):
    """只保留覆盖**真正可变**的观测掩码列；丢掉近常数列（如 RNA 全观测=全 1、
    某模态对几乎所有基因都无测量=近全 0）。常数掩码列零信息，拼进模型只会加噪、翻倍参数。
    mask:(n_nodes × width) 0/1。返回 (n_nodes × k) 仅含 std>阈值 的列；全删则返回 (n_nodes × 0)。"""
    m = np.asarray(mask, dtype=np.float32)
    if m.shape[1] == 0:
        return m
    keep = m.std(axis=0) > std_thresh
    return m[:, keep] if keep.any() else m[:, :0]


def _select_top_genes_for_cell(row, top_k):
    values = row.astype(float).abs()
    if top_k is None or top_k >= len(values):
        top_k = min(10, len(values))
    return values.sort_values(ascending=False).head(top_k).index.tolist()


def _build_hyperedges_from_df(
    df: pd.DataFrame,
    top_k: int = 10,
    top_fraction: Optional[float] = None,
    min_size: int = 2,
) -> Dict[str, List[str]]:
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


def _build_cell_hyperedges(
    view1_dfs: Optional[List[pd.DataFrame]] = None,
    expression_df: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    top_fraction: Optional[float] = None,
    min_size: int = 2,
    merge_strategy: str = "separate",
) -> List[Dict[str, List[str]]]:
    if view1_dfs is None or len(view1_dfs) == 0:
        if expression_df is None:
            raise ValueError("Either view1_dfs or expression_df must be provided")
        view1_dfs = [expression_df]

    cell_hyperedges_list = []
    for view1_df in view1_dfs:
        edges = _build_hyperedges_from_df(view1_df, top_k, top_fraction, min_size)
        cell_hyperedges_list.append(edges)

    return cell_hyperedges_list


# ---------------------------------------------------------------------------
# 新建图逻辑（cell_edge_mode="specific"）：
#   B  细胞–基因观测边：连续模态按 z-score 特异性 top-k；二值模态取该细胞 ==1 的基因；
#      两者都 mask-aware（按缺失掩码排除原本缺测的基因）。
#   C  细胞–细胞 kNN 相似边：在表达空间取每个细胞的 k 近邻细胞成边。
# 这些函数为新增，不影响上面的 legacy 路径。
# ---------------------------------------------------------------------------


def _is_binary_frame(df: pd.DataFrame) -> bool:
    """非缺失值是否落在 {0,1}（用于判定二值模态，如甲基化）。"""
    vals = np.asarray(df.to_numpy(dtype=float))
    obs = vals[~np.isnan(vals)]
    if obs.size == 0:
        return False
    return set(np.unique(obs).tolist()).issubset({0.0, 1.0})


def _build_modality_cell_edges(
    df: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
    *,
    top_k: int = 10,
    binary: bool = False,
    min_size: int = 2,
) -> Dict[str, List[str]]:
    """每个细胞一条 {cell} ∪ {选中基因} 超边，mask-aware。

    binary=True ：选该细胞 value==1 的基因（甲基化"被甲基化"的基因集合）。
    binary=False：按基因方向 z-score 后，选该细胞 z 值最高的 top_k 基因（特异性 marker）。
    mask（True=原始缺失）对齐到 df 后，先排除原本缺测的基因，不让"没测到"进候选。
    """
    genes = list(df.columns)
    n_genes = len(genes)

    mask_aligned = None
    if mask is not None:
        mask_aligned = (
            mask.reindex(index=df.index, columns=df.columns).fillna(False).astype(bool)
        )

    zdf = None
    if not binary:
        mat = df.to_numpy(dtype=float)
        col_mean = np.nanmean(mat, axis=0)
        col_std = np.nanstd(mat, axis=0)
        col_std = np.where(col_std == 0, 1.0, col_std)
        zdf = pd.DataFrame((mat - col_mean) / col_std, index=df.index, columns=df.columns)

    edges: Dict[str, List[str]] = {}
    for cell in df.index:
        missing = (
            mask_aligned.loc[cell].to_numpy()
            if mask_aligned is not None
            else np.zeros(n_genes, dtype=bool)
        )
        if binary:
            row = np.nan_to_num(df.loc[cell].to_numpy(dtype=float), nan=0.0)
            keep = (~missing) & (row == 1.0)
            selected = [genes[i] for i in range(n_genes) if keep[i]]
        else:
            zrow = zdf.loc[cell].to_numpy()
            valid = (~missing) & (~np.isnan(zrow))
            cand = [i for i in range(n_genes) if valid[i]]
            cand.sort(key=lambda i: zrow[i], reverse=True)
            selected = [genes[i] for i in cand[:top_k]]

        members = list(dict.fromkeys(selected + [str(cell)]))
        if len(members) > 1 and len(members) >= min_size:
            edges[str(cell)] = members
    return edges


def _build_cell_knn_edges(
    expression_df: pd.DataFrame,
    *,
    k: int = 10,
) -> Dict[str, List[str]]:
    """C：表达空间 kNN，每个细胞 + 其 k 个最近细胞 = 一条细胞–细胞超边。"""
    cells = [str(c) for c in expression_df.index]
    n = len(cells)
    if n <= 2:
        return {}
    k = max(1, min(k, n - 1))
    X = expression_df.fillna(0.0).to_numpy(dtype=float)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    idx = nbrs.kneighbors(return_distance=False)
    edges: Dict[str, List[str]] = {}
    for i, cell in enumerate(cells):
        neighbors = [cells[j] for j in idx[i] if int(j) != i][:k]
        members = list(dict.fromkeys([cell] + neighbors))
        if len(members) > 1:
            edges[cell] = members
    return edges


def _reindex_impute(
    df: pd.DataFrame,
    common_cells: List[str],
    common_genes: List[str],
    impute: str = "zero",
) -> np.ndarray:
    """对齐到 (common_cells, common_genes)，按 impute 策略填缺失。

    impute="zero"    ：旧行为，NaN→0。
    impute="col_mean"：NaN→该基因列的观测均值（整列全缺则回退 0），避免伪 0 污染。
    """
    aligned = df.reindex(index=common_cells, columns=common_genes)
    if impute == "col_mean":
        col_mean = aligned.mean(axis=0, skipna=True)
        aligned = aligned.fillna(col_mean)
    return aligned.fillna(0.0).values.astype(np.float32)


def _initialize_node_features_prior_only(
    node_list: List[str],
    node_types: List[str],
    common_cells: List[str],
    common_genes: List[str],
    expression_df: pd.DataFrame,
    view1_dfs: Optional[List[pd.DataFrame]] = None,
    feature_dim: int = 64,
    impute: str = "zero",
) -> torch.Tensor:
    expr = _reindex_impute(expression_df, common_cells, common_genes, impute="zero")

    cell_views = [expr]
    gene_views = [expr.T]

    if view1_dfs is not None and len(view1_dfs) > 0:
        for view1_df in view1_dfs:
            view1 = _reindex_impute(view1_df, common_cells, common_genes, impute=impute)
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
    dataset,
    prior,
    feature_dim: int = 64,
    cell_hyperedge_top_k: int = 10,
    cell_hyperedge_top_fraction: Optional[float] = None,
    min_cell_hyperedge_size: int = 2,
    merge_strategy: str = "average",
    cell_edge_mode: str = "legacy",
    add_cell_knn_edges: bool = False,
    cell_knn_k: int = 10,
):
    """Build hypergraph dictionary for prior-only hypergraph mode.

    cell_edge_mode:
      "legacy"   —— 原行为：细胞观测边来自 view1（第二模态）的 |值| top-k。
      "specific" —— 新行为（B）：表达走 z-score 特异性 top-k；第二模态按类型处理
                    （二值/甲基化取 ==1，连续/蛋白走 z-score）；全部 mask-aware。
    add_cell_knn_edges：是否再加 C（表达空间 kNN 的细胞–细胞相似边）。
    """
    gene_set = set(dataset.common_genes)
    cell_set = set(dataset.common_cells)
    specific = cell_edge_mode == "specific"
    node_feature_impute = "col_mean" if specific else "zero"

    # ---- 细胞观测/相似超边：(edges_dict, name_suffix, type_str) 列表 ----
    cell_edge_groups: List = []
    if specific:
        expr_mask = getattr(dataset, "expression_mask", None)
        view1_masks = getattr(dataset, "view1_masks", None) or []
        # B-表达：z-score 特异性 top-k（mask-aware）
        cell_edge_groups.append((
            _build_modality_cell_edges(
                dataset.expression_df, expr_mask,
                top_k=cell_hyperedge_top_k, binary=False,
                min_size=min_cell_hyperedge_size,
            ),
            "expr", "cell_expr",
        ))
        # B-第二模态：二值→==1，连续→z-score（mask-aware）
        for vi, view1_df in enumerate(dataset.view1_dfs or []):
            vmask = view1_masks[vi] if vi < len(view1_masks) else None
            is_bin = _is_binary_frame(view1_df)
            cell_edge_groups.append((
                _build_modality_cell_edges(
                    view1_df, vmask,
                    top_k=cell_hyperedge_top_k, binary=is_bin,
                    min_size=min_cell_hyperedge_size,
                ),
                f"view{vi+1}", f"cell_view{vi+1}",
            ))
    else:
        legacy_edges = _build_cell_hyperedges(
            view1_dfs=dataset.view1_dfs,
            expression_df=dataset.expression_df,
            top_k=cell_hyperedge_top_k,
            top_fraction=cell_hyperedge_top_fraction,
            min_size=min_cell_hyperedge_size,
            merge_strategy=merge_strategy,
        )
        for vi, edges in enumerate(legacy_edges):
            cell_edge_groups.append((edges, f"view{vi+1}", f"cell_view{vi+1}"))

    # C：细胞–细胞 kNN 相似边（两模式都可叠加）
    if add_cell_knn_edges:
        cell_edge_groups.append((
            _build_cell_knn_edges(dataset.expression_df, k=cell_knn_k),
            "knn", "cell_knn",
        ))

    hyperedge_list: List[List[str]] = []
    hyperedge_names: List[str] = []
    hyperedge_types: List[str] = []

    for edge_name, genes in prior.kegg_groups.items():
        members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
        if len(members) > 1:
            hyperedge_list.append(members)
            hyperedge_names.append(str(edge_name))
            hyperedge_types.append("pathway")

    for edge_name, genes in prior.poswin_groups.items():
        members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
        if len(members) > 1:
            hyperedge_list.append(members)
            hyperedge_names.append(str(edge_name))
            hyperedge_types.append("poswin")

    if prior.ppi_groups:
        for edge_name, genes in prior.ppi_groups.items():
            members = list(dict.fromkeys([gene for gene in genes if gene in gene_set]))
            if len(members) > 1:
                hyperedge_list.append(members)
                hyperedge_names.append(str(edge_name))
                hyperedge_types.append("ppi")

    for edges, name_suffix, type_str in cell_edge_groups:
        for cell_name, members in edges.items():
            valid = list(
                dict.fromkeys(
                    [node for node in members if node in gene_set or node in cell_set]
                )
            )
            if len(valid) > 1:
                hyperedge_list.append(valid)
                hyperedge_names.append(f"obs::{cell_name}::{name_suffix}")
                hyperedge_types.append(type_str)

    all_nodes = set(dataset.common_genes) | set(dataset.common_cells)
    node_list = sorted(list(all_nodes))
    node_id_map = {node: idx for idx, node in enumerate(node_list)}

    H = torch.zeros(len(node_list), len(hyperedge_list), dtype=torch.float32)
    edge_weights = torch.ones(len(hyperedge_list), dtype=torch.float32)

    for edge_idx, hyperedge in enumerate(hyperedge_list):
        for node in hyperedge:
            if node not in node_id_map:
                raise KeyError(f"Node {node} in hyperedge not found in node_id_map")
            H[node_id_map[node], edge_idx] = 1.0

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
        impute=node_feature_impute,
    )

    true_sample_labels = torch.zeros(len(node_list), dtype=torch.long)
    label_by_cell = {
        cell: int(dataset.labels[idx]) for idx, cell in enumerate(dataset.common_cells)
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

    edge_type_to_id = {
        "pathway": 0,
        "ppi": 1,
        "poswin": 2,
        "cell": 3,
    }
    hyperedge_type_ids = torch.tensor(
        [edge_type_to_id.get(t, 3) for t in hyperedge_types],
        dtype=torch.long,
    )

    pathway_edge_mask = torch.tensor([t == "pathway" for t in hyperedge_types], dtype=torch.bool)
    ppi_edge_mask = torch.tensor([t == "ppi" for t in hyperedge_types], dtype=torch.bool)
    poswin_edge_mask = torch.tensor([t == "poswin" for t in hyperedge_types], dtype=torch.bool)
    cell_edge_mask = torch.tensor(
        [t == "cell" or t.startswith("cell_") for t in hyperedge_types],
        dtype=torch.bool,
    )

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
        "cell_edge_mask": cell_edge_mask,
    }


# ===========================================================================
# 异构超图（CITE_seq 等真·多模态）：3 类节点 cell / gene / protein。
#   关键：蛋白 ADT 名（CD4/CD8…）与基因符号会**撞名**，故节点用 (type, name)
#   二元组做唯一 ID，不再依赖「名字全局唯一」的旧假设。
#   边族：gene-gene 先验 / cell-gene 观测 / cell-protein 观测 / gene-protein 桥 / cell-cell kNN。
# 完全独立于上面的 legacy 路径，旧行为不受影响。
# ===========================================================================


def _modality_cell_feature_edges(modality, top_k):
    """对一个模态产 {cell: [选中特征名...]}（不含 cell 自身），复用 mask-aware 选边。"""
    edges = _build_modality_cell_edges(
        modality.feature_table, modality.mask,
        top_k=top_k, binary=modality.binary, min_size=1,
    )
    out = {}
    for cell, members in edges.items():
        cellstr = str(cell)
        feats = [m for m in members if m != cellstr]
        if feats:
            out[cellstr] = feats
    return out


def _init_hetero_node_features(bundle, cells, genes, protein_names, feature_dim):
    """按类型分别初始化节点特征（不再共享 common_genes 拼接）。

    cell  ← 各模态 cells×F 标准化后拼接 → 降维
    gene  ← gene-identity(RNA) 模态 genes×cells 画像 → 降维
    protein ← protein 模态 proteins×cells 画像 → 降维
    """
    # cell 视图：所有模态的 cells×F 拼起来
    cell_views = []
    rna_table = None
    prot_table = None
    for m in bundle.modalities:
        tab = m.feature_table.reindex(index=cells)
        cell_views.append(_safe_standardize(tab.to_numpy(dtype=float)))
        if m.node_type == "gene" and m.bridge_to_gene is None and rna_table is None:
            rna_table = tab
        if m.node_type == "protein" and prot_table is None:
            prot_table = m.feature_table.reindex(index=cells)

    cell_raw = np.concatenate(cell_views, axis=1) if cell_views else np.zeros((len(cells), 1))
    cell_features = _safe_standardize(_reduce_to_fixed_dim(cell_raw, feature_dim))

    # gene 视图：RNA genes×cells
    if rna_table is not None:
        gene_mat = rna_table.reindex(columns=genes).to_numpy(dtype=float).T  # genes×cells
    else:
        gene_mat = np.zeros((len(genes), len(cells)), dtype=np.float32)
    gene_features = _safe_standardize(_reduce_to_fixed_dim(_safe_standardize(gene_mat), feature_dim))

    # protein 视图：proteins×cells
    if prot_table is not None and len(protein_names) > 0:
        prot_mat = prot_table.reindex(columns=protein_names).to_numpy(dtype=float).T  # P×cells
        protein_features = _safe_standardize(_reduce_to_fixed_dim(_safe_standardize(prot_mat), feature_dim))
    else:
        protein_features = np.zeros((len(protein_names), feature_dim), dtype=np.float32)

    n_nodes = len(cells) + len(genes) + len(protein_names)
    X = np.zeros((n_nodes, feature_dim), dtype=np.float32)
    X[: len(cells)] = cell_features
    X[len(cells): len(cells) + len(genes)] = gene_features
    X[len(cells) + len(genes):] = protein_features
    return torch.from_numpy(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))


def build_hetero_hypergraph_dict(
    bundle,
    prior,
    *,
    feature_dim: int = 64,
    cell_gene_top_k: int = 10,
    cell_protein_top_k: int = 5,
    add_cell_knn_edges: bool = False,
    cell_knn_k: int = 10,
    ppi_max_members: Optional[int] = None,
):
    """异构超图：cell / gene / protein 三类节点 + 多类边。

    bundle : HeteroBundle（cells / genes / modalities / labels）。
    prior  : PriorBundle（kegg_groups / poswin_groups / ppi_groups，gene-gene）。
    ppi_max_members：PPI 超边成员上限（防超大稠密边），None=不限。
    """
    cells = [str(c) for c in bundle.cells]
    genes = [str(g) for g in bundle.genes]
    gene_set = set(genes)

    protein_names: List[str] = []
    for m in bundle.modalities:
        if m.node_type == "protein":
            protein_names.extend(str(c) for c in m.feature_table.columns)
    protein_set = set(protein_names)

    # ---- 节点：(type, name) 唯一 ID，顺序 cell → gene → protein ----
    node_records = (
        [("cell", c) for c in cells]
        + [("gene", g) for g in genes]
        + [("protein", p) for p in protein_names]
    )
    node_id = {rec: i for i, rec in enumerate(node_records)}
    node_types = [t for t, _ in node_records]
    node_names = [n for _, n in node_records]

    # ---- 边：每条 = (members:[(type,name)], name, type) ----
    edges: List = []

    def _add(members, name, etype):
        members = list(dict.fromkeys(members))
        if len(members) > 1:
            edges.append((members, name, etype))

    # 1) gene–gene 先验
    for gname, gs in (prior.kegg_groups or {}).items():
        _add([("gene", g) for g in gs if g in gene_set], str(gname), "pathway")
    for gname, gs in (prior.poswin_groups or {}).items():
        _add([("gene", g) for g in gs if g in gene_set], str(gname), "poswin")
    for gname, gs in (prior.ppi_groups or {}).items():
        members = [g for g in gs if g in gene_set]
        if ppi_max_members is not None and len(members) > ppi_max_members:
            members = members[:ppi_max_members]
        _add([("gene", g) for g in members], str(gname), "ppi")

    # 2/3) cell–特征观测边（gene 模态 → cell_gene；protein 模态 → cell_protein）
    for m in bundle.modalities:
        if m.node_type == "gene":
            top_k, etype = cell_gene_top_k, "cell_gene"
        elif m.node_type == "protein":
            top_k, etype = cell_protein_top_k, "cell_protein"
        else:
            continue
        per_cell = _modality_cell_feature_edges(m, top_k)
        for cell, feats in per_cell.items():
            members = [(m.node_type, f) for f in feats] + [("cell", cell)]
            _add(members, f"obs::{cell}::{m.name}", etype)

    # 4) gene–protein 桥接边（中心法则）
    for m in bundle.modalities:
        if m.node_type == "protein" and m.bridge_to_gene:
            for p, targets in m.bridge_to_gene.items():
                if p not in protein_set:
                    continue
                gmem = [("gene", g) for g in dict.fromkeys(targets) if g in gene_set]
                _add([("protein", p)] + gmem, f"bridge::{p}", "bridge")

    # 5) cell–cell kNN（用第一个 gene 模态的表达空间）
    if add_cell_knn_edges:
        rna = next((m for m in bundle.modalities if m.node_type == "gene"), None)
        if rna is not None:
            knn = _build_cell_knn_edges(rna.feature_table.reindex(index=cells), k=cell_knn_k)
            for cell, members in knn.items():
                _add([("cell", c) for c in members], f"knn::{cell}", "cell_knn")

    # ---- 关联矩阵 H ----
    H = torch.zeros(len(node_records), len(edges), dtype=torch.float32)
    for ei, (members, _, _) in enumerate(edges):
        for rec in members:
            H[node_id[rec], ei] = 1.0
    edge_weights = torch.ones(len(edges), dtype=torch.float32)

    hyperedge_names = [name for _, name, _ in edges]
    hyperedge_types = [etype for _, _, etype in edges]

    # ---- 节点掩码 / 名称 ----
    cell_mask = torch.tensor([t == "cell" for t in node_types], dtype=torch.bool)
    gene_mask = torch.tensor([t == "gene" for t in node_types], dtype=torch.bool)
    protein_mask = torch.tensor([t == "protein" for t in node_types], dtype=torch.bool)
    sample_node_names = {i: node_names[i] for i in range(len(node_records)) if node_types[i] == "cell"}
    gene_node_names = {i: node_names[i] for i in range(len(node_records)) if node_types[i] == "gene"}
    protein_node_names = {i: node_names[i] for i in range(len(node_records)) if node_types[i] == "protein"}

    # ---- 节点特征 ----
    X = _init_hetero_node_features(bundle, cells, genes, protein_names, feature_dim)

    # ---- 真值标签（仅评估；填到 cell 节点上）----
    true_sample_labels = torch.zeros(len(node_records), dtype=torch.long)
    if bundle.labels is not None:
        label_by_cell = {str(c): int(bundle.labels[i]) for i, c in enumerate(bundle.cells)}
        for i in range(len(node_records)):
            if node_types[i] == "cell":
                true_sample_labels[i] = label_by_cell.get(node_names[i], 0)

    edge_type_to_id = {
        "pathway": 0, "ppi": 1, "poswin": 2,
        "cell_gene": 3, "cell_protein": 4, "bridge": 5, "cell_knn": 6,
    }
    hyperedge_type_ids = torch.tensor(
        [edge_type_to_id.get(t, 3) for t in hyperedge_types], dtype=torch.long
    )

    def _emask(name):
        return torch.tensor([t == name for t in hyperedge_types], dtype=torch.bool)

    return {
        "H": H,
        "X": X,
        "W": edge_weights,
        "node_types": node_types,
        "node_names": node_names,
        "cell_mask": cell_mask,
        "gene_mask": gene_mask,
        "protein_mask": protein_mask,
        # 兼容旧键名（sample==cell）
        "sample_mask": cell_mask,
        "sample_node_names": sample_node_names,
        "gene_node_names": gene_node_names,
        "protein_node_names": protein_node_names,
        "true_sample_labels": true_sample_labels,
        "n_nodes": len(node_records),
        "n_edges": len(edges),
        "n_cells": len(cells),
        "n_genes": len(genes),
        "n_proteins": len(protein_names),
        "hyperedge_names": hyperedge_names,
        "hyperedge_types": hyperedge_types,
        "hyperedge_type_ids": hyperedge_type_ids,
        "edge_type_to_id": edge_type_to_id,
        "pathway_edge_mask": _emask("pathway"),
        "ppi_edge_mask": _emask("ppi"),
        "poswin_edge_mask": _emask("poswin"),
        "cell_gene_edge_mask": _emask("cell_gene"),
        "cell_protein_edge_mask": _emask("cell_protein"),
        "bridge_edge_mask": _emask("bridge"),
        "cell_knn_edge_mask": _emask("cell_knn"),
    }


# ===========================================================================
# 两层级超图（README_两层级超图方案.md）：两并行通道，不是一张扁平图。
#   基因通道 Level 2：H_gene (n_genes × n_gene_edges)，承载 pathway/ppi/poswin 先验。
#   细胞通道 Level 1：H_cell (n_cells × n_cell_edges)，承载 rna_knn/adt_knn 模态相似。
#   两通道各自下游卷积出 cell×128 → concat cell×256 → 分相（耦合=concat，不是 pooling）。
# 关键：H_cell 列数可达上万，**必须稀疏**（返回 scipy.sparse.csr），不物化稠密。
# 完全独立于上面的扁平 builder，旧行为不受影响。
# ===========================================================================


def _assemble_incidence(edges_list, n_nodes):
    """edges_list: [(members, name, etype), ...]，members=[(node_idx, weight), ...]。

    返回 (H_csr[n_nodes × n_edges], W[n_edges], names, types)。空边集返回 (n_nodes × 0)。
    """
    import scipy.sparse as sp

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    names: List[str] = []
    types: List[str] = []
    for ci, (members, name, etype) in enumerate(edges_list):
        for nidx, w in members:
            rows.append(int(nidx))
            cols.append(ci)
            data.append(float(w))
        names.append(str(name))
        types.append(str(etype))
    n_edges = len(edges_list)
    if n_edges == 0:
        H = sp.csr_matrix((n_nodes, 0), dtype=np.float64)
    else:
        H = sp.csr_matrix(
            (np.asarray(data, dtype=np.float64), (rows, cols)),
            shape=(n_nodes, n_edges),
        )
    W = np.ones(n_edges, dtype=np.float64)
    return H, W, names, types


def _pca_reduce_np(X: np.ndarray, dim: int) -> np.ndarray:
    """只为 kNN 定邻居用的降维（不进节点特征）；样本/特征不足时安全回退。"""
    from sklearn.decomposition import PCA

    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    max_k = min(int(dim), X.shape[0] - 1, X.shape[1])
    if max_k < 1:
        return X.astype(np.float32)
    return PCA(n_components=max_k, random_state=42).fit_transform(X).astype(np.float32)


def _knn_edge_members(X: np.ndarray, k: int, knn_weight: bool):
    """X:(n × d)。每行(细胞) → 一条超边 = {自身 ∪ k 近邻}，成员带权（高斯或 1）。"""
    n = X.shape[0]
    if n <= 2:
        return []
    k = max(1, min(int(k), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    dist, idx = nbrs.kneighbors(return_distance=True)
    out = []
    for i in range(n):
        neigh = [(int(j), float(d)) for j, d in zip(idx[i], dist[i]) if int(j) != i][:k]
        members = [(i, 1.0)]
        if knn_weight and neigh:
            ds = np.asarray([d for _, d in neigh], dtype=np.float64)
            med = float(np.median(ds))
            sig = med if med > 0 else 1.0
            members += [(j, float(np.exp(-(d * d) / (2.0 * sig * sig)))) for j, d in neigh]
        else:
            members += [(j, 1.0) for j, _ in neigh]
        out.append(members)
    return out


def _build_cci_edges(
    expr: np.ndarray,
    gene_index: Dict[str, int],
    lr_pairs,
    *,
    sender_q: float = 0.85,
    receiver_q: float = 0.85,
    max_members: Optional[int] = None,
    keep_frac: float = 0.5,
    min_members: int = 2,
):
    """CellChat 式单细胞配体-受体通讯超边（无监督、不用 cluster label，避免 ARI 泄漏）。

    旧版的问题：每个 LR 对都把"所有高配体细胞 ∪ 所有高受体细胞"塞进一条巨边（可达 40%
    细胞），既不区分 LR 对的强弱（非特异共表达也成边），成员还一律 1.0 权 → 严重过平滑。

    新版换成一个更合理、可辩护的方案，借 CellChat / CellPhoneDB 两味核心成分：
      1) 质量作用通讯强度（law of mass action）：
            strength = mean(ligand | sender) · mean(receptor | receiver)
         作为该 LR 对的通讯概率代理，用来跨 LR 对排序。
      2) 特异性筛选：只保留通讯强度 top-`keep_frac` 的 LR 对成边，剔除弱/非特异通讯，
         不再让每个 LR 对都成一条巨边。
    且成员按通讯角色加权（ReLU(z) 归一到 [0.2,1]），让真正高表达的发送/接收细胞在
    超图传播里占更大份额，而非一律 1.0。

    expr 已是按基因 z-score 的表达；sender/receiver 用分位阈值定高表达细胞。
    返回 [members, ...]，members=[(cell_idx, weight), ...]，喂给 _add_knn 复用去重/封顶。
    """
    cands: List = []  # (strength, members)
    n_cells = expr.shape[0]
    for L, R in lr_pairs:
        li = gene_index.get(str(L))
        ri = gene_index.get(str(R))
        if li is None or ri is None:
            continue
        lv = expr[:, li]
        rv = expr[:, ri]
        if lv.std() < 1e-9 or rv.std() < 1e-9:
            continue
        lt = float(np.quantile(lv, sender_q))
        rt = float(np.quantile(rv, receiver_q))
        send = np.where(lv >= lt)[0]
        recv = np.where(rv >= rt)[0]
        if send.size == 0 or recv.size == 0:
            continue
        # 质量作用通讯强度（跨 LR 对可比，用于特异性排序）
        strength = float(lv[send].mean() * rv[recv].mean())
        if not np.isfinite(strength) or strength <= 0:
            continue
        # 成员角色强度：sender 取 ReLU(ligand z)，receiver 取 ReLU(receptor z)，重叠取大
        score: Dict[int, float] = {}
        for c in send.tolist():
            score[c] = max(score.get(c, 0.0), float(max(lv[c], 0.0)))
        for c in recv.tolist():
            score[c] = max(score.get(c, 0.0), float(max(rv[c], 0.0)))
        cells_sorted = sorted(score, key=lambda c: score[c], reverse=True)
        if max_members is not None and len(cells_sorted) > max_members:
            cells_sorted = cells_sorted[:max_members]
        if len(cells_sorted) < min_members:
            continue
        vals = np.asarray([score[c] for c in cells_sorted], dtype=np.float64)
        vmax = vals.max() if vals.max() > 0 else 1.0
        wts = 0.2 + 0.8 * (vals / vmax)  # 归一到 [0.2,1]，避免 0 权成员被算子忽略
        members = [(int(c), float(w)) for c, w in zip(cells_sorted, wts)]
        cands.append((strength, members))

    if not cands:
        return [], 0
    # 特异性筛选：只保留通讯强度最高的 top-keep_frac 个 LR 对
    cands.sort(key=lambda t: t[0], reverse=True)
    n_keep = max(1, int(np.ceil(len(cands) * float(keep_frac))))
    out = [m for _, m in cands[:n_keep]]
    return out, len(out)


def load_grn_regulons(grn_csv, genes, *, min_members: int = 2, max_members: Optional[int] = None):
    """从 OmniPath CollecTRI 缓存(TF,target)建 regulon：每个 TF + 其 panel 内 target = 一组。

    grn_csv：列含 TF,target（data_clean/grn_collectri.csv，omnipath CollecTRI 拉取缓存）。
    返回 {f"regulon::{TF}": [TF, target1, ...]}（仅保留 TF 和 target 都在 panel 内的边）。
    有向性：TF 作为调控中枢列首；超边本身是 regulon 基因集（SCENIC 同款用法）。
    """
    panel = set(str(g) for g in genes)
    df = pd.read_csv(grn_csv)
    cols = {c.lower(): c for c in df.columns}
    tfc = cols.get("tf") or cols.get("source") or df.columns[0]
    tgc = cols.get("target") or df.columns[1]
    df = df[[tfc, tgc]].astype(str)
    df = df[df[tfc].isin(panel) & df[tgc].isin(panel) & (df[tfc] != df[tgc])]
    regs: Dict[str, List[str]] = {}
    for tf, sub in df.groupby(tfc):
        targets = [t for t in dict.fromkeys(sub[tgc].tolist()) if t != tf]
        members = [tf] + targets
        if max_members is not None and len(members) > max_members:
            members = members[:max_members]
        if len(members) >= min_members:
            regs[f"regulon::{tf}"] = members
    return regs


def load_grn_star_edges(grn_csv, genes, *, signed: bool = True, max_targets_per_tf: Optional[int] = None):
    """GRN 星图：每条 TF→target 调控关系 = 一条 2 成员超边 {TF, target}（非 regulon 巨边）。

    相比 regulon set 超边（{TF}∪targets 一条边，卷积里 target 之间被伪连成 clique），
    星图把每个 TF→target 拆成独立 2 成员边：target 彼此**不直连**，只通过共同的 TF 两跳相连。
    - 方向：TF 是该 regulon 所有 target 边的**共享枢纽**（TF 度高、target 度低），度归一化卷积里
      TF 影响天然不对称地大 → 软编码 TF→target 方向（非严格有向传播，那需改卷积）。
    - 符号：signed=True 时按 CollecTRI 的 stim/inhib 分 etype('grn_stim'/'grn_inhib')并返回符号
      ±1（喂给带符号卷积：抑制边传负向消息=反平滑）。stim/inhib 都空或都真→无符号 'grn' +1。

    返回 [(members, name, etype, signW), ...]，members=[(tf_idx_placeholder...)]——实际返回基因名，
    调用方按 gene_index 映射；为简化此处直接返回 (TF_name, target_name, etype, signW)。
    """
    panel = set(str(g) for g in genes)
    df = pd.read_csv(grn_csv)
    cols = {c.lower(): c for c in df.columns}
    tfc = cols.get("tf") or cols.get("source") or df.columns[0]
    tgc = cols.get("target") or df.columns[1]
    stc = cols.get("stim") or cols.get("is_stimulation")
    ihc = cols.get("inhib") or cols.get("is_inhibition")
    df = df[df[tfc].astype(str).isin(panel) & df[tgc].astype(str).isin(panel)]
    df = df[df[tfc].astype(str) != df[tgc].astype(str)]
    edges = []
    per_tf: Dict[str, int] = {}
    for _, row in df.iterrows():
        tf, tg = str(row[tfc]), str(row[tgc])
        if max_targets_per_tf is not None:
            if per_tf.get(tf, 0) >= max_targets_per_tf:
                continue
            per_tf[tf] = per_tf.get(tf, 0) + 1
        if signed and stc is not None and ihc is not None:
            stim = bool(row[stc]) if not pd.isna(row[stc]) else False
            inhib = bool(row[ihc]) if not pd.isna(row[ihc]) else False
            if stim and not inhib:
                etype, sgn = "grn_stim", 1.0
            elif inhib and not stim:
                etype, sgn = "grn_inhib", -1.0
            else:
                etype, sgn = "grn", 1.0   # 无符号/双向歧义 → 不带符号
        else:
            etype, sgn = "grn", 1.0
        edges.append((tf, tg, etype, sgn))
    return edges


def _build_cci_bipartite_edges(
    expr: np.ndarray,
    gene_index: Dict[str, int],
    lr_pairs,
    *,
    sender_q: float = 0.85,
    receiver_q: float = 0.85,
    keep_frac: float = 0.5,
    per_side_cap: int = 10,
):
    """CCI 二阶图：每条 LR 通讯不再塞成一条 sender∪receiver 巨超边，而是拆成
    sender×receiver 的 2 成员 pairwise 边 {s, r}（更像图、边更多，去掉 sender 间/receiver 间伪耦合）。

    防爆炸：每个保留的 LR 对只取通讯角色最强的 top-`per_side_cap` 个 sender 和 receiver，
    成 cap×cap 条 2 成员边。sender 权按 ReLU(ligand z)、receiver 权按 ReLU(receptor z)（软方向）。
    沿用质量作用强度 + top-keep_frac 特异性筛选（与 CellChat 式一致）。
    """
    cands = []  # (strength, send_top, recv_top)
    for L, R in lr_pairs:
        li = gene_index.get(str(L)); ri = gene_index.get(str(R))
        if li is None or ri is None:
            continue
        lv = expr[:, li]; rv = expr[:, ri]
        if lv.std() < 1e-9 or rv.std() < 1e-9:
            continue
        send = np.where(lv >= float(np.quantile(lv, sender_q)))[0]
        recv = np.where(rv >= float(np.quantile(rv, receiver_q)))[0]
        if send.size == 0 or recv.size == 0:
            continue
        strength = float(lv[send].mean() * rv[recv].mean())
        if not np.isfinite(strength) or strength <= 0:
            continue
        s_top = send[np.argsort(-lv[send])[:per_side_cap]]
        r_top = recv[np.argsort(-rv[recv])[:per_side_cap]]
        cands.append((strength, s_top, lv, r_top, rv))
    if not cands:
        return [], 0
    cands.sort(key=lambda t: t[0], reverse=True)
    n_keep = max(1, int(np.ceil(len(cands) * float(keep_frac))))
    out = []
    for _, s_top, lv, r_top, rv in cands[:n_keep]:
        for s in s_top.tolist():
            ws = 0.2 + 0.8 * float(max(lv[s], 0.0)) / (float(lv[s_top].max()) or 1.0)
            for r in r_top.tolist():
                if s == r:
                    continue
                wr = 0.2 + 0.8 * float(max(rv[r], 0.0)) / (float(rv[r_top].max()) or 1.0)
                out.append([(int(s), ws), (int(r), wr)])
    return out, n_keep


def build_two_level_hypergraph(
    bundle,
    prior,
    *,
    # 基因通道（Level 2，细胞内）
    gene_edges=("pathway", "ppi", "poswin"),
    grn_groups: Optional[Dict] = None,
    grn_max_members: Optional[int] = 100,
    grn_star_edges: Optional[List] = None,   # GRN 星图模式：[(tf,target,etype,signW),...]，给定则取代 regulon set 边
    grn_signed: bool = True,                 # 星图是否带 stim/inhib 符号（喂带符号卷积）
    ppi_max_members: Optional[int] = 50,
    gene_node_state: str = "expression",
    gene_multiomics: bool = False,           # 基因节点状态=中心法则整合点：RNA⊕第二模态(同基因对齐)
    gene_mod_coverage_std: bool = False,     # 第二模态覆盖感知标准化：仅对真"无覆盖缺失"模态(甲基化)opt-in；
                                             # 对 LOD 填充(PEA 蛋白=真低丰度)会抹真信号→默认关
    gene_mechanistic_edges: bool = False,    # bridge 共调控基因超边(蛋白→≥2 基因；机制性跨模态边)
    multiview_modules: bool = False,         # 多视图一致共调控模块超边(数据驱动；无基因名数据的边来源)
    multiomics_edges: bool = False,          # 多组学边构建：按模态权威给各超边成员重新赋权(蛋白/RNA)
    mv_corr_thresh: float = 0.3,
    mv_min_obs: int = 20,
    mv_min_views: int = 2,
    mv_top_k: int = 15,
    gene_channel_out: int = 128,
    gene_pool: str = "graph_smooth_mean",
    smooth_steps: int = 2,
    # 细胞通道（Level 1，细胞间）
    cell_edges=("rna_knn", "adt_knn"),
    rna_knn_k: int = 15,
    adt_knn_k: int = 15,
    pca_dim: int = 50,
    knn_weight: bool = True,
    cell_channel_out: int = 128,
    use_wnn: bool = False,
    # CCI（配体-受体通讯）细胞–细胞边
    cci_lr_pairs=None,
    cci_sender_quantile: float = 0.85,
    cci_receiver_quantile: float = 0.85,
    cci_max_members: Optional[int] = None,
    cci_keep_frac: float = 0.5,
    cci_pairwise: bool = False,              # CCI 二阶图：sender×receiver 拆成 2 成员 pairwise 边
    cci_pairwise_cap: int = 10,              # 每个 LR 对每侧取 top-N，防 send×recv 爆炸
    # 耦合 / 通用
    couple: str = "concat",
    cell_feat_with_protein: bool = False,
    max_edge_fraction: float = 0.40,
    # 构建期 CMI 静态剪枝（用原始特征对关联矩阵 H 打分剪冗余边，一次成图）
    build_prune_cell: bool = False,
    build_prune_gene: bool = False,
    build_prune_frac: float = 0.3,
    build_prune_min_keep: float = 0.5,
    build_prune_method: str = "gaussian",   # "gaussian"=高斯偏相关紧凑度；"pc"=PC 式条件独立检验
    pc_ci_threshold: float = 0.02,
    pc_max_order: int = 1,
    pc_max_pairs: int = 6,
    pc_max_cond: int = 4,
    pc_n_bins: int = 6,
    # 因果化超图构建：对(KEGG/PPI/poswin/covar)基因边做 CI 验真→去假边+软权 c_e；
    # 因果性经基因重要性流到细胞层（细胞相似度在因果加权基因空间里算）。
    causal_construct: bool = False,
    causal_cell: bool = False,   # 细胞层因果总开关（默认关：因果只在基因层做，cell 侧不碰）
    causal_drop_zero: bool = True,
    causal_cell_space: bool = True,
    causal_gene_floor: float = 0.1,
    causal_random_baseline: bool = False,  # 对照：随机保留同样数量的边(隔离"因果选得准 vs 只是少边")
    causal_random_seed: int = 0,
    causal_cell_drop_frac: float = 0.0,     # 细胞边按 c_e^cell 硬删的底部比例(0=只软权不硬删)
    causal_cell_method: str = "cohesion",   # "cohesion"=相似度代理；"metacell"=metacell 辅助真 CI(方案A)
    causal_metacell_q: int = 200,           # metacell 数(需<n_genes 才 p<n 良定)
    ci_method: str = "gtest",               # "gtest"=G检验+效应门槛(合理)；"threshold"=固定阈值(旧)
    ci_alpha: float = 0.05,                 # G 检验显著性水平
):
    """两层级超图结构（不训练、不算 embedding，embedding 交给 two_level_s4 / HGNN-VAE）。

    bundle : HeteroBundle（cells / genes / modalities / labels）。
    prior  : PriorBundle（kegg_groups / poswin_groups / ppi_groups，gene-gene 先验）。
    返回 dict：见 README_两层级超图方案.md §11（H_gene/H_cell 为 scipy.sparse.csr）。
    """
    cells = [str(c) for c in bundle.cells]
    genes = [str(g) for g in bundle.genes]
    n_cells, n_genes = len(cells), len(genes)
    gene_index = {g: i for i, g in enumerate(genes)}

    # ---- RNA 表达（基因节点状态，两通道共享的输入；保持原始尺度不预 PCA）----
    rna = next(
        (m for m in bundle.modalities if m.node_type == "gene" and m.bridge_to_gene is None),
        None,
    ) or next((m for m in bundle.modalities if m.node_type == "gene"), None)
    if rna is None:
        raise ValueError("build_two_level_hypergraph 需要至少一个 gene 模态(RNA)。")
    expr = np.nan_to_num(
        rna.feature_table.reindex(index=cells, columns=genes).to_numpy(dtype=np.float64)
    )

    # ---- 第二模态：蛋白 / 甲基化 / 可及性等（细胞通道 kNN + 桥接用）----
    prot = next((m for m in bundle.modalities if m.node_type == "protein"), None)
    protein_names = [str(c) for c in prot.feature_table.columns] if prot is not None else []
    protein = (
        np.nan_to_num(prot.feature_table.reindex(index=cells).to_numpy(dtype=np.float64))
        if prot is not None
        else None
    )
    # 非 protein 的第二模态（甲基化/可及性等），用于建 view2_knn 边
    view2_mod = next(
        (m for m in bundle.modalities
         if m.node_type not in ("gene",) and m.node_type != "protein" and m is not rna),
        None,
    )
    view2_data = None
    if view2_mod is not None:
        v2 = view2_mod.feature_table.reindex(index=cells)
        view2_data = np.nan_to_num(v2.to_numpy(dtype=np.float64), nan=0.0)

    bridge: Dict[str, List[str]] = {}
    if prot is not None and prot.bridge_to_gene:
        for p, targets in prot.bridge_to_gene.items():
            gs = [g for g in dict.fromkeys(targets) if g in gene_index]
            if gs:
                bridge[str(p)] = gs

    # ---- 基因通道 H_gene：pathway / ppi / poswin 先验超边 + 第二模态共变超边 ----
    gmax = max(2, int(max_edge_fraction * n_genes))
    gene_edges_list: List = []
    seen_g = set()
    sources = []
    if "pathway" in gene_edges:
        sources.append(("pathway", prior.kegg_groups or {}, None))
    if "ppi" in gene_edges:
        sources.append(("ppi", prior.ppi_groups or {}, ppi_max_members))
    if "poswin" in gene_edges:
        sources.append(("poswin", prior.poswin_groups or {}, None))
    if "grn" in gene_edges and grn_star_edges is None:   # regulon set 超边（默认）
        sources.append(("grn", grn_groups or {}, grn_max_members))
    grn_sign_map: Dict[str, float] = {}                  # 星图模式：边名→符号 ±1，装配后覆盖 W_gene
    if grn_star_edges:                                   # GRN 星图：每条 TF→target = 一条 2 成员边
        n_star = 0
        for tf, tg, etype, sgn in grn_star_edges:
            if tf not in gene_index or tg not in gene_index:
                continue
            idx = sorted({gene_index[tf], gene_index[tg]})
            if len(idx) != 2:
                continue
            key = frozenset(idx)
            if key in seen_g:
                continue
            seen_g.add(key)
            name = f"grn::{tf}->{tg}"
            gene_edges_list.append(([(i, 1.0) for i in idx], name, etype))
            if grn_signed:
                grn_sign_map[name] = float(sgn)
            n_star += 1
        print(f"[GRN·星图] {n_star} 条 2 成员 TF→target 边"
              + ("（带 stim/inhib 符号）" if grn_signed else "（无符号）"))
    for etype, groups, cap in sources:
        for gname, gs in groups.items():
            idx = [gene_index[g] for g in gs if g in gene_index]
            if cap is not None and len(idx) > cap:
                idx = idx[:cap]
            idx = sorted(set(idx))
            if not (2 <= len(idx) <= gmax):
                continue
            key = frozenset(idx)
            if key in seen_g:
                continue
            seen_g.add(key)
            gene_edges_list.append(([(i, 1.0) for i in idx], str(gname), etype))

    # 机制性跨模态边（中心法则）：covar 裸相关边已删除（第二模态信号改由基因节点状态承载，
    # 见下方 gene_features；相关性建边会让同一信号重复计入且非机制性）。
    # 取而代之：对有 bridge_to_gene 的模态（如 CITE_seq ADT→基因），每个映射到 ≥2 个 panel 内
    # 基因的蛋白 → 一条共调控基因超边（亚基/同源基因被同一蛋白测量，生物上确为共调控单元）。
    # 映射到单基因的蛋白跳过：其中心法则关系已在基因节点状态里（自环无意义）。
    # 1:1 同名模态（PEA/SCoPE2 蛋白、scNMT 甲基化）无 bridge → 不产边，机制全在节点状态。
    if gene_mechanistic_edges:
        n_mech = 0
        for m in bundle.modalities:
            if m is rna or not getattr(m, "bridge_to_gene", None):
                continue
            for p, targets in m.bridge_to_gene.items():
                idx = sorted({gene_index[g] for g in dict.fromkeys(targets) if g in gene_index})
                if not (2 <= len(idx) <= gmax):
                    continue
                key = frozenset(idx)
                if key in seen_g:
                    continue
                seen_g.add(key)
                gene_edges_list.append(
                    ([(i, 1.0) for i in idx], f"mech::{m.name}::{p}", "mech_bridge"))
                n_mech += 1
        print(f"[机制边·bridge] {n_mech} 条共调控基因超边（蛋白→≥2 基因；中心法则）")

    # 多视图一致共调控模块超边（数据驱动，无先验）：RNA∧甲基化∧可及性等多模态视图里
    # 同时强相关的基因 → 共识模块；缺测视图弃权。scNMT(合成名,先验全失配)的唯一基因边来源。
    if multiview_modules:
        mv_views = [expr]  # RNA 视图(细胞×基因)
        for m in bundle.modalities:
            if m is rna:
                continue
            col_ov = len(set(map(str, m.feature_table.columns)) & set(genes))
            if col_ov < 2:
                continue  # 非基因对齐模态(如 CITE 13 ADT)跳过
            mv_views.append(
                m.feature_table.reindex(index=cells, columns=genes).to_numpy(dtype=np.float64))
        mv_edges = _multiview_module_edges(
            mv_views, genes, gene_index, corr_thresh=mv_corr_thresh, min_obs=mv_min_obs,
            min_views=min(mv_min_views, len(mv_views)), top_k=mv_top_k,
            max_members=int(max_edge_fraction * n_genes), gmax=gmax, seen=seen_g)
        gene_edges_list.extend(mv_edges)
        print(f"[多视图模块] {len(mv_views)} 视图 → {len(mv_edges)} 条共识模块超边"
              f"（≥{mv_min_views} 视图一致, |r|≥{mv_corr_thresh}, n_obs≥{mv_min_obs}）")

    # 多组学边构建：按模态权威给已建超边成员重新赋权（蛋白管复合物、RNA 管通路/调控；缺测回退 RNA）
    if multiomics_edges and gene_edges_list:
        mo_views = {"rna": expr}
        for m in bundle.modalities:
            if m is rna:
                continue
            ov = len(set(map(str, m.feature_table.columns)) & set(genes))
            v = m.feature_table.reindex(index=cells, columns=genes).to_numpy(dtype=np.float64)
            key = "protein" if m.node_type == "protein" else m.name
            if ov >= 2:
                mo_views[key] = v
            elif m.node_type == "protein":
                mo_views["protein"] = v   # 非基因对齐(如 CITE ADT)→近全 NaN→该类边回退 RNA
        gene_edges_list = apply_multiomics_edge_weights(gene_edges_list, genes, gene_index, mo_views)
        ws = [w for mem, _, _ in gene_edges_list for _, w in mem]
        print(f"[多组学边构建] {len(gene_edges_list)} 条边成员按模态权威赋权"
              f"（视图 {list(mo_views)}；成员权重 均值{np.mean(ws):.2f} 范围[{min(ws):.2f},{max(ws):.2f}]）")

    H_gene, W_gene, gene_edge_names, gene_edge_types = _assemble_incidence(
        gene_edges_list, n_genes
    )
    if grn_sign_map:   # 星图带符号：抑制边权置 -1，喂带符号卷积（dv 用 |w|、消息按 w 带符号缩放）
        W_gene = np.asarray([grn_sign_map.get(nm, float(w)) for nm, w in
                             zip(gene_edge_names, W_gene)], dtype=np.float64)
        n_inhib = int((W_gene < 0).sum())
        print(f"[GRN·星图] 带符号 W_gene：{n_inhib} 条抑制边(W=-1) / {len(W_gene)} 条基因边")

    # ── 因果化超图构建：对 KEGG/PPI/poswin/covar 全部基因边做 CI 验真 ──
    #   基因变量非共线、细胞当样本(数千)，CI 良定。c_e=不可分离成员对占比(真边置信度)。
    #   硬删 c_e≈0 的假边 → 降边数；保留边的 c_e 作软权 + 可学边门的先验。
    #   基因因果重要性 → 细胞相似度的基因加权(因果性流到细胞层，不在细胞上硬做 CI)。
    gene_causal_prior = None
    gene_importance = None
    cell_space_input = expr
    if causal_construct and H_gene.shape[1] > 0:
        from phasehyper.hypergraph.pc_prune import (
            causal_edge_confidence, node_importance_from_edges)
        c_gene, cinfo = causal_edge_confidence(
            H_gene, expr.T, ci_threshold=pc_ci_threshold, max_order=pc_max_order,
            max_pairs=pc_max_pairs, max_cond=pc_max_cond, n_bins=pc_n_bins,
            ci_method=ci_method, ci_alpha=ci_alpha)
        n0 = H_gene.shape[1]
        if causal_drop_zero:
            keep = c_gene > 1e-9
            if causal_random_baseline:
                # 对照：丢同样多的边，但随机选（c_e 仍按原值保留给软权/门，仅 keep 随机化）
                rng = np.random.RandomState(causal_random_seed)
                n_keep = int(keep.sum())
                keep = np.zeros(n0, dtype=bool)
                keep[rng.choice(n0, n_keep, replace=False)] = True
                print(f"  [对照·随机基线] 随机保留 {n_keep}/{n0} 条基因边（隔离因果选择质量）")
            if keep.sum() < n0:
                ki = np.where(keep)[0]
                H_gene = H_gene.tocsc()[:, ki].tocsr()
                W_gene = np.asarray(W_gene)[ki]
                gene_edge_names = [gene_edge_names[i] for i in ki]
                gene_edge_types = [gene_edge_types[i] for i in ki]
                c_gene = c_gene[ki]
        gene_causal_prior = c_gene.astype(np.float32)
        gene_importance = node_importance_from_edges(H_gene, c_gene)
        print(f"  [因果验真·gene] CI 验真 {n0} 条基因边 → 删假边 {n0 - H_gene.shape[1]}，"
              f"留 {H_gene.shape[1]}（{cinfo['separable_pairs']}/{cinfo['tested_pairs']} 对可分离/假）")
        if causal_cell and causal_cell_space:
            # 因果加权基因空间：重要性低(伪共表达)的基因被压低，细胞相似度由因果中枢基因主导
            gw = causal_gene_floor + (1.0 - causal_gene_floor) * gene_importance
            cell_space_input = expr * gw[np.newaxis, :].astype(np.float32)

    # ---- 细胞通道 H_cell：rna_knn（PCA50→kNN）/ adt_knn（13 维直接 kNN）----
    cmax = max(2, int(max_edge_fraction * n_cells))
    cell_edges_list: List = []
    seen_c = set()

    def _add_knn(members_per_cell, etype):
        for members in members_per_cell:
            idxs = sorted({m[0] for m in members})
            if not (2 <= len(idxs) <= cmax):
                continue
            key = (etype, frozenset(idxs))
            if key in seen_c:
                continue
            seen_c.add(key)
            cell_edges_list.append((members, f"{etype}::{len(cell_edges_list)}", etype))

    if "rna_knn" in cell_edges:
        rna_space = _pca_reduce_np(cell_space_input, pca_dim)  # 因果加权基因空间(若开)，PCA 只定邻居
        _add_knn(_knn_edge_members(rna_space, rna_knn_k, knn_weight), "rna_knn")
    if "adt_knn" in cell_edges and protein is not None:
        _add_knn(_knn_edge_members(protein, adt_knn_k, knn_weight), "adt_knn")
    if "adt_knn" in cell_edges and protein is None and view2_data is not None:
        v2_space = _pca_reduce_np(view2_data, pca_dim)
        _add_knn(_knn_edge_members(v2_space, adt_knn_k, knn_weight), "view2_knn")
    if "cci" in cell_edges and cci_lr_pairs is not None:
        pairs = cci_lr_pairs
        if hasattr(pairs, "columns"):  # DataFrame(ligand, receptor)
            cols = list(pairs.columns)
            pairs = list(zip(pairs[cols[0]], pairs[cols[1]]))
        if cci_pairwise:
            cci_members, n_used = _build_cci_bipartite_edges(
                expr, gene_index, pairs,
                sender_q=cci_sender_quantile, receiver_q=cci_receiver_quantile,
                keep_frac=cci_keep_frac, per_side_cap=cci_pairwise_cap,
            )
            _add_knn(cci_members, "cci")
            print(f"  [CCI·二阶图] 保留 top-{cci_keep_frac:.0%} LR 对，每侧 top-{cci_pairwise_cap} "
                  f"→ {len(cci_members)} 条 2 成员边 (来自 {n_used} 个 LR 对)")
        else:
            cci_members, n_used = _build_cci_edges(
                expr, gene_index, pairs,
                sender_q=cci_sender_quantile,
                receiver_q=cci_receiver_quantile,
                max_members=cci_max_members if cci_max_members is not None else cmax,
                keep_frac=cci_keep_frac,
            )
            _add_knn(cci_members, "cci")
            print(f"  [CCI] CellChat 式：保留 top-{cci_keep_frac:.0%} 强通讯 LR 对 → {n_used} 条边 "
                  f"(sender_q={cci_sender_quantile}, receiver_q={cci_receiver_quantile})")
    if use_wnn:
        pass
    H_cell, W_cell, cell_edge_names, cell_edge_types = _assemble_incidence(
        cell_edges_list, n_cells
    )

    # ── 构建期 CMI 静态剪枝（用原始 expr 当样本对关联矩阵打分，剪冗余整边）──
    #   method="gaussian"：高斯偏相关紧凑度分（快，线性）。
    #   method="pc"      ：PC 式条件独立检验（非参 CMI + 分离集，真因果删边，较慢）。
    def _prune_channel(H, W, names, types, profiles, prunable):
        if build_prune_method == "pc":
            from phasehyper.hypergraph.pc_prune import prune_incidence_pc
            H2, W2, n2, t2, _k, pcinfo = prune_incidence_pc(
                H, W, names, types, profiles, prunable_mask=prunable,
                prune_frac=build_prune_frac, min_keep_frac=build_prune_min_keep,
                ci_threshold=pc_ci_threshold, max_order=pc_max_order,
                max_pairs=pc_max_pairs, max_cond=pc_max_cond, n_bins=pc_n_bins)
            return H2, W2, n2, t2, pcinfo
        H2, W2, n2, t2, _k = prune_incidence(
            H, W, names, types, profiles, prunable_mask=prunable,
            prune_frac=build_prune_frac, min_keep_frac=build_prune_min_keep)
        return H2, W2, n2, t2, None

    # 细胞通道：profile=expr（细胞×基因，基因维当样本）。
    if build_prune_cell and H_cell.shape[1] > 0:
        n0 = H_cell.shape[1]
        H_cell, W_cell, cell_edge_names, cell_edge_types, pcinfo = _prune_channel(
            H_cell, W_cell, cell_edge_names, cell_edge_types, expr, None)
        tail = f"（PC: {pcinfo['separable_pairs']}/{pcinfo['tested_pairs']} 对可分离）" if pcinfo else ""
        print(f"  [构建剪枝·cell·{build_prune_method}] 剪冗余边 {n0 - H_cell.shape[1]}/{n0} "
              f"→ 留 {H_cell.shape[1]} {tail}")
    # 基因通道：只剪 covar_* 共变边（保护 KEGG/PPI/poswin 先验），profile=expr.T（基因×细胞）。
    if build_prune_gene and H_gene.shape[1] > 0:
        gt = np.asarray(gene_edge_types)
        prunable = np.array([str(t).startswith("covar_") for t in gt], dtype=bool)
        if prunable.any():
            n0 = H_gene.shape[1]
            H_gene, W_gene, gene_edge_names, gene_edge_types, pcinfo = _prune_channel(
                H_gene, W_gene, gene_edge_names, gene_edge_types, expr.T, prunable)
            tail = f"（PC: {pcinfo['separable_pairs']}/{pcinfo['tested_pairs']} 对可分离）" if pcinfo else ""
            print(f"  [构建剪枝·gene·{build_prune_method}] 剪 covar 冗余边 {n0 - H_gene.shape[1]}/{n0} {tail}")

    # ── 细胞边因果验真：c_e^cell = 因果加权基因空间内聚度 / 原始空间内聚度 ──
    #   不碰病态的 cell-cell CI；真边在只看因果基因时仍紧致，假边一上因果权就散。
    cell_causal_prior = None
    if causal_construct and causal_cell and H_cell.shape[1] > 0 and gene_importance is not None:
        if causal_cell_method == "metacell":
            # 方案 A：metacell 辅助真 CI（p<n 良定，CASCAT 级严谨度）
            from phasehyper.hypergraph.pc_prune import metacell_ci_cell_confidence
            c_cell, _mlabels, _q = metacell_ci_cell_confidence(
                H_cell, expr, space=_pca_reduce_np(cell_space_input, pca_dim),
                q=causal_metacell_q, ci_threshold=pc_ci_threshold, max_order=pc_max_order,
                max_cond=pc_max_cond, n_bins=pc_n_bins, seed=causal_random_seed,
                ci_method=ci_method, ci_alpha=ci_alpha)
            _cell_tag = f"metacell-CI(q={_q})"
        else:
            from phasehyper.hypergraph.pc_prune import cell_edge_causal_cohesion
            c_cell = cell_edge_causal_cohesion(H_cell, expr, cell_space_input)
            _cell_tag = "cohesion"
        if causal_random_baseline:
            rng = np.random.RandomState(causal_random_seed + 1)
            c_cell = c_cell[rng.permutation(len(c_cell))]  # 打乱→隔离细胞因果选择质量
        n0 = H_cell.shape[1]
        if causal_cell_drop_frac > 0:
            keep = c_cell >= np.quantile(c_cell, causal_cell_drop_frac)
            if keep.sum() < n0:
                ki = np.where(keep)[0]
                H_cell = H_cell.tocsc()[:, ki].tocsr()
                W_cell = np.asarray(W_cell)[ki]
                cell_edge_names = [cell_edge_names[i] for i in ki]
                cell_edge_types = [cell_edge_types[i] for i in ki]
                c_cell = c_cell[ki]
            print(f"  [因果验真·cell·{_cell_tag}] 验真 → 删低置信边 {n0 - H_cell.shape[1]}/{n0}，"
                  f"留 {H_cell.shape[1]}")
        else:
            print(f"  [因果验真·cell·{_cell_tag}] 软权 c_e^cell（均值 {c_cell.mean():.3f}），不硬删")
        cell_causal_prior = c_cell.astype(np.float32)

    g_types = np.asarray(gene_edge_types)
    c_types = np.asarray(cell_edge_types)

    def _gmask(name):
        return (g_types == name) if g_types.size else np.zeros(0, dtype=bool)

    def _cmask(name):
        return (c_types == name) if c_types.size else np.zeros(0, dtype=bool)

    true_cell_labels = (
        np.asarray(bundle.labels, dtype=np.int64) if bundle.labels is not None else None
    )

    # ---- 多模态细胞特征：拼接所有模态供 cell_in 用（比纯 expr 更丰富）----
    #   不插补：缺失值置 0（中性占位），并并行输出观测掩码（1=观测,0=缺失）供 mask-aware 输入用。
    cell_feat_parts = [expr.astype(np.float32)]
    cell_mask_parts = [np.ones_like(expr, dtype=np.float32)]   # RNA 视为观测（稠密骨架）
    for m in bundle.modalities:
        if m is rna:
            continue
        raw = m.feature_table.reindex(index=cells).to_numpy(dtype=np.float64)
        cell_feat_parts.append(np.nan_to_num(raw, nan=0.0).astype(np.float32))
        cell_mask_parts.append(np.isfinite(raw).astype(np.float32))   # NaN(无测量)→0
    cell_features = np.concatenate(cell_feat_parts, axis=1) if len(cell_feat_parts) > 1 else expr.astype(np.float32)
    cell_features_mask = (np.concatenate(cell_mask_parts, axis=1)
                          if len(cell_mask_parts) > 1 else cell_mask_parts[0])
    cell_features_mask = _informative_mask_cols(cell_features_mask)   # 只留覆盖真可变的列

    # ---- 基因节点状态（中心法则整合点）：基因通道输入 gene_in 的特征矩阵 ----
    #   始终输出（关或单模态时 == expr.T，保证 ratio/sim 与旧行为逐字节一致、构造器不 KeyError）。
    #   gene_multiomics=True：每个基因 g 的"细胞画像"扩为 RNA_g ⊕ 第二模态_g（同基因对齐）。
    #     - 1:1 同名模态（PEA/SCoPE2 蛋白、scNMT 甲基化）：reindex(columns=genes) 直接对齐。
    #     - bridge 模态（CITE_seq ADT）：按 bridge_to_gene 把蛋白列填到对应基因列，无蛋白基因补零。
    #   不插补：第二/三模态"无测量"(NaN)与 bridge 未映射基因 → 值置 0(中性) + 掩码=0，
    #   并行输出 gene_features_mask（1=观测,0=缺失）供模型 mask-aware 显式区分"缺失"vs"真低值"。
    gene_feat_parts = [expr.T.astype(np.float32)]                 # RNA: (n_genes × n_cells)
    gene_mask_parts = [np.ones((n_genes, n_cells), dtype=np.float32)]  # RNA 视为观测
    if gene_multiomics:
        for m in bundle.modalities:
            if m is rna:
                continue
            if getattr(m, "bridge_to_gene", None):
                blk = np.zeros((n_cells, n_genes), dtype=np.float32)
                obs = np.zeros((n_cells, n_genes), dtype=np.float32)   # bridge 未映射基因→0(缺失)
                pcols = {str(c): i for i, c in enumerate(m.feature_table.columns)}
                praw = m.feature_table.reindex(index=cells).to_numpy(dtype=np.float64)
                pmat = np.nan_to_num(praw, nan=0.0)
                pobs = np.isfinite(praw)
                for p, targets in m.bridge_to_gene.items():
                    pi = pcols.get(str(p))
                    if pi is None:
                        continue
                    for g in targets:
                        gi = gene_index.get(g)
                        if gi is not None:
                            blk[:, gi] = pmat[:, pi]
                            obs[:, gi] = pobs[:, pi].astype(np.float32)
                blk2 = _coverage_aware_standardize(blk) if gene_mod_coverage_std else blk.astype(np.float32)
                gene_feat_parts.append(blk2.T)          # (n_genes × n_cells)
                gene_mask_parts.append(obs.T)
            else:
                raw = m.feature_table.reindex(index=cells, columns=genes).to_numpy(dtype=np.float64)
                v = np.nan_to_num(raw, nan=0.0).astype(np.float32)
                # 覆盖感知标准化：甲基化等"无覆盖均值填充"占位置 0，观测值按自身分布 z-score。
                v2 = _coverage_aware_standardize(v) if gene_mod_coverage_std else v
                gene_feat_parts.append(v2.T)            # (n_genes × n_cells)
                gene_mask_parts.append(np.isfinite(raw).astype(np.float32).T)   # NaN(无测量)→0
    gene_features = (np.concatenate(gene_feat_parts, axis=1)
                     if len(gene_feat_parts) > 1 else gene_feat_parts[0])
    gene_features_mask = (np.concatenate(gene_mask_parts, axis=1)
                          if len(gene_mask_parts) > 1 else gene_mask_parts[0])
    gene_features_mask = _informative_mask_cols(gene_features_mask)   # 只留覆盖真可变的列

    return {
        # 公共
        "cells": cells,
        "genes": genes,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "expr": expr.astype(np.float32),  # (n_cells × n_genes) RNA 表达（重构目标）
        "cell_features": cell_features,   # (n_cells × n_total_features) 多模态拼接（cell_in 输入）
        "n_cell_features": int(cell_features.shape[1]),
        "cell_features_mask": cell_features_mask,  # 同形观测掩码(1=观测,0=缺失)，mask-aware 输入用
        "gene_features": gene_features,   # (n_genes × n_cells*n_mod) 中心法则节点状态（gene_in 输入）
        "n_gene_features": int(gene_features.shape[1]),
        "gene_features_mask": gene_features_mask,  # 同形观测掩码，不插补：显式标记缺失而非猜值
        "protein": (protein.astype(np.float32) if protein is not None else None),
        "protein_names": protein_names,
        # 基因通道（Level 2）
        "H_gene": H_gene,
        "W_gene": W_gene,
        "gene_edge_names": gene_edge_names,
        "gene_edge_types": gene_edge_types,
        "n_gene_edges": int(H_gene.shape[1]),
        # 因果化超图：基因边因果置信度 c_e（可学边门先验）+ 基因因果重要性
        "gene_causal_prior": gene_causal_prior,
        "gene_importance": gene_importance,
        "cell_causal_prior": cell_causal_prior,
        "gene_pathway_mask": _gmask("pathway"),
        "gene_ppi_mask": _gmask("ppi"),
        "gene_poswin_mask": _gmask("poswin"),
        "gene_grn_mask": _gmask("grn"),
        # 细胞通道（Level 1）
        "H_cell": H_cell,
        "W_cell": W_cell,
        "cell_edge_names": cell_edge_names,
        "cell_edge_types": cell_edge_types,
        "n_cell_edges": int(H_cell.shape[1]),
        "cell_rna_knn_mask": _cmask("rna_knn"),
        "cell_adt_knn_mask": _cmask("adt_knn"),
        "cell_cci_mask": _cmask("cci"),
        # 耦合 & 通道配置（下游据此产 cell×128 ⊕ cell×128 → cell×256）
        "couple": couple,
        "gene_channel_out": int(gene_channel_out),
        "cell_channel_out": int(cell_channel_out),
        "gene_pool": gene_pool,
        "gene_node_state": gene_node_state,
        "smooth_steps": int(smooth_steps),
        "cell_feat_with_protein": bool(cell_feat_with_protein),
        # 桥 & 真值
        "bridge": bridge,
        "true_cell_labels": true_cell_labels,
        "label_map": bundle.label_map,
    }
