"""模拟数据（ratio 任务）的加载与无监督先验构建。

只读合法输入（mixed_expression / cell_info / gene_info），不碰 ground_truth/double_check：
  - load_sim_bundle              : mixed_expression + cell_info → HeteroBundle
  - self_cluster_cells           : log-expr PCA→KMeans，silhouette 选 k（不读 cell_type）
  - build_gene_prior_from_geneinfo: gene_info 位置/通路 → 基因相似度 + 超边分组
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from phasehyper.schemas import HeteroBundle, ModalitySpec


def load_sim_bundle(sim_dir: Path, log_transform: bool = True) -> HeteroBundle:
    """从 simulation input 目录构建 HeteroBundle。

    log_transform=True：对 mixed_expression 做 log(x+1)，对齐 log 空间混合生成机制
        log(expr) = w·log(pat) + (1-w)·log(mat) + noise
    """
    sim_dir = Path(sim_dir)
    expr_df = pd.read_csv(sim_dir / "mixed_expression.csv", index_col=0)
    expr_df.index = expr_df.index.astype(str).str.strip()
    expr_df.columns = [str(c).strip() for c in expr_df.columns]

    cell_info = pd.read_csv(sim_dir / "cell_info.csv")
    cell_info["cell_id"] = cell_info["cell_id"].astype(str).str.strip()

    cells = list(cell_info["cell_id"])
    expr_df = expr_df.reindex(index=cells)
    genes = list(expr_df.columns)

    ct = cell_info.set_index("cell_id").loc[cells, "cell_type"].values
    unique_ct = sorted(set(ct))
    ct_to_idx = {v: i for i, v in enumerate(unique_ct)}
    labels = np.array([ct_to_idx[c] for c in ct])

    if log_transform:
        arr = np.log1p(np.clip(expr_df.values.astype(np.float64), 0, None))
        expr_df = pd.DataFrame(arr, index=expr_df.index, columns=expr_df.columns)
        print(f"  [log-transform] log(expr+1): range=[{arr.min():.3f}, {arr.max():.3f}]")

    modality = ModalitySpec(name="RNA", node_type="gene", feature_table=expr_df)
    return HeteroBundle(
        cells=cells, genes=genes, modalities=[modality], dataset_type="ratio_sim",
        labels=labels, label_names=[str(v) for v in unique_ct],
        label_map={i: str(v) for i, v in enumerate(unique_ct)},
    )


def self_cluster_cells(log_expr, k_range=range(4, 16), seed=42):
    """log-expr PCA → KMeans，silhouette 选 k。返回 (labels, k)。不读 cell_type。"""
    Z = PCA(n_components=min(50, log_expr.shape[0] - 1, log_expr.shape[1]),
            random_state=seed).fit_transform(log_expr)
    best_k, best_s, best_lab = None, -1, None
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        s = silhouette_score(Z, lab)
        if s > best_s:
            best_k, best_s, best_lab = k, s, lab
    print(f"  自聚类: 选 k={best_k} (silhouette={best_s:.3f})")
    return best_lab.astype(np.int64), best_k


def build_gene_prior_from_geneinfo(gene_info_path, genes, scale=6.0, path_bump=0.5, cross_chr=0.0):
    """位置衰减 + 同通路加成 → (cov_proxy, kegg_groups, poswin_groups)。盲选默认参数，不读真参。

      similarity = exp(-dist / (chr_span/scale))  同染色体位置衰减
                 + path_bump · [同通路]            同通路加成
    """
    gi = pd.read_csv(gene_info_path)
    gi["gene_id"] = gi["gene_id"].astype(str).str.strip()
    gi = gi.set_index("gene_id").reindex(genes)
    G = len(genes)

    chrom = gi["chromosome"].astype(str).values
    center = gi["center_pos"].astype(float).values
    pathway = gi["pathway"].astype(str).values

    same_chr = chrom[:, None] == chrom[None, :]
    dist = np.abs(center[:, None] - center[None, :])

    chr_span = {}
    for c in np.unique(chrom):
        m = chrom == c
        chr_span[c] = center[m].max() - center[m].min() if m.sum() > 1 else 1.0
    span_vec = np.array([chr_span[c] for c in chrom])
    span_pair = np.maximum(span_vec[:, None], span_vec[None, :])
    decay = span_pair / scale
    decay[decay == 0] = 1.0

    sim = np.full((G, G), cross_chr, dtype=np.float64)
    sim[same_chr] = np.exp(-dist[same_chr] / decay[same_chr])
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)

    P = (pathway[:, None] == pathway[None, :]).astype(np.float64)
    cov_proxy = sim + path_bump * P
    cov_proxy = (cov_proxy + cov_proxy.T) / 2.0
    np.fill_diagonal(cov_proxy, 1.0 + path_bump)
    ev = la.eigvalsh(cov_proxy)
    if ev.min() < 1e-6:
        cov_proxy += np.eye(G) * (abs(ev.min()) + 1e-6)

    poswin_groups, kegg_groups = {}, {}
    thr = 0.3
    for i, g in enumerate(genes):
        nb = [genes[j] for j in range(G) if j != i and sim[i, j] > thr]
        if nb:
            poswin_groups[f"pos::{g}"] = [g] + nb
    pw_to_genes = {}
    for g, pw in zip(genes, pathway):
        pw_to_genes.setdefault(pw, []).append(g)
    for pw, gs in pw_to_genes.items():
        if len(gs) > 1:
            kegg_groups[f"path::{pw}"] = gs

    return cov_proxy, kegg_groups, poswin_groups
