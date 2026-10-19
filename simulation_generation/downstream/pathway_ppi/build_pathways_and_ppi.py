#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 V6 模拟数据:
  A. 对基因做表达聚类 -> 生成模拟 KEGG 通路数据
  B. 基于共表达 / 同通路 / 同模块 / GRN 关系 -> 生成 PPI (蛋白互作) 网络

输入 (从 V6/ 读取):
  combined_true_expression.csv, gene_info.csv, gene_module_membership.csv,
  GRN_edge_table.csv, shared_TF_info.csv
输出 (写到 V6/pathway_ppi/):
  kegg_pathways.csv, gene_pathway_membership.csv, pathway_gene_lists.csv
  ppi_edges.csv, ppi_nodes.csv
  + 若干 PNG 图

用法:
    python3 build_pathways_and_ppi.py
纯 numpy/pandas/scipy/matplotlib 实现 (力导向布局与图指标自实现, 不依赖 networkx).
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, ".."))     # V6/
OUT_DIR = os.path.join(HERE)                             # V6/pathway_ppi/  (脚本自身目录)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_PATHWAYS = 10               # 聚类簇数 = 模拟通路数

# PPI logistic 模型权重 (调稀疏使网络接近真实 PPI: 稀疏 + 无标度)
PPI_INTERCEPT = -7.0
PPI_W_COEXPR = 4.0            # 共表达绝对相关
PPI_W_SAME_PATHWAY = 1.8     # 同通路
PPI_W_SAME_MODULE = 1.4      # 同调控模块
PPI_W_GRN = 3.0              # 存在 GRN 调控关系
PPI_NOISE_SD = 0.4
PPI_SCORE_CUTOFF = 500       # combined_score (0-1000) 保留阈值

# 模拟 KEGG 通路名称池 (name, category, kegg-like id 前缀号)
PATHWAY_POOL = [
    ("MAPK signaling pathway", "Signal transduction", "04010"),
    ("Cell cycle", "Cell growth and death", "04110"),
    ("Wnt signaling pathway", "Signal transduction", "04310"),
    ("Apoptosis", "Cell growth and death", "04210"),
    ("Oxidative phosphorylation", "Energy metabolism", "00190"),
    ("Ribosome biogenesis", "Translation", "03008"),
    ("p53 signaling pathway", "Signal transduction", "04115"),
    ("Citrate cycle (TCA)", "Carbohydrate metabolism", "00020"),
    ("Notch signaling pathway", "Signal transduction", "04330"),
    ("DNA replication", "Replication and repair", "03030"),
    ("Purine metabolism", "Nucleotide metabolism", "00230"),
    ("PI3K-Akt signaling pathway", "Signal transduction", "04151"),
    ("Spliceosome", "Transcription", "03040"),
    ("Fatty acid metabolism", "Lipid metabolism", "01212"),
    ("TGF-beta signaling pathway", "Signal transduction", "04350"),
]

CAT_COLORS = {
    "Signal transduction": "#1f77b4",
    "Cell growth and death": "#ff7f0e",
    "Energy metabolism": "#2ca02c",
    "Translation": "#d62728",
    "Carbohydrate metabolism": "#9467bd",
    "Replication and repair": "#8c564b",
    "Nucleotide metabolism": "#e377c2",
    "Transcription": "#7f7f7f",
    "Lipid metabolism": "#bcbd22",
}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    expr = pd.read_csv(os.path.join(DATA_DIR, "combined_true_expression.csv"),
                       index_col=0)                       # cells x genes
    genes = pd.read_csv(os.path.join(DATA_DIR, "gene_info.csv"))
    membership = pd.read_csv(os.path.join(DATA_DIR, "gene_module_membership.csv"))
    grn = pd.read_csv(os.path.join(DATA_DIR, "GRN_edge_table.csv"))
    tf_info = pd.read_csv(os.path.join(DATA_DIR, "shared_TF_info.csv"))
    cells = pd.read_csv(os.path.join(DATA_DIR, "cell_metadata.csv"))
    return expr, genes, membership, grn, tf_info, cells


# ---------------------------------------------------------------------------
# A. 基因聚类 -> 模拟 KEGG 通路
# ---------------------------------------------------------------------------
def cluster_genes_to_pathways(expr, genes, k, rng):
    gene_ids = list(expr.columns)
    X = expr[gene_ids].to_numpy().T            # genes x cells
    # 每基因 log + z-score
    Xl = np.log1p(X)
    Z = (Xl - Xl.mean(1, keepdims=True)) / (Xl.std(1, keepdims=True) + 1e-8)

    corr = np.corrcoef(Z)
    corr = np.nan_to_num(corr)

    # KMeans 聚类 (在 z-score 表达谱上, 得到大小均衡的通路; 避免层次聚类链式效应)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=int(rng.integers(1e6)), n_init=10)
    labels = km.fit_predict(Z) + 1    # 1-based

    # 给每个簇分配一个 KEGG 通路 (随机不重复)
    uniq = sorted(set(labels))
    pool_idx = rng.permutation(len(PATHWAY_POOL))[:len(uniq)]
    clust_to_path = {}
    for i, c in enumerate(uniq):
        name, cat, num = PATHWAY_POOL[pool_idx[i]]
        pid = f"sim{num}"
        clust_to_path[c] = (pid, name, cat)

    # 组织成表
    rows = []
    for gid, lab in zip(gene_ids, labels):
        pid, name, cat = clust_to_path[lab]
        rows.append({"gene_id": gid, "cluster": int(lab),
                     "pathway_id": pid, "pathway_name": name, "category": cat})
    gp = pd.DataFrame(rows)

    # membership_score = 基因与所属簇质心表达谱的相关
    zdf = pd.DataFrame(Z, index=gene_ids)
    scores = []
    for lab in uniq:
        members = gp[gp["cluster"] == lab]["gene_id"].tolist()
        centroid = zdf.loc[members].mean(0).to_numpy()
        for g in members:
            r = np.corrcoef(zdf.loc[g].to_numpy(), centroid)[0, 1]
            scores.append((g, float(np.nan_to_num(r))))
    smap = dict(scores)
    gp["membership_score"] = gp["gene_id"].map(smap).round(4)

    # pathway 汇总表
    psum = (gp.groupby(["pathway_id", "pathway_name", "category"])
            .agg(n_genes=("gene_id", "count"),
                 mean_membership=("membership_score", "mean"))
            .reset_index())
    psum["mean_membership"] = psum["mean_membership"].round(4)

    # gene lists (分号分隔)
    plist = (gp.groupby(["pathway_id", "pathway_name"])["gene_id"]
             .apply(lambda s: ";".join(s)).reset_index()
             .rename(columns={"gene_id": "genes"}))

    return gp, psum, plist, corr, gene_ids, labels, Z


def plot_pathway_figures(gp, psum, corr, gene_ids, labels, expr, cells, out_dir):
    order = np.argsort(labels)
    corr_o = corr[np.ix_(order, order)]
    labs_o = np.array(labels)[order]
    gids_o = [gene_ids[i] for i in order]
    pmap = gp.set_index("gene_id")

    # 1. 基因相关性热图 (按簇排序) + 通路侧条
    fig = plt.figure(figsize=(9.5, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.04, 1], wspace=0.02)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(corr_o, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"基因表达相关性 (按聚类/通路排序, {len(set(labels))} 个模拟通路)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("Pearson 相关", fontsize=8)

    cat_o = [pmap.loc[g, "category"] for g in gids_o]
    cats = list(CAT_COLORS.keys())
    cat_idx = np.array([cats.index(c) for c in cat_o])
    ax_bar.imshow(cat_idx[:, None], aspect="auto",
                  cmap=ListedColormap([CAT_COLORS[c] for c in cats]),
                  vmin=0, vmax=len(cats) - 1)
    # 簇分界线
    bnd = np.where(np.diff(labs_o) != 0)[0] + 1
    for b in bnd:
        ax.axhline(b - 0.5, color="k", lw=0.5)
        ax.axvline(b - 0.5, color="k", lw=0.5)
        ax_bar.axhline(b - 0.5, color="k", lw=0.5)
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.set_title("通路", fontsize=8)
    used_cats = sorted(set(cat_o))
    fig.legend(handles=[Patch(facecolor=CAT_COLORS[c], label=c) for c in used_cats],
               loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    p = os.path.join(out_dir, "pathway_gene_correlation_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)

    # 2. 每通路基因数条形图
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ps = psum.sort_values("n_genes", ascending=False)
    colors = [CAT_COLORS[c] for c in ps["category"]]
    ax.bar(range(len(ps)), ps["n_genes"], color=colors)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{r.pathway_id}\n{r.pathway_name}" for r in ps.itertuples()],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("基因数"); ax.set_title("模拟 KEGG 通路基因数分布")
    fig.legend(handles=[Patch(facecolor=CAT_COLORS[c], label=c)
                        for c in sorted(set(ps["category"]))],
               loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "pathway_gene_counts.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)

    # 3. 通路 x 细胞类型 平均表达热图
    ct = cells.set_index("cell_id")["cell_type"]
    common = [c for c in expr.index if c in ct.index]
    ex = expr.loc[common]
    exz = (ex - ex.mean(0)) / (ex.std(0) + 1e-8)
    path_ids = sorted(gp["pathway_id"].unique())
    ctypes = sorted(ct.unique())
    M = np.zeros((len(path_ids), len(ctypes)))
    for i, pid in enumerate(path_ids):
        pgenes = [g for g in gp[gp["pathway_id"] == pid]["gene_id"] if g in exz.columns]
        for j, cty in enumerate(ctypes):
            ccells = [c for c in common if ct[c] == cty]
            M[i, j] = exz.loc[ccells, pgenes].to_numpy().mean()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ctypes))); ax.set_xticklabels(ctypes, rotation=30, ha="right")
    pname = psum.set_index("pathway_id")["pathway_name"]
    ax.set_yticks(range(len(path_ids)))
    ax.set_yticklabels([f"{p} {pname[p]}" for p in path_ids], fontsize=8)
    ax.set_title("通路 x 细胞类型 平均表达 (z-score)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("平均 z-score", fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "pathway_celltype_activity.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)


# ---------------------------------------------------------------------------
# B. PPI 网络构建
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_ppi(expr, gp, membership, grn, tf_info, rng):
    gene_ids = list(expr.columns)
    n = len(gene_ids)
    gidx = {g: i for i, g in enumerate(gene_ids)}

    # 共表达绝对相关
    X = np.log1p(expr[gene_ids].to_numpy().T)
    corr = np.abs(np.nan_to_num(np.corrcoef(X)))

    # 同通路
    path_of = gp.set_index("gene_id")["pathway_id"].to_dict()
    # 同模块
    mod_of = membership.set_index("gene_id")["module_id"].to_dict()
    # GRN 基因-基因关系 (TF_id -> 其 gene_id; target_gene 已是 gene_id)
    tf2gene = tf_info.set_index("TF_id")["gene_id"].to_dict()
    grn_pairs = set()
    for r in grn.itertuples():
        src = tf2gene.get(r.TF_id)
        dst = r.target_gene
        if src in gidx and dst in gidx and src != dst:
            a, b = sorted((gidx[src], gidx[dst]))
            grn_pairs.add((a, b))

    # 逐对计算互作概率 (无向, i<j)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = gene_ids[i], gene_ids[j]
            co = corr[i, j]
            same_p = int(path_of.get(gi) is not None and path_of.get(gi) == path_of.get(gj))
            same_m = int(mod_of.get(gi) is not None and mod_of.get(gi) == mod_of.get(gj))
            grn_l = int((i, j) in grn_pairs)
            logit = (PPI_INTERCEPT + PPI_W_COEXPR * co + PPI_W_SAME_PATHWAY * same_p
                     + PPI_W_SAME_MODULE * same_m + PPI_W_GRN * grn_l
                     + rng.normal(0, PPI_NOISE_SD))
            prob = sigmoid(logit)
            score = int(round(prob * 1000))
            if score >= PPI_SCORE_CUTOFF:
                edges.append((gi, gj, round(co, 4), same_p, same_m, grn_l, score))

    edf = pd.DataFrame(edges, columns=[
        "protein_a", "protein_b", "coexpression", "same_pathway",
        "same_module", "grn_link", "combined_score"])

    # 邻接矩阵 (加权 = score/1000)
    A = np.zeros((n, n))
    for r in edf.itertuples():
        a, b = gidx[r.protein_a], gidx[r.protein_b]
        A[a, b] = A[b, a] = r.combined_score / 1000.0

    deg = (A > 0).sum(1)
    wdeg = A.sum(1)
    # 局部聚类系数
    clust = np.zeros(n)
    Abin = (A > 0).astype(int)
    for i in range(n):
        nb = np.where(Abin[i] > 0)[0]
        if len(nb) < 2:
            continue
        sub = Abin[np.ix_(nb, nb)]
        clust[i] = sub.sum() / (len(nb) * (len(nb) - 1))

    tf_set = set(tf_info["gene_id"])
    hub_thr = np.percentile(deg, 90) if n else 0
    nodes = pd.DataFrame({
        "gene_id": gene_ids,
        "pathway_id": [path_of.get(g, "NA") for g in gene_ids],
        "degree": deg.astype(int),
        "weighted_degree": wdeg.round(4),
        "clustering_coef": clust.round(4),
        "is_TF": [int(g in tf_set) for g in gene_ids],
        "is_hub": (deg >= max(hub_thr, 1)).astype(int),
    })
    nodes = nodes.merge(gp[["gene_id", "pathway_name", "category"]], on="gene_id", how="left")

    return edf, nodes, A, gene_ids


def spring_layout(A, iterations=500, seed=0, k_scale=2.5, gravity=0.08):
    """网络布局: 优先用 networkx (spring/kamada), 否则回退到自实现 FR."""
    n = A.shape[0]
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    G.add_edge(i, j, weight=float(A[i, j]))
        pos_d = nx.spring_layout(G, k=0.9, iterations=300, weight="weight",
                                 seed=seed)
        pos = np.array([pos_d[i] for i in range(n)])
        pos -= pos.mean(0)
        span = np.percentile(np.abs(pos), 98) + 1e-9
        return np.clip(pos / span, -1.3, 1.3)
    except Exception as e:  # 回退
        print("[warn] networkx 不可用, 回退自实现布局:", e)
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))
    k = k_scale / np.sqrt(max(n, 1))
    t0 = 0.1
    W = A
    for it in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9
        u = delta / dist[..., None]
        rep = (k * k / dist)[..., None] * u
        rep = rep.sum(1)
        att = ((dist * dist / k) * W)[..., None] * u
        att = att.sum(1)
        disp = rep - att - gravity * pos
        dl = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        t = t0 * (1 - it / iterations) + 1e-3
        step = np.minimum(dl, t)
        pos += disp / dl[..., None] * step[..., None]
    pos -= pos.mean(0)
    span = np.percentile(np.abs(pos), 97) + 1e-9
    return np.clip(pos / span, -1.3, 1.3)


def plot_ppi(edf, nodes, A, gene_ids, out_dir, seed):
    gidx = {g: i for i, g in enumerate(gene_ids)}
    pos = spring_layout(A, iterations=250, seed=seed)

    cats = nodes.set_index("gene_id")["category"].fillna("Signal transduction")
    node_color = [CAT_COLORS.get(cats.get(g, ""), "#cccccc") for g in gene_ids]
    deg = nodes.set_index("gene_id")["degree"]
    sizes = 30 + 25 * np.sqrt(deg.reindex(gene_ids).fillna(0).to_numpy())
    is_tf = nodes.set_index("gene_id")["is_TF"].reindex(gene_ids).fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 11))
    # 边
    for r in edf.itertuples():
        a, b = gidx[r.protein_a], gidx[r.protein_b]
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                color="gray", lw=0.3 + 1.2 * r.combined_score / 1000.0,
                alpha=0.10 + 0.35 * r.combined_score / 1000.0, zorder=1)
    # 节点
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=node_color,
               edgecolors="black", linewidths=0.4, zorder=2)
    # TF 用方框标记
    tf_mask = is_tf > 0
    ax.scatter(pos[tf_mask, 0], pos[tf_mask, 1], s=sizes[tf_mask] + 40,
               facecolors="none", edgecolors="black", linewidths=1.3,
               marker="s", zorder=3, label="TF")
    # hub 标注基因名
    hub_ids = nodes.sort_values("degree", ascending=False).head(8)["gene_id"]
    for g in hub_ids:
        i = gidx[g]
        ax.text(pos[i, 0], pos[i, 1], g.replace("gene_", "g"), fontsize=6,
                ha="center", va="center", zorder=4)

    ax.set_title(f"PPI 蛋白互作网络 ({len(gene_ids)} 节点, {len(edf)} 边; 颜色=通路类别, 大小=度, 方框=TF)",
                 fontsize=12)
    ax.axis("off")
    used = sorted(set(cats.dropna()))
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in used if c in CAT_COLORS]
    handles.append(plt.Line2D([0], [0], marker="s", color="black", lw=0,
                              markerfacecolor="none", label="TF"))
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout()
    p = os.path.join(out_dir, "ppi_network.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)

    # 度分布
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    d = nodes["degree"].to_numpy()
    axs[0].hist(d, bins=20, color="#4c72b0", edgecolor="white")
    axs[0].set_xlabel("度 (degree)"); axs[0].set_ylabel("节点数")
    axs[0].set_title("PPI 度分布")
    vals, cnts = np.unique(d[d > 0], return_counts=True)
    axs[1].scatter(np.log10(vals), np.log10(cnts), color="#c44e52")
    axs[1].set_xlabel("log10(度)"); axs[1].set_ylabel("log10(节点数)")
    axs[1].set_title("度分布 (log-log, 检查无标度性)")
    fig.tight_layout()
    p = os.path.join(out_dir, "ppi_degree_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)

    # 邻接热图 (按通路排序)
    order = np.argsort([nodes.set_index("gene_id")["pathway_id"].get(g, "") for g in gene_ids])
    Ao = A[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Ao, cmap="magma", aspect="auto")
    ax.set_title("PPI 邻接矩阵 (按通路排序, 值=combined_score/1000)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("互作强度", fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "ppi_adjacency_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("[saved]", p)


# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    expr, genes, membership, grn, tf_info, cells = load_data()
    print(f"[info] 表达 {expr.shape[0]} 细胞 x {expr.shape[1]} 基因; "
          f"GRN {len(grn)} 边; TF {len(tf_info)}")

    # A. 聚类 -> 通路
    gp, psum, plist, corr, gene_ids, labels, Z = cluster_genes_to_pathways(
        expr, genes, N_PATHWAYS, rng)
    gp.to_csv(os.path.join(OUT_DIR, "gene_pathway_membership.csv"), index=False)
    psum.to_csv(os.path.join(OUT_DIR, "kegg_pathways.csv"), index=False)
    plist.to_csv(os.path.join(OUT_DIR, "pathway_gene_lists.csv"), index=False)
    print(f"[A] 生成 {len(psum)} 个模拟通路, 覆盖 {len(gp)} 基因")
    plot_pathway_figures(gp, psum, corr, gene_ids, labels, expr, cells, OUT_DIR)

    # B. PPI
    edf, nodes, A, gene_ids2 = build_ppi(expr, gp, membership, grn, tf_info, rng)
    edf.to_csv(os.path.join(OUT_DIR, "ppi_edges.csv"), index=False)
    nodes.to_csv(os.path.join(OUT_DIR, "ppi_nodes.csv"), index=False)
    dens = 2 * len(edf) / (len(gene_ids2) * (len(gene_ids2) - 1))
    print(f"[B] PPI: {len(gene_ids2)} 节点, {len(edf)} 边, 密度 {dens:.3f}, "
          f"平均度 {nodes['degree'].mean():.1f}, hub {int(nodes['is_hub'].sum())}")
    plot_ppi(edf, nodes, A, gene_ids2, OUT_DIR, SEED)

    print("\n完成. 输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
