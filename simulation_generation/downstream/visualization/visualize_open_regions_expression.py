#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 V6 模拟输出:
  1. 每条染色体上的开放区域轨道图 (父本 / 母本 / 混合), 沿基因组坐标.
  2. 每条染色体的基因表达相关性 (基因按位置排序) + 开放区域轨道注释 (父本/母本/混合).
  3. 基因 x 细胞 表达热图 + 染色体开放区域轨道注释 (父本/母本/混合).

用法:
    python3 visualize_open_regions_expression.py
默认从上级目录 (V6) 读取 CSV, 输出 PNG 到本目录 (visualization/).
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

# 中文字体 (macOS)
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, ".."))   # V6/
OUT_DIR = HERE                                          # V6/visualization/

# 使用哪套表达矩阵做热图与相关性 (combined = 双等位合并)
EXPR_FILE = "combined_true_expression.csv"

# 开放状态到父/母/混合的映射
PAT_OPEN_STATES = {"shared_open", "paternal_open"}
MAT_OPEN_STATES = {"shared_open", "maternal_open"}

# 轨道配色
COL_CLOSED = "#e8e8e8"
COL_PAT = "#1f77b4"    # 父本 蓝
COL_MAT = "#d62728"    # 母本 红
COL_COMB = "#2ca02c"   # 混合 绿


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data():
    chrom = pd.read_csv(os.path.join(DATA_DIR, "chromosome_info.csv"))
    genes = pd.read_csv(os.path.join(DATA_DIR, "gene_info.csv"))
    regions = pd.read_csv(os.path.join(DATA_DIR, "chromatin_regions.csv"))
    expr = pd.read_csv(os.path.join(DATA_DIR, EXPR_FILE), index_col=0)  # cells x genes
    cells = pd.read_csv(os.path.join(DATA_DIR, "cell_metadata.csv"))

    # region 开放度 (父/母/混合)
    regions["pat_open"] = regions["allelic_state"].isin(PAT_OPEN_STATES).astype(int)
    regions["mat_open"] = regions["allelic_state"].isin(MAT_OPEN_STATES).astype(int)
    regions["comb_open"] = (regions["pat_open"] + regions["mat_open"]) / 2.0

    # 基因 -> 所在 region 的开放度
    rmap = regions.set_index("region_id")[["pat_open", "mat_open", "comb_open"]]
    genes = genes.merge(rmap, left_on="chromatin_region_id", right_index=True, how="left")
    genes[["pat_open", "mat_open", "comb_open"]] = genes[
        ["pat_open", "mat_open", "comb_open"]].fillna(0.0)

    # 染色体排序 (按数字)
    def chr_key(c):
        s = str(c).replace("chr", "")
        return int(s) if s.isdigit() else 999
    chrom = chrom.sort_values("chromosome", key=lambda s: s.map(chr_key)).reset_index(drop=True)
    return chrom, genes, regions, expr, cells


# ---------------------------------------------------------------------------
# 图1: 每条染色体开放区域轨道图 (父/母/混合)
# ---------------------------------------------------------------------------
def plot_open_region_tracks(chrom, genes, regions, out_path):
    chrom_ids = chrom["chromosome"].tolist()
    n = len(chrom_ids)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.9 * n + 1), squeeze=False)
    axes = axes[:, 0]

    track_defs = [("父本 paternal", "pat_open", COL_PAT),
                  ("母本 maternal", "mat_open", COL_MAT),
                  ("混合 combined", "comb_open", COL_COMB)]

    for ax, cid in zip(axes, chrom_ids):
        crow = chrom[chrom["chromosome"] == cid].iloc[0]
        L = float(crow["chromosome_length"])
        rsub = regions[regions["chromosome"] == cid]
        gsub = genes[genes["chromosome"] == cid]

        track_h = 0.6
        gap = 0.4
        for ti, (label, col_key, color) in enumerate(track_defs):
            y0 = ti * (track_h + gap)
            # 底色 (closed)
            ax.broken_barh([(0, L)], (y0, track_h), facecolors=COL_CLOSED,
                           edgecolors="none", zorder=1)
            # 开放段
            if col_key == "comb_open":
                # 分两级: 0.5 (单等位) 与 1.0 (双等位)
                for val, alpha in [(0.5, 0.45), (1.0, 1.0)]:
                    seg = rsub[np.isclose(rsub["comb_open"], val)]
                    bars = [(r.start, r.end - r.start) for r in seg.itertuples()]
                    if bars:
                        ax.broken_barh(bars, (y0, track_h), facecolors=color,
                                       alpha=alpha, edgecolors="none", zorder=2)
            else:
                seg = rsub[rsub[col_key] > 0]
                bars = [(r.start, r.end - r.start) for r in seg.itertuples()]
                if bars:
                    ax.broken_barh(bars, (y0, track_h), facecolors=color,
                                   edgecolors="none", zorder=2)
            ax.text(-0.012 * L, y0 + track_h / 2, label, ha="right", va="center",
                    fontsize=8)

        # 着丝粒
        cs, ce = float(crow["centromere_start"]), float(crow["centromere_end"])
        top = len(track_defs) * (track_h + gap)
        ax.axvspan(cs, ce, color="#888888", alpha=0.35, zorder=3)
        # 基因 TSS 刻度 (最上方)
        ax.scatter(gsub["TSS"], np.full(len(gsub), top - 0.05), marker="|",
                   s=40, c="black", linewidths=0.5, zorder=4)

        ax.set_xlim(-0.02 * L, L)
        ax.set_ylim(-0.2, top + 0.2)
        ax.set_yticks([])
        ax.set_title(f"{cid}  (长度 {L/1e6:.1f} Mb, {len(gsub)} 基因)",
                     fontsize=10, loc="left")
        ax.set_xlabel("基因组坐标 (bp)", fontsize=8)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)

    legend = [Patch(facecolor=COL_CLOSED, label="关闭 closed"),
              Patch(facecolor=COL_PAT, label="父本开放"),
              Patch(facecolor=COL_MAT, label="母本开放"),
              Patch(facecolor=COL_COMB, label="混合开放 (深=双等位)"),
              Patch(facecolor="#888888", alpha=0.35, label="着丝粒"),
              plt.Line2D([0], [0], marker="|", color="black", lw=0, label="基因 TSS")]
    fig.legend(handles=legend, loc="upper center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("各染色体开放区域轨道图 (父本 / 母本 / 混合)", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_path)


# ---------------------------------------------------------------------------
# 开放度注释条 (给定基因顺序, 画 3 行 pat/mat/comb 色带)
# ---------------------------------------------------------------------------
def _draw_open_annotation(ax, gene_df, orient="horizontal"):
    """gene_df 按目标顺序排列, 含 pat_open/mat_open/comb_open. 画 3 条注释带."""
    n = len(gene_df)
    pat = gene_df["pat_open"].to_numpy()
    mat = gene_df["mat_open"].to_numpy()
    comb = gene_df["comb_open"].to_numpy()

    pat_cmap = ListedColormap([COL_CLOSED, COL_PAT])
    mat_cmap = ListedColormap([COL_CLOSED, COL_MAT])
    comb_cmap = LinearSegmentedColormap.from_list("comb", [COL_CLOSED, "#a6dba0", COL_COMB])

    if orient == "horizontal":
        # 3 行, x = 基因
        mat_img = np.vstack([comb, mat, pat])  # 顶行 comb
        ax.imshow(np.zeros_like(mat_img), aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.imshow(pat[None, :], aspect="auto", cmap=pat_cmap, vmin=0, vmax=1,
                  extent=[0, n, 0, 1])
        ax.imshow(mat[None, :], aspect="auto", cmap=mat_cmap, vmin=0, vmax=1,
                  extent=[0, n, 1, 2])
        ax.imshow(comb[None, :], aspect="auto", cmap=comb_cmap, vmin=0, vmax=1,
                  extent=[0, n, 2, 3])
        ax.set_xlim(0, n)
        ax.set_ylim(0, 3)
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(["父本", "母本", "混合"], fontsize=7)
        ax.set_xticks([])
    else:  # vertical: 3 列, y = 基因 (从上到下)
        ax.imshow(pat[:, None], aspect="auto", cmap=pat_cmap, vmin=0, vmax=1,
                  extent=[0, 1, n, 0])
        ax.imshow(mat[:, None], aspect="auto", cmap=mat_cmap, vmin=0, vmax=1,
                  extent=[1, 2, n, 0])
        ax.imshow(comb[:, None], aspect="auto", cmap=comb_cmap, vmin=0, vmax=1,
                  extent=[2, 3, n, 0])
        ax.set_ylim(n, 0)
        ax.set_xlim(0, 3)
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(["父", "母", "混"], fontsize=7)
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# 图2: 每条染色体基因相关性 + 开放轨道
# ---------------------------------------------------------------------------
def plot_chromosome_correlation(chrom, genes, expr, out_dir):
    for cid in chrom["chromosome"]:
        gsub = genes[genes["chromosome"] == cid].sort_values("start")
        gids = [g for g in gsub["gene_id"] if g in expr.columns]
        gsub = gsub[gsub["gene_id"].isin(gids)]
        if len(gids) < 3:
            continue
        sub_expr = expr[gids]
        corr = np.corrcoef(sub_expr.to_numpy().T)
        corr = np.nan_to_num(corr)

        fig = plt.figure(figsize=(8.6, 8.6))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[0.05, 1], height_ratios=[0.05, 1],
            wspace=0.02, hspace=0.02)
        ax_main = fig.add_subplot(gs[1, 1])
        ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
        ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)

        im = ax_main.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                            extent=[0, len(gids), len(gids), 0])
        ax_main.set_xlabel("基因 (按位置排序) →", fontsize=9)
        ax_main.set_yticks([])
        ax_main.set_xticks([])
        cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.02)
        cbar.set_label("Pearson 相关", fontsize=8)

        _draw_open_annotation(ax_top, gsub, orient="horizontal")
        _draw_open_annotation(ax_left, gsub, orient="vertical")

        fig.suptitle(f"{cid} 基因表达相关性 (n={len(gids)}) + 开放区域轨道",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(out_dir, f"corr_{cid}_open_track.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[saved]", out_path)


# ---------------------------------------------------------------------------
# 图3: 基因 x 细胞 表达热图 + 开放轨道
# ---------------------------------------------------------------------------
def plot_expression_heatmap(chrom, genes, expr, cells, out_path):
    # 基因排序: 染色体 -> 位置
    def chr_key(c):
        s = str(c).replace("chr", "")
        return int(s) if s.isdigit() else 999
    gorder = genes.copy()
    gorder["_ck"] = gorder["chromosome"].map(chr_key)
    gorder = gorder.sort_values(["_ck", "start"])
    gids = [g for g in gorder["gene_id"] if g in expr.columns]
    gorder = gorder[gorder["gene_id"].isin(gids)].reset_index(drop=True)

    # 细胞排序: 按 cell_type
    cell_order = cells.sort_values(["cell_type", "cell_id"])
    cids = [c for c in cell_order["cell_id"] if c in expr.index]
    cell_order = cell_order[cell_order["cell_id"].isin(cids)]

    mat = expr.loc[cids, gids].to_numpy().T  # genes x cells
    # 每基因 z-score (便于观察模式)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True) + 1e-8
    z = (mat - mu) / sd
    z = np.clip(z, -3, 3)

    n_gene = len(gids)
    n_cell = len(cids)

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[0.14, 0.5, 3, 0.12], height_ratios=[0.05, 3],
        wspace=0.03, hspace=0.03)
    ax_open = fig.add_subplot(gs[1, 0])       # 开放轨道 (基因)
    ax_chr = fig.add_subplot(gs[1, 1])        # 染色体色条
    ax_main = fig.add_subplot(gs[1, 2])       # 主热图
    ax_ctype = fig.add_subplot(gs[0, 2])      # 细胞类型条
    ax_cbar = fig.add_subplot(gs[1, 3])       # colorbar

    # 主热图
    im = ax_main.imshow(z, aspect="auto", cmap="magma", vmin=-3, vmax=3,
                        extent=[0, n_cell, n_gene, 0])
    ax_main.set_xlabel("细胞 (按类型分组) →", fontsize=9)
    ax_main.set_yticks([])
    ax_main.set_xticks([])
    fig.colorbar(im, cax=ax_cbar).set_label("表达 z-score", fontsize=8)

    # 开放轨道 (基因方向, vertical)
    _draw_open_annotation(ax_open, gorder, orient="vertical")

    # 染色体色条
    chr_ids = chrom["chromosome"].tolist()
    chr_to_idx = {c: i for i, c in enumerate(chr_ids)}
    chr_cmap = plt.get_cmap("tab10")
    chr_vec = np.array([chr_to_idx.get(c, 0) for c in gorder["chromosome"]])
    ax_chr.imshow(chr_vec[:, None], aspect="auto",
                  cmap=ListedColormap([chr_cmap(i) for i in range(len(chr_ids))]),
                  vmin=0, vmax=len(chr_ids) - 1, extent=[0, 1, n_gene, 0])
    ax_chr.set_xticks([0.5]); ax_chr.set_xticklabels(["染色体"], fontsize=7)
    ax_chr.set_yticks([])
    for sp in ax_chr.spines.values():
        sp.set_visible(False)
    # 染色体分界线
    boundaries = np.where(np.diff(chr_vec) != 0)[0] + 1
    for b in boundaries:
        ax_main.axhline(b, color="white", lw=0.6)

    # 细胞类型条
    ctypes = cell_order["cell_type"].tolist()
    ct_ids = sorted(set(ctypes))
    ct_to_idx = {c: i for i, c in enumerate(ct_ids)}
    ct_vec = np.array([ct_to_idx[c] for c in ctypes])
    ct_cmap = plt.get_cmap("Set2")
    ax_ctype.imshow(ct_vec[None, :], aspect="auto",
                    cmap=ListedColormap([ct_cmap(i) for i in range(len(ct_ids))]),
                    vmin=0, vmax=len(ct_ids) - 1, extent=[0, n_cell, 0, 1])
    ax_ctype.set_yticks([]); ax_ctype.set_xticks([])
    ax_ctype.set_title("基因 x 细胞 表达热图 (行=基因 按染色体/位置, 列=细胞 按类型) + 开放区域轨道",
                       fontsize=11, loc="left")
    for sp in ax_ctype.spines.values():
        sp.set_visible(False)

    # 图例
    chr_legend = [Patch(facecolor=chr_cmap(i), label=c) for i, c in enumerate(chr_ids)]
    ct_legend = [Patch(facecolor=ct_cmap(i), label=c) for i, c in enumerate(ct_ids)]
    open_legend = [Patch(facecolor=COL_PAT, label="父本开放"),
                   Patch(facecolor=COL_MAT, label="母本开放"),
                   Patch(facecolor=COL_COMB, label="混合开放"),
                   Patch(facecolor=COL_CLOSED, label="关闭")]
    fig.legend(handles=chr_legend + ct_legend + open_legend,
               loc="lower center", ncol=8, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_path)


# ---------------------------------------------------------------------------
def main():
    chrom, genes, regions, expr, cells = load_data()
    print(f"[info] {len(chrom)} 染色体, {len(genes)} 基因, {len(regions)} regions, "
          f"表达矩阵 {expr.shape[0]} 细胞 x {expr.shape[1]} 基因")

    plot_open_region_tracks(
        chrom, genes, regions,
        os.path.join(OUT_DIR, "open_region_tracks_by_chromosome.png"))

    plot_chromosome_correlation(chrom, genes, expr, OUT_DIR)

    plot_expression_heatmap(
        chrom, genes, expr, cells,
        os.path.join(OUT_DIR, "expression_heatmap_open_track.png"))

    print("\n完成. 输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
