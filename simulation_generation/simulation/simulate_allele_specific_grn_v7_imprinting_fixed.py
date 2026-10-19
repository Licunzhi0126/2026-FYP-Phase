#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
父本 / 母本等位基因特异 GRN 与基因表达模拟器 (V7)
=================================================

从共享基因组坐标出发, 依次生成:
  染色体坐标 -> 基因位置 -> 调控元件 -> chromatin regions ->
  父/母本染色质开放状态 -> 共享 TF 与 TF activity -> TF motif ->
  元件-基因 contact -> paternal/maternal GRN -> 调控模块 ->
  父/母本基因表达 -> combined 表达 -> RNA-seq counts

设计要点:
  * 父母本共享坐标 (染色体/基因/元件), 差异来自 cis 状态 (开放/甲基化/接触).
  * TF 节点与每个细胞的 TF activity 在父母本之间共享.
  * 未开放区域基因不强制为 0, 由 promoter gate + basal leakage 决定低水平表达.
  * 模块内基因共享主调控信号 (相关), 但保留基因特异 loading 差异.

本文件默认以“小规模测试”参数运行 (5 条染色体, 100 个基因), 全部结果写入 V7 文件夹.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse  # 预留稀疏支持 (大规模 motif/contact)
from scipy.special import expit  # 数值稳定 sigmoid
from scipy.stats import nbinom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 一、基础参数
# ---------------------------------------------------------------------------
# 说明: 以下为“测试规模”参数 (5 染色体 / 100 基因). 染色体长度按测试尺度缩小,
# 以便 100 个基因仍能呈现 gene-rich / gene-poor 的成簇结构.
PARAMS: Dict[str, object] = {
    "seed": 42,

    # Genome (测试尺度)
    "n_chromosomes": 5,
    "n_genes": 100,
    "chromosome_length_min": 6_000_000,
    "chromosome_length_max": 15_000_000,
    "telomere_length": 300_000,

    # Cells
    "n_cell_types": 4,
    "cells_per_type": 40,

    # TF (测试尺度, 必须 << n_genes)
    "n_tfs": 15,
    "master_regulator_fraction": 0.15,

    # Regulatory elements
    "enhancers_per_gene_mean": 3.0,
    "silencers_per_gene_mean": 1.0,
    "insulators_per_mb": 0.5,
    "promoter_upstream": 2000,
    "promoter_downstream": 500,

    # Chromatin regions
    "chromatin_region_length_min": 50_000,
    "chromatin_region_length_max": 2_000_000,
    "region_ar1_rho": 0.8,          # 相邻 region 活性自回归系数

    # Allelic chromatin states (四类比例之和必须为 1)
    "shared_open_fraction": 0.50,
    "shared_closed_fraction": 0.25,
    "paternal_open_fraction": 0.125,
    "maternal_open_fraction": 0.125,

    # Accessibility
    "paternal_maternal_accessibility_correlation": 0.75,
    "closed_state_leak": 0.03,

    # Genomic imprinting (随机非 TF 基因, 偏父或偏母表达)
    "imprinting_gene_count": 6,
    "imprinting_strength_min": 0.75,
    "imprinting_strength_max": 0.95,
    "imprinting_nonpreferred_leak": 0.03,
    "imprinting_min_preferred_ratio": 3.0,
    "imprinting_min_direction_accuracy": 0.80,

    # Contact
    "contact_distance_scale": 200_000,
    "same_region_bonus": 1.0,
    "same_compartment_bonus": 0.5,
    "insulator_penalty": 1.5,
    "enhancer_window": 1_000_000,   # enhancer/silencer 可作用的最大距离

    # GRN
    "activating_tf_fraction": 0.70,
    "target_spectral_radius": 0.85,
    "grn_edge_min_abs": 1e-4,
    "promoter_tf_gate_scale": 1.2,
    "enhancer_fold_clip": 2.0,

    # Modules
    "n_regulatory_modules": 8,
    "functional_module_fraction": 0.70,
    "local_module_fraction": 0.30,
    "unassigned_gene_fraction": 0.10,

    # Expression
    "closed_promoter_basal_leak": 0.03,
    "gene_noise_sd": 0.30,
    "module_noise_sd": 0.20,
    "negative_binomial_dispersion_mean": 5.0,
    "library_size_mean": 1_500,
    "dropout_midpoint": 1.0,
    "solve_max_iter": 200,
    "solve_tol": 1e-4,
    "solve_damping": 0.5,

    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__))),
    "plot_dpi": 130,
}

EPS = 1e-8

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("sim_v7")


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
def softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定 softplus: log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def clip01(x: np.ndarray) -> np.ndarray:
    """裁剪到 [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def validate_params(params: Dict[str, object]) -> None:
    """校验参数: 比例在 0-1, 四类染色质比例和为 1, 长度为正."""
    ratio_keys = [
        "shared_open_fraction", "shared_closed_fraction",
        "paternal_open_fraction", "maternal_open_fraction",
        "paternal_maternal_accessibility_correlation", "closed_state_leak",
        "activating_tf_fraction", "functional_module_fraction",
        "local_module_fraction", "unassigned_gene_fraction",
        "closed_promoter_basal_leak", "master_regulator_fraction",
        "imprinting_strength_min", "imprinting_strength_max",
        "imprinting_nonpreferred_leak", "imprinting_min_direction_accuracy",
    ]
    for k in ratio_keys:
        v = float(params[k])
        assert 0.0 <= v <= 1.0, f"参数 {k}={v} 必须位于 0-1"

    four = (float(params["shared_open_fraction"]) + float(params["shared_closed_fraction"])
            + float(params["paternal_open_fraction"]) + float(params["maternal_open_fraction"]))
    assert abs(four - 1.0) < 1e-9, f"四类染色质状态比例之和必须为 1, 当前={four}"

    length_keys = [
        "chromosome_length_min", "chromosome_length_max", "telomere_length",
        "chromatin_region_length_min", "chromatin_region_length_max",
        "contact_distance_scale", "enhancer_window", "library_size_mean",
    ]
    for k in length_keys:
        assert float(params[k]) > 0, f"长度参数 {k} 必须为正"
    assert int(params["n_tfs"]) < int(params["n_genes"]), "TF 数必须小于基因数"
    assert int(params["imprinting_gene_count"]) <= int(params["n_genes"]), "imprinting 基因数不能超过基因数"
    assert float(params["imprinting_strength_min"]) <= float(params["imprinting_strength_max"]), "imprinting 强度范围无效"
    assert float(params["imprinting_min_preferred_ratio"]) > 1.0, "imprinting 表达比阈值必须大于 1"
    log.info("参数校验通过.")


# ---------------------------------------------------------------------------
# 二、共享坐标: 2.1 染色体
# ---------------------------------------------------------------------------
def generate_chromosome_info(params: Dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    """生成染色体信息: 长度(高编号略短)、着丝粒(中部附近偏移)、端粒."""
    n = int(params["n_chromosomes"])
    lo, hi = int(params["chromosome_length_min"]), int(params["chromosome_length_max"])
    tel = int(params["telomere_length"])

    # 高编号整体略短: 基准长度线性下降 + 噪声
    frac = np.linspace(1.0, 0.6, n)
    base = lo + (hi - lo) * frac
    jitter = rng.normal(0, (hi - lo) * 0.05, size=n)
    lengths = np.clip(base + jitter, lo, hi).astype(int)

    records = []
    for c in range(n):
        L = int(lengths[c])
        cen_center = int(L * 0.5 + rng.normal(0, L * 0.08))
        cen_center = int(np.clip(cen_center, int(L * 0.3), int(L * 0.7)))
        cen_half = int(L * 0.02)
        records.append({
            "chromosome": f"chr{c + 1}",
            "chromosome_length": L,
            "centromere_start": cen_center - cen_half,
            "centromere_end": cen_center + cen_half,
            "left_telomere_end": tel,
            "right_telomere_start": L - tel,
        })
    return pd.DataFrame(records)


def _special_region_penalty(pos: np.ndarray, chrom_row: pd.Series) -> np.ndarray:
    """着丝粒/端粒附近的密度惩罚 (0-1, 越接近 1 越应稀疏)."""
    L = float(chrom_row["chromosome_length"])
    pen = np.zeros_like(pos, dtype=float)
    # 端粒
    pen = np.where(pos < chrom_row["left_telomere_end"], 0.9, pen)
    pen = np.where(pos > chrom_row["right_telomere_start"], 0.9, pen)
    # 着丝粒 (高斯)
    cen = 0.5 * (chrom_row["centromere_start"] + chrom_row["centromere_end"])
    cen_sd = 0.05 * L
    pen = np.maximum(pen, 0.9 * np.exp(-0.5 * ((pos - cen) / cen_sd) ** 2))
    return pen


# ---------------------------------------------------------------------------
# 二、共享坐标: 2.2 基因 (分层生成)
# ---------------------------------------------------------------------------
def generate_gene_info(
    params: Dict[str, object],
    rng: np.random.Generator,
    chrom_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    分层生成基因坐标: 染色体 -> genomic blocks(密度分级) -> cluster center -> gene position.
    避免全染色体均匀随机; gene-rich block 生成更多基因并成簇.
    """
    n_genes = int(params["n_genes"])
    n_chr = int(params["n_chromosomes"])

    # 按染色体长度分配基因数
    lengths = chrom_info["chromosome_length"].to_numpy(dtype=float)
    weights = lengths / lengths.sum()
    per_chr = rng.multinomial(n_genes, weights)

    records: List[Dict[str, object]] = []
    gid = 1
    for c in range(n_chr):
        row = chrom_info.iloc[c]
        L = int(row["chromosome_length"])
        n_c = int(per_chr[c])
        if n_c == 0:
            continue

        # 将染色体切成 blocks, 每个 block 赋密度等级
        block_len = max(500_000, L // 12)
        block_starts = np.arange(int(row["left_telomere_end"]), L - int(params["telomere_length"]), block_len)
        if len(block_starts) == 0:
            block_starts = np.array([int(row["left_telomere_end"])])
        density_class = rng.choice(
            ["gene_rich", "intermediate", "gene_poor"],
            size=len(block_starts), p=[0.3, 0.4, 0.3],
        )
        base_w = {"gene_rich": 4.0, "intermediate": 1.5, "gene_poor": 0.4}
        bw = np.array([base_w[d] for d in density_class])
        # 特殊区域惩罚 block 权重
        centers = block_starts + block_len / 2
        bw = bw * (1.0 - _special_region_penalty(centers, row))
        bw = np.clip(bw, 1e-3, None)
        bw = bw / bw.sum()

        # 每个基因先选 block, 再在 block 内围绕若干 cluster center 采样
        chosen_blocks = rng.choice(len(block_starts), size=n_c, p=bw)
        for b in np.unique(chosen_blocks):
            idx = np.where(chosen_blocks == b)[0]
            bstart = int(block_starts[b])
            bend = min(L - int(params["telomere_length"]), bstart + block_len)
            n_in_block = len(idx)
            # cluster 中心数量: gene-rich 更少中心 (更成簇)
            n_clusters = max(1, n_in_block // 4)
            cluster_centers = rng.uniform(bstart, bend, size=n_clusters)
            spread = block_len * 0.08
            for _ in idx:
                cc = cluster_centers[rng.integers(0, n_clusters)]
                tss = int(np.clip(rng.normal(cc, spread), bstart, bend))
                # 基因长度: 截断对数正态 (右偏, 几 kb - 几百 kb)
                glen = int(np.clip(rng.lognormal(mean=9.4, sigma=0.9), 500, 500_000))
                strand = "+" if rng.random() < 0.5 else "-"
                if strand == "+":
                    start, end = tss, min(L, tss + glen)
                else:
                    start, end = max(1, tss - glen), tss
                records.append({
                    "gene_id": f"gene_{gid:04d}",
                    "chromosome": row["chromosome"],
                    "start": int(start),
                    "end": int(end),
                    "TSS": int(tss),
                    "strand": strand,
                    "gene_length": int(end - start),
                    "gene_type": "protein_coding",
                    "is_TF": 0,
                    "is_housekeeping": int(rng.random() < 0.15),
                    "density_class": density_class[b],
                })
                gid += 1

    gene = pd.DataFrame(records).sort_values(["chromosome", "start"]).reset_index(drop=True)

    # local_gene_density: 每 1Mb 窗口内基因数
    dens = np.zeros(len(gene))
    for chrom, sub in gene.groupby("chromosome"):
        tss = sub["TSS"].to_numpy()
        for i, t in zip(sub.index, tss):
            dens[i] = np.sum(np.abs(tss - t) < 500_000)
    gene["local_gene_density"] = dens.astype(int)
    return gene


# ---------------------------------------------------------------------------
# 三、chromatin regions (AR(1) 活性, A/B compartment, 完整非重叠覆盖)
# ---------------------------------------------------------------------------
def generate_chromatin_regions(
    params: Dict[str, object],
    rng: np.random.Generator,
    chrom_info: pd.DataFrame,
    gene: pd.DataFrame,
) -> pd.DataFrame:
    """把每条染色体切成连续、不重叠、完整覆盖的 region; 局部活性用 AR(1) 空间相关."""
    rlo = int(params["chromatin_region_length_min"])
    rhi = int(params["chromatin_region_length_max"])
    rho = float(params["region_ar1_rho"])

    records: List[Dict[str, object]] = []
    rid = 1
    for _, row in chrom_info.iterrows():
        chrom = row["chromosome"]
        L = int(row["chromosome_length"])
        gsub = gene[gene["chromosome"] == chrom]

        cur = 1
        z_prev = 0.0
        while cur <= L:
            # 右偏 region 长度
            seg = int(np.clip(rng.lognormal(mean=np.log(300_000), sigma=0.8), rlo, rhi))
            start = cur
            end = min(L, cur + seg - 1)

            # AR(1) 局部活性
            z = rho * z_prev + rng.normal(0, 1.0)
            z_prev = z

            # 特殊区域惩罚
            center = 0.5 * (start + end)
            pen = float(_special_region_penalty(np.array([center]), row)[0])
            repeat_fraction = float(np.clip(0.15 + 0.6 * pen + rng.normal(0, 0.08), 0, 0.95))
            hetero = float(np.clip(pen + 0.3 * (z < 0) + rng.normal(0, 0.1), 0, 1))

            # 基因密度分级
            n_gene_in = int(np.sum((gsub["TSS"] >= start) & (gsub["TSS"] <= end)))
            gd_per_mb = n_gene_in / max(seg / 1e6, EPS)
            if gd_per_mb > 4:
                gd_class = "gene_rich"
            elif gd_per_mb > 1:
                gd_class = "intermediate"
            else:
                gd_class = "gene_poor"

            local_activity = float(expit(z - 1.5 * pen))
            compartment = "A" if (z - 1.2 * pen) > 0 else "B"
            if local_activity > 0.55:
                state_class = "active"
            elif local_activity > 0.3:
                state_class = "intermediate"
            else:
                state_class = "repressed"

            records.append({
                "region_id": f"region_{rid:05d}",
                "chromosome": chrom,
                "start": int(start),
                "end": int(end),
                "region_length": int(end - start + 1),
                "compartment": compartment,
                "gene_density_class": gd_class,
                "repeat_fraction": round(repeat_fraction, 4),
                "heterochromatin_score": round(hetero, 4),
                "local_activity_score": round(local_activity, 4),
                "region_state_class": state_class,
            })
            rid += 1
            cur = end + 1
    return pd.DataFrame(records)


def _region_lookup(regions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {c: sub.reset_index(drop=True) for c, sub in regions.groupby("chromosome")}


def assign_region_to_positions(
    chrom: np.ndarray, pos: np.ndarray, region_by_chr: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """把坐标映射到所属 region_id 与 compartment (region 完整覆盖染色体)."""
    rid_out = np.empty(len(pos), dtype=object)
    comp_out = np.empty(len(pos), dtype=object)
    for i in range(len(pos)):
        sub = region_by_chr[chrom[i]]
        mask = (sub["start"].to_numpy() <= pos[i]) & (sub["end"].to_numpy() >= pos[i])
        j = np.argmax(mask) if mask.any() else int(
            np.abs(((sub["start"] + sub["end"]) / 2).to_numpy() - pos[i]).argmin())
        rid_out[i] = sub.iloc[j]["region_id"]
        comp_out[i] = sub.iloc[j]["compartment"]
    return rid_out, comp_out


# ---------------------------------------------------------------------------
# 二、共享坐标: 2.3-2.7 调控元件
# ---------------------------------------------------------------------------
def generate_regulatory_elements(
    params: Dict[str, object],
    rng: np.random.Generator,
    chrom_info: pd.DataFrame,
    gene: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    """生成 promoter/enhancer/silencer (围绕基因) 与全基因组少量 insulator."""
    region_by_chr = _region_lookup(regions)
    chr_len = dict(zip(chrom_info["chromosome"], chrom_info["chromosome_length"]))
    recs: List[Dict[str, object]] = []
    eid = 1
    win = int(params["enhancer_window"])

    def add(etype, chrom, s, e, target):
        nonlocal eid
        s = int(max(1, s)); e = int(min(chr_len[chrom], e))
        if e <= s:
            e = s + 1
        recs.append({
            "element_id": f"elem_{eid:06d}", "chromosome": chrom,
            "start": s, "end": e, "center": (s + e) // 2,
            "element_length": e - s, "element_type": etype,
            "candidate_target_gene": target,
        })
        eid += 1

    for _, g in gene.iterrows():
        chrom = g["chromosome"]; tss = int(g["TSS"])
        # promoter: 实际 peak 长度 100-800 bp (housekeeping 略宽稳定)
        plen = int(np.clip(rng.lognormal(mean=np.log(320), sigma=0.4), 100, 800))
        add("promoter", chrom, tss - plen // 2, tss + plen // 2, g["gene_id"])
        # enhancer: Poisson 数量, 距离右偏 (log-normal)
        n_enh = int(rng.poisson(float(params["enhancers_per_gene_mean"])))
        for _ in range(n_enh):
            dist = int(np.clip(rng.lognormal(mean=np.log(50_000), sigma=1.1), 1000, win))
            direction = 1 if rng.random() < 0.5 else -1
            center = tss + direction * dist
            elen = int(np.clip(rng.lognormal(mean=np.log(400), sigma=0.6), 150, 1500))
            add("enhancer", chrom, center - elen // 2, center + elen // 2, g["gene_id"])
        # silencer: 少于 enhancer
        n_sil = int(rng.poisson(float(params["silencers_per_gene_mean"])))
        for _ in range(n_sil):
            dist = int(np.clip(rng.lognormal(mean=np.log(60_000), sigma=1.1), 1000, win))
            direction = 1 if rng.random() < 0.5 else -1
            center = tss + direction * dist
            elen = int(np.clip(rng.lognormal(mean=np.log(700), sigma=0.5), 200, 2000))
            add("silencer", chrom, center - elen // 2, center + elen // 2, g["gene_id"])

    # insulator: 优先 region 边界
    for chrom, sub in region_by_chr.items():
        L = chr_len[chrom]
        n_ins = max(1, int(params["insulators_per_mb"] * L / 1e6))
        boundaries = sub["end"].to_numpy()[:-1]
        if len(boundaries) == 0:
            continue
        pick = rng.choice(boundaries, size=min(n_ins, len(boundaries)), replace=False)
        for b in pick:
            ilen = int(np.clip(rng.lognormal(mean=np.log(400), sigma=0.4), 100, 1000))
            add("insulator", chrom, b - ilen // 2, b + ilen // 2, "NA")

    elements = pd.DataFrame(recs)

    # nearest_gene + distance_to_TSS
    tss_by_chr = {c: s[["gene_id", "TSS"]].reset_index(drop=True) for c, s in gene.groupby("chromosome")}
    nearest, dist = [], []
    for _, e in elements.iterrows():
        td = tss_by_chr[e["chromosome"]]
        k = int(np.abs(td["TSS"].to_numpy() - e["center"]).argmin())
        nearest.append(td.iloc[k]["gene_id"])
        dist.append(int(e["center"] - td.iloc[k]["TSS"]))
    elements["nearest_gene"] = nearest
    elements["distance_to_TSS"] = dist

    # region_id + compartment + repeat/mappability
    rid, comp = assign_region_to_positions(
        elements["chromosome"].to_numpy(), elements["center"].to_numpy(), region_by_chr)
    elements["chromatin_region_id"] = rid
    elements["compartment"] = comp
    rep_map = dict(zip(regions["region_id"], regions["repeat_fraction"]))
    elements["repeat_fraction"] = [float(rep_map[r]) for r in rid]
    elements["mappability_score"] = clip01(1.0 - 0.5 * elements["repeat_fraction"].to_numpy()
                                           + rng.normal(0, 0.05, len(elements)))
    return elements


def finalize_gene_annotation(
    gene: pd.DataFrame, regions: pd.DataFrame, tf_ids: List[str]
) -> pd.DataFrame:
    """给 gene 补充 chromatin_region_id / compartment / is_TF."""
    region_by_chr = _region_lookup(regions)
    rid, comp = assign_region_to_positions(
        gene["chromosome"].to_numpy(), gene["TSS"].to_numpy(), region_by_chr)
    out = gene.copy()
    out["chromatin_region_id"] = rid
    out["compartment"] = comp
    out["is_TF"] = out["gene_id"].isin(tf_ids).astype(int)
    return out


def assign_imprinting(
    params: Dict[str, object], rng: np.random.Generator, gene: pd.DataFrame
) -> pd.DataFrame:
    """随机指定非 TF 基因的父本或母本表达偏好."""
    out = gene.copy()
    out["imprinting_status"] = "none"
    out["imprinting_strength"] = 0.0
    candidates = out.loc[out["is_TF"] == 0, "gene_id"].to_numpy()
    count = min(int(params["imprinting_gene_count"]), len(candidates))
    selected = rng.choice(candidates, size=count, replace=False)
    status = rng.choice(["paternal_expressed", "maternal_expressed"], size=count)
    strength = rng.uniform(
        float(params["imprinting_strength_min"]),
        float(params["imprinting_strength_max"]), size=count,
    )
    selected_info = pd.DataFrame({
        "gene_id": selected, "imprinting_status": status, "imprinting_strength": strength,
    })
    out = out.drop(columns=["imprinting_status", "imprinting_strength"]).merge(
        selected_info, on="gene_id", how="left"
    )
    out["imprinting_status"] = out["imprinting_status"].fillna("none")
    out["imprinting_strength"] = out["imprinting_strength"].fillna(0.0).round(4)
    nonpreferred = float(params["imprinting_nonpreferred_leak"])
    out["paternal_imprinting_gate"] = np.where(
        out["imprinting_status"] == "maternal_expressed", nonpreferred, 1.0)
    out["maternal_imprinting_gate"] = np.where(
        out["imprinting_status"] == "paternal_expressed", nonpreferred, 1.0)
    return out


# ---------------------------------------------------------------------------
# 四、父母本 region 等位状态
# ---------------------------------------------------------------------------
def assign_allelic_region_states(
    params: Dict[str, object],
    rng: np.random.Generator,
    regions: pd.DataFrame,
    elements: pd.DataFrame,
    gene: pd.DataFrame,
) -> pd.DataFrame:
    """
    为每个 region 赋等位状态 (shared_open/shared_closed/paternal_open/maternal_open).
    倾向: promoter/housekeeping 所在 region 偏 shared_open; enhancer 区更易单亲开放;
    repeat/hetero 区偏 shared_closed. 由 compartment/活性调制比例.
    """
    f_so = float(params["shared_open_fraction"])
    f_sc = float(params["shared_closed_fraction"])
    f_po = float(params["paternal_open_fraction"])
    f_mo = float(params["maternal_open_fraction"])

    hk_genes = set(gene.loc[gene["is_housekeeping"] == 1, "chromatin_region_id"])
    prom_regions = set(elements.loc[elements["element_type"] == "promoter", "chromatin_region_id"])
    enh_regions = set(elements.loc[elements["element_type"] == "enhancer", "chromatin_region_id"])

    imprint_by_gene = gene.set_index("gene_id")[["imprinting_status", "imprinting_strength"]]
    regional_imprints: Dict[str, List[Tuple[str, float]]] = {}
    for _, e in elements.iterrows():
        target = e["candidate_target_gene"]
        if target in imprint_by_gene.index:
            info = imprint_by_gene.loc[target]
            if info["imprinting_status"] != "none":
                regional_imprints.setdefault(e["chromatin_region_id"], []).append(
                    (str(info["imprinting_status"]), float(info["imprinting_strength"]))
                )

    states = []
    for _, r in regions.iterrows():
        act = float(r["local_activity_score"])
        hetero = float(r["heterochromatin_score"])
        w = np.array([f_so, f_sc, f_po, f_mo], dtype=float)  # [so, sc, po, mo]
        # 活性高 -> 更 shared_open; hetero 高 -> 更 shared_closed
        w[0] *= (0.5 + act)
        w[1] *= (0.5 + hetero + 0.5 * (r["compartment"] == "B"))
        if r["region_id"] in prom_regions:
            w[0] *= 1.8
        if r["region_id"] in hk_genes:
            w[0] *= 2.5
        if r["region_id"] in enh_regions:
            w[2] *= 1.5; w[3] *= 1.5
        w = w / w.sum()
        preferred = regional_imprints.get(r["region_id"], [])
        if preferred:
            status, strength = max(preferred, key=lambda x: x[1])
            forced = "paternal_open" if status == "paternal_expressed" else "maternal_open"
            if rng.random() < strength:
                states.append(forced)
                continue
        states.append(str(rng.choice(["shared_open", "shared_closed", "paternal_open", "maternal_open"], p=w)))

    out = regions.copy()
    out["allelic_state"] = states
    return out


# ---------------------------------------------------------------------------
# 五、父母本元件状态
# ---------------------------------------------------------------------------
def generate_element_state(
    params: Dict[str, object],
    rng: np.random.Generator,
    elements: pd.DataFrame,
    regions: pd.DataFrame,
    gene: pd.DataFrame,
    allele: str,
    shared_latent: np.ndarray,
) -> pd.DataFrame:
    """
    生成某一亲本的元件状态: open_state/open_strength/methylation/accessibility/activity.
    父母本通过 shared_latent + 等位噪声 达到目标 accessibility 相关性.
    """
    assert allele in {"paternal", "maternal"}
    corr = float(params["paternal_maternal_accessibility_correlation"])
    leak = float(params["closed_state_leak"])
    reg = regions.set_index("region_id")
    imprint = gene.set_index("gene_id")

    n = len(elements)
    # 父母本共享一部分潜变量, 加独立噪声 -> 控制相关性
    allele_noise = rng.normal(0, 1.0, n)
    latent = corr * shared_latent + np.sqrt(max(1 - corr ** 2, 0)) * allele_noise

    recs = []
    for i, (_, e) in enumerate(elements.iterrows()):
        rinfo = reg.loc[e["chromatin_region_id"]]
        astate = rinfo["allelic_state"]
        # region 是否对该亲本开放
        if astate == "shared_open":
            open_prob = 0.9
        elif astate == "shared_closed":
            open_prob = 0.08
        elif astate == "paternal_open":
            open_prob = 0.85 if allele == "paternal" else 0.2
        else:  # maternal_open
            open_prob = 0.85 if allele == "maternal" else 0.2

        etype = e["element_type"]
        type_bias = {"promoter": 0.15, "enhancer": 0.0, "silencer": -0.02, "insulator": -0.1}[etype]
        comp_bias = 0.1 if rinfo["compartment"] == "A" else -0.1
        dens = float(rinfo["local_activity_score"])
        score = (expit(latent[i]) * open_prob
                 + type_bias + comp_bias + 0.2 * dens - 0.3 * float(rinfo["repeat_fraction"]))
        open_state = int(score > 0.5)

        target = e["candidate_target_gene"]
        imprint_gate = 1.0
        if etype == "promoter" and target in imprint.index:
            imprint_gate = float(imprint.loc[target, f"{allele}_imprinting_gate"])
            if imprint.loc[target, "imprinting_status"] != "none":
                open_state = int(imprint_gate >= 1.0)

        # open_strength: 开放高、关闭低, 但保留弱开放
        if open_state == 1:
            open_strength = float(np.clip(rng.beta(5, 2) * (0.7 + 0.3 * dens), 0, 1))
        else:
            open_strength = float(np.clip(rng.beta(1.3, 6) + leak, 0, 1))
        if etype == "promoter" and imprint_gate < 1.0:
            open_strength *= imprint_gate

        # methylation: 与开放负相关 + 噪声, 元件类型不同分布
        base_meth = {"promoter": 0.25, "enhancer": 0.4, "silencer": 0.45, "insulator": 0.4}[etype]
        methylation = float(np.clip(
            base_meth - 0.35 * open_state - 0.25 * open_strength
            + 0.3 * float(rinfo["heterochromatin_score"]) + rng.normal(0, 0.08), 0, 1))

        # accessibility: sigmoid 综合
        acc = float(expit(
            -0.5 + 2.2 * open_state + 1.5 * open_strength - 2.0 * methylation
            + (0.4 if rinfo["compartment"] == "A" else -0.4)
            - 1.0 * float(rinfo["repeat_fraction"])
            + 0.8 * float(e["mappability_score"]) + rng.normal(0, 0.15)))
        if etype == "promoter" and imprint_gate < 1.0:
            acc *= imprint_gate

        # activity
        meth_penalty = float(np.exp(-2.0 * methylation))
        type_factor = {"promoter": 1.0, "enhancer": 1.0, "silencer": 1.0, "insulator": 0.2}[etype]
        activity = float(np.clip(acc * (0.3 + 0.7 * open_strength) * meth_penalty
                                 * type_factor * (0.5 + 0.5 * dens), 0, 1))

        recs.append({
            "element_id": e["element_id"], "chromosome": e["chromosome"],
            "start": e["start"], "end": e["end"], "element_type": etype,
            "chromatin_region_id": e["chromatin_region_id"],
            "region_state_class": rinfo["region_state_class"],
            "open_state": open_state, "open_strength": round(open_strength, 4),
            "methylation": round(methylation, 4), "accessibility": round(acc, 4),
            "activity": round(activity, 4),
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 六、共享 TF 信息
# ---------------------------------------------------------------------------
def generate_tf_info(
    params: Dict[str, object], rng: np.random.Generator, gene: pd.DataFrame
) -> pd.DataFrame:
    """从基因集合中选出 TF; 赋 activator/repressor 概率与 master regulator 标记."""
    n_tf = int(params["n_tfs"])
    tf_genes = rng.choice(gene["gene_id"].to_numpy(), size=n_tf, replace=False)
    act_frac = float(params["activating_tf_fraction"])
    gmap = gene.set_index("gene_id")
    recs = []
    for i, gid in enumerate(sorted(tf_genes)):
        g = gmap.loc[gid]
        p_act = float(np.clip(rng.normal(act_frac, 0.15), 0, 1))
        recs.append({
            "TF_id": f"TF_{i + 1:03d}", "gene_id": gid,
            "chromosome": g["chromosome"], "start": int(g["start"]), "end": int(g["end"]),
            "TF_family": f"family_{rng.integers(1, 6)}",
            "activator_probability": round(p_act, 3),
            "repressor_probability": round(1 - p_act, 3),
            "is_master_regulator": int(rng.random() < float(params["master_regulator_fraction"])),
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 七、共享 TF activity (按细胞类型 + 噪声)
# ---------------------------------------------------------------------------
def generate_cell_metadata(params: Dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    """生成细胞元数据: cell_type / library_size / batch / quality."""
    n_type = int(params["n_cell_types"])
    per = int(params["cells_per_type"])
    recs = []
    cid = 1
    for t in range(n_type):
        for _ in range(per):
            ls = int(np.clip(rng.lognormal(mean=np.log(params["library_size_mean"]), sigma=0.3),
                             1000, None))
            recs.append({
                "cell_id": f"cell_{cid:04d}", "cell_type": f"type_{t + 1}",
                "library_size": ls, "batch": f"batch_{rng.integers(1, 3)}",
                "quality_score": round(float(np.clip(rng.beta(6, 2), 0, 1)), 3),
            })
            cid += 1
    return pd.DataFrame(recs)


def generate_tf_activity(
    params: Dict[str, object], rng: np.random.Generator,
    tf_info: pd.DataFrame, cell_meta: pd.DataFrame,
) -> pd.DataFrame:
    """每个细胞的 TF activity (父母本共享): cell-type 基线 + 细胞噪声."""
    tf_ids = tf_info["TF_id"].tolist()
    types = cell_meta["cell_type"].unique()
    # 每个 cell type 的 TF 基线 (log-normal latent)
    type_mean = {t: rng.normal(0, 1.0, len(tf_ids)) for t in types}
    mat = np.zeros((len(cell_meta), len(tf_ids)))
    for i, (_, c) in enumerate(cell_meta.iterrows()):
        mat[i] = type_mean[c["cell_type"]] + rng.normal(0, 0.35, len(tf_ids))
    act = np.exp(mat)  # 正值 activity
    df = pd.DataFrame(act, index=cell_meta["cell_id"], columns=tf_ids)
    df.index.name = "cell_id"
    return df


# ---------------------------------------------------------------------------
# 八、调控模块
# ---------------------------------------------------------------------------
def generate_modules(
    params: Dict[str, object], rng: np.random.Generator,
    gene: pd.DataFrame, tf_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 functional / local modules, 分配 core/aux TF 与基因 membership+loading."""
    n_mod = int(params["n_regulatory_modules"])
    tf_ids = tf_info["TF_id"].tolist()
    gene_ids = gene["gene_id"].tolist()
    n_func = int(round(n_mod * float(params["functional_module_fraction"])))

    mod_recs, tf_link_recs = [], []
    module_kind = {}
    module_core_tf = {}
    for k in range(n_mod):
        kind = "functional" if k < n_func else "local"
        module_kind[f"module_{k + 1:03d}"] = kind
        n_core = int(rng.integers(1, 4))
        n_aux = int(rng.integers(1, 6))
        core = list(rng.choice(tf_ids, size=min(n_core, len(tf_ids)), replace=False))
        remain = [t for t in tf_ids if t not in core]
        aux = list(rng.choice(remain, size=min(n_aux, len(remain)), replace=False)) if remain else []
        module_core_tf[f"module_{k + 1:03d}"] = core
        strength = rng.choice(["strong", "moderate", "weak"], p=[0.3, 0.4, 0.3])
        mod_recs.append({
            "module_id": f"module_{k + 1:03d}", "module_type": kind,
            "n_core_tf": len(core), "n_aux_tf": len(aux), "strength_class": strength,
        })
        for t in core:
            w = float(rng.uniform(0.6, 1.0))
            tf_link_recs.append({"module_id": f"module_{k + 1:03d}", "TF_id": t,
                                 "role": "core", "weight": round(w, 3)})
        for t in aux:
            w = float(rng.uniform(0.1, 0.5))
            tf_link_recs.append({"module_id": f"module_{k + 1:03d}", "TF_id": t,
                                 "role": "aux", "weight": round(w, 3)})

    # 基因 membership: 部分基因 unassigned; local module 优先同染色体邻近基因
    unassigned = float(params["unassigned_gene_fraction"])
    module_ids = [f"module_{k + 1:03d}" for k in range(n_mod)]
    mem_recs = []
    # 给 local module 预分配一个“中心染色体”
    local_home = {m: rng.choice(gene["chromosome"].unique())
                  for m in module_ids if module_kind[m] == "local"}
    for gid in gene_ids:
        if rng.random() < unassigned:
            continue
        gchr = gene.loc[gene["gene_id"] == gid, "chromosome"].iloc[0]
        # 依 module 类型给权重
        w = []
        for m in module_ids:
            if module_kind[m] == "local":
                w.append(3.0 if local_home[m] == gchr else 0.2)
            else:
                w.append(1.0)
        w = np.array(w); w = w / w.sum()
        m = str(rng.choice(module_ids, p=w))
        shared_loading = float(rng.uniform(0.6, 1.2))
        dev = float(rng.normal(0, 0.25))
        mem_recs.append({
            "gene_id": gid, "module_id": m, "module_type": module_kind[m],
            "module_shared_loading": round(shared_loading, 3),
            "gene_specific_deviation": round(dev, 3),
            "gene_module_loading": round(max(shared_loading + dev, 0.05), 3),
        })

    return (pd.DataFrame(mod_recs), pd.DataFrame(mem_recs),
            pd.DataFrame(tf_link_recs))


def compute_module_activity(
    params: Dict[str, object], rng: np.random.Generator,
    modules: pd.DataFrame, tf_links: pd.DataFrame,
    tf_activity: pd.DataFrame, cell_meta: pd.DataFrame,
) -> pd.DataFrame:
    """module_activity[k,cell] = sum(core_TF_weight*TF_activity) + cell_type_effect + noise."""
    cells = tf_activity.index.tolist()
    type_of = dict(zip(cell_meta["cell_id"], cell_meta["cell_type"]))
    types = cell_meta["cell_type"].unique()
    out = {}
    for _, m in modules.iterrows():
        mid = m["module_id"]
        core = tf_links[(tf_links["module_id"] == mid) & (tf_links["role"] == "core")]
        ct_effect = {t: rng.normal(0, 0.5) for t in types}
        vals = np.zeros(len(cells))
        for i, cid in enumerate(cells):
            drive = 0.0
            for _, r in core.iterrows():
                drive += r["weight"] * float(np.log1p(tf_activity.loc[cid, r["TF_id"]]))
            vals[i] = drive + ct_effect[type_of[cid]] + rng.normal(0, float(params["module_noise_sd"]))
        out[mid] = vals
    df = pd.DataFrame(out, index=cells)
    df.index.name = "cell_id"
    return df


# ---------------------------------------------------------------------------
# 九、TF motif (稀疏长表)
# ---------------------------------------------------------------------------
def generate_motif(
    params: Dict[str, object], rng: np.random.Generator,
    tf_info: pd.DataFrame, elements: pd.DataFrame,
) -> pd.DataFrame:
    """稀疏 TF-element motif; affinity 为正值 (log-normal); sign 取决于 TF 激活/抑制倾向."""
    elems = elements[elements["element_type"].isin(["promoter", "enhancer", "silencer"])]
    elem_ids = elems["element_id"].to_numpy()
    recs = []
    for _, tf in tf_info.iterrows():
        # 每个 TF 只在少量元件上有 motif (稀疏, 右偏出度)
        k = int(np.clip(rng.lognormal(mean=np.log(max(3, len(elem_ids) * 0.05)), sigma=0.6),
                        1, len(elem_ids)))
        if tf["is_master_regulator"] == 1:
            k = min(len(elem_ids), int(k * 2.5))
        chosen = rng.choice(elem_ids, size=k, replace=False)
        sign = 1 if rng.random() < tf["activator_probability"] else -1
        for e in chosen:
            aff = float(np.clip(rng.lognormal(mean=np.log(0.6), sigma=0.5), 0.01, 3.0))
            # 元件级方向: silencer 上的 motif 更偏抑制
            etype = elems.loc[elems["element_id"] == e, "element_type"].iloc[0]
            s = sign
            if etype == "silencer":
                s = -abs(sign) if rng.random() < 0.8 else sign
            recs.append({
                "TF_id": tf["TF_id"], "element_id": e, "has_motif": 1,
                "motif_affinity": round(aff, 4), "motif_sign": int(s),
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 十、元件-基因 contact (稀疏长表, 父母本各一)
# ---------------------------------------------------------------------------
def generate_contact(
    params: Dict[str, object], rng: np.random.Generator,
    elements: pd.DataFrame, gene: pd.DataFrame, regions: pd.DataFrame,
    p_state: pd.DataFrame, m_state: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成 element-gene contact (paternal/maternal). 候选:
      promoter -> 自身基因; enhancer/silencer -> window 内基因.
    logit 模型综合距离、同 region、同 compartment、中间 insulator、元件活性.
    """
    win = int(params["enhancer_window"])
    dscale = float(params["contact_distance_scale"])
    gene_by_chr = {c: s.reset_index(drop=True) for c, s in gene.groupby("chromosome")}
    reg = regions.set_index("region_id")
    p_act = dict(zip(p_state["element_id"], p_state["activity"]))
    m_act = dict(zip(m_state["element_id"], m_state["activity"]))

    # 每条染色体 insulator 位置
    ins = elements[elements["element_type"] == "insulator"]
    ins_by_chr = {c: s["center"].to_numpy() for c, s in ins.groupby("chromosome")}

    # 每个 region 的 allelic state (用于父母本 contact 偏置)
    astate = dict(zip(regions["region_id"], regions["allelic_state"]))
    gene_region = dict(zip(gene["gene_id"], gene["chromatin_region_id"]))
    gene_comp = dict(zip(gene["gene_id"], gene["compartment"]))
    gene_tss = dict(zip(gene["gene_id"], gene["TSS"]))

    p_recs, m_recs = [], []
    for _, e in elements.iterrows():
        etype = e["element_type"]
        if etype == "insulator":
            continue
        chrom = e["chromosome"]
        if etype == "promoter":
            targets = [e["candidate_target_gene"]]
        else:
            gsub = gene_by_chr[chrom]
            d = np.abs(gsub["TSS"].to_numpy() - e["center"])
            targets = list(gsub.loc[d < win, "gene_id"])
        e_region = e["chromatin_region_id"]
        e_comp = e["compartment"]
        ins_pos = ins_by_chr.get(chrom, np.array([]))

        for tg in targets:
            dist = abs(int(e["center"]) - int(gene_tss[tg]))
            same_region = int(gene_region[tg] == e_region)
            same_comp = int(gene_comp[tg] == e_comp)
            lo, hi = sorted([int(e["center"]), int(gene_tss[tg])])
            ins_between = int(np.any((ins_pos > lo) & (ins_pos < hi)))
            # promoter-自身默认高
            prom_self = int(etype == "promoter")

            logit = (2.0 * prom_self
                     - 1.1 * np.log1p(dist / dscale)
                     + float(params["same_region_bonus"]) * same_region
                     + float(params["same_compartment_bonus"]) * same_comp
                     - float(params["insulator_penalty"]) * ins_between
                     + rng.normal(0, 0.3))
            base_p = float(expit(logit))

            st = astate[e_region]
            for allele, act_map, recs in (("paternal", p_act, p_recs), ("maternal", m_act, m_recs)):
                bias = 0.0
                if st == "paternal_open":
                    bias = 0.5 if allele == "paternal" else -0.5
                elif st == "maternal_open":
                    bias = 0.5 if allele == "maternal" else -0.5
                strength = float(np.clip(base_p * (0.4 + 0.6 * float(act_map[e["element_id"]]))
                                         + 0.1 * bias, 0, 1))
                recs.append({
                    "element_id": e["element_id"], "target_gene": tg, "chromosome": chrom,
                    "distance_to_TSS": dist, "same_chromatin_region": same_region,
                    "same_compartment": same_comp, "insulator_between": ins_between,
                    "element_type": etype,
                    "base_contact_probability": round(base_p, 4),
                    "allelic_contact_strength": round(strength, 4),
                })
    return pd.DataFrame(p_recs), pd.DataFrame(m_recs)


# ---------------------------------------------------------------------------
# 十三、构建父母本 GRN
# ---------------------------------------------------------------------------
def build_grn(
    params: Dict[str, object],
    tf_info: pd.DataFrame, gene: pd.DataFrame, elements: pd.DataFrame,
    motif: pd.DataFrame, p_state: pd.DataFrame, m_state: pd.DataFrame,
    p_contact: pd.DataFrame, m_contact: pd.DataFrame,
    membership: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, pd.DataFrame, List[str], List[str]]:
    """
    对每个 TF-gene 汇总 promoter/enhancer/silencer contribution -> effective weight.
    父母本差异来自 element activity 与 contact 的不同 (motif/拓扑共享).
    返回三类元件矩阵、合并矩阵、edge_table、TF 与基因 ID。
    """
    tf_ids = tf_info["TF_id"].tolist()
    gene_ids = gene["gene_id"].tolist()
    tf_idx = {t: i for i, t in enumerate(tf_ids)}
    gene_idx = {g: j for j, g in enumerate(gene_ids)}
    n_tf, n_gene = len(tf_ids), len(gene_ids)

    etype_map = dict(zip(elements["element_id"], elements["element_type"]))
    p_actv = dict(zip(p_state["element_id"], p_state["activity"]))
    m_actv = dict(zip(m_state["element_id"], m_state["activity"]))
    type_w = {"promoter": 1.0, "enhancer": 0.9, "silencer": 0.9}

    # motif: element -> list of (TF, affinity, sign)
    motif_by_elem: Dict[str, List] = {}
    for _, r in motif.iterrows():
        motif_by_elem.setdefault(r["element_id"], []).append(
            (r["TF_id"], float(r["motif_affinity"]), int(r["motif_sign"])))

    # 分部件累积: 每个 (tf, gene) 的 promoter/enh/sil 贡献 (父母本)
    parts = {}  # (tf,gene) -> dict

    def contact_iter(contact_df, act_map):
        act_local = dict(zip(contact_df["element_id"].to_numpy(),
                             contact_df["allelic_contact_strength"].to_numpy()))
        return act_local

    p_contact_strength = p_contact.groupby(["element_id", "target_gene"])["allelic_contact_strength"].first()
    m_contact_strength = m_contact.groupby(["element_id", "target_gene"])["allelic_contact_strength"].first()

    imprint = gene.set_index("gene_id")[["imprinting_status", "imprinting_strength"]]

    # 遍历 contact (稀疏), 用 motif 生成 TF 贡献
    def accumulate(contact_df, act_map, contact_series, key):
        for (eid, tg), cs in contact_series.items():
            etype = etype_map.get(eid)
            if etype not in type_w:
                continue
            motifs = motif_by_elem.get(eid, [])
            if not motifs:
                continue
            act = float(act_map.get(eid, 0.0))
            for (tf, aff, sign) in motifs:
                base = aff * act * float(cs) * type_w[etype]
                status = imprint.loc[tg, "imprinting_status"]
                strength = float(imprint.loc[tg, "imprinting_strength"])
                if ((key == "p" and status == "maternal_expressed")
                        or (key == "m" and status == "paternal_expressed")):
                    base *= 1.0 - strength
                if etype == "silencer":
                    contrib = -abs(base)
                elif etype == "promoter":
                    contrib = sign * base  # promoter 可含少量 repressor
                else:  # enhancer
                    contrib = sign * base
                d = parts.setdefault((tf, tg), {
                    "prom_p": 0.0, "enh_p": 0.0, "sil_p": 0.0,
                    "prom_m": 0.0, "enh_m": 0.0, "sil_m": 0.0})
                slot = {"promoter": "prom", "enhancer": "enh", "silencer": "sil"}[etype]
                d[f"{slot}_{key}"] += contrib

    accumulate(p_contact, p_actv, p_contact_strength, "p")
    accumulate(m_contact, m_actv, m_contact_strength, "m")

    W_prom_pat = np.zeros((n_tf, n_gene))
    W_prom_mat = np.zeros((n_tf, n_gene))
    W_enh_pat = np.zeros((n_tf, n_gene))
    W_enh_mat = np.zeros((n_tf, n_gene))
    W_sil_pat = np.zeros((n_tf, n_gene))
    W_sil_mat = np.zeros((n_tf, n_gene))
    mod_of_gene = dict(zip(membership["gene_id"], membership["module_id"]))
    edge_recs = []
    emin = float(params["grn_edge_min_abs"])
    for (tf, tg), d in parts.items():
        if tf not in tf_idx or tg not in gene_idx:
            continue
        wp = d["prom_p"] + d["enh_p"] + d["sil_p"]
        wm = d["prom_m"] + d["enh_m"] + d["sil_m"]
        if abs(wp) < emin and abs(wm) < emin:
            continue
        ti, gi = tf_idx[tf], gene_idx[tg]
        W_prom_pat[ti, gi] = d["prom_p"]
        W_prom_mat[ti, gi] = d["prom_m"]
        W_enh_pat[ti, gi] = d["enh_p"]
        W_enh_mat[ti, gi] = d["enh_m"]
        W_sil_pat[ti, gi] = d["sil_p"]
        W_sil_mat[ti, gi] = d["sil_m"]
        shared = int(abs(wp) >= emin and abs(wm) >= emin)
        status = str(imprint.loc[tg, "imprinting_status"])
        strength = float(imprint.loc[tg, "imprinting_strength"])
        edge_recs.append({
            "TF_id": tf, "target_gene": tg, "module_id": mod_of_gene.get(tg, "NA"),
            "regulatory_sign": int(np.sign(wp + wm) if (wp + wm) != 0 else np.sign(wp)),
            "imprinting_status": status, "imprinting_strength": round(strength, 4),
            "imprinting_adjustment_applied": int(status != "none"),
            "promoter_contribution_paternal": round(d["prom_p"], 5),
            "enhancer_contribution_paternal": round(d["enh_p"], 5),
            "silencer_contribution_paternal": round(d["sil_p"], 5),
            "paternal_weight": round(wp, 5),
            "promoter_contribution_maternal": round(d["prom_m"], 5),
            "enhancer_contribution_maternal": round(d["enh_m"], 5),
            "silencer_contribution_maternal": round(d["sil_m"], 5),
            "maternal_weight": round(wm, 5),
            "weight_difference": round(wp - wm, 5),
            "shared_edge": shared,
            "paternal_specific_edge": int(abs(wp) >= emin and abs(wm) < emin),
            "maternal_specific_edge": int(abs(wm) >= emin and abs(wp) < emin),
        })
    edge_table = pd.DataFrame(edge_recs)
    W_pat = W_prom_pat + W_enh_pat + W_sil_pat
    W_mat = W_prom_mat + W_enh_mat + W_sil_mat
    return (W_prom_pat, W_prom_mat, W_enh_pat, W_enh_mat, W_sil_pat, W_sil_mat,
            W_pat, W_mat, edge_table, tf_ids, gene_ids)


def scale_spectral_radius(A: np.ndarray, target: float) -> np.ndarray:
    """把方阵谱半径缩放到 target, 防止表达迭代爆炸."""
    if A.size == 0:
        return A
    r = float(np.max(np.abs(np.linalg.eigvals(A))))
    if r > EPS:
        return A * (target / r)
    return A


def generate_expression_background(
    params: Dict[str, object], rng: np.random.Generator,
    gene_ids: List[str], cell_meta: pd.DataFrame, cells: List[str],
) -> Dict[str, object]:
    n_gene = len(gene_ids)
    n_cell = len(cells)
    types = cell_meta.set_index("cell_id").loc[cells, "cell_type"].to_numpy()
    type_levels = {cell_type: rng.normal(0, 0.3, n_gene) for cell_type in np.unique(types)}
    celltype_effect = np.column_stack([type_levels[cell_type] for cell_type in types])
    return {
        "basal": rng.normal(-0.5, 0.3, n_gene),
        "celltype_effect": celltype_effect,
        "noise": rng.normal(0, float(params["gene_noise_sd"]), (n_gene, n_cell)),
    }


# ---------------------------------------------------------------------------
# 十四-十六、表达求解 (迭代 + damping + 谱半径缩放)
# ---------------------------------------------------------------------------
def solve_expression(
    params: Dict[str, object], rng: np.random.Generator,
    W_prom: np.ndarray, W_enh: np.ndarray, W_sil: np.ndarray,
    tf_info: pd.DataFrame, gene: pd.DataFrame,
    tf_activity: pd.DataFrame, module_activity: pd.DataFrame,
    membership: pd.DataFrame, elem_state: pd.DataFrame, elements: pd.DataFrame,
    regions: pd.DataFrame, cell_meta: pd.DataFrame, allele: str,
    expression_background: Dict[str, object],
) -> pd.DataFrame:
    """
    单亲本表达: 迭代 damped fixed point.
            z <- (1-d)z + d(basal + gate*(enhancer_fold*(module + feedback))*silencer_factor + chromatin + noise)
        promoter TF 信号决定 gate，enhancer TF 信号决定表达倍数，silencer TF 信号决定独立抑制项。
    未开放基因由 promoter_gate + basal_leak 决定低表达, enhancer 不能绕过关闭 promoter.
    true = softplus(z).
    """
    gene_ids = gene["gene_id"].tolist()
    tf_ids = tf_info["TF_id"].tolist()
    n_gene = len(gene_ids)
    cells = tf_activity.index.tolist()
    n_cell = len(cells)
    gidx = {g: i for i, g in enumerate(gene_ids)}
    imprinting_gate = gene.set_index("gene_id").loc[
        gene_ids, f"{allele}_imprinting_gate"].to_numpy(dtype=float)

    # promoter gate per gene: 由该基因 promoter accessibility 决定
    prom = elements[elements["element_type"] == "promoter"][["element_id", "candidate_target_gene"]]
    acc_map = dict(zip(elem_state["element_id"], elem_state["accessibility"]))
    leak = float(params["closed_promoter_basal_leak"])
    accessibility_gate = np.full(n_gene, leak)
    for _, r in prom.iterrows():
        g = r["candidate_target_gene"]
        if g in gidx:
            a = float(acc_map.get(r["element_id"], 0.0))
            accessibility_gate[gidx[g]] = leak + (1 - leak) * a

    # 三类元件的 TF 调控信号: TF x cell -> gene x cell
    TFA = tf_activity.loc[cells, tf_ids].to_numpy().T  # TF x cell
    tf_signal = np.log1p(TFA)
    promoter_signal = W_prom.T @ tf_signal
    enhancer_signal = W_enh.T @ tf_signal
    silencer_signal = W_sil.T @ tf_signal
    promoter_gate = expit(float(params["promoter_tf_gate_scale"]) * promoter_signal)
    gate = leak + (1 - leak) * accessibility_gate[:, None] * promoter_gate * imprinting_gate[:, None]
    fold_clip = float(params["enhancer_fold_clip"])
    enhancer_fold = np.exp(np.clip(enhancer_signal, -fold_clip, fold_clip))
    silencer_factor = np.exp(-np.clip(-silencer_signal, 0.0, fold_clip))

    # module effect: loading * module_activity
    mod_of_gene = dict(zip(membership["gene_id"], membership["module_id"]))
    load_of_gene = dict(zip(membership["gene_id"], membership["gene_module_loading"]))
    module_effect = np.zeros((n_gene, n_cell))
    ma = module_activity.loc[cells]
    for g in gene_ids:
        if g in mod_of_gene and mod_of_gene[g] in ma.columns:
            module_effect[gidx[g]] = float(load_of_gene[g]) * ma[mod_of_gene[g]].to_numpy()

    # chromatin effect per gene (亲本特异, 由基因所在 region 活性)
    reg_act = dict(zip(regions["region_id"], regions["local_activity_score"]))
    gene_region = dict(zip(gene["gene_id"], gene["chromatin_region_id"]))
    chromatin_effect = np.array([reg_act.get(gene_region[g], 0.5) - 0.5 for g in gene_ids])

    # feedback matrix A (gene x gene) from W restricted to TF genes
    tf_gene = dict(zip(tf_info["TF_id"], tf_info["gene_id"]))
    A = np.zeros((n_gene, n_gene))
    for ti, t in enumerate(tf_ids):
        gj = tf_gene[t]
        if gj in gidx:
            A[:, gidx[gj]] += W_prom[ti, :] + W_enh[ti, :] + W_sil[ti, :]
    A = scale_spectral_radius(A, float(params["target_spectral_radius"]))

    # 迭代
    d = float(params["solve_damping"])
    z = np.zeros((n_gene, n_cell))
    basal = np.asarray(expression_background["basal"])
    celltype_effect = np.asarray(expression_background["celltype_effect"])
    noise = np.asarray(expression_background["noise"])
    for it in range(int(params["solve_max_iter"])):
        drive = enhancer_fold * (module_effect + A @ np.tanh(z)) * silencer_factor
        gated = gate * drive
        z_new = ((1 - d) * z + d * (
            basal[:, None] + gated + chromatin_effect[:, None] + celltype_effect + noise))
        diff = np.max(np.abs(z_new - z))
        z = z_new
        if diff < float(params["solve_tol"]):
            break
    else:
        log.warning("[%s] 表达迭代未在 %d 步内收敛 (diff=%.4g)", allele, params["solve_max_iter"], diff)

    true_expr = softplus(z + np.log(imprinting_gate[:, None]))  # gene x cell
    df = pd.DataFrame(true_expr.T, index=cells, columns=gene_ids)
    df.index.name = "cell_id"
    return df


# ---------------------------------------------------------------------------
# 十七、RNA-seq counts (负二项)
# ---------------------------------------------------------------------------
def expression_to_counts(
    params: Dict[str, object], rng: np.random.Generator,
    true_expr: pd.DataFrame, cell_meta: pd.DataFrame, gene_ids: List[str],
) -> pd.DataFrame:
    """true expression -> NB counts. library size per cell, gene-specific dispersion."""
    cells = true_expr.index.tolist()
    expr = true_expr.to_numpy()
    # 每个细胞归一化 -> 比例
    prop = expr / (expr.sum(axis=1, keepdims=True) + EPS)
    ls = cell_meta.set_index("cell_id").loc[cells, "library_size"].to_numpy()
    mean = prop * ls[:, None]  # cell x gene
    disp = rng.gamma(shape=2.0, scale=float(params["negative_binomial_dispersion_mean"]) / 2.0,
                     size=len(gene_ids))
    disp = np.clip(disp, 0.5, None)
    r = disp[None, :]
    p = r / (r + mean + EPS)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    counts = nbinom.rvs(r, p, random_state=rng).astype(int)
    # 转录爆发/dropout: 低表达基因随机静默, 产生生物学零值
    mid = float(params["dropout_midpoint"])
    expr_prob = mean / (mean + mid)
    keep = rng.random(size=counts.shape) < expr_prob
    counts = counts * keep
    df = pd.DataFrame(counts, index=cells, columns=gene_ids)
    df.index.name = "cell_id"
    return df


# ---------------------------------------------------------------------------
# 十九、质量控制 summary
# ---------------------------------------------------------------------------
def compute_summary(
    ctx: Dict[str, object]
) -> pd.DataFrame:
    """计算关键 QC 指标 (坐标/开放/contact/GRN/表达/模块)."""
    m = {}
    gene = ctx["gene"]; regions = ctx["regions"]; elements = ctx["elements"]
    p_state = ctx["p_state"]; m_state = ctx["m_state"]
    edge = ctx["edge_table"]; p_expr = ctx["p_true"]; m_expr = ctx["m_true"]
    comb_counts = ctx["comb_counts"]; membership = ctx["membership"]

    m["n_genes"] = len(gene)
    m["n_elements"] = len(elements)
    m["n_regions"] = len(regions)
    m["gene_length_median"] = float(gene["gene_length"].median())
    for et in ["promoter", "enhancer", "silencer", "insulator"]:
        s = elements[elements["element_type"] == et]["element_length"]
        m[f"{et}_length_median"] = float(s.median()) if len(s) else 0.0
    m["region_length_median"] = float(regions["region_length"].median())

    # 开放状态
    st = regions["allelic_state"].value_counts(normalize=True)
    for k in ["shared_open", "shared_closed", "paternal_open", "maternal_open"]:
        m[f"frac_{k}"] = float(st.get(k, 0.0))
    for et in ["promoter", "enhancer", "silencer"]:
        m[f"{et}_open_frac"] = float(p_state[p_state["element_type"] == et]["open_state"].mean())
    m["accessibility_corr_pat_mat"] = float(np.corrcoef(
        p_state["accessibility"], m_state["accessibility"])[0, 1])
    m["activity_corr_pat_mat"] = float(np.corrcoef(
        p_state["activity"], m_state["activity"])[0, 1])
    for et in ["promoter", "enhancer"]:
        diff = np.abs(p_state[p_state["element_type"] == et]["accessibility"].to_numpy()
                      - m_state[m_state["element_type"] == et]["accessibility"].to_numpy())
        m[f"{et}_allelic_acc_diff_mean"] = float(diff.mean()) if len(diff) else 0.0

    # GRN
    m["paternal_edges"] = int((edge["paternal_weight"].abs() > 0).sum())
    m["maternal_edges"] = int((edge["maternal_weight"].abs() > 0).sum())
    m["shared_edge_frac"] = float(edge["shared_edge"].mean()) if len(edge) else 0.0
    m["paternal_specific_frac"] = float(edge["paternal_specific_edge"].mean()) if len(edge) else 0.0
    m["maternal_specific_frac"] = float(edge["maternal_specific_edge"].mean()) if len(edge) else 0.0
    if len(edge):
        m["grn_weight_corr"] = float(np.corrcoef(edge["paternal_weight"], edge["maternal_weight"])[0, 1])

    # 表达
    pv = p_expr.to_numpy().ravel(); mv = m_expr.to_numpy().ravel()
    m["expr_corr_pat_mat"] = float(np.corrcoef(pv, mv)[0, 1])
    m["count_sparsity"] = float((comb_counts.to_numpy() == 0).mean())

    # 模块内/间相关
    within, between = module_correlations(p_expr, membership)
    m["within_module_corr"] = within
    m["between_module_corr"] = between
    m["within_minus_between"] = within - between

    # 关闭区表达
    gene_region_state = ctx["gene_region_state"]
    closed_genes = [g for g in gene["gene_id"] if gene_region_state.get(g) == "shared_closed"]
    if closed_genes:
        m["shared_closed_mean_expr"] = float(p_expr[closed_genes].mean().mean())
    open_genes = [g for g in gene["gene_id"] if gene_region_state.get(g) == "shared_open"]
    if open_genes:
        m["shared_open_mean_expr"] = float(p_expr[open_genes].mean().mean())

    imprint_validation = ctx["imprinting_validation"]
    m["imprinted_gene_count"] = len(imprint_validation)
    if len(imprint_validation):
        m["imprinting_direction_accuracy"] = float(imprint_validation["direction_correct"].mean())
        m["imprinting_preferred_expression_ratio_median"] = float(
            imprint_validation["preferred_expression_ratio"].median())
        m["imprinting_preferred_expression_ratio_mean"] = float(
            imprint_validation["preferred_expression_ratio"].mean())

    return pd.DataFrame({"metric": list(m.keys()), "value": list(m.values())})


def build_imprinting_validation(ctx: Dict[str, object]) -> pd.DataFrame:
    gene = ctx["gene"]
    imprinted = gene[gene["imprinting_status"] != "none"].copy()
    if imprinted.empty:
        return pd.DataFrame()
    elements = ctx["elements"]
    promoters = elements[elements["element_type"] == "promoter"][
        ["element_id", "candidate_target_gene"]]
    p_acc = ctx["p_state"].set_index("element_id")["accessibility"]
    m_acc = ctx["m_state"].set_index("element_id")["accessibility"]
    p_expr = ctx["p_true"]
    m_expr = ctx["m_true"]
    records = []
    for _, row in imprinted.iterrows():
        gene_id = row["gene_id"]
        element_ids = promoters.loc[
            promoters["candidate_target_gene"] == gene_id, "element_id"]
        paternal_mean = float(p_expr[gene_id].mean())
        maternal_mean = float(m_expr[gene_id].mean())
        paternal_preferred = row["imprinting_status"] == "paternal_expressed"
        preferred = paternal_mean if paternal_preferred else maternal_mean
        nonpreferred = maternal_mean if paternal_preferred else paternal_mean
        records.append({
            "gene_id": gene_id,
            "imprinting_status": row["imprinting_status"],
            "imprinting_strength": row["imprinting_strength"],
            "paternal_imprinting_gate": row["paternal_imprinting_gate"],
            "maternal_imprinting_gate": row["maternal_imprinting_gate"],
            "paternal_promoter_accessibility": float(p_acc.reindex(element_ids).mean()),
            "maternal_promoter_accessibility": float(m_acc.reindex(element_ids).mean()),
            "paternal_mean_true_expression": paternal_mean,
            "maternal_mean_true_expression": maternal_mean,
            "preferred_expression_ratio": preferred / max(nonpreferred, EPS),
            "direction_correct": int(preferred > nonpreferred),
        })
    return pd.DataFrame(records)


def module_correlations(expr: pd.DataFrame, membership: pd.DataFrame) -> Tuple[float, float]:
    """计算模块内平均相关与模块间平均相关 (基因-基因 pearson)."""
    genes = [g for g in membership["gene_id"] if g in expr.columns]
    if len(genes) < 4:
        return 0.0, 0.0
    sub = expr[genes]
    C = np.corrcoef(sub.to_numpy().T)
    mod = membership.set_index("gene_id").loc[genes, "module_id"].to_numpy()
    n = len(genes)
    win_vals, bet_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(C[i, j]):
                continue
            (win_vals if mod[i] == mod[j] else bet_vals).append(C[i, j])
    within = float(np.mean(win_vals)) if win_vals else 0.0
    between = float(np.mean(bet_vals)) if bet_vals else 0.0
    return within, between


# ---------------------------------------------------------------------------
# 二十一、可视化
# ---------------------------------------------------------------------------
def make_plots(ctx: Dict[str, object], out_dir: str, params: Dict[str, object]) -> None:
    """生成核心可视化图 (>=18 张)."""
    dpi = int(params["plot_dpi"])
    gene = ctx["gene"]; regions = ctx["regions"]; elements = ctx["elements"]
    p_state = ctx["p_state"]; m_state = ctx["m_state"]; edge = ctx["edge_table"]
    p_expr = ctx["p_true"]; m_expr = ctx["m_true"]; membership = ctx["membership"]
    p_contact = ctx["p_contact"]; module_activity = ctx["module_activity"]
    comb_counts = ctx["comb_counts"]; cell_meta = ctx["cell_meta"]

    def save(name):
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, name), dpi=dpi); plt.close()

    # 1 gene density track (chr1)
    plt.figure(figsize=(10, 2.5))
    c0 = gene["chromosome"].iloc[0]
    plt.hist(gene[gene["chromosome"] == c0]["TSS"], bins=40, color="#4C78A8")
    plt.title(f"Gene density track {c0}"); plt.xlabel("pos"); save("plot_01_gene_density.png")

    # 2 chromatin region distribution
    plt.figure(figsize=(8, 3))
    plt.hist(regions["region_length"], bins=30, color="#72B7B2")
    plt.title("Chromatin region length"); save("plot_02_region_dist.png")

    # 3 paternal/maternal open regions track
    plt.figure(figsize=(10, 3))
    sub = regions[regions["chromosome"] == c0]
    for _, r in sub.iterrows():
        col = {"shared_open": "green", "shared_closed": "grey",
               "paternal_open": "blue", "maternal_open": "red"}[r["allelic_state"]]
        plt.axvspan(r["start"], r["end"], color=col, alpha=0.5)
    plt.title(f"Allelic open regions {c0}"); save("plot_03_open_regions.png")

    # 4 element length dist
    plt.figure(figsize=(8, 4))
    for et in ["promoter", "enhancer", "silencer", "insulator"]:
        s = elements[elements["element_type"] == et]["element_length"]
        if len(s):
            plt.hist(s, bins=25, alpha=0.5, label=et)
    plt.legend(); plt.title("Element length dist"); save("plot_04_element_length.png")

    # 5 region length (log)
    plt.figure(figsize=(7, 3))
    plt.hist(np.log10(regions["region_length"]), bins=30, color="#E45756")
    plt.title("log10 region length"); save("plot_05_region_log.png")

    # 6 pat vs mat accessibility
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(p_state["accessibility"], m_state["accessibility"], s=6, alpha=0.4)
    plt.xlabel("paternal acc"); plt.ylabel("maternal acc"); plt.title("Accessibility pat vs mat")
    save("plot_06_accessibility_scatter.png")

    # 7 methylation vs accessibility
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(p_state["methylation"], p_state["accessibility"], s=6, alpha=0.4, color="#B279A2")
    plt.xlabel("methylation"); plt.ylabel("accessibility"); save("plot_07_meth_acc.png")

    # 8 activity by element type
    plt.figure(figsize=(6, 4))
    data = [p_state[p_state["element_type"] == et]["activity"].to_numpy()
            for et in ["promoter", "enhancer", "silencer", "insulator"]]
    plt.boxplot(data, labels=["prom", "enh", "sil", "ins"])
    plt.title("Activity by element type"); save("plot_08_activity_box.png")

    # 9 contact vs distance
    plt.figure(figsize=(5, 4))
    plt.scatter(np.log1p(p_contact["distance_to_TSS"]), p_contact["allelic_contact_strength"],
                s=5, alpha=0.3)
    plt.xlabel("log(1+dist)"); plt.ylabel("contact"); plt.title("Contact vs distance")
    save("plot_09_contact_distance.png")

    # 10 insulator effect on contact
    plt.figure(figsize=(5, 4))
    g0 = p_contact[p_contact["insulator_between"] == 0]["allelic_contact_strength"]
    g1 = p_contact[p_contact["insulator_between"] == 1]["allelic_contact_strength"]
    plt.boxplot([g0.to_numpy(), g1.to_numpy()], labels=["no insulator", "insulator"])
    plt.title("Insulator effect on contact"); save("plot_10_insulator.png")

    # 11 pat vs mat GRN weight
    if len(edge):
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(edge["paternal_weight"], edge["maternal_weight"], s=6, alpha=0.4)
        plt.xlabel("paternal W"); plt.ylabel("maternal W"); plt.title("GRN weight pat vs mat")
        save("plot_11_grn_weight.png")

    # 12 GRN in/out degree
    if len(edge):
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1); plt.hist(edge["TF_id"].value_counts(), bins=15, color="#4C78A8")
        plt.title("TF out-degree")
        plt.subplot(1, 2, 2); plt.hist(edge["target_gene"].value_counts(), bins=15, color="#F58518")
        plt.title("gene in-degree"); save("plot_12_grn_degree.png")

    # 13 module activity heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(module_activity.to_numpy().T, aspect="auto", cmap="viridis")
    plt.colorbar(); plt.title("Module activity (module x cell)"); save("plot_13_module_activity.png")

    # 14 gene corr matrix sorted by module
    genes_sorted = membership.sort_values("module_id")["gene_id"].tolist()
    genes_sorted = [g for g in genes_sorted if g in p_expr.columns]
    if len(genes_sorted) > 3:
        C = np.corrcoef(p_expr[genes_sorted].to_numpy().T)
        plt.figure(figsize=(6, 5)); plt.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(); plt.title("Gene corr (module sorted)"); save("plot_14_gene_corr.png")

    # 15 within vs between module corr
    within, between = module_correlations(p_expr, membership)
    plt.figure(figsize=(4, 4))
    plt.bar(["within", "between"], [within, between], color=["#54A24B", "#B279A2"])
    plt.title("Module correlation"); save("plot_15_module_corr_bar.png")

    # 16 pat vs mat expression
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(p_expr.to_numpy().ravel()[::7], m_expr.to_numpy().ravel()[::7], s=4, alpha=0.3)
    plt.xlabel("paternal expr"); plt.ylabel("maternal expr"); save("plot_16_expr_scatter.png")

    # 17 differential accessibility vs differential expression (per gene, promoter)
    prom = elements[elements["element_type"] == "promoter"][["element_id", "candidate_target_gene"]]
    pa = dict(zip(p_state["element_id"], p_state["accessibility"]))
    ma = dict(zip(m_state["element_id"], m_state["accessibility"]))
    dacc, dexpr = [], []
    for _, r in prom.iterrows():
        g = r["candidate_target_gene"]
        if g in p_expr.columns:
            dacc.append(pa.get(r["element_id"], 0) - ma.get(r["element_id"], 0))
            dexpr.append(p_expr[g].mean() - m_expr[g].mean())
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(dacc, dexpr, s=8, alpha=0.5)
    plt.xlabel("Δaccessibility"); plt.ylabel("Δexpression"); plt.title("DA vs DE")
    save("plot_17_da_de.png")

    # 18 promoter accessibility vs expression
    pacc, pexp = [], []
    for _, r in prom.iterrows():
        g = r["candidate_target_gene"]
        if g in p_expr.columns:
            pacc.append(pa.get(r["element_id"], 0)); pexp.append(p_expr[g].mean())
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(pacc, pexp, s=8, alpha=0.5, color="#F58518")
    plt.xlabel("promoter accessibility"); plt.ylabel("mean expression")
    save("plot_18_prom_acc_expr.png")

    # 19 PCA of combined counts colored by cell type
    X = np.log1p(comb_counts.to_numpy())
    if X.shape[0] > 3:
        pcs = PCA(n_components=2).fit_transform(X)
        plt.figure(figsize=(5, 4))
        types = cell_meta.set_index("cell_id").loc[comb_counts.index, "cell_type"].to_numpy()
        for t in np.unique(types):
            mask = types == t
            plt.scatter(pcs[mask, 0], pcs[mask, 1], s=10, label=t, alpha=0.6)
        plt.legend(); plt.title("PCA combined counts"); save("plot_19_pca.png")


# ---------------------------------------------------------------------------
# 主流程 (含核心约束检查与重试)
# ---------------------------------------------------------------------------
def run_simulation(params: Dict[str, object], rng: np.random.Generator) -> Dict[str, object]:
    """执行一次完整模拟, 返回所有 dataframe 与上下文."""
    log.info("生成染色体信息...")
    chrom_info = generate_chromosome_info(params, rng)
    log.info("分层生成基因坐标...")
    gene = generate_gene_info(params, rng, chrom_info)
    log.info("生成 chromatin regions...")
    regions = generate_chromatin_regions(params, rng, chrom_info, gene)
    log.info("生成调控元件...")
    elements = generate_regulatory_elements(params, rng, chrom_info, gene, regions)
    log.info("选择 TF 并补充基因注释...")
    tf_info = generate_tf_info(params, rng, gene)
    gene = finalize_gene_annotation(gene, regions, tf_info["gene_id"].tolist())
    gene = assign_imprinting(params, rng, gene)

    log.info("赋 region 等位状态...")
    regions = assign_allelic_region_states(params, rng, regions, elements, gene)
    log.info("生成父母本元件状态...")
    shared_latent = rng.normal(0, 1, len(elements))
    p_state = generate_element_state(params, rng, elements, regions, gene, "paternal", shared_latent)
    m_state = generate_element_state(params, rng, elements, regions, gene, "maternal", shared_latent)

    log.info("生成细胞元数据与共享 TF activity...")
    cell_meta = generate_cell_metadata(params, rng)
    tf_activity = generate_tf_activity(params, rng, tf_info, cell_meta)

    log.info("生成调控模块与 module activity...")
    modules, membership, tf_links = generate_modules(params, rng, gene, tf_info)
    module_activity = compute_module_activity(params, rng, modules, tf_links, tf_activity, cell_meta)

    log.info("生成 TF motif (稀疏)...")
    motif = generate_motif(params, rng, tf_info, elements)
    log.info("生成父母本 contact (稀疏)...")
    p_contact, m_contact = generate_contact(params, rng, elements, gene, regions, p_state, m_state)

    log.info("构建父母本 GRN...")
    (W_prom_pat, W_prom_mat, W_enh_pat, W_enh_mat, W_sil_pat, W_sil_mat,
     W_pat, W_mat, edge_table, tf_ids, gene_ids) = build_grn(
        params, tf_info, gene, elements, motif, p_state, m_state, p_contact, m_contact, membership)

    log.info("迭代求解父母本表达...")
    expression_background = generate_expression_background(
        params, rng, gene_ids, cell_meta, tf_activity.index.tolist())
    p_true = solve_expression(params, rng, W_prom_pat, W_enh_pat, W_sil_pat,
                              tf_info, gene, tf_activity, module_activity,
                              membership, p_state, elements, regions, cell_meta, "paternal", expression_background)
    m_true = solve_expression(params, rng, W_prom_mat, W_enh_mat, W_sil_mat,
                              tf_info, gene, tf_activity, module_activity,
                              membership, m_state, elements, regions, cell_meta, "maternal", expression_background)
    comb_true = p_true + m_true

    log.info("生成 RNA-seq counts...")
    p_counts = expression_to_counts(params, rng, p_true, cell_meta, gene_ids)
    m_counts = expression_to_counts(params, rng, m_true, cell_meta, gene_ids)
    comb_counts = p_counts + m_counts

    gene_region_state = {
        g: regions.set_index("region_id").loc[r, "allelic_state"]
        for g, r in zip(gene["gene_id"], gene["chromatin_region_id"])
    }

    ctx = dict(
        chrom_info=chrom_info, gene=gene, regions=regions, elements=elements,
        tf_info=tf_info, cell_meta=cell_meta, tf_activity=tf_activity,
        modules=modules, membership=membership, tf_links=tf_links,
        module_activity=module_activity, motif=motif,
        p_state=p_state, m_state=m_state, p_contact=p_contact, m_contact=m_contact,
        W_prom_pat=W_prom_pat, W_prom_mat=W_prom_mat,
        W_enh_pat=W_enh_pat, W_enh_mat=W_enh_mat,
        W_sil_pat=W_sil_pat, W_sil_mat=W_sil_mat,
        W_pat=W_pat, W_mat=W_mat, edge_table=edge_table,
        p_true=p_true, m_true=m_true, comb_true=comb_true,
        p_counts=p_counts, m_counts=m_counts, comb_counts=comb_counts,
        gene_region_state=gene_region_state, tf_ids=tf_ids, gene_ids=gene_ids,
    )
    ctx["imprinting_validation"] = build_imprinting_validation(ctx)
    return ctx


def core_constraints_ok(summary: pd.DataFrame) -> Tuple[bool, List[str]]:
    """检查核心约束: 模块内>模块间相关, 关闭区表达<开放区, count 有零值."""
    s = dict(zip(summary["metric"], summary["value"]))
    fails = []
    if s.get("within_minus_between", 0) <= 0.02:
        fails.append("模块内相关未显著高于模块间")
    if ("shared_closed_mean_expr" in s and "shared_open_mean_expr" in s
            and s["shared_closed_mean_expr"] >= s["shared_open_mean_expr"]):
        fails.append("关闭区平均表达未低于开放区")
    if s.get("count_sparsity", 0) <= 0:
        fails.append("count 无零值 (缺乏稀疏性)")
    if s.get("imprinting_direction_accuracy", 1.0) < float(PARAMS["imprinting_min_direction_accuracy"]):
        fails.append("imprinting 基因表达方向正确率不足")
    if s.get("imprinting_preferred_expression_ratio_median", np.inf) < float(PARAMS["imprinting_min_preferred_ratio"]):
        fails.append("imprinting 基因偏好表达比不足")
    return len(fails) == 0, fails


def save_all(ctx: Dict[str, object], out_dir: str) -> List[str]:
    """保存所有 CSV 输出."""
    paths = []

    def w(df, name, index=False):
        p = os.path.join(out_dir, name)
        df.to_csv(p, index=index)
        paths.append(p)

    w(ctx["chrom_info"], "chromosome_info.csv")
    w(ctx["gene"], "gene_info.csv")
    w(ctx["gene"].loc[ctx["gene"]["imprinting_status"] != "none", [
        "gene_id", "chromosome", "chromatin_region_id", "imprinting_status", "imprinting_strength",
        "paternal_imprinting_gate", "maternal_imprinting_gate",
    ]], "imprinting_gene_info.csv")
    w(ctx["imprinting_validation"], "imprinting_expression_validation.csv")
    w(ctx["elements"], "regulatory_elements.csv")
    w(ctx["regions"], "chromatin_regions.csv")
    w(ctx["p_state"], "paternal_element_state.csv")
    w(ctx["m_state"], "maternal_element_state.csv")
    w(ctx["tf_info"], "shared_TF_info.csv")
    w(ctx["tf_activity"], "shared_TF_activity.csv", index=True)
    w(ctx["modules"], "regulatory_modules.csv")
    w(ctx["membership"], "gene_module_membership.csv")
    w(ctx["tf_links"], "TF_module_links.csv")
    w(ctx["module_activity"], "module_activity.csv", index=True)
    w(ctx["motif"], "TF_motif_matrix.csv")
    w(ctx["p_contact"], "paternal_contact_matrix.csv")
    w(ctx["m_contact"], "maternal_contact_matrix.csv")
    pd.DataFrame(ctx["W_pat"], index=ctx["tf_ids"], columns=ctx["gene_ids"]).to_csv(
        os.path.join(out_dir, "paternal_GRN.csv")); paths.append("paternal_GRN.csv")
    pd.DataFrame(ctx["W_mat"], index=ctx["tf_ids"], columns=ctx["gene_ids"]).to_csv(
        os.path.join(out_dir, "maternal_GRN.csv")); paths.append("maternal_GRN.csv")
    for allele, suffix in (("paternal", "pat"), ("maternal", "mat")):
        for element, key in (("promoter", "W_prom"), ("enhancer", "W_enh"), ("silencer", "W_sil")):
            name = f"{allele}_GRN_{element}.csv"
            pd.DataFrame(ctx[f"{key}_{suffix}"], index=ctx["tf_ids"], columns=ctx["gene_ids"]).to_csv(
                os.path.join(out_dir, name))
            paths.append(name)
    w(ctx["edge_table"], "GRN_edge_table.csv")
    w(ctx["p_true"], "paternal_true_expression.csv", index=True)
    w(ctx["m_true"], "maternal_true_expression.csv", index=True)
    w(ctx["comb_true"], "combined_true_expression.csv", index=True)
    w(ctx["p_counts"], "paternal_counts.csv", index=True)
    w(ctx["m_counts"], "maternal_counts.csv", index=True)
    w(ctx["comb_counts"], "combined_counts.csv", index=True)
    w(ctx["cell_meta"], "cell_metadata.csv")
    return paths


def main() -> None:
    params = dict(PARAMS)
    validate_params(params)
    out_dir = str(params["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    ctx = None
    summary = None
    for attempt in range(1, 4):
        log.info("=== 模拟尝试 %d ===", attempt)
        rng = np.random.default_rng(int(params["seed"]) + attempt - 1)
        ctx = run_simulation(params, rng)
        summary = compute_summary(ctx)
        ok, fails = core_constraints_ok(summary)
        if ok:
            log.info("核心约束满足.")
            break
        log.warning("核心约束未满足: %s -> 调整参数重试", fails)
        # 自动调整关键参数
        params["module_noise_sd"] = max(0.05, float(params["module_noise_sd"]) * 0.8)
        params["closed_promoter_basal_leak"] = max(0.01, float(params["closed_promoter_basal_leak"]) * 0.7)
        params["gene_noise_sd"] = max(0.1, float(params["gene_noise_sd"]) * 0.9)
        params["dropout_midpoint"] = float(params["dropout_midpoint"]) * 1.5
    paths = save_all(ctx, out_dir)
    summary.to_csv(os.path.join(out_dir, "simulation_summary.csv"), index=False)
    log.info("生成可视化...")
    make_plots(ctx, out_dir, params)

    # 结尾打印
    s = dict(zip(summary["metric"], summary["value"]))
    print("\n" + "=" * 60)
    print("模拟完成. 输出目录:", out_dir)
    print("-" * 60)
    print("CSV 输出文件:")
    for p in paths + [os.path.join(out_dir, "simulation_summary.csv")]:
        print("  ", os.path.basename(p))
    print("-" * 60)
    print(f"基因数量           : {s['n_genes']:.0f}")
    print(f"调控元件数量       : {s['n_elements']:.0f}")
    print(f"chromatin region数 : {s['n_regions']:.0f}")
    print(f"TF 数量            : {len(ctx['tf_ids'])}")
    print(f"paternal GRN 边数  : {s['paternal_edges']:.0f}")
    print(f"maternal GRN 边数  : {s['maternal_edges']:.0f}")
    print(f"shared edge 比例   : {s['shared_edge_frac']:.3f}")
    print(f"accessibility 相关 : {s['accessibility_corr_pat_mat']:.3f}")
    print(f"expression 相关    : {s['expr_corr_pat_mat']:.3f}")
    print(f"模块内相关         : {s['within_module_corr']:.3f}")
    print(f"模块间相关         : {s['between_module_corr']:.3f}")
    print(f"count 稀疏度       : {s['count_sparsity']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
