"""由根目录**全量参考**重建 gene-gene 先验文件（KEGG / 位置 / PPI）。

动机（见仓库 README_数据清洗.md 第 7 节）：数据集自带的 `hsa00001.txt` /
`gene_positions_*.txt` 是**不完整子集**，导致先验命中"零增益"。CITE_seq 干脆不带这些，
所以从根目录全量参考解析：
  - `hsa00001.json`                       —— 全 KEGG BRITE，解析 gene→pathway
  - `gencode.v38.basic.annotation.gtf(.gz)` —— 全基因坐标，解析 gene→(chrom,start,end,strand)
  - `human_ppi.csv`                        —— 11745² 互作矩阵，裁 gene 子矩阵（复用 cleaning.trim_ppi_submatrix）

**只写出既有 `priors.py` 能直接读的 tab 格式**，下游 priors 解析逻辑一字不改：
  - KEGG：`build_gene_prior_features`（priors.py:23）读 col0=gene、col2=pathway。
  - 位置：`_load_position_prior`（priors.py:155）读每行前 5 个 tab 字段 gene/chrom/start/end/strand。
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from phasehyper.data.cleaning import trim_ppi_submatrix


# ======================================================================
# KEGG：hsa00001.json（BRITE 树）-> gene -> pathway
# ======================================================================

def _parse_kegg_gene_symbol(leaf_name: str) -> Optional[str]:
    """KEGG 基因叶子名 -> 基因符号。

    叶子形如 ``"3098 HK1; hexokinase 1\\tK00844 HK; hexokinase [EC:2.7.1.1]"``。
    取制表符前的首段 ``"3098 HK1; hexokinase 1"``：首 token 是 Entrez 数字 ID，
    其后到第一个 ';' 之间是基因符号。非"数字开头"的叶子（非基因）返回 None。
    """
    head = str(leaf_name).split("\t", 1)[0].strip()
    parts = head.split(" ", 1)
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    symbol = parts[1].split(";", 1)[0].strip()
    return symbol or None


def _iter_kegg_gene_pathway(node: dict):
    """DFS 产出 (gene_symbol, pathway_name)。

    pathway = 基因叶子的**父节点名**（KEGG BRITE 中 C 层即通路，D 层即基因）。
    """
    children = node.get("children")
    if not children:
        return
    pathway_name = str(node.get("name", "")).strip()
    for child in children:
        if child.get("children"):
            yield from _iter_kegg_gene_pathway(child)
        else:
            gene = _parse_kegg_gene_symbol(child.get("name", ""))
            if gene:
                yield gene, pathway_name


def parse_kegg_json(
    json_path, gene_set: Optional[set] = None, *, pathway_only: bool = True
) -> Dict[str, List[str]]:
    """解析 hsa00001.json -> {gene: [pathway, ...]}（去重，可选按 gene_set 过滤）。

    pathway_only=True：只保留真正的代谢/信号通路（名字含 ``[PATH:``），
    丢弃 BRITE 功能分类（``[BR:``，如 "CD molecules"/"Membrane trafficking"）——
    后者会形成超大低信息超边。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    gene_to_pathways: Dict[str, List[str]] = {}
    for gene, pathway in _iter_kegg_gene_pathway(root):
        if gene_set is not None and gene not in gene_set:
            continue
        if not pathway:
            continue
        if pathway_only and "[PATH:" not in pathway:
            continue
        lst = gene_to_pathways.setdefault(gene, [])
        if pathway not in lst:
            lst.append(pathway)
    return gene_to_pathways


def write_kegg_txt(gene_to_pathways: Dict[str, List[str]], out_path) -> int:
    """写 tab 文件，每 (gene, pathway) 一行：`gene\\tpathway\\tpathway`。

    第 2 列与第 3 列同为 pathway——只为满足 priors.build_gene_prior_features 用 col2 取 pathway。
    返回写出的行数。
    """
    lines: List[str] = []
    for gene, pathways in gene_to_pathways.items():
        for pathway in pathways:
            lines.append(f"{gene}\t{pathway}\t{pathway}")
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


# ======================================================================
# 位置：gencode GTF -> gene -> (chrom, start, end, strand)
# ======================================================================

_GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_gtf_positions(
    gtf_path, gene_set: Optional[set] = None
) -> Dict[str, Dict[str, object]]:
    """流式扫 GTF 的 ``feature==gene`` 行 -> {gene: {chrom,start,end,strand}}。

    重名基因保留首次出现（与 priors._load_position_prior 的 keep-first 语义一致）。
    """
    positions: Dict[str, Dict[str, object]] = {}
    with _open_text(gtf_path) as fh:
        for raw in fh:
            if raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            m = _GENE_NAME_RE.search(fields[8])
            if not m:
                continue
            gene = m.group(1).strip()
            if gene in positions:
                continue
            if gene_set is not None and gene not in gene_set:
                continue
            positions[gene] = {
                "chrom": fields[0].strip(),
                "start": fields[3].strip(),
                "end": fields[4].strip(),
                "strand": fields[6].strip(),
            }
    return positions


def write_positions_txt(positions: Dict[str, Dict[str, object]], out_path) -> int:
    """写 tab 文件：`gene\\tchrom\\tstart\\tend\\tstrand`（priors 读前 5 字段）。"""
    lines = [
        f"{g}\t{p['chrom']}\t{p['start']}\t{p['end']}\t{p['strand']}"
        for g, p in positions.items()
    ]
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


# ======================================================================
# 一站式：为给定 gene 集重建三先验，写进 out_dir
# ======================================================================

def build_priors_for_genes(
    genes: List[str],
    *,
    kegg_json,
    gtf_path,
    ppi_csv,
    out_dir,
    kegg_out_name: str = "hsa00001.txt",
    pos_out_name: str = "gene_positions_cite.txt",
    ppi_out_name: str = "human_ppi.csv",
) -> Dict[str, object]:
    """从全量参考重建 KEGG / 位置 / PPI 三先验文件到 out_dir，返回命中统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_list = [str(g).strip() for g in genes]
    gene_set = set(gene_list)

    # KEGG
    kegg_map = parse_kegg_json(kegg_json, gene_set=gene_set)
    n_kegg_rows = write_kegg_txt(kegg_map, out_dir / kegg_out_name)
    kegg_hit = len(set(kegg_map.keys()))

    # 位置
    positions = parse_gtf_positions(gtf_path, gene_set=gene_set)
    n_pos = write_positions_txt(positions, out_dir / pos_out_name)
    pos_hit = len(positions)

    # PPI（裁子矩阵）
    ppi_df = trim_ppi_submatrix(ppi_csv, gene_list)
    ppi_df.to_csv(out_dir / ppi_out_name)
    ppi_genes = int(ppi_df.shape[0])
    ppi_edges = int((ppi_df.to_numpy() != 0).sum())

    n_total = len(gene_set)
    return {
        "n_genes": n_total,
        "kegg": {"hit": kegg_hit, "total": n_total, "rows": n_kegg_rows,
                 "missing": sorted(gene_set - set(kegg_map.keys()))},
        "position": {"hit": pos_hit, "total": n_total, "rows": n_pos,
                     "missing": sorted(gene_set - set(positions.keys()))},
        "ppi": {"genes_in_ppi": ppi_genes, "nonzero_edges": ppi_edges,
                "shape": list(ppi_df.shape)},
    }
