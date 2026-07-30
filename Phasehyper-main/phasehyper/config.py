from __future__ import annotations

from pathlib import Path
from typing import Dict, List

REAL_PHASE_DATASETS = (
    "PEA_STA",
    "sc_GEM",
    "CITE_seq",
    "SCoPE2",
    "scNMT",
)

DATASET_CONFIG = {
    "PEA_STA": {
        "name": "PEA_STA",
        "description": "Protein Expression and STA data",
        "root": Path(r"PEA_STA"),
        "files": {
            "expression": "expression_data.csv",
            "view": ["protein_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_pea.txt",
            "ppi_prior": "human_ppi.csv",
        },
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "protein", "file": "protein_data.csv",
             "node_type": "protein", "bridge": None, "binary": False},
        ],
        "labels": {
            "file": "cell_stage.csv",
            "kind": "stage_or_treatment",
            "normalizer": "pea_sta",
            "optional_header_tokens": [],
            "expected_names": ["0h_control", "6d_control", "6d_BMP4"],
        },
        "has_ppi": True,
        "have_answer": True,
    },
    "sc_GEM": {
        "name": "sc_GEM",
        "description": "Single-cell Gene Expression Methylation data",
        "root": Path(r"sc_GEM"),
        "files": {
            "expression": "expression_data.csv",
            "view": ["methylation_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_sc.txt",
            "ppi_prior": "",
        },
        # 两层级超图路径需要 modalities：RNA 为 gene 节点（identity），
        # 甲基化为 methylation 节点（binary；build_two_level_hypergraph 不直接使用，
        # 但保留在 HeteroBundle 供后续扩展）。
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "methylation", "file": "methylation_data.csv",
             "node_type": "methylation", "bridge": None, "binary": True},
        ],
        "labels": {
            "file": "cell_stage.csv",
            "kind": "stage",
            "normalizer": "identity",
            "optional_header_tokens": [],
        },
        "has_ppi": False,
        "have_answer": False,
    },
    "CITE_seq": {
        "name": "CITE_seq",
        "description": "CITE-seq CBMC: RNA counts + 13 ADT proteins + celltype truth",
        "root": Path(r"CITE_seq"),
        "files": {
            "expression": "expression_data.csv",
            "view": ["protein_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_cite.txt",
            "ppi_prior": "human_ppi.csv",
            "protein_gene_map": "protein_gene_map.csv",
        },
        # 异构多模态清单（cell 共享轴；RNA=gene 节点/identity 桥，protein=protein 节点/中心法则桥）。
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "protein", "file": "protein_data.csv",
             "node_type": "protein", "bridge": "protein_gene_map.csv", "binary": False},
        ],
        "labels": {
            "file_candidates": ["cell_type.csv", "cell_stage.csv"],
            "kind": "cell_type",
            "normalizer": "identity",
            "optional_header_tokens": ["celltype", "cell_type"],
        },
        "has_ppi": True,
        "have_answer": True,
    },
    "sim_gene100_alpha_1_beta_2": {
        "name": "sim_gene100_alpha_1_beta_2",
        "description": "Simulated Gene Expression data with alpha=0.1, beta=0.2",
        "root": Path(r"sim_gene100/alpha_1_beta_2"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "",
        },
        "has_ppi": False,
        "have_answer": True,
    },
    "sim_gene100_alpha_1_beta_1": {
        "name": "sim_gene100_alpha_1_beta_1",
        "description": "Simulated Gene Expression data with alpha=0.1, beta=0.1",
        "root": Path(r"sim_gene100/alpha_1_beta_1"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "",
        },
        "has_ppi": False,
        "have_answer": True,
    },
    "sim_gene100_alpha_2_beta_1": {
        "name": "sim_gene100_alpha_2_beta_1",
        "description": "Simulated Gene Expression data with alpha=0.2, beta=0.1",
        "root": Path(r"sim_gene100/alpha_2_beta_1"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "",
        },
        "has_ppi": False,
        "have_answer": True,
    },
    "sim_gene100_alpha_2_beta_2": {
        "name": "sim_gene100_alpha_2_beta_2",
        "description": "Simulated Gene Expression data with alpha=0.2, beta=0.2",
        "root": Path(r"sim_gene100/alpha_2_beta_2"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "kegg_prior.txt",
            "poswin_prior": "poswin_prior.txt",
            "ppi_prior": "",
        },
        "has_ppi": False,
        "have_answer": True,
    },
    # ratio_correlation_diffcov 模拟数据（单模态 RNA；由 prepare_ratio_sim.py 转入
    # data_clean/）。两层级 HGNN-VAE 走 modalities 异构路径，仅 RNA=gene identity 节点，
    # 无 protein/methylation。先验：KEGG(hsa00001.txt) + 位置(gene_positions_pea.txt,
    # 命名沿用 priors._position_file_for_dataset 对未知数据集的默认名) + 合成 PPI(human_ppi.csv)。
    "ratio_genepos": {
        "name": "ratio_genepos",
        "description": "ratio_correlation_diffcov / gene_position: 比例位置相关，1000 基因 100 细胞 10 类",
        "root": Path(r"ratio_genepos"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_pea.txt",
            "ppi_prior": "human_ppi.csv",
        },
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
        ],
        "has_ppi": True,
        "have_answer": True,
    },
    "ratio_kegg": {
        "name": "ratio_kegg",
        "description": "ratio_correlation_diffcov / position_with_kegg: 比例位置+KEGG 相关，1000 基因 100 细胞 10 类",
        "root": Path(r"ratio_kegg"),
        "files": {
            "expression": "expression_data.csv",
            "view": [],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_pea.txt",
            "ppi_prior": "human_ppi.csv",
        },
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
        ],
        "has_ppi": True,
        "have_answer": True,
    },
    "SCoPE2": {
        "name": "SCoPE2",
        "description": "SCoPE2 单细胞蛋白组学：mRNA 表达 + 蛋白定量，1490 细胞 2 类(sc_m0/sc_u)",
        "root": Path(r"SCoPE2"),
        "files": {
            "expression": "expression_data.csv",
            "view": ["protein_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_scope2.txt",
            "ppi_prior": "human_ppi.csv",
        },
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "protein", "file": "protein_data.csv",
             "node_type": "protein", "bridge": None, "binary": False},
        ],
        "labels": {
            "file": "cell_stage.csv",
            "kind": "cell_type",
            "normalizer": "identity",
            "optional_header_tokens": ["celltype", "cell_type"],
        },
        "has_ppi": True,
        "have_answer": True,
    },
    "scNMT": {
        "name": "scNMT",
        "description": "scNMT-seq 小鼠胚胎：RNA 表达 + 启动子甲基化 + 启动子可及性，1940 细胞 3 类(E5.5/E6.5/E7.5)",
        "root": Path(r"scNMT"),
        "files": {
            "expression": "expression_data.csv",
            "view": ["methylation_data.csv", "accessibility_data.csv"],
            "stage": "cell_stage.csv",
            "kegg_prior": "hsa00001.txt",
            "poswin_prior": "gene_positions_scnmt.txt",
            "ppi_prior": "human_ppi.csv",
        },
        "modalities": [
            {"name": "rna", "file": "expression_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "methylation", "file": "methylation_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
            {"name": "accessibility", "file": "accessibility_data.csv",
             "node_type": "gene", "bridge": None, "binary": False},
        ],
        "labels": {
            "file": "cell_stage.csv",
            "kind": "stage",
            "normalizer": "identity",
            "optional_header_tokens": [],
        },
        "has_ppi": False,
        "have_answer": True,
    },
}

CLUSTER_VERSION_NAMES = {
    "kmeans": "TV_PHASE_v11",
    "leiden": "TV_PHASE_v11",
    "louvain": "TV_PHASE_v11",
}

DEFAULT_LEIDEN_RESOLUTION = {
    "sc_GEM": 1.5,
    "PEA_STA": 0.5,
}

DEFAULT_CLUSTER_METHODS = tuple(CLUSTER_VERSION_NAMES.keys())
GENE_POSITION_MODES = ("none", "feature", "chrom", "near", "full")
DEFAULT_GENE_POSITION_MODE = "full"
DEFAULT_PHASE_RESIDUAL_SCALE_METHYLATION = 1.0
DEFAULT_PHASE_RESIDUAL_SCALE_PPI = 0.05

# 基因名别名表：数据里出现的名字（KEY，统一大写）-> 候选 canonical 名（VALUE，首个为首选）。
# 同时用于：位置先验匹配、KEGG/PPI 匹配（清洗阶段把数据列名规范成 canonical 后，
# 下游 priors.py 的精确匹配即可命中），故由原 GENE_POSITION_ALIASES 提升为通用 GENE_ALIASES。
GENE_ALIASES: Dict[str, List[str]] = {
    # sc_GEM 干细胞标记 / 旧符号
    "OCT4": ["POU5F1"],
    "NESTIN": ["NES"],
    "LEFTY": ["LEFTY1"],
    "TMEM173": ["STING1"],
    # PEA_STA R 点号名 / 旧符号
    "CXCL8.IL.8": ["CXCL8", "IL8"],
    "IGFBP.2": ["IGFBP2"],
    "TENASCIN.C": ["TNC"],
    "SNAL1": ["SNAI1"],
    "CASPR1": ["CNTNAP1"],
    "HS1": ["HCLS1"],
    # CITE_seq ADT 标记 -> 编码基因（中心法则桥；用于把蛋白名对到基因符号）
    "CD8": ["CD8A"],
    "CD16": ["FCGR3A"],
    "CD45RA": ["PTPRC"],
    "CD56": ["NCAM1"],
    "CD11C": ["ITGAX"],
    "CD10": ["MME"],
    "CD3": ["CD3D", "CD3E", "CD3G"],
}

# 向后兼容：priors.py 仍以 GENE_POSITION_ALIASES 名引用同一张表。
GENE_POSITION_ALIASES: Dict[str, List[str]] = GENE_ALIASES

METH_PHASE_LABEL_MAP = {
    0: "BJ",
    1: "IPS",
    2: "ES",
    3: "d24T+",
    4: "d24T-",
    5: "d16T+",
    6: "d16T-",
    7: "d8",
    8: "Other",
}

PPI_PHASE_LABEL_MAP = {
    0: "0h_control",
    1: "6d_BMP4",
    2: "6d_control",
}
