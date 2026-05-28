from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import leidenalg
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    umap = None

from .config import *

def _methylation_stage_name(cell_name: str) -> str:
    cell_name = str(cell_name)
    if cell_name.startswith("BJ_"):
        return "BJ"
    if cell_name.startswith("IPS_"):
        return "IPS"
    if cell_name.startswith("ES_"):
        return "ES"
    if "d24T+" in cell_name:
        return "d24T+"
    if "d24T-" in cell_name:
        return "d24T-"
    if "d16T+" in cell_name:
        return "d16T+"
    if "d16T-" in cell_name:
        return "d16T-"
    if "d8" in cell_name:
        return "d8"
    return "Other"

def _ppi_stage_name(cell_name: str) -> str:
    name = str(cell_name).strip()
    low = name.lower()
    has_control = "contol" in low or "control" in low
    if "6d" in low and "bmp4" in low:
        return "6d_BMP4"
    if "6d" in low and has_control:
        return "6d_control"
    if "0h" in low and has_control:
        return "0h_control"
    if "6d" in low:
        return "6d_control"
    if "0h" in low:
        return "0h_control"
    return "6d_BMP4"

def _normalize_ppi_stage_token(token: str) -> str:
    token = str(token).strip()
    low = token.lower()
    if "_" in token or "bmp4" in low or "0h" in low or "6d" in low:
        return _ppi_stage_name(token)
    if token:
        return "6d_BMP4"
    return "6d_BMP4"

def _load_feature_frame(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id").fillna(0)
    df.columns = [str(col).strip() for col in df.columns]

    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
    
    return df

def _read_stage_tokens(stage_path: Path) -> List[str]:
    if not stage_path.exists():
        return []
    try:
        text = stage_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        line = text.splitlines()[0].strip()
        tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
        if tokens and tokens[0].lower() in {"celltype", "label", "stage"}:
            tokens = tokens[1:]
        return tokens
    except Exception as exc:
        warnings.warn(f"Failed to read stage tokens from {stage_path}: {exc}")
        return []

def _labels_to_ids(label_names: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    ordered = []
    seen = set()

    for name in label_names:
        name = str(name).strip()
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    mapping = {name: idx for idx, name in enumerate(ordered)}
    inverse = {idx: name for name, idx in mapping.items()}

    labels = np.asarray(
        [mapping[str(name).strip()] for name in label_names],
        dtype=np.int64
    )

    return labels, inverse

def _load_stage_names_simple(stage_path: Path, ordered_cells: List[str]) -> Dict[str, str]:
    tokens = _read_stage_tokens(stage_path)

    if not tokens:
        raise ValueError("Stage file is empty or missing")

    if len(tokens) != len(ordered_cells):
        raise ValueError(
            "Stage count does not match number of cells. "
            "Please ensure one-to-one alignment."
        )
    
    return {
        cell: tok
        for cell, tok in zip(ordered_cells, tokens)
    }

def _resolve_dataset_type(base_dir: Path, dataset_type: str) -> str:
    if dataset_type in DATASET_CONFIG:
        return dataset_type
    raise ValueError(f"Unknown dataset type: {dataset_type}. Valid types: {list(DATASET_CONFIG.keys())}")

def load_dataset(base_dir, dataset_type):
    base_dir = Path(base_dir)
    dataset_type = _resolve_dataset_type(base_dir, dataset_type)
    
    # 根据 dataset_type 获取对应的配置
    config = DATASET_CONFIG[dataset_type]
    files = config["files"]
    root = config["root"]
    
    # 始终使用 DATASET_CONFIG 中配置的 root 作为 base_dir
    # 这样可以确保 dataset_type 和 base_dir 的一致性
    base_dir = root
    
    view1_dfs = []
    view1_name = config.get("view_name", dataset_type)
    
    # 加载 expression 数据
    expression_file = base_dir / files["expression"]
    expression_df = _load_feature_frame(expression_file)
    
    # 加载 view1 数据 - 从 DATASET_CONFIG 配置中读取
    view1_file_names = files.get("view", [])
    if view1_file_names:
        for view1_file_name in view1_file_names:
            view1_dfs.append(_load_feature_frame(base_dir / view1_file_name))
        if len(view1_file_names) > 1:
            view1_name = f"{dataset_type}_multi"
    
    # 加载 stage 数据
    stage_file = base_dir / files["stage"]
    
    # 获取 cell 列表（优先从 view1，没有则从 expression）
    if view1_dfs:
        cells = view1_dfs[0].index.tolist()
    else:
        cells = expression_df.index.tolist()
    
    stage_name_by_cell = _load_stage_names_simple(stage_file, cells)
    
    # PPI 数据集需要特殊处理 stage
    if config.get("has_ppi", False):
        stage_name_by_cell = {
            cell: _normalize_ppi_stage_token(stage)
            for cell, stage in stage_name_by_cell.items()
        }
    
    return view1_dfs, expression_df, stage_name_by_cell, view1_name

__all__ = [name for name in globals() if not name.startswith("__")]
