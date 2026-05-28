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

def clean_nan(x):
    if isinstance(x, torch.Tensor):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _standardize_matrix(array: np.ndarray):
    standardized = StandardScaler().fit_transform(np.asarray(array, dtype=np.float32))
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _reduce_to_fixed_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n_samples, n_features = array.shape
    if n_samples == 0:
        return np.zeros((0, target_dim), dtype=np.float32)

    max_components = min(n_samples, n_features)
    if max_components <= 1:
        reduced = array[:, :1] if n_features > 0 else np.zeros((n_samples, 1), dtype=np.float32)
    else:
        n_components = min(target_dim, max_components)
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(array).astype(np.float32)

    if reduced.shape[1] < target_dim:
        pad = np.zeros((n_samples, target_dim - reduced.shape[1]), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)
    elif reduced.shape[1] > target_dim:
        reduced = reduced[:, :target_dim]
    return reduced.astype(np.float32, copy=False)

__all__ = [name for name in globals() if not name.startswith("__")]
