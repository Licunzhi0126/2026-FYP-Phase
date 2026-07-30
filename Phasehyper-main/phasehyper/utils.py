from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def clean_nan(x):
    if isinstance(x, torch.Tensor):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _standardize_matrix(array: np.ndarray) -> np.ndarray:
    standardized = StandardScaler().fit_transform(np.asarray(array, dtype=np.float32))
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _safe_standardize(x: np.ndarray) -> np.ndarray:
    return _standardize_matrix(
        np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    )


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


def _match_embedding_to_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    current_dim = int(array.shape[1]) if array.ndim == 2 else 0
    if current_dim == target_dim:
        return array.astype(np.float32, copy=False)
    if current_dim > target_dim:
        return array[:, :target_dim].astype(np.float32, copy=False)
    pad = np.zeros((array.shape[0], target_dim - current_dim), dtype=np.float32)
    return np.concatenate([array, pad], axis=1).astype(np.float32, copy=False)
