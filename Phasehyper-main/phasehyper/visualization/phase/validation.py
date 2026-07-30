"""Validation and numeric safety helpers for phase visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import PhaseVisualizationBundle


REQUIRED_REPRESENTATIONS = ("Raw_RNA", "cell_h", "Phase_A", "Phase_B")
REQUIRED_METRICS = ("NMI", "FMI", "ARI", "ASW", "PredClusters")


def finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape={matrix.shape}")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} has insufficient shape={matrix.shape}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def validate_bundle(bundle: PhaseVisualizationBundle) -> None:
    n_cells = len(bundle.cell_ids)
    n_genes = len(bundle.genes)
    if len(set(bundle.cell_ids)) != n_cells:
        raise ValueError("cell_ids contain duplicates")
    if len(set(bundle.genes)) != n_genes:
        raise ValueError("genes contain duplicates")
    if bundle.labels.reshape(-1).shape[0] != n_cells:
        raise ValueError("label length does not match cell count")
    if len(bundle.label_names) != n_cells:
        raise ValueError("label_names length does not match cell count")
    for name, values in {
        "raw_rna": bundle.raw_rna,
        "phase_a": bundle.phase_a,
        "phase_b": bundle.phase_b,
    }.items():
        if values.shape != (n_cells, n_genes):
            raise ValueError(
                f"{name} shape {values.shape} does not match {(n_cells, n_genes)}"
            )
    if bundle.cell_h.shape[0] != n_cells:
        raise ValueError("cell_h row count does not match cell count")
    for representation in REQUIRED_REPRESENTATIONS:
        fields = bundle.metrics.get(representation)
        if not isinstance(fields, dict):
            raise ValueError(f"metrics missing representation {representation}")
        missing = [name for name in REQUIRED_METRICS if name not in fields]
        if missing:
            raise ValueError(f"metrics[{representation}] missing {missing}")
    if not bundle.training_history.empty and "epoch" in bundle.training_history:
        epoch = bundle.training_history["epoch"].to_numpy()
        if np.any(np.diff(epoch) <= 0):
            raise ValueError("training epochs must be strictly increasing")


def safe_corrcoef(values: np.ndarray) -> np.ndarray:
    values = finite_matrix(values, "correlation input")
    keep = np.nanstd(values, axis=0) > 1e-12
    if keep.sum() < 2:
        raise ValueError("correlation requires at least two non-constant features")
    corr = np.corrcoef(values[:, keep], rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    median = float(values.median())
    mad = float((values - median).abs().median())
    if not np.isfinite(mad) or mad <= 1e-12:
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - float(values.mean())) / std
    return (values - median) / (1.4826 * mad)
