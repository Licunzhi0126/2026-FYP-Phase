"""Strict alignment checks; never align on row position silently."""
import numpy as np
import pandas as pd

def matrix_from_csv(frame: pd.DataFrame, id_col='cell_id'):
    if id_col not in frame:
        raise ValueError(f"Missing {id_col} in matrix")
    if frame[id_col].duplicated().any():
        raise ValueError(f"Duplicate {id_col}")
    mat = frame.set_index(id_col)
    if mat.columns.duplicated().any():
        raise ValueError("Duplicate matrix columns")
    mat = mat.astype(float)
    if not np.isfinite(mat.to_numpy()).all():
        raise ValueError("Nonfinite numeric data")
    return mat

def require_pair(a, b, name):
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError(f"{name}: index/columns differ; positional comparison forbidden")
    if not np.isfinite(a.to_numpy(dtype=float)).all() or not np.isfinite(b.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: NaN/inf")

def require_unique(df, key, name):
    if key not in df or df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f"{name}: missing/duplicate key {key}")
