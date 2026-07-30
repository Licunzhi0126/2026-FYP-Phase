from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _match_embedding_dim(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
    embedding = embedding.float()
    emb_dim = int(embedding.shape[1])
    if emb_dim == target_dim:
        return embedding
    if emb_dim > target_dim:
        return embedding[:, :target_dim]
    return F.pad(embedding, (0, target_dim - emb_dim))


def _align_embedding_to_names(
    source_names: List[str], embedding: np.ndarray, target_names: List[str]
) -> np.ndarray:
    source_names = [str(name) for name in source_names]
    target_names = [str(name) for name in target_names]
    embedding = np.asarray(embedding, dtype=np.float32)
    source_map = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_map]
    if missing:
        raise KeyError(f"Missing cell names during embedding alignment: {missing[:5]}")
    return np.stack(
        [embedding[source_map[name]] for name in target_names], axis=0
    ).astype(np.float32, copy=False)


def _expression_only_original_embedding(
    expression_df: pd.DataFrame, sample_names: List[str]
) -> np.ndarray:
    sample_names = [str(name) for name in sample_names]
    missing = [name for name in sample_names if name not in expression_df.index]
    if missing:
        raise KeyError(
            f"Missing sample names during original embedding export: {missing[:5]}"
        )
    return expression_df.loc[sample_names].values.astype(np.float32)


def build_expression_space_phase_embeddings(
    df_cell: pd.DataFrame,
    expression_df: pd.DataFrame,
    sample_names: List[str],
    common_genes: List[str],
) -> Dict[str, np.ndarray]:
    sample_names = [str(name) for name in sample_names]
    common_genes = [str(gene) for gene in common_genes]

    original_expression_embedding = (
        expression_df.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)
    )

    maternal_pivot = df_cell.pivot(index="cell", columns="gene", values="maternal_expression")
    paternal_pivot = df_cell.pivot(index="cell", columns="gene", values="paternal_expression")

    maternal_expression_embedding = (
        maternal_pivot.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)
    )
    paternal_expression_embedding = (
        paternal_pivot.loc[sample_names, common_genes].fillna(0).values.astype(np.float32)
    )

    shapes = {
        "original_expression_embedding": original_expression_embedding.shape,
        "maternal_expression_embedding": maternal_expression_embedding.shape,
        "paternal_expression_embedding": paternal_expression_embedding.shape,
    }

    if not all(shape == shapes["original_expression_embedding"] for shape in shapes.values()):
        shape_msg = "\n".join([f"  {name}: {shape}" for name, shape in shapes.items()])
        raise ValueError(
            f"Embedding shape mismatch after alignment:\n{shape_msg}\n"
            f"sample_names count: {len(sample_names)}\n"
            f"common_genes count: {len(common_genes)}\n"
            f"maternal_pivot shape: {maternal_pivot.shape}, index: {list(maternal_pivot.index[:5])}, columns: {list(maternal_pivot.columns[:5])}\n"
            f"paternal_pivot shape: {paternal_pivot.shape}, index: {list(paternal_pivot.index[:5])}, columns: {list(paternal_pivot.columns[:5])}"
        )

    return {
        "original_expression_embedding": original_expression_embedding,
        "maternal_expression_embedding": maternal_expression_embedding,
        "paternal_expression_embedding": paternal_expression_embedding,
    }


def _save_embedding_npz(path: Path, embedding: np.ndarray, cell_names: List[str]) -> None:
    np.savez(
        path,
        embedding=np.asarray(embedding, dtype=np.float32),
        cell_names=np.asarray([str(name) for name in cell_names], dtype=object),
    )
