"""Load and align a saved real-data phase experiment."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .schemas import PhaseVisualizationBundle
from .validation import finite_matrix, validate_bundle


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _reload_dataset(dataset_name: str):
    from run_phase import load_real_dataset

    return load_real_dataset(dataset_name, logging.getLogger("phase_visualization"))


def load_phase_visualization_bundle(
    result_dir: Path | str,
    *,
    dataset_name: str | None = None,
    raw_rna: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_names: Sequence[str] | None = None,
    cell_ids: Sequence[str] | None = None,
    genes: Sequence[str] | None = None,
) -> PhaseVisualizationBundle:
    result_dir = Path(result_dir)
    phase_a_frame = _read_matrix(result_dir / "phase_A.csv")
    phase_b_frame = _read_matrix(result_dir / "phase_B.csv")
    cell_h_frame = _read_matrix(result_dir / "cell_h.csv")
    config = _read_json(result_dir / "config.json")
    metrics = _read_json(result_dir / "metrics.json")

    canonical_cells = [str(x) for x in (cell_ids or phase_a_frame.index.tolist())]
    canonical_genes = [str(x) for x in (genes or phase_a_frame.columns.tolist())]
    for name, frame in (("phase_B", phase_b_frame), ("cell_h", cell_h_frame)):
        missing_cells = sorted(set(canonical_cells) - set(frame.index))
        if missing_cells:
            raise ValueError(f"{name} is missing {len(missing_cells)} cells")
    phase_a_frame = phase_a_frame.reindex(index=canonical_cells, columns=canonical_genes)
    phase_b_frame = phase_b_frame.reindex(index=canonical_cells, columns=canonical_genes)
    cell_h_frame = cell_h_frame.reindex(index=canonical_cells)

    metadata = _read_optional(result_dir / "cell_metadata.csv")
    if labels is None and not metadata.empty:
        metadata["cell_id"] = metadata["cell_id"].astype(str)
        aligned = metadata.set_index("cell_id").reindex(canonical_cells)
        labels = aligned["label_id"].to_numpy()
        label_names = aligned["label_name"].astype(str).tolist()

    loaded_dataset = None
    if raw_rna is None or labels is None or label_names is None:
        name = dataset_name or str(config.get("dataset", ""))
        if not name:
            raise ValueError("dataset name is required to reload missing raw data")
        loaded_dataset = _reload_dataset(name)
        cell_pos = {cell: i for i, cell in enumerate(loaded_dataset.cell_ids)}
        gene_pos = {gene: i for i, gene in enumerate(loaded_dataset.genes)}
        try:
            row_index = [cell_pos[cell] for cell in canonical_cells]
            col_index = [gene_pos[gene] for gene in canonical_genes]
        except KeyError as exc:
            raise ValueError(f"saved result cannot be aligned to dataset: {exc}") from exc
        if raw_rna is None:
            raw_rna = loaded_dataset.rna[np.ix_(row_index, col_index)]
        if labels is None:
            labels = loaded_dataset.labels[row_index]
        if label_names is None:
            label_map = loaded_dataset.metadata.get("label_map", {})
            label_names = [str(label_map.get(int(v), v)) for v in np.asarray(labels)]

    if raw_rna is None or labels is None or label_names is None:
        raise ValueError("raw RNA and label metadata could not be resolved")

    bundle = PhaseVisualizationBundle(
        result_dir=result_dir,
        raw_rna=finite_matrix(raw_rna, "raw_rna"),
        cell_h=finite_matrix(cell_h_frame.to_numpy(), "cell_h"),
        phase_a=finite_matrix(phase_a_frame.to_numpy(), "phase_a"),
        phase_b=finite_matrix(phase_b_frame.to_numpy(), "phase_b"),
        cell_ids=canonical_cells,
        genes=canonical_genes,
        labels=np.asarray(labels).reshape(-1),
        label_names=[str(x) for x in label_names],
        metrics=metrics,
        edge_gates=_read_optional(result_dir / "edge_gates.csv"),
        edge_summary=_read_optional(result_dir / "edge_summary.csv"),
        training_history=_read_optional(result_dir / "training_history.csv"),
        run_config=config,
        gene_annotation=_read_optional(result_dir / "gene_annotation.csv"),
        pathway_membership=_read_optional(result_dir / "pathway_membership.csv"),
        ppi_membership=_read_optional(result_dir / "ppi_membership.csv"),
        hyperedge_membership=_read_optional(result_dir / "hyperedge_membership.csv"),
    )
    validate_bundle(bundle)
    return bundle
