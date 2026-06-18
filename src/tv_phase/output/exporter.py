from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from ..config import DatasetBundle, PriorBundle
from .layout import RunOutputLayout


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def export_data_contract(
    layout: RunOutputLayout,
    dataset: DatasetBundle,
    dataset_config: Dict[str, Any],
    dataset_root: Path,
) -> None:
    files = dataset_config.get("files", {})
    label_counts = pd.Series(dataset.label_names, dtype=str).value_counts().to_dict()
    payload = {
        "dataset_type": dataset.dataset_type,
        "description": dataset_config.get("description", ""),
        "root": str(Path(dataset_root).resolve()),
        "files": {key: str((Path(dataset_root) / value).resolve()) if isinstance(value, str) and value else value for key, value in files.items()},
        "view1_name": dataset.view1_name,
        "n_cells": len(dataset.common_cells),
        "n_genes": len(dataset.common_genes),
        "expression_shape": list(dataset.expression_df.shape),
        "view_shapes": [] if dataset.view1_dfs is None else [list(df.shape) for df in dataset.view1_dfs],
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_map": {str(k): str(v) for k, v in (dataset.label_map or {}).items()},
        "cell_order_policy": "aligned by cell_id index",
        "missing_value_policy": "NaN/inf values are filled with 0 during loading",
    }
    if dataset.metadata:
        payload.update(dataset.metadata)
    save_json(layout.config / "data_contract.json", payload)


def _prior_summary_rows(prior: PriorBundle, prior_name: str) -> pd.DataFrame:
    rows = []
    for edge_type, groups in [
        ("kegg", prior.kegg_groups),
        ("poswin", prior.poswin_groups),
        ("ppi", prior.ppi_groups or {}),
        ("data", prior.data_groups or {}),
    ]:
        genes = {gene for members in groups.values() for gene in members}
        pair_count = sum(max(0, len(set(members)) - 1) for members in groups.values())
        rows.append(
            {
                "prior_name": prior_name,
                "edge_type": edge_type,
                "group_count": len(groups),
                "unique_gene_count": len(genes),
                "membership_edge_count": pair_count,
            }
        )
    return pd.DataFrame(rows)


def export_prior(layout: RunOutputLayout, prior: PriorBundle, params: Dict[str, Any]) -> None:
    metadata = dict(prior.metadata or {})
    prior_name = str(metadata.get("prior_name", params.get("name", "unknown")))
    metadata.update(
        {
            "prior_name": prior_name,
            "kegg_groups": len(prior.kegg_groups),
            "poswin_groups": len(prior.poswin_groups),
            "ppi_groups": len(prior.ppi_groups or {}),
            "data_groups": len(prior.data_groups or {}),
            "gene_prior_matrix_shape": None if prior.gene_prior_matrix is None else list(prior.gene_prior_matrix.shape),
            "params": params,
        }
    )
    save_json(layout.config / "prior_metadata.json", metadata)
    _prior_summary_rows(prior, prior_name).to_csv(
        layout.plot_data / "prior_edge_summary.csv", index=False, encoding="utf-8-sig"
    )
    if prior.edge_table is not None and not prior.edge_table.empty:
        prior.edge_table.to_csv(layout.tables / "prior_edges.csv", index=False, encoding="utf-8-sig")


def export_training_outputs(
    layout: RunOutputLayout,
    *,
    phase_a: pd.DataFrame,
    phase_b: pd.DataFrame,
    gene_gate: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    sample_names: Iterable[str],
    loss_history: Iterable[Dict[str, float]],
) -> pd.DataFrame:
    phase_a.to_csv(layout.tables / "phaseA_expression.csv", encoding="utf-8-sig")
    phase_b.to_csv(layout.tables / "phaseB_expression.csv", encoding="utf-8-sig")
    gene_gate.to_csv(layout.tables / "gene_gate.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        layout.tables / "embeddings.npz",
        cell_names=np.asarray(list(sample_names), dtype=str),
        **{name: np.asarray(value, dtype=np.float32) for name, value in embeddings.items()},
    )
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(layout.logs / "training_loss.csv", index=False, encoding="utf-8-sig")
    return loss_df


def export_run_config(layout: RunOutputLayout, config: Any, extra: Dict[str, Any]) -> None:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    payload.update(extra)
    save_json(layout.config / "run_config.json", payload)


def copy_adapter_manifest(layout: RunOutputLayout, dataset_root: Path, filename: str = "adapter_manifest.json") -> None:
    source = Path(dataset_root) / filename
    if source.exists():
        shutil.copy2(source, layout.config / "adapter_manifest.json")

