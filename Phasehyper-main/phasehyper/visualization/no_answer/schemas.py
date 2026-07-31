"""Shared data structures for real-data phase visualizations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhaseVisualizationConfig:
    dpi: int = 300
    top_genes: int = 40
    projection_seed: int = 0
    cluster_seed: int = 0
    output_formats: tuple[str, ...] = ("png",)
    gene_clusters: int = 6
    low_detection: float = 0.10
    low_variance: float = 1e-8
    collapse_balance: float = 0.10
    well_resolved_balance: float = 0.20
    correlation_gene_limit: int = 200
    min_module_genes: int = 3
    genomic_rolling_window: int = 21
    detail_genes_per_class: int = 4
    fdr_alpha: float = 0.05
    min_enrichment_overlap: int = 2
    permutation_count: int = 200


@dataclass
class PhaseVisualizationBundle:
    result_dir: Path
    raw_rna: np.ndarray
    cell_h: np.ndarray
    phase_a: np.ndarray
    phase_b: np.ndarray
    cell_ids: list[str]
    genes: list[str]
    labels: np.ndarray
    label_names: list[str]
    metrics: dict[str, Any]
    edge_gates: pd.DataFrame
    edge_summary: pd.DataFrame
    training_history: pd.DataFrame
    run_config: dict[str, Any]
    gene_annotation: pd.DataFrame
    pathway_membership: pd.DataFrame
    ppi_membership: pd.DataFrame
    hyperedge_membership: pd.DataFrame


@dataclass
class VisualizationResult:
    output_dir: Path
    generated: list[str]
    skipped: list[dict[str, str]]
    failed: dict[str, str]

    @property
    def status(self) -> str:
        if self.generated and not self.failed:
            return "success" if not self.skipped else "partial"
        if self.generated:
            return "partial"
        return "failed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated": self.generated,
            "skipped": self.skipped,
            "failed": self.failed,
            "output_dir": str(self.output_dir),
        }
