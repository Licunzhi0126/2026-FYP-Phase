from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunOutputLayout:
    root: Path
    tables: Path
    embeddings: Path
    figures_training: Path
    figures_embeddings: Path
    eval_root: Path
    eval_metrics: Path
    eval_figures_metrics: Path
    eval_figures_expression_heatmaps: Path
    eval_figures_gate_aggregate: Path
    eval_figures_gate_by_chr_predicted: Path
    eval_figures_gate_by_chr_truth: Path
    eval_figures_gate_by_chr_compare: Path
    eval_figures_correlation_aggregate: Path
    eval_figures_correlation_by_chr: Path
    eval_tables_gate: Path
    eval_tables_correlation: Path
    legacy_eval: Path


def _mkdirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def make_run_output_layout(out_dir: Path) -> RunOutputLayout:
    root = Path(out_dir)
    layout = RunOutputLayout(
        root=root,
        tables=root / "tables",
        embeddings=root / "embeddings",
        figures_training=root / "figures" / "training",
        figures_embeddings=root / "figures" / "embeddings",
        eval_root=root / "evaluation",
        eval_metrics=root / "evaluation" / "metrics",
        eval_figures_metrics=root / "evaluation" / "figures" / "metrics",
        eval_figures_expression_heatmaps=root / "evaluation" / "figures" / "expression_heatmaps",
        eval_figures_gate_aggregate=root / "evaluation" / "figures" / "gate" / "aggregate",
        eval_figures_gate_by_chr_predicted=root / "evaluation" / "figures" / "gate" / "by_chromosome" / "predicted",
        eval_figures_gate_by_chr_truth=root / "evaluation" / "figures" / "gate" / "by_chromosome" / "truth",
        eval_figures_gate_by_chr_compare=root / "evaluation" / "figures" / "gate" / "by_chromosome" / "compare",
        eval_figures_correlation_aggregate=root / "evaluation" / "figures" / "correlation" / "aggregate",
        eval_figures_correlation_by_chr=root / "evaluation" / "figures" / "correlation" / "by_chromosome",
        eval_tables_gate=root / "evaluation" / "tables" / "gate",
        eval_tables_correlation=root / "evaluation" / "tables" / "correlation",
        legacy_eval=root / "evaluation_visualization",
    )
    _mkdirs(
        [
            layout.tables,
            layout.embeddings,
            layout.figures_training,
            layout.figures_embeddings,
            layout.eval_metrics,
            layout.eval_figures_metrics,
            layout.eval_figures_expression_heatmaps,
            layout.eval_figures_gate_aggregate,
            layout.eval_figures_gate_by_chr_predicted,
            layout.eval_figures_gate_by_chr_truth,
            layout.eval_figures_gate_by_chr_compare,
            layout.eval_figures_correlation_aggregate,
            layout.eval_figures_correlation_by_chr,
            layout.eval_tables_gate,
            layout.eval_tables_correlation,
            layout.legacy_eval,
        ]
    )
    return layout


__all__ = ["RunOutputLayout", "make_run_output_layout"]
