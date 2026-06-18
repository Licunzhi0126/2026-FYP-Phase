from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutputLayout:
    root: Path
    figures: Path
    figures_chromosomes: Path
    plot_data: Path
    tables: Path
    logs: Path
    config: Path

    def _legacy_dir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Lazily created compatibility paths. Default runs never access these.
    @property
    def embeddings(self) -> Path:
        return self._legacy_dir("embeddings")

    @property
    def figures_training(self) -> Path:
        return self._legacy_dir("figures", "training")

    @property
    def figures_embeddings(self) -> Path:
        return self._legacy_dir("figures", "embeddings")

    @property
    def eval_root(self) -> Path:
        return self._legacy_dir("evaluation")

    @property
    def eval_metrics(self) -> Path:
        return self._legacy_dir("evaluation", "metrics")

    @property
    def eval_figures_metrics(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "metrics")

    @property
    def eval_figures_expression_heatmaps(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "expression_heatmaps")

    @property
    def eval_figures_gate_aggregate(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "gate", "aggregate")

    @property
    def eval_figures_gate_by_chr_predicted(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "gate", "by_chromosome", "predicted")

    @property
    def eval_figures_gate_by_chr_truth(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "gate", "by_chromosome", "truth")

    @property
    def eval_figures_gate_by_chr_compare(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "gate", "by_chromosome", "compare")

    @property
    def eval_figures_correlation_aggregate(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "correlation", "aggregate")

    @property
    def eval_figures_correlation_by_chr(self) -> Path:
        return self._legacy_dir("evaluation", "figures", "correlation", "by_chromosome")

    @property
    def eval_tables_gate(self) -> Path:
        return self._legacy_dir("evaluation", "tables", "gate")

    @property
    def eval_tables_correlation(self) -> Path:
        return self._legacy_dir("evaluation", "tables", "correlation")

    @property
    def legacy_eval(self) -> Path:
        return self._legacy_dir("evaluation_visualization")


def make_run_output_layout(out_dir: Path) -> RunOutputLayout:
    root = Path(out_dir)
    layout = RunOutputLayout(
        root=root,
        figures=root / "figures",
        figures_chromosomes=root / "figures" / "chromosomes",
        plot_data=root / "plot_data",
        tables=root / "tables",
        logs=root / "logs",
        config=root / "config",
    )
    for path in [layout.figures, layout.figures_chromosomes, layout.plot_data, layout.tables, layout.logs, layout.config]:
        path.mkdir(parents=True, exist_ok=True)
    return layout


__all__ = ["RunOutputLayout", "make_run_output_layout"]
