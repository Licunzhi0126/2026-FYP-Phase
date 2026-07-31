import matplotlib
from pathlib import Path
from typing import Sequence

matplotlib.use("Agg")

__all__ = ["run_have_answer_visualization", "run_phase_visualization"]


def run_have_answer_visualization(
    sim_dir: Path,
    result_dir: Path,
    *,
    output_dir: Path | None = None,
    dpi: int = 100,
    genes_to_plot: Sequence[str] | None = None,
    make_summary: bool = True,
    make_chromosome: bool = True,
    make_imbalance: bool = True,
    make_figure3: bool = True,
    max_correlation_genes: int | None = None,
    max_genome_correlation_genes: int | None = None,
):
    """Lazy public entry point for datasets with known ground truth."""
    from .have_answer import run_visualization as implementation

    return implementation(
        sim_dir=sim_dir,
        result_dir=result_dir,
        output_dir=output_dir,
        dpi=dpi,
        genes_to_plot=genes_to_plot,
        make_summary=make_summary,
        make_chromosome=make_chromosome,
        make_imbalance=make_imbalance,
        make_figure3=make_figure3,
        max_correlation_genes=max_correlation_genes,
        max_genome_correlation_genes=max_genome_correlation_genes,
    )


def run_phase_visualization(
    result_dir: Path | str,
    *,
    dataset_name: str | None = None,
    raw_rna=None,
    labels=None,
    label_names: Sequence[str] | None = None,
    cell_ids: Sequence[str] | None = None,
    genes: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    dpi: int = 300,
    top_genes: int = 40,
    projection_seed: int = 0,
    cluster_seed: int = 0,
):
    """Lazy public entry point for datasets without phasing ground truth."""
    from .no_answer import run_phase_visualization as implementation

    return implementation(
        result_dir=result_dir,
        dataset_name=dataset_name,
        raw_rna=raw_rna,
        labels=labels,
        label_names=label_names,
        cell_ids=cell_ids,
        genes=genes,
        output_dir=output_dir,
        dpi=dpi,
        top_genes=top_genes,
        projection_seed=projection_seed,
        cluster_seed=cluster_seed,
    )
