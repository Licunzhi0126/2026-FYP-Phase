import matplotlib
from pathlib import Path
from typing import Sequence

matplotlib.use("Agg")

__all__ = ["run_simulation_visualization"]


def run_simulation_visualization(
    sim_dir: Path,
    result_dir: Path,
    *,
    output_dir: Path | None = None,
    dpi: int = 300,
    genes_to_plot: Sequence[str] | None = None,
    make_summary: bool = True,
    make_chromosome: bool = True,
    make_imbalance: bool = True,
    make_figure3: bool = True,
):
    """Lazy public entry point; keeps ``python -m ...simulation_pipeline`` clean."""
    from .simulation_pipeline import run_simulation_visualization as implementation

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
    )
