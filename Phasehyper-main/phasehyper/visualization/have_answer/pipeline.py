"""Visualization pipeline for data with known ground truth."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Sequence

from .figure3 import (
    build_figure3_data,
    plot_figure3a,
    plot_figure3b,
    plot_figure3c,
    plot_figure3d,
    plot_figure3e,
    plot_figure3f,
    write_figure3_source_data,
)
from .diagnostics import (
    load_visualization_bundle,
    plot_delta_correlation,
    plot_gene_correlation_track,
    plot_gene_detail,
    plot_genome_delta_correlation,
    plot_phase_contrast_track,
    plot_phase_correlation_heatmap,
    plot_phase_expression_heatmap,
    write_aligned_annotation,
)
from .imbalance import (
    plot_chromosome_imbalance_heatmap,
    plot_chromosome_imbalance_track,
    plot_genome_imbalance_heatmap,
    write_gene_level_imbalance,
)
from .summary import (
    plot_expression_metrics,
    plot_grn_decomposition,
    plot_grn_metrics,
)

LOGGER = logging.getLogger("phasehyper.visualization.have_answer")


class VisualizationError(RuntimeError):
    """Raised after all independent figures have been attempted."""

    def __init__(self, failures: list[tuple[str, Exception]]):
        self.failures = failures
        details = "; ".join(
            f"{name}: {type(error).__name__}: {error}" for name, error in failures
        )
        super().__init__(f"{len(failures)} visualization task(s) failed: {details}")


def _collect_paths(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def _attempt(
    name: str,
    callback: Callable[[], object],
    outputs: list[Path],
    failures: list[tuple[str, Exception]],
) -> None:
    try:
        paths = _collect_paths(callback())
        outputs.extend(paths)
        for path in paths:
            print(f"  [visualization] wrote {path}")
    except Exception as error:
        failures.append((name, error))
        print(f"  [visualization warning] {name}: {type(error).__name__}: {error}")


def _legacy_pngs(result_dir: Path, output_dir: Path) -> list[Path]:
    output_resolved = output_dir.resolve()
    legacy = []
    for path in result_dir.rglob("*.png"):
        resolved = path.resolve()
        try:
            resolved.relative_to(output_resolved)
        except ValueError:
            legacy.append(path)
    return legacy


def run_visualization(
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
) -> list[Path]:
    """Generate every requested figure from one aligned data bundle."""
    sim_dir = Path(sim_dir)
    result_dir = Path(result_dir)
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [visualization] loading aligned bundle from {sim_dir} and {result_dir}")
    bundle = load_visualization_bundle(sim_dir, result_dir)
    print(
        "  [visualization] global orientation: "
        f"{bundle.phase_mapping.get('assign', bundle.phase_mapping.get('level'))}"
    )
    outputs: list[Path] = []
    failures: list[tuple[str, Exception]] = []

    _attempt(
        "source_data/aligned_gene_annotation",
        lambda: write_aligned_annotation(
            bundle, source_dir / "aligned_gene_annotation.csv"
        ),
        outputs,
        failures,
    )
    _attempt(
        "source_data/gene_level_imbalance",
        lambda: write_gene_level_imbalance(
            bundle, source_dir / "gene_level_imbalance.csv"
        ),
        outputs,
        failures,
    )

    if make_summary:
        summary_dir = output_dir / "summary"
        _attempt(
            "summary/expression_metrics",
            lambda: plot_expression_metrics(
                bundle, summary_dir / "expression_metrics.png", dpi
            ),
            outputs,
            failures,
        )
        grn_dir = result_dir / "grn"
        if (grn_dir / "differential.csv").exists():
            _attempt(
                "summary/grn_metrics",
                lambda: plot_grn_metrics(
                    bundle, summary_dir / "grn_metrics.png", dpi
                ),
                outputs,
                failures,
            )
        else:
            print("  [visualization] skipped summary/grn_metrics: no reference GRN")
        if (grn_dir / "edges.npz").exists():
            _attempt(
                "summary/grn_decomposition",
                lambda: plot_grn_decomposition(
                    bundle, summary_dir / "grn_decomposition.png", dpi
                ),
                outputs,
                failures,
            )
        else:
            print(
                "  [visualization] skipped summary/grn_decomposition: "
                "no reference GRN"
            )

    if make_chromosome or make_imbalance:
        for chromosome in bundle.chromosomes:
            chromosome_dir = output_dir / "chromosomes" / chromosome
            if make_chromosome:
                chromosome_tasks = (
                    (
                        "phase_expression_heatmap",
                        plot_phase_expression_heatmap,
                        False,
                    ),
                    ("phase_contrast_track", plot_phase_contrast_track, False),
                    ("gene_correlation_track", plot_gene_correlation_track, False),
                    (
                        "phase_correlation_heatmap",
                        plot_phase_correlation_heatmap,
                        True,
                    ),
                    ("delta_correlation", plot_delta_correlation, True),
                )
                for suffix, function, limited in chromosome_tasks:
                    _attempt(
                        f"{chromosome}/{suffix}",
                        (
                            lambda function=function, suffix=suffix: function(
                                bundle,
                                chromosome,
                                chromosome_dir / f"{chromosome}_{suffix}.png",
                                dpi,
                                max_correlation_genes,
                            )
                        )
                        if limited
                        else (
                            lambda function=function, suffix=suffix: function(
                                bundle,
                                chromosome,
                                chromosome_dir / f"{chromosome}_{suffix}.png",
                                dpi,
                            )
                        ),
                        outputs,
                        failures,
                    )
            if make_imbalance:
                for suffix, function in (
                    ("imbalance_track", plot_chromosome_imbalance_track),
                    ("imbalance_heatmap", plot_chromosome_imbalance_heatmap),
                ):
                    _attempt(
                        f"{chromosome}/{suffix}",
                        lambda function=function, suffix=suffix: function(
                            bundle,
                            chromosome,
                            chromosome_dir / f"{chromosome}_{suffix}.png",
                            dpi,
                        ),
                        outputs,
                        failures,
                    )

    if make_chromosome:
        _attempt(
            "genome/all_chromosomes_delta_correlation",
            lambda: plot_genome_delta_correlation(
                bundle,
                output_dir / "genome" / "all_chromosomes_delta_correlation.png",
                dpi,
                max_genome_correlation_genes,
            ),
            outputs,
            failures,
        )
        for gene in genes_to_plot or ():
            _attempt(
                f"genes/{gene}",
                lambda gene=gene: plot_gene_detail(
                    bundle, gene, output_dir / "genes" / f"{gene}_detail.png", dpi
                ),
                outputs,
                failures,
            )

    if make_imbalance:
        _attempt(
            "genome/all_chromosomes_imbalance_heatmap",
            lambda: plot_genome_imbalance_heatmap(
                bundle,
                output_dir / "genome" / "all_chromosomes_imbalance_heatmap.png",
                dpi,
            ),
            outputs,
            failures,
        )

    if make_figure3:
        try:
            figure3_data = build_figure3_data(bundle)
        except Exception as error:
            failures.append(("figure3/source_data", error))
            print(
                "  [visualization warning] figure3/source_data: "
                f"{type(error).__name__}: {error}"
            )
        else:
            _attempt(
                "figure3/source_data",
                lambda: write_figure3_source_data(figure3_data, source_dir),
                outputs,
                failures,
            )
            figure3_dir = output_dir / "figure3"
            for label, function, filename in (
                ("3A", plot_figure3a, "fig3A_imbalance_calibration.png"),
                ("3B", plot_figure3b, "fig3B_phasefit_mse_heatmap.png"),
                ("3C", plot_figure3c, "fig3C_paired_gene_mse.png"),
                ("3D", plot_figure3d, "fig3D_precision_recall.png"),
                ("3E", plot_figure3e, "fig3E_orientation_audit.png"),
                ("3F", plot_figure3f, "fig3F_detection_metrics.png"),
            ):
                _attempt(
                    f"figure3/{label}",
                    lambda function=function, filename=filename: function(
                        figure3_data, figure3_dir / filename, dpi
                    ),
                    outputs,
                    failures,
                )

    legacy = _legacy_pngs(result_dir, output_dir)
    if legacy:
        print("  [visualization warning] legacy PNG files remain outside visualization/:")
        for path in legacy:
            print(f"    - {path}")

    if failures:
        raise VisualizationError(failures)
    print(f"  [visualization] completed successfully ({len(outputs)} files)")
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=Path("simulation_data"))
    parser.add_argument("--result-dir", type=Path, default=Path("result_simulation"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--max-correlation-genes", type=int)
    parser.add_argument("--max-genome-correlation-genes", type=int)
    parser.add_argument(
        "--gene",
        action="append",
        dest="genes",
        default=None,
        help="gene ID to plot; may be supplied more than once",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_visualization(
        sim_dir=args.sim_dir,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        genes_to_plot=args.genes,
        max_correlation_genes=args.max_correlation_genes,
        max_genome_correlation_genes=args.max_genome_correlation_genes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
