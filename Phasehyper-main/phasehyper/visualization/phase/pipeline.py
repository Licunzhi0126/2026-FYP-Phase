"""Fault-isolated orchestration for all real-data phase visualizations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .allocation import compute_phase_allocation, plot_phase_allocation
from .cell_gene_maps import (
    build_cell_gene_contrast,
    build_cellgroup_phase_matrices,
    plot_cell_gene_contrast,
    plot_cellgroup_phase_triptych,
)
from .correlation_blocks import (
    compute_phase_correlation_blocks,
    correlation_source_frames,
    plot_correlation_block_heatmaps,
)
from .edge_gate import merge_edge_gate_data, plot_edge_gates
from .gene_contrast import compute_gene_contrast, plot_gene_contrast_heatmap
from .gene_details import plot_gene_detail_panels, select_representative_genes
from .gene_resolution import (
    cluster_gene_resolution,
    compute_gene_resolution_metrics,
    plot_gene_resolution_atlas,
)
from .genomic_tracks import prepare_genomic_tracks, plot_genome_resolution_tracks
from .io import prepare_output_dirs, save_figure_formats, write_source_data
from .loader import load_phase_visualization_bundle
from .metric_association import (
    compute_gene_metric_associations,
    plot_metric_association_heatmap,
)
from .metrics import build_metrics_table, plot_representation_metrics
from .module_maps import compute_module_phase_metrics, plot_module_phase_maps
from .prior_exposure import compute_gene_prior_exposure, plot_gene_prior_exposure
from .resolution_enrichment import (
    compute_resolution_cluster_enrichment,
    plot_resolution_cluster_enrichment,
)
from .schemas import PhaseVisualizationConfig, VisualizationResult
from .training import prepare_training_diagnostics, plot_training_diagnostics
from .pca import compute_pca_data, plot_four_representation_pca


LOGGER = logging.getLogger("phase_visualization")


class SkipVisualization(RuntimeError):
    """A figure is inapplicable because its required real input is absent."""


def run_phase_visualization(
    result_dir: Path | str,
    *,
    dataset_name: str | None = None,
    raw_rna: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_names: Sequence[str] | None = None,
    cell_ids: Sequence[str] | None = None,
    genes: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    dpi: int = 300,
    top_genes: int = 40,
    projection_seed: int = 0,
    cluster_seed: int = 0,
) -> dict:
    """Generate all applicable figures without rolling back saved model results."""
    result_dir = Path(result_dir)
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "visualization"
    config = PhaseVisualizationConfig(
        dpi=dpi, top_genes=top_genes,
        projection_seed=projection_seed, cluster_seed=cluster_seed,
    )
    result = VisualizationResult(output_dir, [], [], {})
    paths = prepare_output_dirs(output_dir)
    try:
        bundle = load_phase_visualization_bundle(
            result_dir,
            dataset_name=dataset_name,
            raw_rna=raw_rna,
            labels=labels,
            label_names=label_names,
            cell_ids=cell_ids,
            genes=genes,
        )
    except Exception as exc:
        result.failed["bundle"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("phase visualization bundle failed")
        return result.as_dict()

    def source(frame: pd.DataFrame, filename: str, *, index: bool = False) -> Path:
        path = write_source_data(frame, paths["source_data"] / filename, index=index)
        result.generated.append(str(path))
        return path

    def figure(fig, subdir: str, filename: str) -> None:
        saved = save_figure_formats(
            fig, paths[subdir] / filename, dpi=config.dpi, formats=config.output_formats
        )
        result.generated.extend(str(path) for path in saved)

    def task(name: str, action: Callable[[], None]) -> None:
        try:
            action()
        except SkipVisualization as exc:
            result.skipped.append({"name": name, "reason": str(exc)})
            LOGGER.warning("visualization skipped: %s: %s", name, exc)
        except Exception as exc:
            result.failed[name] = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("visualization failed: %s", name)

    task("01_four_representation_pca", lambda: _task_pca(bundle, config, source, figure))
    task("02_representation_metrics", lambda: _task_metrics(bundle, source, figure))
    allocation_box: dict[str, pd.DataFrame] = {}
    task("03_phase_allocation", lambda: _task_allocation(bundle, source, figure, allocation_box))
    contrast_box: dict[str, pd.DataFrame] = {}
    task("04_gene_contrast", lambda: _task_contrast(bundle, config, source, figure, contrast_box))
    task("05_edge_gate", lambda: _task_edge_gate(bundle, source, figure))
    task("06_training_diagnostics", lambda: _task_training(bundle, source, figure))

    shared: dict[str, pd.DataFrame] = {}
    task("07_gene_resolution_atlas", lambda: _task_resolution(bundle, config, source, figure, shared))
    task("08_correlation_blocks", lambda: _task_correlations(bundle, config, source, figure, shared))
    task("09_genome_phase_resolution_tracks", lambda: _task_genome(bundle, config, source, figure, shared))
    task(
        "10_cellgroup_gene_phase_triptych",
        lambda: _task_cell_maps(bundle, config, source, figure, shared, allocation_box, contrast_box),
    )
    task(
        "11_module_phase_maps",
        lambda: _task_modules(bundle, config, source, figure, result.skipped),
    )
    task("12_gene_prior_exposure", lambda: _task_prior_exposure(config, source, figure, shared))
    task(
        "13_cell_gene_contrast",
        lambda: _task_cell_contrast(bundle, config, source, figure, shared, allocation_box, contrast_box),
    )
    task("14_gene_details", lambda: _task_gene_details(bundle, config, source, figure, shared))
    task("15_gene_metric_association", lambda: _task_associations(source, figure, shared))
    task("16_resolution_cluster_enrichment", lambda: _task_enrichment(bundle, config, source, figure, shared))
    return result.as_dict()


def _task_pca(bundle, config, source, figure):
    data = compute_pca_data(bundle, seed=config.projection_seed)
    source(data, "pca_coordinates.csv")
    figure(plot_four_representation_pca(data, bundle.metrics), "overview", "01_four_representation_pca")


def _task_metrics(bundle, source, figure):
    data = build_metrics_table(bundle.metrics)
    source(data, "representation_metrics.csv")
    figure(plot_representation_metrics(data), "overview", "02_representation_metrics")


def _task_allocation(bundle, source, figure, box):
    data = compute_phase_allocation(bundle)
    box["data"] = data
    source(data, "phase_allocation_per_cell.csv")
    figure(plot_phase_allocation(data), "phase", "03_phase_allocation_violin")


def _task_contrast(bundle, config, source, figure, box):
    contrast, selected = compute_gene_contrast(bundle, top_genes=config.top_genes)
    box["contrast"], box["selected"] = contrast, selected
    source(contrast, "gene_contrast_matrix.csv")
    source(selected, "selected_gene_contrasts.csv")
    figure(plot_gene_contrast_heatmap(contrast, selected), "phase", "04_gene_contrast_heatmap")


def _task_edge_gate(bundle, source, figure):
    merged = merge_edge_gate_data(bundle.edge_gates, bundle.edge_summary)
    source(merged, "edge_gate_merged.csv")
    figure(plot_edge_gates(merged), "structure", "05_edge_gate")


def _task_training(bundle, source, figure):
    data = prepare_training_diagnostics(bundle.training_history)
    source(data, "training_diagnostics.csv")
    best_epoch = int(bundle.metrics.get("best_epoch", data.loc[data["loss"].idxmin(), "epoch"]))
    figure(plot_training_diagnostics(data, best_epoch=best_epoch), "training", "06_training_diagnostics")


def _task_resolution(bundle, config, source, figure, shared):
    exposure = compute_gene_prior_exposure(bundle)
    metrics = compute_gene_resolution_metrics(bundle, exposure, config)
    clusters = cluster_gene_resolution(metrics, config)
    shared.update(exposure=exposure, resolution=metrics, clusters=clusters)
    source(metrics, "gene_resolution_metrics.csv")
    source(clusters, "gene_resolution_clusters.csv")
    if not exposure.empty:
        source(exposure, "gene_edge_exposure.csv")
        write_source_data(exposure, bundle.result_dir / "gene_edge_exposure.csv")
    figure(plot_gene_resolution_atlas(metrics, clusters), "phase", "07_gene_resolution_atlas")


def _require_shared(shared, *names):
    missing = [name for name in names if name not in shared]
    if missing:
        raise SkipVisualization(f"dependency unavailable: {', '.join(missing)}")


def _task_correlations(bundle, config, source, figure, shared):
    _require_shared(shared, "resolution")
    genes, matrices, orders = compute_phase_correlation_blocks(
        bundle, shared["resolution"], limit=config.correlation_gene_limit
    )
    for name, frame in correlation_source_frames(genes, matrices).items():
        source(frame, f"correlation_{name}.csv", index=True)
    order_rows = [
        {"ordering": order_name, "order": rank, "gene": gene}
        for order_name, order in orders.items() for rank, gene in enumerate(order, 1)
    ]
    source(pd.DataFrame(order_rows), "gene_correlation_order.csv")
    for order_name, order in orders.items():
        figure(
            plot_correlation_block_heatmaps(
                genes, matrices, order, f"Aligned correlation blocks: {order_name} ordering"
            ),
            "correlation", f"08_correlation_blocks_{order_name}",
        )


def _task_genome(bundle, config, source, figure, shared):
    _require_shared(shared, "resolution")
    data, summary, null = prepare_genomic_tracks(
        shared["resolution"], bundle.gene_annotation,
        window=config.genomic_rolling_window,
        permutations=config.permutation_count,
        seed=config.cluster_seed,
    )
    source(data, "gene_genomic_resolution.csv")
    source(summary, "chromosome_resolution_summary.csv")
    source(null, "chromosome_cluster_null.csv")
    for name, fig in plot_genome_resolution_tracks(data, summary):
        target = pathsafe_split(name)
        figure(fig, "genome", target)


def pathsafe_split(name: str) -> str:
    return name.replace("\\", "/")


def _selected_genes(shared, contrast_box, config):
    if "selected" in contrast_box:
        return contrast_box["selected"]["gene"].head(config.top_genes).tolist()
    _require_shared(shared, "resolution")
    return shared["resolution"].nlargest(config.top_genes, "resolution_score")["gene"].tolist()


def _task_cell_maps(bundle, config, source, figure, shared, allocation_box, contrast_box):
    genes = _selected_genes(shared, contrast_box, config)
    data = build_cellgroup_phase_matrices(bundle, genes)
    source(data, "cellgroup_gene_phase_matrix.csv")
    figure(plot_cellgroup_phase_triptych(data), "phase", "10_cellgroup_gene_phase_triptych")


def _task_modules(bundle, config, source, figure, skipped):
    tables = []
    for family, membership, filename in (
        ("pathway", bundle.pathway_membership, "11_pathway_phase_map"),
        ("ppi", bundle.ppi_membership, "11_ppi_phase_map"),
    ):
        data = compute_module_phase_metrics(
            bundle, membership, family, min_genes=config.min_module_genes
        )
        if data.empty:
            reason = f"{family} has no module with at least {config.min_module_genes} aligned genes"
            skipped.append({"name": f"11_{family}_phase_map", "reason": reason})
            LOGGER.warning("module family skipped: %s", reason)
            continue
        tables.append(data)
        figure(plot_module_phase_maps(data, family), "modules", filename)
    if not tables:
        raise SkipVisualization("no pathway or PPI module has enough aligned genes")
    source(pd.concat(tables, ignore_index=True), "module_phase_metrics.csv")


def _task_prior_exposure(config, source, figure, shared):
    _require_shared(shared, "exposure", "clusters")
    exposure = shared["exposure"]
    if exposure.empty:
        raise SkipVisualization("hyperedge membership is unavailable")
    clusters = shared["clusters"].sort_values("atlas_order")
    matrix = exposure.pivot_table(
        index="gene", columns=["channel", "edge_type"], values="structural_exposure",
        aggfunc="sum", fill_value=0,
    ).reindex(clusters["gene"], fill_value=0)
    source(matrix.reset_index(), "gene_edge_exposure_matrix.csv")
    figure(
        plot_gene_prior_exposure(exposure, clusters["gene"].tolist(), clusters),
        "structure", "12_gene_prior_exposure",
    )


def _allocation(bundle, allocation_box):
    if "data" not in allocation_box:
        allocation_box["data"] = compute_phase_allocation(bundle)
    return allocation_box["data"]


def _task_cell_contrast(bundle, config, source, figure, shared, allocation_box, contrast_box):
    genes = _selected_genes(shared, contrast_box, config)
    data = build_cell_gene_contrast(bundle, genes, _allocation(bundle, allocation_box))
    source(data, "cell_gene_contrast_matrix.csv")
    figure(plot_cell_gene_contrast(data), "phase", "13_cell_gene_contrast_heatmap")


def _task_gene_details(bundle, config, source, figure, shared):
    _require_shared(shared, "resolution", "clusters")
    selected = select_representative_genes(
        shared["resolution"], shared["clusters"], per_class=config.detail_genes_per_class
    )
    if selected.empty:
        raise SkipVisualization("no representative gene class is available")
    source(selected, "selected_gene_details.csv")
    for name, fig in plot_gene_detail_panels(bundle, shared["resolution"], selected).items():
        figure(fig, "genes", name)


def _task_associations(source, figure, shared):
    _require_shared(shared, "resolution")
    data = compute_gene_metric_associations(shared["resolution"])
    source(data, "gene_metric_associations.csv")
    figure(plot_metric_association_heatmap(data), "associations", "15_gene_metric_association")


def _task_enrichment(bundle, config, source, figure, shared):
    _require_shared(shared, "resolution", "clusters")
    data, members = compute_resolution_cluster_enrichment(
        bundle, shared["resolution"], shared["clusters"]
    )
    source(members, "resolution_cluster_members.csv")
    if data.empty:
        raise SkipVisualization("no usable annotation family for enrichment")
    source(data, "resolution_cluster_enrichment.csv")
    shown = data[
        (data["fdr"] <= config.fdr_alpha)
        & (data["overlap_count"] >= config.min_enrichment_overlap)
    ]
    if shown.empty:
        raise SkipVisualization("no enrichment passes FDR and overlap thresholds")
    figure(
        plot_resolution_cluster_enrichment(
            data, fdr_alpha=config.fdr_alpha, min_overlap=config.min_enrichment_overlap
        ),
        "modules", "16_resolution_cluster_enrichment",
    )
