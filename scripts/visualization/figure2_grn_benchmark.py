"""Create the simulation-data Figure 2 GRN benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visualization.figure2_io import (  # type: ignore
        GRNBundle,
        load_external_grn_prediction,
        load_grn_bundle,
    )
    from visualization.figure2_metrics import (  # type: ignore
        make_preview_methods,
        pair_metrics,
        project_baseline_methods,
        safe_correlation,
    )
    from visualization.figure2_style import (  # type: ignore
        BLUE,
        CMAP_PRED_A,
        CMAP_PRED_B,
        CMAP_TOTAL,
        CMAP_TRUTH_A,
        CMAP_TRUTH_B,
        GOLD,
        GRID,
        INK,
        PRED_A,
        PRED_B,
        TRUTH_A,
        TRUTH_B,
        add_caption,
        apply_style,
        clean_axis,
        method_color,
        panel_label,
        save_figure,
    )
else:
    from .figure2_io import (
        GRNBundle,
        load_external_grn_prediction,
        load_grn_bundle,
    )
    from .figure2_metrics import (
        make_preview_methods,
        pair_metrics,
        project_baseline_methods,
        safe_correlation,
    )
    from .figure2_style import (
        BLUE,
        CMAP_PRED_A,
        CMAP_PRED_B,
        CMAP_TOTAL,
        CMAP_TRUTH_A,
        CMAP_TRUTH_B,
        GOLD,
        GRID,
        INK,
        PRED_A,
        PRED_B,
        TRUTH_A,
        TRUTH_B,
        add_caption,
        apply_style,
        clean_axis,
        method_color,
        panel_label,
        save_figure,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark two-channel GRN reconstruction on thresholded data."
    )
    parser.add_argument("--per-cell-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pred-npz", type=Path)
    parser.add_argument("--model-name", default="HyperPhase")
    parser.add_argument("--primary-method")
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-prevalence", type=float, default=0.95)
    parser.add_argument("--heatmap-edges", type=int, default=50)
    parser.add_argument("--heatmap-cells", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _contexts(n_cells: int) -> list[tuple[str, np.ndarray]]:
    all_cells = np.arange(n_cells)
    contexts: list[tuple[str, np.ndarray]] = [("All cells", all_cells)]
    if n_cells >= 16:
        for index, cell_indices in enumerate(np.array_split(all_cells, 4), start=1):
            if index <= 3:
                contexts.append((f"Cell quartile {index}", cell_indices))
    return contexts


def _evaluate(
    bundle: GRNBundle,
    external: tuple[np.ndarray, np.ndarray] | None,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
]:
    rows: list[dict[str, object]] = []
    oriented: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for context_index, (context_name, indices) in enumerate(
        _contexts(len(bundle.cells))
    ):
        combined = bundle.combined[indices]
        maternal = bundle.maternal[indices]
        paternal = bundle.paternal[indices]
        methods = project_baseline_methods(
            combined,
            seed=args.seed + context_index,
        )
        if external is not None:
            methods[args.model_name] = (external[0][indices], external[1][indices])

        oriented[context_name] = {}
        for method, (pred_a, pred_b) in methods.items():
            metrics, aligned_a, aligned_b = pair_metrics(
                pred_a,
                pred_b,
                maternal,
                paternal,
                total=combined,
                binary_truth=True,
            )
            oriented[context_name][method] = (aligned_a, aligned_b)
            row: dict[str, object] = {
                "context_order": context_index,
                "context": context_name,
                "method": method,
                "is_preview_control": method != args.model_name,
                "n_cells": len(indices),
                "n_edges": bundle.combined.shape[1],
                "combined_density": float(combined.mean()),
                "maternal_density": float(maternal.mean()),
                "paternal_density": float(paternal.mean()),
            }
            row.update(metrics.to_dict())
            rows.append(row)
    return pd.DataFrame(rows), oriented


def _choose_primary_method(
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> str:
    if args.primary_method:
        if args.primary_method not in methods:
            available = ", ".join(methods)
            raise ValueError(
                f"Primary method {args.primary_method!r} is unavailable. "
                f"Available methods: {available}"
            )
        return args.primary_method
    if args.model_name in methods:
        return args.model_name
    if "NMF2Factor" in methods:
        return "NMF2Factor"
    return next(iter(methods))


def _plot_heatmap_panel(
    figure: plt.Figure,
    subplotspec,
    bundle: GRNBundle,
    prediction: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> None:
    cell_order = np.argsort(bundle.combined.mean(axis=1))
    if len(cell_order) > args.heatmap_cells:
        positions = np.linspace(
            0, len(cell_order) - 1, args.heatmap_cells
        ).astype(int)
        cell_order = cell_order[positions]
    prevalence = bundle.combined[cell_order].mean(axis=0)
    variability = prevalence * (1.0 - prevalence)
    edge_order = np.argsort(-variability)[
        : min(args.heatmap_edges, bundle.combined.shape[1])
    ]

    arrays = [
        bundle.combined[np.ix_(cell_order, edge_order)],
        bundle.maternal[np.ix_(cell_order, edge_order)],
        bundle.paternal[np.ix_(cell_order, edge_order)],
        prediction[0][np.ix_(cell_order, edge_order)],
        prediction[1][np.ix_(cell_order, edge_order)],
    ]
    titles = [
        "Combined observation",
        "Truth maternal",
        "Truth paternal",
        "Predicted channel 1",
        "Predicted channel 2",
    ]
    cmaps = [
        CMAP_TOTAL,
        CMAP_TRUTH_A,
        CMAP_TRUTH_B,
        CMAP_PRED_A,
        CMAP_PRED_B,
    ]
    nested = subplotspec.subgridspec(1, 5, wspace=0.10)
    axes = []
    for index, (array, title, cmap) in enumerate(zip(arrays, titles, cmaps)):
        axis = figure.add_subplot(nested[0, index])
        axes.append(axis)
        axis.imshow(
            array.transpose(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=max(float(np.percentile(array, 99)), 1.0),
        )
        axis.set_title(title, fontsize=8.5, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(
        f"{len(edge_order)} variable directed edges", fontsize=8, color=INK
    )
    axes[0].text(
        0.0,
        -0.08,
        f"{len(cell_order)} cells →",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color="#667085",
    )
    panel_label(axes[0], "A", x=-0.24, y=1.16)


def _plot_metric_over_contexts(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    panel: str,
) -> None:
    context_order = (
        metrics[["context_order", "context"]]
        .drop_duplicates()
        .sort_values("context_order")
    )
    methods = metrics["method"].drop_duplicates().tolist()
    width = 0.72 / max(len(methods), 1)
    x = np.arange(len(context_order))
    context_lookup = dict(zip(context_order["context"], x))
    for method_index, method in enumerate(methods):
        subset = metrics.loc[metrics["method"] == method]
        positions = np.array([context_lookup[name] for name in subset["context"]])
        offset = (method_index - (len(methods) - 1) / 2) * width
        axis.scatter(
            positions + offset,
            subset[metric],
            s=24,
            color=method_color(method, method_index),
            edgecolor="white",
            linewidth=0.4,
            label=method,
            zorder=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [name.replace(" ", "\n", 1) for name in context_order["context"]],
        fontsize=7.5,
    )
    axis.set_xlim(-0.6, len(x) - 0.4)
    axis.set_ylabel(ylabel)
    clean_axis(axis)
    panel_label(axis, panel)


def _plot_representative_cell(
    axis: plt.Axes,
    bundle: GRNBundle,
    prediction: tuple[np.ndarray, np.ndarray],
) -> None:
    phase_signal = np.abs(bundle.maternal - bundle.paternal).sum(axis=1)
    cell_index = int(np.argmax(phase_signal))
    informative = np.flatnonzero(
        bundle.maternal[cell_index] != bundle.paternal[cell_index]
    )
    if informative.size == 0:
        informative = np.argsort(-bundle.combined.mean(axis=0))[:16]
    else:
        prevalence = bundle.combined[:, informative].mean(axis=0)
        informative = informative[np.argsort(-prevalence)[:16]]

    x = np.arange(len(informative))
    width = 0.35
    axis.bar(
        x - width / 2,
        bundle.maternal[cell_index, informative],
        width,
        color=TRUTH_A,
        alpha=0.72,
        label="truth maternal",
    )
    axis.bar(
        x + width / 2,
        bundle.paternal[cell_index, informative],
        width,
        color=TRUTH_B,
        alpha=0.72,
        label="truth paternal",
    )
    axis.scatter(
        x - width / 2,
        prediction[0][cell_index, informative],
        s=24,
        facecolor="none",
        edgecolor=PRED_A,
        linewidth=1.1,
        label="predicted 1",
        zorder=4,
    )
    axis.scatter(
        x + width / 2,
        prediction[1][cell_index, informative],
        s=24,
        marker="s",
        facecolor="none",
        edgecolor=PRED_B,
        linewidth=1.1,
        label="predicted 2",
        zorder=4,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [bundle.edge_names[index] for index in informative],
        rotation=70,
        ha="right",
        fontsize=6.2,
    )
    axis.set_ylim(-0.05, 1.15)
    axis.set_ylabel("Edge support")
    axis.set_title(f"Representative cell: {bundle.cells[cell_index]}", loc="left")
    axis.legend(ncol=2, fontsize=6.8, loc="upper right")
    clean_axis(axis)
    panel_label(axis, "D")


def _safe_binary_auc(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if np.unique(truth).size < 2:
        return float("nan")
    return float(roc_auc_score(truth, score))


def _plot_union_shared(
    axis: plt.Axes,
    bundle: GRNBundle,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    held_out_count = max(1, int(round(0.20 * len(bundle.cells))))
    held_out = np.sort(
        rng.choice(len(bundle.cells), size=held_out_count, replace=False)
    )
    truth_union = np.maximum(
        bundle.maternal[held_out], bundle.paternal[held_out]
    )
    truth_shared = np.minimum(
        bundle.maternal[held_out], bundle.paternal[held_out]
    )
    methods = list(predictions)
    x = np.arange(len(methods))
    width = 0.34
    union_auc = []
    shared_auc = []
    for method in methods:
        pred_a, pred_b = predictions[method]
        pred_a = pred_a[held_out]
        pred_b = pred_b[held_out]
        union_auc.append(_safe_binary_auc(truth_union, np.maximum(pred_a, pred_b)))
        shared_auc.append(
            _safe_binary_auc(truth_shared, np.minimum(pred_a, pred_b))
        )

    axis.bar(
        x - width / 2,
        union_auc,
        width,
        color=BLUE,
        alpha=0.82,
        label="parental union",
    )
    axis.bar(
        x + width / 2,
        shared_auc,
        width,
        color=GOLD,
        alpha=0.82,
        label="shared support",
    )
    axis.axhline(0.5, color="#8F98A7", linewidth=0.8, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("AUROC")
    axis.set_title(
        f"Held-out parental support recovery ({held_out_count} cells)",
        loc="left",
    )
    axis.legend(loc="upper right", fontsize=7)
    clean_axis(axis)
    panel_label(axis, "E")


# Legacy preview layout retained for callers that imported it directly.
# The command-line workflow below uses _create_paper_grn_figure.
def _create_figure(
    bundle: GRNBundle,
    metrics: pd.DataFrame,
    oriented: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[plt.Figure, str]:
    apply_style()
    primary_predictions = oriented["All cells"]
    primary_method = _choose_primary_method(primary_predictions, args)
    primary_prediction = primary_predictions[primary_method]

    figure = plt.figure(figsize=(17.2, 9.2))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.08, 1.08, 1.0),
        height_ratios=(0.85, 1.0),
        left=0.045,
        right=0.985,
        top=0.91,
        bottom=0.13,
        wspace=0.34,
        hspace=0.52,
    )
    _plot_heatmap_panel(
        figure, grid[0, :2], bundle, primary_prediction, args
    )

    axis_b = figure.add_subplot(grid[0, 2])
    _plot_metric_over_contexts(axis_b, metrics, "nmse", "Normalized MSE", "B")
    axis_b.set_title("Phase reconstruction error", loc="left")
    axis_b.legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7
    )

    axis_c = figure.add_subplot(grid[1, 0])
    _plot_metric_over_contexts(
        axis_c, metrics, "differential_r", "Differential correlation", "C"
    )
    axis_c.axhline(0.0, color="#AAB2BF", linewidth=0.8)
    axis_c.set_title("Parent-specific edge agreement", loc="left")

    axis_d = figure.add_subplot(grid[1, 1])
    _plot_representative_cell(axis_d, bundle, primary_prediction)

    axis_e = figure.add_subplot(grid[1, 2])
    _plot_union_shared(axis_e, bundle, primary_predictions, args.seed)

    figure.suptitle(
        "Figure 2-style parent-resolved GRN benchmark",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.045,
        0.942,
        f"Input: independently thresholded per-cell networks · displayed method: "
        f"{primary_method}",
        ha="left",
        fontsize=9,
        color="#667085",
    )
    add_caption(
        figure,
        "The combined thresholded matrix is treated as an independent observation, "
        "not as the binary union of maternal and paternal matrices. Union/shared "
        "labels in panel E are derived from the two parental truth channels only. "
        "Preview controls are not PhaseHyper model results.",
    )
    return figure, primary_method


def _paper_top_nodes(bundle: GRNBundle, maximum: int = 28) -> np.ndarray:
    degree = np.zeros(len(bundle.genes), dtype=int)
    for source, target in bundle.edge_index:
        degree[int(source)] += 1
        degree[int(target)] += 1
    return np.argsort(-degree)[: min(maximum, len(degree))]


def _paper_edge_matrix(
    values: np.ndarray,
    bundle: GRNBundle,
    nodes: np.ndarray,
) -> np.ndarray:
    local = {int(node): index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for value, (source, target) in zip(values, bundle.edge_index):
        if int(source) in local and int(target) in local:
            matrix[local[int(source)], local[int(target)]] = float(value)
    return matrix


def _paper_grn_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    cmap,
    title: str,
) -> None:
    axis.imshow(
        matrix,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_title(title, pad=5, fontsize=9.2)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _paper_grn_arrow(axis: plt.Axes, text: str = "") -> None:
    axis.set_axis_off()
    axis.annotate(
        "",
        xy=(0.92, 0.5),
        xytext=(0.08, 0.5),
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.05,
            "color": "#8B95A5",
        },
        xycoords="axes fraction",
    )
    if text:
        axis.text(
            0.5,
            0.68,
            text,
            ha="center",
            fontsize=7.0,
            color="#657084",
        )


def _plot_paper_grn_panel_a(
    figure: plt.Figure,
    slot,
    bundle: GRNBundle,
    prediction: tuple[np.ndarray, np.ndarray],
    method: str,
    seed: int,
) -> None:
    nested = GridSpecFromSubplotSpec(
        1,
        9,
        subplot_spec=slot,
        width_ratios=(1, 1, 0.34, 1.3, 0.34, 1, 1, 0.28, 1.8),
        wspace=0.18,
    )
    density = bundle.combined.mean(axis=1)
    cell_index = int(np.argsort(density)[len(density) // 2])
    nodes = _paper_top_nodes(bundle)
    pred_a, pred_b = prediction
    matrices = [
        _paper_edge_matrix(bundle.maternal[cell_index], bundle, nodes),
        _paper_edge_matrix(bundle.paternal[cell_index], bundle, nodes),
        _paper_edge_matrix(bundle.combined[cell_index], bundle, nodes),
        _paper_edge_matrix(pred_a[cell_index], bundle, nodes),
        _paper_edge_matrix(pred_b[cell_index], bundle, nodes),
    ]

    axes = [figure.add_subplot(nested[0, index]) for index in range(9)]
    _paper_grn_heatmap(axes[0], matrices[0], CMAP_TRUTH_A, "Maternal truth")
    _paper_grn_heatmap(axes[1], matrices[1], CMAP_TRUTH_B, "Paternal truth")
    _paper_grn_arrow(axes[2], "independent\nthresholding")
    _paper_grn_heatmap(axes[3], matrices[2], CMAP_TOTAL, "Combined GRN")
    _paper_grn_arrow(axes[4], f"{method} fit\n(no truth)")
    _paper_grn_heatmap(axes[5], matrices[3], CMAP_PRED_A, "Recovered P1")
    _paper_grn_heatmap(axes[6], matrices[4], CMAP_PRED_B, "Recovered P2")
    _paper_grn_arrow(axes[7], "post-hoc\nscore")

    truth = bundle.maternal.ravel()
    score = pred_a.ravel()
    rng = np.random.default_rng(seed)
    sample_size = min(7000, len(truth))
    selected = rng.choice(len(truth), size=sample_size, replace=False)
    jitter = rng.normal(0.0, 0.035, sample_size)
    axes[8].scatter(
        truth[selected] + jitter,
        score[selected],
        s=7,
        alpha=0.20,
        color=BLUE,
        linewidths=0,
        rasterized=True,
    )
    auroc = _safe_binary_auc(truth, score)
    pearson = safe_correlation(truth, score)
    axes[8].text(
        0.96,
        0.07,
        f"Pearson r = {pearson:.3f}\nAUROC = {auroc:.3f}",
        transform=axes[8].transAxes,
        ha="right",
        color=BLUE,
        fontweight="bold",
    )
    axes[8].set_xticks([0, 1])
    axes[8].set_xlim(-0.18, 1.18)
    axes[8].set_ylim(-0.03, 1.03)
    axes[8].set_xlabel("Held-out maternal edge")
    axes[8].set_ylabel("Oriented P1 score")
    axes[8].set_title("Post-hoc edge scoring")
    clean_axis(axes[8], grid_axis="both")

    first_position = axes[0].get_position()
    second_position = axes[1].get_position()
    fifth_position = axes[5].get_position()
    sixth_position = axes[6].get_position()
    figure.text(
        first_position.x0 - 0.024,
        first_position.y1 + 0.018,
        "A",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    figure.text(
        (first_position.x0 + second_position.x1) / 2,
        first_position.y1 + 0.018,
        "Parent-specific held-out network truth",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )
    figure.text(
        (fifth_position.x0 + sixth_position.x1) / 2,
        fifth_position.y1 + 0.018,
        f"{method}-recovered exchangeable edge scores",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )


def _plot_paper_grn_panel_b(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    methods: list[str],
    contexts: list[str],
) -> None:
    width = min(0.16, 0.78 / max(1, len(methods)))
    x = np.arange(len(contexts))
    for method_index, method in enumerate(methods):
        values = []
        for context in contexts:
            subset = metrics[
                (metrics["context"] == context)
                & (metrics["method"] == method)
            ]
            values.append(float(subset["nmse"].iloc[0]) if len(subset) else np.nan)
        offset = (method_index - (len(methods) - 1) / 2) * width
        axis.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=method_color(method, method_index),
            label=method,
            zorder=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [context.replace("Cell quartile ", "Q") for context in contexts]
    )
    axis.set_ylabel("Binary phase-fit normalized MSE")
    axis.set_title("Network-phase reconstruction error")
    clean_axis(axis)
    panel_label(axis, "B")
    axis.legend(
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0, -0.23),
        columnspacing=1.0,
        handlelength=1.3,
        fontsize=7,
    )


def _plot_paper_grn_panel_c(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    methods: list[str],
    contexts: list[str],
    model_name: str,
) -> None:
    y = np.arange(len(contexts))
    offsets = (
        np.linspace(-0.18, 0.18, len(methods))
        if len(methods) > 1
        else np.asarray([0.0])
    )
    finite_values = metrics["pearson"].to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    minimum = (
        max(-0.05, float(np.min(finite_values)) - 0.08)
        if len(finite_values)
        else -0.05
    )
    maximum = (
        min(1.02, float(np.max(finite_values)) + 0.14)
        if len(finite_values)
        else 1.02
    )
    for context_index, context in enumerate(contexts):
        axis.hlines(
            context_index,
            minimum,
            maximum,
            color=GRID,
            linewidth=1.0,
        )
        for method_index, method in enumerate(methods):
            subset = metrics[
                (metrics["context"] == context)
                & (metrics["method"] == method)
            ]
            if subset.empty:
                continue
            value = float(subset["pearson"].iloc[0])
            yy = context_index + offsets[method_index]
            axis.scatter(
                value,
                yy,
                s=32,
                color=method_color(method, method_index),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            axis.text(
                value + 0.012,
                yy,
                f"r={value:.3f}",
                va="center",
                fontsize=6.8,
                color=method_color(method, method_index),
                fontweight="bold" if method == model_name else "normal",
            )
    axis.set_yticks(y)
    axis.set_yticklabels(
        [context.replace("Cell quartile ", "Quartile ") for context in contexts]
    )
    axis.set_xlim(minimum, maximum)
    axis.set_xlabel("Held-out edge-score Pearson r")
    axis.set_title("Phase-wise edge agreement (all values shown)")
    clean_axis(axis, grid_axis="x")
    panel_label(axis, "C")


def _plot_paper_grn_panel_d(
    axis: plt.Axes,
    bundle: GRNBundle,
    prediction: tuple[np.ndarray, np.ndarray],
) -> None:
    pred_a, pred_b = prediction
    exclusive = np.logical_xor(
        bundle.maternal > 0,
        bundle.paternal > 0,
    ).sum(axis=1)
    cell_index = int(np.argmax(exclusive))
    observed = bundle.combined[cell_index] > 0
    edge_indices = np.flatnonzero(observed)
    if edge_indices.size == 0:
        edge_indices = np.argsort(-bundle.combined.mean(axis=0))[:180]
    category = (
        2 * bundle.maternal[cell_index, edge_indices]
        + bundle.paternal[cell_index, edge_indices]
    )
    edge_indices = edge_indices[np.argsort(-category)]
    if edge_indices.size > 180:
        edge_indices = edge_indices[:180]
    x = np.arange(len(edge_indices))
    axis.plot(
        x,
        bundle.maternal[cell_index, edge_indices],
        color=TRUTH_A,
        label="Maternal truth",
        linewidth=1.3,
    )
    axis.plot(
        x,
        bundle.paternal[cell_index, edge_indices],
        color=TRUTH_B,
        label="Paternal truth",
        linewidth=1.3,
    )
    axis.scatter(
        x,
        pred_a[cell_index, edge_indices],
        color=PRED_A,
        label="Recovered P1",
        s=14,
        alpha=0.75,
        linewidths=0,
    )
    axis.scatter(
        x,
        pred_b[cell_index, edge_indices],
        color=PRED_B,
        label="Recovered P2",
        s=14,
        alpha=0.75,
        linewidths=0,
    )
    r_a = safe_correlation(
        pred_a[cell_index, edge_indices],
        bundle.maternal[cell_index, edge_indices],
    )
    r_b = safe_correlation(
        pred_b[cell_index, edge_indices],
        bundle.paternal[cell_index, edge_indices],
    )
    axis.text(
        0.03,
        0.95,
        f"rA={r_a:.3f}\nrB={r_b:.3f}",
        transform=axis.transAxes,
        va="top",
        fontweight="bold",
        color=INK,
    )
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Observed edges sorted by held-out phase class")
    axis.set_ylabel("Edge presence / phase score")
    axis.set_title(f"Representative per-cell GRN: {bundle.cells[cell_index]}")
    clean_axis(axis)
    panel_label(axis, "D")
    axis.legend(ncol=2, loc="upper right", fontsize=7)


def _plot_paper_grn_panel_e(
    axis: plt.Axes,
    bundle: GRNBundle,
    prediction: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    held_out_count = max(1, int(round(0.20 * len(bundle.cells))))
    held_out = np.sort(
        rng.choice(len(bundle.cells), size=held_out_count, replace=False)
    )
    pred_a = prediction[0][held_out]
    pred_b = prediction[1][held_out]
    true_union = np.maximum(
        bundle.maternal[held_out],
        bundle.paternal[held_out],
    ).ravel()
    true_shared = np.minimum(
        bundle.maternal[held_out],
        bundle.paternal[held_out],
    ).ravel()
    pred_union = np.maximum(pred_a, pred_b).ravel()
    pred_shared = np.minimum(pred_a, pred_b).ravel()

    sample_size = min(6500, len(true_union))
    selected = rng.choice(len(true_union), size=sample_size, replace=False)
    union_x = true_union[selected] + rng.normal(0.0, 0.035, sample_size)
    shared_x = (
        true_shared[selected] + 2.6 + rng.normal(0.0, 0.035, sample_size)
    )
    axis.scatter(
        union_x,
        pred_union[selected],
        s=7,
        alpha=0.18,
        color=BLUE,
        linewidths=0,
        label="Union / major",
        rasterized=True,
    )
    axis.scatter(
        shared_x,
        pred_shared[selected],
        s=7,
        alpha=0.18,
        color=GOLD,
        linewidths=0,
        label="Shared / minor",
        rasterized=True,
    )
    axis.set_xticks([0, 1, 2.6, 3.6])
    axis.set_xticklabels(
        ["Union\nabsent", "Union\npresent", "Shared\nabsent", "Shared\npresent"]
    )
    axis.set_ylim(-0.04, 1.04)
    axis.set_ylabel("Recovered label-invariant score")
    axis.set_title(
        f"Held-out union/shared-edge recovery ({held_out_count} cells)"
    )
    axis.text(
        0.04,
        0.95,
        f"union AUROC = {_safe_binary_auc(true_union, pred_union):.2f}\n"
        f"shared AUROC = {_safe_binary_auc(true_shared, pred_shared):.2f}",
        transform=axis.transAxes,
        va="top",
        fontweight="bold",
        color=INK,
    )
    clean_axis(axis, grid_axis="y")
    panel_label(axis, "E")
    axis.legend(loc="lower right")


def _create_paper_grn_figure(
    bundle: GRNBundle,
    metrics: pd.DataFrame,
    oriented: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[plt.Figure, str]:
    apply_style()
    primary_predictions = oriented["All cells"]
    primary_method = _choose_primary_method(primary_predictions, args)
    primary_prediction = primary_predictions[primary_method]
    preferred_methods = [
        args.model_name,
        "NMF2Factor",
        "RandomSplit",
        "MeanFractionShrinkage",
    ]
    available = metrics["method"].drop_duplicates().tolist()
    methods = [method for method in preferred_methods if method in available]
    methods.extend(method for method in available if method not in methods)
    context_names = [name for name, _ in _contexts(len(bundle.cells))]

    figure = plt.figure(figsize=(15.6, 10.7))
    figure.suptitle(
        "Figure 2-style HyperPhase recovery for thresholded per-cell GRNs",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.04,
        0.968,
        f"Primary display: threshold 0.1 · {len(bundle.cells)} cells · "
        f"{len(bundle.edge_names)} scoreable edges · {primary_method}",
        ha="left",
        fontsize=9.3,
        color="#606B7D",
    )
    outer = GridSpec(
        3,
        2,
        figure=figure,
        height_ratios=(1.17, 1.0, 1.08),
        hspace=0.47,
        wspace=0.27,
        left=0.05,
        right=0.985,
        top=0.92,
        bottom=0.08,
    )
    _plot_paper_grn_panel_a(
        figure,
        outer[0, :],
        bundle,
        primary_prediction,
        primary_method,
        args.seed,
    )
    axis_b = figure.add_subplot(outer[1, 0])
    axis_c = figure.add_subplot(outer[1, 1])
    axis_d = figure.add_subplot(outer[2, 0])
    axis_e = figure.add_subplot(outer[2, 1])
    _plot_paper_grn_panel_b(
        axis_b,
        metrics,
        methods,
        context_names,
    )
    _plot_paper_grn_panel_c(
        axis_c,
        metrics,
        methods,
        context_names,
        args.model_name,
    )
    _plot_paper_grn_panel_d(axis_d, bundle, primary_prediction)
    _plot_paper_grn_panel_e(
        axis_e,
        bundle,
        primary_prediction,
        args.seed,
    )
    add_caption(
        figure,
        "A, maternal/paternal truth is excluded from fitting and introduced "
        "only for post-hoc orientation/scoring. The combined GRN is an "
        "independently thresholded observation, not an assumed parental union. "
        "B-C, phase-fit error/correlation with all Pearson values shown. "
        "D, per-cell edge reconstruction. E, label-invariant union/shared "
        "recovery on deterministic held-out cells. Controls are called directly "
        "from phasehyper/evaluation/saber.py.",
        y=0.014,
    )
    return figure, primary_method


def _data_summary(bundle: GRNBundle, root: Path) -> pd.DataFrame:
    truth_union = np.maximum(bundle.maternal, bundle.paternal)
    truth_shared = np.minimum(bundle.maternal, bundle.paternal)
    row: dict[str, object] = {
        "n_cells": len(bundle.cells),
        "n_genes": len(bundle.genes),
        "n_selected_directed_edges": len(bundle.edge_names),
        "combined_density_selected": float(bundle.combined.mean()),
        "maternal_density_selected": float(bundle.maternal.mean()),
        "paternal_density_selected": float(bundle.paternal.mean()),
        "parental_union_density_selected": float(truth_union.mean()),
        "parental_shared_density_selected": float(truth_shared.mean()),
        "combined_union_mismatch_count_all_offdiagonal": (
            bundle.combined_union_mismatch_count
        ),
        "combined_union_match_rate_all_offdiagonal": (
            bundle.combined_union_match_rate
        ),
        "combined_semantics": "independently thresholded observation",
    }

    threshold_summary = root / "threshold_summary.csv"
    if threshold_summary.exists():
        summary = pd.read_csv(threshold_summary)
        row["threshold_summary_rows"] = len(summary)
        if "density" in summary.columns:
            row["threshold_summary_mean_density"] = float(
                pd.to_numeric(summary["density"], errors="coerce").mean()
            )
    return pd.DataFrame([row])


def main() -> None:
    args = _parse_args()
    bundle = load_grn_bundle(
        args.per_cell_root,
        max_edges=args.max_edges,
        min_prevalence=args.min_prevalence,
        max_prevalence=args.max_prevalence,
    )
    external = (
        load_external_grn_prediction(args.pred_npz, bundle)
        if args.pred_npz is not None
        else None
    )
    metrics, oriented = _evaluate(bundle, external, args)
    figure, primary_method = _create_paper_grn_figure(
        bundle,
        metrics,
        oriented,
        args,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        figure,
        args.output_dir / "simulationdata_figure2_hyperphase",
        dpi=args.dpi,
    )
    plt.close(figure)

    metrics.sort_values(["context_order", "method"]).to_csv(
        args.output_dir / "simulationdata_metrics.csv", index=False
    )
    _data_summary(bundle, args.per_cell_root).to_csv(
        args.output_dir / "simulationdata_data_summary.csv", index=False
    )

    pred_a, pred_b = oriented["All cells"][primary_method]
    np.savez_compressed(
        args.output_dir / "simulationdata_hyperphase_arrays.npz",
        combined=bundle.combined,
        truth_maternal=bundle.maternal,
        truth_paternal=bundle.paternal,
        pred_a=pred_a,
        pred_b=pred_b,
        cells=np.asarray(bundle.cells, dtype=str),
        genes=np.asarray(bundle.genes, dtype=str),
        edge_names=np.asarray(bundle.edge_names, dtype=str),
        edge_index=bundle.edge_index,
        method=np.asarray(primary_method),
        combined_union_mismatch_count=np.asarray(
            bundle.combined_union_mismatch_count
        ),
        combined_union_match_rate=np.asarray(bundle.combined_union_match_rate),
    )
    print(f"Simulation-data outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
