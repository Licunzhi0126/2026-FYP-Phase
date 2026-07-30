"""Create the simulation-data Figure 2 GRN benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    )
    from visualization.figure2_style import (  # type: ignore
        BLUE,
        CMAP_PRED_A,
        CMAP_PRED_B,
        CMAP_TOTAL,
        CMAP_TRUTH_A,
        CMAP_TRUTH_B,
        GOLD,
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
    from .figure2_metrics import make_preview_methods, pair_metrics
    from .figure2_style import (
        BLUE,
        CMAP_PRED_A,
        CMAP_PRED_B,
        CMAP_TOTAL,
        CMAP_TRUTH_A,
        CMAP_TRUTH_B,
        GOLD,
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
    parser.add_argument("--model-name", default="PhaseHyper")
    parser.add_argument("--primary-method")
    parser.add_argument("--max-edges", type=int, default=1200)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-prevalence", type=float, default=0.95)
    parser.add_argument("--heatmap-edges", type=int, default=50)
    parser.add_argument("--heatmap-cells", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _contexts(n_cells: int) -> list[tuple[str, np.ndarray]]:
    all_cells = np.arange(n_cells)
    contexts: list[tuple[str, np.ndarray]] = [("All cells", all_cells)]
    if n_cells >= 16:
        for index, cell_indices in enumerate(np.array_split(all_cells, 4), start=1):
            contexts.append((f"Cell block {index}", cell_indices))
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
        methods = make_preview_methods(combined, seed=args.seed + context_index)
        if "NMF2" in methods:
            methods["NMF2 preview"] = methods.pop("NMF2")
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
    if "NMF2 preview" in methods:
        return "NMF2 preview"
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
    figure, primary_method = _create_figure(bundle, metrics, oriented, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        figure,
        args.output_dir / "simulationdata_figure2",
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
        args.output_dir / "simulationdata_primary_arrays.npz",
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
