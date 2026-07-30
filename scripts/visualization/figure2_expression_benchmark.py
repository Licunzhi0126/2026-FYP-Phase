"""Create the answer-data Figure 2 expression benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visualization.figure2_io import (  # type: ignore
        ExpressionContext,
        load_answerdata_contexts,
        read_phase_matrix,
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
        ExpressionContext,
        load_answerdata_contexts,
        read_phase_matrix,
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
        description="Benchmark two-channel expression reconstruction on answerdata."
    )
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gse45719-pred-a", type=Path)
    parser.add_argument("--gse45719-pred-b", type=Path)
    parser.add_argument("--gse80810-pred-a", type=Path)
    parser.add_argument("--gse80810-pred-b", type=Path)
    parser.add_argument("--model-name", default="PhaseHyper")
    parser.add_argument("--primary-method")
    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--heatmap-genes", type=int, default=30)
    parser.add_argument("--heatmap-cells", type=int, default=60)
    parser.add_argument("--min-allelic-reads", type=int, default=2)
    parser.add_argument("--min-gse80810-reads", type=int, default=8)
    parser.add_argument("--min-scoreable-genes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _prediction_paths(
    args: argparse.Namespace,
) -> dict[str, tuple[Path | None, Path | None]]:
    return {
        "GSE45719": (args.gse45719_pred_a, args.gse45719_pred_b),
        "GSE80810": (args.gse80810_pred_a, args.gse80810_pred_b),
    }


def _methods_for_context(
    context: ExpressionContext,
    args: argparse.Namespace,
    external_prediction: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    methods = make_preview_methods(context.total.to_numpy(dtype=float), args.seed)
    if "NMF2" in methods:
        methods["NMF2 preview"] = methods.pop("NMF2")

    if external_prediction is not None:
        methods[args.model_name] = external_prediction
    return methods


def _load_external_predictions(
    contexts: list[ExpressionContext],
    args: argparse.Namespace,
) -> dict[
    str,
    tuple[
        ExpressionContext,
        np.ndarray,
        np.ndarray,
    ],
]:
    cache: dict[
        str,
        tuple[ExpressionContext, np.ndarray, np.ndarray],
    ] = {}
    for dataset, (pred_a_path, pred_b_path) in _prediction_paths(args).items():
        if (pred_a_path is None) != (pred_b_path is None):
            raise ValueError(
                f"{dataset} requires both prediction paths, not just one."
            )
        if pred_a_path is None or pred_b_path is None:
            continue

        candidates = [context for context in contexts if context.dataset == dataset]
        if not candidates:
            continue
        reference = max(candidates, key=lambda context: context.total.shape[0])
        pred_a = read_phase_matrix(
            pred_a_path, reference.total.index, reference.total.columns
        )
        pred_b = read_phase_matrix(
            pred_b_path, reference.total.index, reference.total.columns
        )
        cache[dataset] = (reference, pred_a, pred_b)
    return cache


def _subset_external_prediction(
    context: ExpressionContext,
    cached: tuple[ExpressionContext, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if cached is None:
        return None
    reference, pred_a, pred_b = cached
    cell_positions = reference.total.index.get_indexer(context.total.index)
    gene_positions = reference.total.columns.get_indexer(context.total.columns)
    if np.any(cell_positions < 0) or np.any(gene_positions < 0):
        raise ValueError(
            f"{context.name} is not a subset of the external prediction axes."
        )
    return (
        pred_a[np.ix_(cell_positions, gene_positions)],
        pred_b[np.ix_(cell_positions, gene_positions)],
    )


def _evaluate(
    contexts: list[ExpressionContext],
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
]:
    rows: list[dict[str, object]] = []
    oriented: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    external_cache = _load_external_predictions(contexts, args)
    for context_index, context in enumerate(contexts):
        external = _subset_external_prediction(
            context, external_cache.get(context.dataset)
        )
        methods = _methods_for_context(context, args, external)
        oriented[context.name] = {}
        truth_a = context.truth_a.to_numpy(dtype=float)
        truth_b = context.truth_b.to_numpy(dtype=float)
        total = context.total.to_numpy(dtype=float)
        mask = context.score_mask.to_numpy(dtype=bool)

        for method, (pred_a, pred_b) in methods.items():
            metrics, aligned_a, aligned_b = pair_metrics(
                pred_a,
                pred_b,
                truth_a,
                truth_b,
                mask=mask,
                total=total,
            )
            oriented[context.name][method] = (aligned_a, aligned_b)
            row: dict[str, object] = {
                "context_order": context_index,
                "context": context.name,
                "dataset": context.dataset,
                "method": method,
                "is_preview_control": method != args.model_name,
                "n_cells": context.total.shape[0],
                "n_genes": context.total.shape[1],
                "n_scoreable_entries": int(mask.sum()),
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


def _select_heatmap_axes(
    context: ExpressionContext,
    max_cells: int,
    max_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = context.total.to_numpy(dtype=float)
    cell_order = np.argsort(total.sum(axis=1))
    if len(cell_order) > max_cells:
        positions = np.linspace(0, len(cell_order) - 1, max_cells).astype(int)
        cell_order = cell_order[positions]

    variability = np.var(np.log1p(total[cell_order]), axis=0)
    gene_order = np.argsort(-variability)[: min(max_genes, total.shape[1])]
    return cell_order, gene_order


def _representative_gene(
    context: ExpressionContext,
) -> int:
    truth_a = context.truth_a.to_numpy(dtype=float)
    truth_b = context.truth_b.to_numpy(dtype=float)
    mask = context.score_mask.to_numpy(dtype=bool)
    denominator = truth_a + truth_b
    imbalance = np.divide(
        np.abs(truth_a - truth_b),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    imbalance = np.where(mask, imbalance, np.nan)
    support = np.sum(mask, axis=0)
    mean_imbalance = np.divide(
        np.nansum(imbalance, axis=0),
        support,
        out=np.zeros(imbalance.shape[1], dtype=float),
        where=support > 0,
    )
    score = mean_imbalance * np.log1p(support)
    score = np.nan_to_num(score, nan=-1.0)
    return int(np.argmax(score))


def _plot_heatmap_panel(
    figure: plt.Figure,
    subplotspec,
    context: ExpressionContext,
    prediction: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> None:
    cell_order, gene_order = _select_heatmap_axes(
        context, args.heatmap_cells, args.heatmap_genes
    )
    total = context.total.to_numpy(dtype=float)[np.ix_(cell_order, gene_order)]
    truth_a = context.truth_a.to_numpy(dtype=float)[
        np.ix_(cell_order, gene_order)
    ]
    truth_b = context.truth_b.to_numpy(dtype=float)[
        np.ix_(cell_order, gene_order)
    ]
    mask = context.score_mask.to_numpy(dtype=bool)[np.ix_(cell_order, gene_order)]
    pred_a = prediction[0][np.ix_(cell_order, gene_order)]
    pred_b = prediction[1][np.ix_(cell_order, gene_order)]

    truth_a = np.where(mask, truth_a, np.nan)
    truth_b = np.where(mask, truth_b, np.nan)
    arrays = [
        np.log1p(total),
        np.log1p(truth_a),
        np.log1p(truth_b),
        np.log1p(np.clip(pred_a, 0.0, None)),
        np.log1p(np.clip(pred_b, 0.0, None)),
    ]
    titles = [
        "Observed total",
        f"Truth {context.truth_a_label}",
        f"Truth {context.truth_b_label}",
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
        local_cmap = cmap.copy()
        local_cmap.set_bad("#E8ECF2")
        finite = array[np.isfinite(array)]
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        axis.imshow(
            array.transpose(),
            aspect="auto",
            interpolation="nearest",
            cmap=local_cmap,
            vmin=0.0,
            vmax=max(vmax, 1e-9),
        )
        axis.set_title(title, fontsize=8.5, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(
        f"{len(gene_order)} variable genes", fontsize=8, color=INK
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
        [name.replace("GSE", "GSE\n", 1) for name in context_order["context"]],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.6, len(x) - 0.4)
    clean_axis(axis)
    panel_label(axis, panel)


def _plot_representative_gene(
    axis: plt.Axes,
    context: ExpressionContext,
    prediction: tuple[np.ndarray, np.ndarray],
) -> None:
    gene_index = _representative_gene(context)
    gene = context.total.columns[gene_index]
    truth_a = context.truth_a.iloc[:, gene_index].to_numpy(dtype=float)
    truth_b = context.truth_b.iloc[:, gene_index].to_numpy(dtype=float)
    pred_a = prediction[0][:, gene_index]
    pred_b = prediction[1][:, gene_index]
    mask = context.score_mask.iloc[:, gene_index].to_numpy(dtype=bool)
    order = np.argsort(truth_a - truth_b)
    x = np.arange(len(order))

    axis.plot(x, truth_a[order], color=TRUTH_A, label=context.truth_a_label)
    axis.plot(x, truth_b[order], color=TRUTH_B, label=context.truth_b_label)
    axis.plot(
        x,
        pred_a[order],
        color=PRED_A,
        linestyle="--",
        label="predicted 1",
    )
    axis.plot(
        x,
        pred_b[order],
        color=PRED_B,
        linestyle="--",
        label="predicted 2",
    )
    unscored = np.flatnonzero(~mask[order])
    if unscored.size:
        axis.scatter(
            unscored,
            np.zeros_like(unscored, dtype=float),
            marker="x",
            s=16,
            color="#9AA3B2",
            label="low read support",
            zorder=4,
        )
    axis.set_title(f"Representative gene: {gene}", loc="left")
    axis.set_xlabel("Cells ordered by truth imbalance")
    axis.set_ylabel("Expression")
    axis.legend(ncol=2, fontsize=7, loc="upper left")
    clean_axis(axis)
    panel_label(axis, "D")


def _plot_major_minor(
    axis: plt.Axes,
    context: ExpressionContext,
    prediction: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> None:
    truth_a = context.truth_a.to_numpy(dtype=float)
    truth_b = context.truth_b.to_numpy(dtype=float)
    pred_a, pred_b = prediction
    valid = context.score_mask.to_numpy(dtype=bool)
    truth_major = np.maximum(truth_a, truth_b)[valid]
    truth_minor = np.minimum(truth_a, truth_b)[valid]
    pred_major = np.maximum(pred_a, pred_b)[valid]
    pred_minor = np.minimum(pred_a, pred_b)[valid]

    rng = np.random.default_rng(seed)
    if len(truth_major) > 2500:
        keep = rng.choice(len(truth_major), size=2500, replace=False)
        truth_major = truth_major[keep]
        truth_minor = truth_minor[keep]
        pred_major = pred_major[keep]
        pred_minor = pred_minor[keep]

    axis.scatter(
        truth_major,
        pred_major,
        s=7,
        alpha=0.30,
        color=BLUE,
        label="major",
        rasterized=True,
    )
    axis.scatter(
        truth_minor,
        pred_minor,
        s=7,
        alpha=0.30,
        color=GOLD,
        label="minor",
        rasterized=True,
    )
    maximum = max(
        float(np.nanpercentile(truth_major, 99)),
        float(np.nanpercentile(pred_major, 99)),
        1e-6,
    )
    axis.plot([0, maximum], [0, maximum], color="#7B8493", linewidth=0.8)
    axis.set_xlim(0, maximum)
    axis.set_ylim(0, maximum)
    axis.set_xlabel("Truth expression")
    axis.set_ylabel("Predicted expression")
    axis.legend(loc="upper left")
    clean_axis(axis, grid_axis=None)
    panel_label(axis, "E")


def _create_figure(
    contexts: list[ExpressionContext],
    metrics: pd.DataFrame,
    oriented: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> tuple[plt.Figure, str]:
    apply_style()
    primary = contexts[0]
    primary_method = _choose_primary_method(oriented[primary.name], args)
    primary_prediction = oriented[primary.name][primary_method]

    figure = plt.figure(figsize=(17.2, 9.2))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.08, 1.08, 1.0),
        height_ratios=(0.85, 1.0),
        left=0.045,
        right=0.985,
        top=0.91,
        bottom=0.12,
        wspace=0.34,
        hspace=0.52,
    )
    _plot_heatmap_panel(
        figure, grid[0, :2], primary, primary_prediction, args
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
    axis_c.set_title("Allelic differential agreement", loc="left")

    axis_d = figure.add_subplot(grid[1, 1])
    _plot_representative_gene(axis_d, primary, primary_prediction)

    axis_e = figure.add_subplot(grid[1, 2])
    _plot_major_minor(axis_e, primary, primary_prediction, args.seed)
    axis_e.set_title("Major/minor reconstruction", loc="left")

    figure.suptitle(
        "Figure 2-style allele-resolved expression benchmark",
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
        f"Primary view: {primary.name} · displayed method: {primary_method}",
        ha="left",
        fontsize=9,
        color="#667085",
    )
    add_caption(
        figure,
        "Preview controls use observed totals only and are not PhaseHyper model "
        "results. Grey heatmap entries and × marks are excluded for insufficient "
        "allelic read support. One global channel swap is allowed per context.",
    )
    return figure, primary_method


def _data_summary(contexts: list[ExpressionContext]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context in contexts:
        mask = context.score_mask.to_numpy(dtype=bool)
        stage_counts = context.cell_metadata["stage"].value_counts().to_dict()
        rows.append(
            {
                "context": context.name,
                "dataset": context.dataset,
                "n_cells": context.total.shape[0],
                "n_genes": context.total.shape[1],
                "n_scoreable_entries": int(mask.sum()),
                "scoreable_fraction": float(mask.mean()),
                "truth_channel_a": context.truth_a_label,
                "truth_channel_b": context.truth_b_label,
                "stage_counts": "; ".join(
                    f"{stage}:{count}"
                    for stage, count in sorted(stage_counts.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    contexts = load_answerdata_contexts(
        args.answer_root,
        n_genes=args.n_genes,
        min_allelic_reads=args.min_allelic_reads,
        min_gse80810_reads=args.min_gse80810_reads,
        min_scoreable_genes=args.min_scoreable_genes,
    )
    metrics, oriented = _evaluate(contexts, args)
    figure, primary_method = _create_figure(contexts, metrics, oriented, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        figure,
        args.output_dir / "answerdata_figure2",
        dpi=args.dpi,
    )
    plt.close(figure)

    metrics.sort_values(["context_order", "method"]).to_csv(
        args.output_dir / "answerdata_metrics.csv", index=False
    )
    _data_summary(contexts).to_csv(
        args.output_dir / "answerdata_data_summary.csv", index=False
    )

    primary = contexts[0]
    pred_a, pred_b = oriented[primary.name][primary_method]
    np.savez_compressed(
        args.output_dir / "answerdata_primary_arrays.npz",
        total=primary.total.to_numpy(dtype=float),
        truth_a=primary.truth_a.to_numpy(dtype=float),
        truth_b=primary.truth_b.to_numpy(dtype=float),
        score_mask=primary.score_mask.to_numpy(dtype=bool),
        pred_a=pred_a,
        pred_b=pred_b,
        cells=np.asarray(primary.total.index, dtype=str),
        genes=np.asarray(primary.total.columns, dtype=str),
        dataset=np.asarray(primary.dataset),
        method=np.asarray(primary_method),
        truth_a_label=np.asarray(primary.truth_a_label),
        truth_b_label=np.asarray(primary.truth_b_label),
    )
    print(f"Answer-data outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
