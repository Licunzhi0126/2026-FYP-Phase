"""Summary figures for simulation expression and GRN evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import GRID, INK, INK_2, NEG, SERIES, SURFACE, save_figure, style_axis
from .simulation_diagnostics import SimulationBundle, safe_corr


METHOD_ORDER = ["phasehyper", "RandomSplit", "NMF2Factor"]
METHOD_LABELS = {"phasehyper": "HyperPhase"}


def _grouped_bars(ax, frame: pd.DataFrame, metrics, labels, title) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    if not methods:
        raise ValueError(f"none of the expected methods are present: {METHOD_ORDER}")
    x = np.arange(len(metrics))
    width = 0.78 / len(methods)
    for index, method in enumerate(methods):
        row = frame.loc[frame["method"] == method].iloc[0]
        values = [pd.to_numeric(row.get(metric), errors="coerce") for metric in metrics]
        offset = (index - (len(methods) - 1) / 2) * (width + 0.02)
        ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS.get(method, method),
            color=SERIES[method],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        for position, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3f}",
                    (position, value),
                    ha="center",
                    fontsize=6.2,
                    color=INK_2,
                    rotation=90,
                    xytext=(0, 3 if value >= 0 else -3),
                    textcoords="offset points",
                    va="bottom" if value >= 0 else "top",
                    zorder=4,
                )
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=12, fontweight="bold")
    style_axis(ax)


def plot_expression_metrics(bundle: SimulationBundle, path: Path, dpi: int) -> Path:
    metrics_path = bundle.result_dir / "expression" / "metrics.csv"
    frame = pd.read_csv(metrics_path)
    columns = ["pcc_global", "pcc_cell", "pcc_A", "pcc_B", "imb", "imb_gene"]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    _grouped_bars(
        ax,
        frame,
        columns,
        ["PCC", "PCC\nper-cell", "PCC\nphase A", "PCC\nphase B", "imbalance", "imbalance\nper-gene"],
        "Expression decomposition",
    )
    trivial = frame.loc[frame["method"] == "[trivial] combined/2"]
    if len(trivial):
        row = trivial.iloc[0]
        for index, column in enumerate(columns):
            value = pd.to_numeric(row.get(column), errors="coerce")
            if np.isfinite(value):
                ax.plot(
                    [index - 0.42, index + 0.42],
                    [value, value],
                    color=NEG,
                    linewidth=1.6,
                    linestyle="--",
                    zorder=5,
                )
        ax.plot([], [], color=NEG, linewidth=1.6, linestyle="--", label="combined/2")
    ax.set_ylim(-0.25, 0.88)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.tight_layout()
    return save_figure(fig, path, dpi)


def plot_grn_metrics(bundle: SimulationBundle, path: Path, dpi: int) -> Path:
    frame = pd.read_csv(bundle.result_dir / "grn" / "differential.csv")
    frame = frame.loc[frame["name"].isin(METHOD_ORDER)].rename(columns={"name": "method"})
    frame["auroc_adv"] = pd.to_numeric(frame["auroc"], errors="coerce") - 0.5
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    _grouped_bars(
        ax,
        frame,
        ["pcc", "pcc_cell", "spearman", "auroc_adv", "skill_cal"],
        ["PCC(D)", "PCC(D)\nper-cell", "Spearman", "AUROC\n− 0.5", "skill\ncalibrated"],
        "GRN decomposition — differential component",
    )
    ax.set_ylim(-0.03, 0.235)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.tight_layout()
    return save_figure(fig, path, dpi)


def _dense_matrix(values: np.ndarray, edge_index: np.ndarray, n_genes: int) -> np.ndarray:
    matrix = np.zeros((n_genes, n_genes), dtype=float)
    matrix[edge_index[:, 0], edge_index[:, 1]] = values
    return matrix


def _matrix_panel(ax, matrix, title, *, limit, labels=False):
    ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    count = matrix.shape[0]
    ticks = sorted(set([0, count // 4, count // 2, 3 * count // 4, count - 1]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([index + 1 for index in ticks], fontsize=6)
    ax.set_yticklabels([index + 1 for index in ticks], fontsize=6)
    ax.tick_params(length=2, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.set_title(title, color=INK, fontsize=8.8, loc="left", pad=5)
    if labels:
        ax.set_xlabel("target gene", fontsize=7.5)
        ax.set_ylabel("source gene", fontsize=7.5)


def plot_grn_decomposition(bundle: SimulationBundle, path: Path, dpi: int) -> Path:
    archive_path = bundle.result_dir / "grn" / "edges.npz"
    with np.load(archive_path, allow_pickle=True) as archive:
        edge_index = archive["edge_index"]
        combined = archive["combined"]
        true_a, true_b = archive["true_A"], archive["true_B"]
        pred_a, pred_b = archive["pred_A"], archive["pred_B"]
        genes = list(archive["genes"])
        cell_type = archive["cell_type"]

    cell_scores = np.array(
        [
            safe_corr(
                (pred_a[cell] - pred_b[cell]) / 2,
                (true_a[cell] - true_b[cell]) / 2,
            )
            for cell in range(combined.shape[0])
        ]
    )
    cell = int(np.argsort(cell_scores)[combined.shape[0] // 2])
    n_genes = len(genes)
    combined_limit = max(float(np.max(np.abs(combined[cell]))), 1e-9)

    fig = plt.figure(figsize=(10.8, 7.4))
    grid = fig.add_gridspec(
        2, 3, hspace=0.24, wspace=0.28, left=0.07, right=0.985, top=0.86, bottom=0.07
    )
    fig.text(
        0.008,
        0.955,
        f"Per-cell gene × gene GRN decomposition — cell {cell} ({cell_type[cell]})",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    _matrix_panel(
        fig.add_subplot(grid[0, 0]),
        _dense_matrix(combined[cell], edge_index, n_genes),
        "INPUT   combined GRN",
        limit=combined_limit,
        labels=True,
    )
    for column, (values, label) in enumerate(
        ((pred_a[cell], "predicted phase A"), (pred_b[cell], "predicted phase B")), start=1
    ):
        _matrix_panel(
            fig.add_subplot(grid[0, column]),
            _dense_matrix(values, edge_index, n_genes),
            f"OUTPUT   {label}",
            limit=combined_limit / 2,
        )
    for column, (values, label) in enumerate(
        ((true_a[cell], "reference maternal"), (true_b[cell], "reference paternal")), start=1
    ):
        _matrix_panel(
            fig.add_subplot(grid[1, column]),
            _dense_matrix(values, edge_index, n_genes),
            f"ANSWER   {label}",
            limit=combined_limit / 2,
            labels=column == 1,
        )
    return save_figure(fig, path, dpi)
