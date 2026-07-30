"""Saber-style Figure 3 panels for simulation results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from phasehyper.evaluation import saber
from .plot_style import GRID, INK, INK_2, NEG, POS, SERIES, save_figure, style_axis
from .simulation_diagnostics import SimulationBundle
from .simulation_imbalance import signed_imbalance


METHOD_ORDER = ["HyperPhase", "RandomSplit", "NMF2Factor", "combined/2"]
DETECTION_ORDER = [
    "HyperPhase imbalance",
    "phase separability",
    "mean expression",
    "detected cells",
]


@dataclass(frozen=True)
class Figure3Data:
    methods: dict[str, tuple[np.ndarray, np.ndarray]]
    gene_level: pd.DataFrame
    context_mse: pd.DataFrame
    detection_scores: pd.DataFrame
    orientation: pd.DataFrame


def _orient_pair(pair, bundle: SimulationBundle):
    first, second, _ = saber.orient(
        pair[0],
        pair[1],
        bundle.maternal.to_numpy(),
        bundle.paternal.to_numpy(),
        "global",
    )
    return first, second


def build_figure3_data(bundle: SimulationBundle) -> Figure3Data:
    combined = bundle.combined.to_numpy()
    truth_m = bundle.maternal.to_numpy()
    truth_p = bundle.paternal.to_numpy()
    methods = {
        "HyperPhase": (
            bundle.pred_maternal.to_numpy(),
            bundle.pred_paternal.to_numpy(),
        ),
        "RandomSplit": _orient_pair(
            saber.baseline_random_split(combined, bundle.seed), bundle
        ),
        "NMF2Factor": _orient_pair(
            saber.baseline_nmf2(combined, bundle.seed), bundle
        ),
        "combined/2": (combined / 2, combined / 2),
    }

    truth_imb = signed_imbalance(truth_m, truth_p)
    pred_imb = signed_imbalance(*methods["HyperPhase"])
    truth_magnitude = np.mean(np.abs(truth_imb), axis=0)
    pred_magnitude = np.median(np.abs(pred_imb), axis=0)
    low_cut, high_cut = np.quantile(truth_magnitude, [1 / 3, 2 / 3])
    groups = np.where(
        truth_magnitude <= low_cut,
        "Low",
        np.where(truth_magnitude <= high_cut, "Mid", "High"),
    )
    hyper_gene_mse = 0.5 * (
        np.mean((methods["HyperPhase"][0] - truth_m) ** 2, axis=0)
        + np.mean((methods["HyperPhase"][1] - truth_p) ** 2, axis=0)
    )
    baseline_gene_mse = 0.5 * (
        np.mean((methods["combined/2"][0] - truth_m) ** 2, axis=0)
        + np.mean((methods["combined/2"][1] - truth_p) ** 2, axis=0)
    )
    gene_level = pd.DataFrame(
        {
            "gene_id": bundle.genes,
            "truth_mean_absolute_imbalance": truth_magnitude,
            "hyperphase_median_absolute_imbalance": pred_magnitude,
            "imbalance_group": groups,
            "combined_half_mse": baseline_gene_mse,
            "hyperphase_mse": hyper_gene_mse,
        }
    )

    contexts = [("All", np.ones(len(bundle.cells), dtype=bool))]
    labels = bundle.metadata["cell_type"].to_numpy()
    contexts.extend(
        (context, labels == context) for context in dict.fromkeys(labels.tolist())
    )
    mse_rows = []
    for context, mask in contexts:
        for method, (first, second) in methods.items():
            mse_rows.append(
                {
                    "context": context,
                    "method": method,
                    "phasefit_mse": 0.5
                    * (
                        np.mean((first[mask] - truth_m[mask]) ** 2)
                        + np.mean((second[mask] - truth_p[mask]) ** 2)
                    ),
                }
            )
    context_mse = pd.DataFrame(mse_rows)

    high_threshold = np.quantile(truth_magnitude, saber.HIGH_IMB_QUANTILE)
    detection_scores = pd.DataFrame(
        {
            "gene_id": bundle.genes,
            "is_high_imbalance": (truth_magnitude >= high_threshold).astype(int),
            "truth_mean_absolute_imbalance": truth_magnitude,
            "HyperPhase imbalance": np.mean(np.abs(pred_imb), axis=0),
            "phase separability": np.mean(
                np.abs(methods["HyperPhase"][0] - methods["HyperPhase"][1]), axis=0
            ),
            "mean expression": np.mean(combined, axis=0),
            "detected cells": np.mean(combined > 0, axis=0),
        }
    )
    orientation = pd.DataFrame(
        saber.orientation_audit(
            bundle.pred_a_raw.to_numpy(),
            bundle.pred_b_raw.to_numpy(),
            truth_m,
            truth_p,
            "phasehyper",
        )
    )
    return Figure3Data(
        methods=methods,
        gene_level=gene_level,
        context_mse=context_mse,
        detection_scores=detection_scores,
        orientation=orientation,
    )


def write_figure3_source_data(
    data: Figure3Data, source_dir: Path
) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        source_dir / "figure3_gene_level.csv",
        source_dir / "figure3_context_method_mse.csv",
        source_dir / "figure3_detection_scores.csv",
        source_dir / "figure3_orientation.csv",
    ]
    for frame, path in zip(
        (data.gene_level, data.context_mse, data.detection_scores, data.orientation),
        paths,
    ):
        frame.to_csv(path, index=False)
    return paths


def plot_figure3a(data: Figure3Data, path: Path, dpi: int) -> Path:
    frame = data.gene_level
    x = frame["truth_mean_absolute_imbalance"].to_numpy()
    y = frame["hyperphase_median_absolute_imbalance"].to_numpy()
    rho = float(spearmanr(x, y)[0]) if np.std(x) and np.std(y) else 0.0
    bins = pd.qcut(x, q=min(8, len(np.unique(x))), duplicates="drop")
    binned = (
        pd.DataFrame({"x": x, "y": y, "bin": bins})
        .groupby("bin", observed=True)[["x", "y"]]
        .median()
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(x, y, s=22, color=POS, alpha=0.55, edgecolor="none", label="genes")
    ax.plot(binned["x"], binned["y"], color=NEG, marker="o", linewidth=1.8, label="binned median")
    limit = max(float(np.max(x)), float(np.max(y)), 1e-6)
    ax.plot([0, limit], [0, limit], color=INK_2, linestyle="--", linewidth=0.8)
    ax.text(0.03, 0.96, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes, va="top")
    ax.set_xlabel("held-out truth mean |imbalance|")
    ax.set_ylabel("HyperPhase median |imbalance|")
    ax.set_title("Figure 3A — imbalance calibration", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    return save_figure(fig, path, dpi)


def plot_figure3b(data: Figure3Data, path: Path, dpi: int) -> Path:
    pivot = data.context_mse.pivot(index="context", columns="method", values="phasefit_mse")
    row_order = ["All"] + [row for row in pivot.index if row != "All"]
    pivot = pivot.reindex(index=row_order, columns=METHOD_ORDER)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, max(3.5, 0.65 * len(pivot))))
    image = ax.imshow(values, aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Figure 3B — phase-fit MSE", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="MSE")
    return save_figure(fig, path, dpi)


def plot_figure3c(data: Figure3Data, path: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, group in zip(axes, ("Low", "Mid", "High")):
        subset = data.gene_level.loc[data.gene_level["imbalance_group"] == group]
        for _, row in subset.iterrows():
            ax.plot(
                [0, 1],
                [row["combined_half_mse"], row["hyperphase_mse"]],
                color=GRID,
                linewidth=0.7,
                zorder=1,
            )
            ax.scatter(
                [0, 1],
                [row["combined_half_mse"], row["hyperphase_mse"]],
                color=[NEG, POS],
                s=13,
                zorder=2,
            )
        medians = [
            subset["combined_half_mse"].median(),
            subset["hyperphase_mse"].median(),
        ]
        ax.plot([0, 1], medians, color=INK, linewidth=2.2, marker="o", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["combined/2", "HyperPhase"], rotation=20)
        ax.set_title(f"{group} truth imbalance")
        style_axis(ax)
    axes[0].set_ylabel("per-gene phase-fit MSE")
    fig.suptitle("Figure 3C — paired gene MSE", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def _detection_metrics(frame: pd.DataFrame):
    truth = frame["is_high_imbalance"].to_numpy(dtype=int)
    rows = []
    curves = {}
    for name in DETECTION_ORDER:
        score = frame[name].to_numpy(dtype=float)
        if np.std(score) < 1e-12:
            auroc = 0.5
            auprc = float(np.mean(truth))
            recall = np.array([0.0, 1.0])
            precision = np.array([auprc, auprc])
        else:
            auroc = float(roc_auc_score(truth, score))
            auprc = float(average_precision_score(truth, score))
            precision, recall, _ = precision_recall_curve(truth, score)
        curves[name] = (recall, precision, auprc)
        rows.append({"predictor": name, "AUROC": auroc, "AUPRC": auprc})
    return pd.DataFrame(rows), curves


def plot_figure3d(data: Figure3Data, path: Path, dpi: int) -> Path:
    _, curves = _detection_metrics(data.detection_scores)
    prevalence = float(data.detection_scores["is_high_imbalance"].mean())
    colors = [POS, "#008300", "#e87ba4", INK_2]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for name, color in zip(DETECTION_ORDER, colors):
        recall, precision, auprc = curves[name]
        ax.plot(recall, precision, color=color, linewidth=1.7, label=f"{name} ({auprc:.3f})")
    ax.axhline(prevalence, color=NEG, linestyle="--", label=f"random expected ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_title("Figure 3D — high-imbalance detection", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    return save_figure(fig, path, dpi)


def plot_figure3e(data: Figure3Data, path: Path, dpi: int) -> Path:
    frame = data.orientation.set_index("level").reindex(["raw", "global", "per_gene"])
    labels = ["Raw labels", "Global reference\norientation", "Per-gene reference\norientation"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(np.arange(3), frame["mse"], color=[INK_2, POS, NEG])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel("phase-fit MSE")
    ax.set_title("Figure 3E — orientation audit", loc="left", fontweight="bold")
    style_axis(ax)
    return save_figure(fig, path, dpi)


def plot_figure3f(data: Figure3Data, path: Path, dpi: int) -> Path:
    metrics, _ = _detection_metrics(data.detection_scores)
    prevalence = float(data.detection_scores["is_high_imbalance"].mean())
    random_row = pd.DataFrame(
        [{"predictor": "random expected", "AUROC": 0.5, "AUPRC": prevalence}]
    )
    metrics = pd.concat([metrics, random_row], ignore_index=True)
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, metrics["AUROC"], width, label="AUROC", color=POS)
    ax.bar(x + width / 2, metrics["AUPRC"], width, label="AUPRC", color=NEG)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["predictor"], rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3F — detection metrics", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    return save_figure(fig, path, dpi)
