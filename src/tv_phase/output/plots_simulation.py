from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .layout import RunOutputLayout
from .plots_common import plot_metric_bars, save_umap, save_umap_overview


METRIC_NAMES = {
    "original_expression_embedding": "original",
    "phase_A_expression_embedding": "phaseA",
    "phase_B_expression_embedding": "phaseB",
    "truth_total_expression_embedding": "total",
    "truth_maternal_expression_embedding": "maternal",
    "truth_paternal_expression_embedding": "paternal",
}

UMAP_NAMES = {
    "original_expression_embedding": "original",
    "phase_A_expression_embedding": "phaseA",
    "phase_B_expression_embedding": "phaseB",
    "truth_maternal_expression_embedding": "truth_maternal",
    "truth_paternal_expression_embedding": "truth_paternal",
}


def _read_matrix(root: Path, filename: str) -> Optional[pd.DataFrame]:
    path = root / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str.strip()
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _read_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["gene", "chromosome", "start", "end", "midpoint"])
    frame = pd.read_csv(path, sep="\t", header=None, names=["gene", "chromosome", "start", "end", "strand"])
    frame["gene"] = frame["gene"].astype(str).str.strip()
    frame["chromosome"] = frame["chromosome"].astype(str).str.replace("chr", "", regex=False)
    frame["midpoint"] = (pd.to_numeric(frame["start"]) + pd.to_numeric(frame["end"])) / 2.0
    return frame


def _metric_frame(metric_df: pd.DataFrame) -> pd.DataFrame:
    sub = metric_df[metric_df["embedding"].isin(METRIC_NAMES)].copy()
    sub["representation"] = sub["embedding"].map(METRIC_NAMES)
    return sub[["source", "representation", "ari", "nmi", "fmi"]].rename(
        columns={"ari": "ARI", "nmi": "NMI", "fmi": "FMI"}
    )


def _plot_heatmap_grid(matrices: Dict[str, pd.DataFrame], path: Path, *, cmap: str = "viridis", symmetric: bool = False) -> None:
    if not matrices:
        return
    values = [frame.to_numpy(dtype=np.float32) for frame in matrices.values()]
    if symmetric:
        limit = max(float(np.nanmax(np.abs(value))) for value in values)
        vmin, vmax = -limit, limit
    else:
        vmin = min(float(np.nanmin(value)) for value in values)
        vmax = max(float(np.nanmax(value)) for value in values)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5), squeeze=False)
    image = None
    for ax, (name, frame) in zip(axes[0], matrices.items()):
        image = ax.imshow(frame.to_numpy(dtype=np.float32), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_gate_outputs(
    layout: RunOutputLayout,
    *,
    gene_names: List[str],
    gate_g: np.ndarray,
    truth_gate: pd.Series,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    pred = pd.DataFrame({"gene": gene_names, "pred_gate": np.asarray(gate_g, dtype=np.float32)})
    truth = pd.DataFrame({"gene": truth_gate.index.astype(str), "truth_gate": truth_gate.to_numpy(dtype=np.float32)})
    merged = pred.merge(truth, on="gene", how="inner").merge(positions, on="gene", how="left")
    merged["pred_phase"] = np.where(merged["pred_gate"] >= 0.5, "phaseB", "phaseA")
    merged["truth_phase"] = np.where(merged["truth_gate"] >= 0.5, "paternal", "maternal")
    merged["abs_gate_error"] = (merged["pred_gate"] - merged["truth_gate"]).abs()
    merged.to_csv(layout.plot_data / "gene_position_gate_pred_vs_truth.csv", index=False, encoding="utf-8-sig")
    merged[["gene", "pred_gate", "pred_phase", "chromosome", "start", "end", "midpoint"]].to_csv(
        layout.plot_data / "gene_position_gate_pred.csv", index=False, encoding="utf-8-sig"
    )
    merged[["gene", "truth_gate", "truth_phase", "chromosome", "start", "end", "midpoint"]].to_csv(
        layout.plot_data / "gene_position_gate_truth.csv", index=False, encoding="utf-8-sig"
    )

    ordered = merged.sort_values(["chromosome", "midpoint"], na_position="last").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(ordered.index, ordered["pred_gate"], linewidth=0.8, label="predicted")
    ax.plot(ordered.index, ordered["truth_gate"], linewidth=0.8, alpha=0.8, label="truth")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Genes ordered by chromosome and position")
    ax.set_ylabel("Gate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(layout.figures / "gene_position_gate_pred_vs_truth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return merged


def _save_phase_distribution(layout: RunOutputLayout, merged_gate: pd.DataFrame) -> None:
    pred = merged_gate["pred_phase"].value_counts().rename_axis("phase").reset_index(name="count")
    truth = merged_gate["truth_phase"].value_counts().rename_axis("phase").reset_index(name="count")
    pred["source"] = "predicted"
    truth["source"] = "truth"
    pred.to_csv(layout.plot_data / "phase_distribution_pred.csv", index=False, encoding="utf-8-sig")
    truth.to_csv(layout.plot_data / "phase_distribution_truth.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([pred, truth], ignore_index=True)
    combined.to_csv(layout.plot_data / "phase_distribution_pred_vs_truth.csv", index=False, encoding="utf-8-sig")
    pivot = combined.pivot(index="phase", columns="source", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Gene count")
    ax.set_title("Predicted vs truth phase distribution")
    ax.figure.tight_layout()
    ax.figure.savefig(layout.figures / "phase_distribution_pred_vs_truth.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def _save_chromosome_outputs(
    layout: RunOutputLayout,
    merged_gate: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
) -> None:
    for chromosome, subset in merged_gate.dropna(subset=["chromosome"]).groupby("chromosome", sort=False):
        safe = str(chromosome).replace(".0", "")
        try:
            prefix = f"chr{int(float(safe)):02d}"
        except ValueError:
            prefix = f"chr{safe}"
        subset = subset.sort_values("midpoint")
        subset.to_csv(layout.plot_data / f"{prefix}_gene_position_gate_pred_vs_truth.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["midpoint"], subset["pred_gate"], marker=".", linewidth=0.8, label="predicted")
        ax.plot(subset["midpoint"], subset["truth_gate"], marker=".", linewidth=0.8, label="truth")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Chromosome {safe}")
        ax.set_xlabel("Genomic midpoint")
        ax.set_ylabel("Gate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(layout.figures_chromosomes / f"{prefix}_gene_position_gate_pred_vs_truth.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

        genes = [gene for gene in subset["gene"] if gene in next(iter(matrices.values())).columns][:100]
        if len(genes) >= 2:
            corr = {name: frame.loc[:, genes].corr() for name, frame in matrices.items()}
            pd.concat(corr, names=["representation", "gene"]).to_csv(
                layout.plot_data / f"{prefix}_correlation_heatmap_pred_vs_truth.csv", encoding="utf-8-sig"
            )
            _plot_heatmap_grid(
                corr,
                layout.figures_chromosomes / f"{prefix}_correlation_heatmap_pred_vs_truth.png",
                cmap="coolwarm",
                symmetric=True,
            )


def _save_reconstruction_tables(
    layout: RunOutputLayout,
    phase_a: pd.DataFrame,
    phase_b: pd.DataFrame,
    maternal: pd.DataFrame,
    paternal: pd.DataFrame,
) -> None:
    def mse(a, b):
        return float(np.mean((a.to_numpy(dtype=np.float32) - b.to_numpy(dtype=np.float32)) ** 2))

    direct = (mse(phase_a, maternal) + mse(phase_b, paternal)) / 2.0
    flipped = (mse(phase_a, paternal) + mse(phase_b, maternal)) / 2.0
    pd.DataFrame(
        [
            {"comparison": "phaseA_vs_maternal", "MSE": mse(phase_a, maternal)},
            {"comparison": "phaseB_vs_paternal", "MSE": mse(phase_b, paternal)},
            {"comparison": "phaseA_vs_paternal", "MSE": mse(phase_a, paternal)},
            {"comparison": "phaseB_vs_maternal", "MSE": mse(phase_b, maternal)},
        ]
    ).to_csv(layout.tables / "reconstruction_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"direct_MSE": direct, "flipped_MSE": flipped, "best_orientation": "direct" if direct <= flipped else "flipped"}]
    ).to_csv(layout.tables / "orientation_metrics.csv", index=False, encoding="utf-8-sig")


def render_simulation_outputs(
    layout: RunOutputLayout,
    *,
    metric_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    cell_names: List[str],
    labels: List[str],
    gene_names: List[str],
    gate_g: np.ndarray,
    phase_a: pd.DataFrame,
    phase_b: pd.DataFrame,
    dataset_root: Path,
    files: Dict[str, str],
    seed: int,
) -> None:
    metrics = _metric_frame(metric_df)
    metrics.to_csv(layout.plot_data / "metric_pred_vs_truth.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars(metrics, layout.figures / "metric_pred_vs_truth.png", source_rows=True)

    umap_frames = {}
    for embedding_name, display_name in UMAP_NAMES.items():
        if embedding_name not in embeddings:
            continue
        umap_frames[display_name] = save_umap(
            embeddings[embedding_name],
            cell_names=cell_names,
            labels=labels,
            title=display_name,
            figure_path=layout.figures / f"umap_{display_name}.png",
            data_path=layout.plot_data / f"umap_{display_name}.csv",
            seed=seed,
        )
    save_umap_overview(umap_frames, layout.figures / "umap_overview.png")

    maternal = _read_matrix(dataset_root, files.get("truth_maternal", "E_M.csv"))
    paternal = _read_matrix(dataset_root, files.get("truth_paternal", "E_P.csv"))
    ratio = _read_matrix(dataset_root, files.get("truth_ratio", "ratio.csv"))
    if maternal is None or paternal is None:
        return
    maternal = maternal.loc[cell_names, gene_names]
    paternal = paternal.loc[cell_names, gene_names]
    if ratio is not None:
        truth_gate = ratio.loc[cell_names, gene_names].mean(axis=0).clip(0, 1)
    else:
        total = (maternal.mean(axis=0) + paternal.mean(axis=0)).replace(0, np.nan)
        truth_gate = (paternal.mean(axis=0) / total).fillna(0.5).clip(0, 1)

    positions = _read_positions(dataset_root / files.get("poswin_prior", "poswin_prior.txt"))
    merged_gate = _save_gate_outputs(
        layout,
        gene_names=gene_names,
        gate_g=gate_g,
        truth_gate=truth_gate,
        positions=positions,
    )
    _save_phase_distribution(layout, merged_gate)
    _save_reconstruction_tables(layout, phase_a, phase_b, maternal, paternal)

    selected_genes = [gene for gene in positions.sort_values(["chromosome", "midpoint"])["gene"] if gene in gene_names][:200]
    if not selected_genes:
        selected_genes = gene_names[:200]
    selected_cells = cell_names[:100]
    expression_matrices = {
        "pred_phaseA": phase_a.loc[selected_cells, selected_genes],
        "pred_phaseB": phase_b.loc[selected_cells, selected_genes],
        "truth_maternal": maternal.loc[selected_cells, selected_genes],
        "truth_paternal": paternal.loc[selected_cells, selected_genes],
    }
    for name, frame in expression_matrices.items():
        frame.to_csv(layout.plot_data / f"expression_heatmap_{name}.csv", encoding="utf-8-sig")
    _plot_heatmap_grid(expression_matrices, layout.figures / "expression_heatmap_pred_vs_truth.png")

    difference_matrices = {
        "phaseA_vs_maternal": expression_matrices["pred_phaseA"] - expression_matrices["truth_maternal"],
        "phaseB_vs_paternal": expression_matrices["pred_phaseB"] - expression_matrices["truth_paternal"],
    }
    for name, frame in difference_matrices.items():
        frame.to_csv(layout.plot_data / f"difference_heatmap_{name}.csv", encoding="utf-8-sig")
    _plot_heatmap_grid(
        difference_matrices,
        layout.figures / "difference_heatmap_pred_vs_truth.png",
        cmap="coolwarm",
        symmetric=True,
    )

    full_matrices = {
        "pred_phaseA": phase_a,
        "pred_phaseB": phase_b,
        "truth_maternal": maternal,
        "truth_paternal": paternal,
    }
    correlation_matrices = {name: frame.loc[:, selected_genes].corr() for name, frame in full_matrices.items()}
    for name, frame in correlation_matrices.items():
        frame.to_csv(layout.plot_data / f"correlation_heatmap_{name}.csv", encoding="utf-8-sig")
    _plot_heatmap_grid(
        correlation_matrices,
        layout.figures / "correlation_heatmap_pred_vs_truth.png",
        cmap="coolwarm",
        symmetric=True,
    )
    _save_chromosome_outputs(layout, merged_gate, full_matrices)


__all__ = ["render_simulation_outputs"]
