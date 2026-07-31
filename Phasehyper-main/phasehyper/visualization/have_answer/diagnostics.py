"""Ground-truth data alignment and chromosome-level diagnostic figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phasehyper.evaluation import saber
from ..plot_style import INK, INK_2, NEG, POS, save_figure, style_axis


EPS = 1e-9


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required matrix not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def _natural_key(value: object) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return 0.0
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        return np.eye(values.shape[1], dtype=float)
    if np.isfinite(values).all():
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(values, rowvar=False)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    n_features = values.shape[1]
    matrix = np.eye(n_features, dtype=float)
    for left in range(n_features):
        for right in range(left + 1, n_features):
            value = safe_corr(values[:, left], values[:, right])
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


@dataclass(frozen=True)
class VisualizationBundle:
    sim_dir: Path
    result_dir: Path
    combined: pd.DataFrame
    maternal: pd.DataFrame
    paternal: pd.DataFrame
    pred_a_raw: pd.DataFrame
    pred_b_raw: pd.DataFrame
    pred_maternal: pd.DataFrame
    pred_paternal: pd.DataFrame
    observed_mask: pd.DataFrame
    annotation: pd.DataFrame
    metadata: pd.DataFrame
    phase_mapping: dict
    phase_a_label: str
    phase_b_label: str
    seed: int

    @property
    def genes(self) -> list[str]:
        return self.combined.columns.tolist()

    @property
    def cells(self) -> list[str]:
        return self.combined.index.tolist()

    @property
    def chromosomes(self) -> list[str]:
        values = self.annotation["chromosome"].dropna().astype(str).unique()
        return sorted(values.tolist(), key=_natural_key)

    def genes_for_chromosome(self, chromosome: str) -> list[str]:
        mask = self.annotation["chromosome"].astype(str) == str(chromosome)
        rows = self.annotation.loc[mask].copy()
        sort_columns = [column for column in ("start", "TSS", "end") if column in rows]
        if sort_columns:
            rows = rows.sort_values(sort_columns, kind="stable")
        return [gene for gene in rows.index.astype(str) if gene in self.combined.columns]

    def ordered_genes(self) -> list[str]:
        ordered: list[str] = []
        for chromosome in self.chromosomes:
            ordered.extend(self.genes_for_chromosome(chromosome))
        ordered.extend(gene for gene in self.genes if gene not in set(ordered))
        return ordered


def load_visualization_bundle(
    sim_dir: Path,
    result_dir: Path,
) -> VisualizationBundle:
    """Load and align all plotting inputs once, preserving combined matrix order."""
    sim_dir = Path(sim_dir)
    result_dir = Path(result_dir)
    combined = _read_matrix(sim_dir / "input" / "combined_true_expression.csv")
    generic_a = sim_dir / "groundtruth" / "phase_A_true.csv"
    generic_b = sim_dir / "groundtruth" / "phase_B_true.csv"
    uses_generic_labels = generic_a.exists() and generic_b.exists()
    if uses_generic_labels:
        maternal = _read_matrix(generic_a)
        paternal = _read_matrix(generic_b)
        phase_a_label, phase_b_label = "Phase A", "Phase B"
    else:
        maternal = _read_matrix(
            sim_dir / "groundtruth" / "maternal_true_expression.csv"
        )
        paternal = _read_matrix(
            sim_dir / "groundtruth" / "paternal_true_expression.csv"
        )
        phase_a_label, phase_b_label = "Maternal", "Paternal"
    pred_a = _read_matrix(result_dir / "expression" / "phase_A.csv")
    pred_b = _read_matrix(result_dir / "expression" / "phase_B.csv")

    matrices = (maternal, paternal, pred_a, pred_b)
    cells = [
        cell
        for cell in combined.index
        if all(cell in matrix.index for matrix in matrices)
    ]
    genes = [
        gene
        for gene in combined.columns
        if all(gene in matrix.columns for matrix in matrices)
    ]
    if not cells or not genes:
        raise ValueError("visualization inputs have no common cells or genes")

    combined = combined.loc[cells, genes].astype(float)
    maternal = maternal.loc[cells, genes].astype(float)
    paternal = paternal.loc[cells, genes].astype(float)
    pred_a = pred_a.loc[cells, genes].astype(float)
    pred_b = pred_b.loc[cells, genes].astype(float)
    mask_path = sim_dir / "observed_mask.csv"
    if not mask_path.exists():
        mask_path = sim_dir / "groundtruth" / "observed_mask.csv"
    if mask_path.exists():
        mask_frame = _read_matrix(mask_path).reindex(index=cells, columns=genes)
        observed_mask = mask_frame.fillna(0).astype(float).ne(0)
    else:
        observed_mask = maternal.notna() & paternal.notna()
    observed_mask &= maternal.notna() & paternal.notna()
    if not observed_mask.to_numpy().any():
        raise ValueError("visualization observed mask contains no truth entries")
    maternal = maternal.where(observed_mask)
    paternal = paternal.where(observed_mask)

    annotation_path = sim_dir / "input" / "gene_info.csv"
    annotation = pd.read_csv(annotation_path)
    if "gene_id" not in annotation or "chromosome" not in annotation:
        raise ValueError(f"{annotation_path} must contain gene_id and chromosome")
    annotation["gene_id"] = annotation["gene_id"].astype(str)
    annotation = annotation.drop_duplicates("gene_id").set_index("gene_id")
    annotation = annotation.reindex(genes)

    metadata_path = sim_dir / "input" / "cell_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    if "cell_id" not in metadata:
        raise ValueError(f"{metadata_path} must contain cell_id")
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    metadata = metadata.drop_duplicates("cell_id").set_index("cell_id").reindex(cells)
    if "cell_type" not in metadata:
        metadata["cell_type"] = "All"
    metadata["cell_type"] = metadata["cell_type"].fillna("Unknown").astype(str)

    oriented_a, oriented_b, mapping = saber.orient(
        pred_a.to_numpy(),
        pred_b.to_numpy(),
        maternal.to_numpy(),
        paternal.to_numpy(),
        "global",
        observed_mask.to_numpy(),
    )
    pred_maternal = pd.DataFrame(oriented_a, index=cells, columns=genes)
    pred_paternal = pd.DataFrame(oriented_b, index=cells, columns=genes)

    seed = 0
    metrics_path = result_dir / "expression" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "method" in metrics:
            row = metrics.loc[metrics["method"] == "phasehyper"]
            if len(row) and "seed" in row:
                seed = int(row.iloc[0]["seed"])

    return VisualizationBundle(
        sim_dir=sim_dir,
        result_dir=result_dir,
        combined=combined,
        maternal=maternal,
        paternal=paternal,
        pred_a_raw=pred_a,
        pred_b_raw=pred_b,
        pred_maternal=pred_maternal,
        pred_paternal=pred_paternal,
        observed_mask=observed_mask,
        annotation=annotation,
        metadata=metadata,
        phase_mapping=mapping,
        phase_a_label=phase_a_label,
        phase_b_label=phase_b_label,
        seed=seed,
    )


def _limit_genes(
    bundle: VisualizationBundle,
    genes: Sequence[str],
    max_genes: int | None,
) -> list[str]:
    genes = list(genes)
    if max_genes is None or len(genes) <= max_genes:
        return genes
    support = bundle.observed_mask[genes].sum(axis=0)
    order = {gene: index for index, gene in enumerate(genes)}
    return sorted(
        genes,
        key=lambda gene: (-int(support[gene]), order[gene]),
    )[:max(2, int(max_genes))]


def write_aligned_annotation(bundle: VisualizationBundle, path: Path) -> Path:
    frame = bundle.annotation.copy()
    frame.index.name = "gene_id"
    frame["plot_order"] = frame.index.map(
        {gene: index for index, gene in enumerate(bundle.ordered_genes())}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(path, index=False)
    return path


def _image(ax, values, title: str, *, cmap="viridis", vmin=None, vmax=None):
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("genes")
    ax.set_ylabel("cells")
    return image


def plot_phase_expression_heatmap(
    bundle: VisualizationBundle, chromosome: str, path: Path, dpi: int
) -> Path:
    genes = bundle.genes_for_chromosome(chromosome)
    if not genes:
        raise ValueError(f"{chromosome} has no aligned genes")
    truth_m = bundle.maternal[genes].to_numpy()
    truth_p = bundle.paternal[genes].to_numpy()
    pred_m = bundle.pred_maternal[genes].to_numpy()
    pred_p = bundle.pred_paternal[genes].to_numpy()
    panels = [
        (truth_m, f"Truth {bundle.phase_a_label}", "viridis"),
        (pred_m, f"Pred {bundle.phase_a_label}", "viridis"),
        (pred_m - truth_m, "Residual (pred − truth)", "RdBu_r"),
        (truth_p, f"Truth {bundle.phase_b_label}", "viridis"),
        (pred_p, f"Pred {bundle.phase_b_label}", "viridis"),
        (pred_p - truth_p, "Residual (pred − truth)", "RdBu_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (values, title, cmap) in zip(axes.flat, panels):
        limit = np.nanmax(np.abs(values)) if "Residual" in title else None
        image = _image(
            ax,
            values,
            title,
            cmap=cmap,
            vmin=-limit if limit else None,
            vmax=limit if limit else None,
        )
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"{chromosome} phase expression", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_phase_contrast_track(
    bundle: VisualizationBundle, chromosome: str, path: Path, dpi: int
) -> Path:
    genes = bundle.genes_for_chromosome(chromosome)
    truth_m = bundle.maternal[genes].mean(axis=0).to_numpy()
    truth_p = bundle.paternal[genes].mean(axis=0).to_numpy()
    pred_m = (
        bundle.pred_maternal[genes]
        .where(bundle.observed_mask[genes])
        .mean(axis=0)
        .to_numpy()
    )
    pred_p = (
        bundle.pred_paternal[genes]
        .where(bundle.observed_mask[genes])
        .mean(axis=0)
        .to_numpy()
    )
    x = np.arange(len(genes))
    truth_ratio = truth_m / (np.abs(truth_p) + EPS)
    pred_ratio = pred_m / (np.abs(pred_p) + EPS)
    truth_delta, pred_delta = truth_m - truth_p, pred_m - pred_p

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, values, label, color in (
        (
            axes[0],
            truth_ratio,
            f"Truth {bundle.phase_a_label}/{bundle.phase_b_label}",
            INK_2,
        ),
        (
            axes[1],
            pred_ratio,
            f"Pred {bundle.phase_a_label}/{bundle.phase_b_label}",
            POS,
        ),
    ):
        ax.plot(x, values, color=color, linewidth=1.4)
        ax.set_ylabel(label)
        style_axis(ax)
    axes[2].plot(x, truth_delta, color=INK_2, label="truth M−P")
    axes[2].plot(x, pred_delta, color=POS, label="pred M−P")
    axes[2].legend(frameon=False)
    axes[2].set_ylabel("contrast")
    style_axis(axes[2])
    axes[3].plot(x, np.abs(pred_delta - truth_delta), color=NEG)
    axes[3].set_ylabel("|residual|")
    axes[3].set_xlabel("genes in genomic order")
    style_axis(axes[3])
    fig.suptitle(f"{chromosome} phase contrast", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_gene_correlation_track(
    bundle: VisualizationBundle, chromosome: str, path: Path, dpi: int
) -> Path:
    genes = bundle.genes_for_chromosome(chromosome)
    correlations = {
        "maternal": np.array(
            [safe_corr(bundle.maternal[g], bundle.pred_maternal[g]) for g in genes]
        ),
        "paternal": np.array(
            [safe_corr(bundle.paternal[g], bundle.pred_paternal[g]) for g in genes]
        ),
        "contrast": np.array(
            [
                safe_corr(
                    bundle.maternal[g] - bundle.paternal[g],
                    bundle.pred_maternal[g] - bundle.pred_paternal[g],
                )
                for g in genes
            ]
        ),
    }
    window = max(1, min(9, len(genes) // 2 * 2 + 1))
    x = np.arange(len(genes))
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for ax, (name, values), color in zip(
        axes, correlations.items(), (POS, "#008300", NEG)
    ):
        rolling = pd.Series(values).rolling(window, center=True, min_periods=1).median()
        ax.scatter(x, values, s=12, alpha=0.55, color=color)
        ax.plot(x, rolling, color=INK, linewidth=1.5, label=f"rolling median ({window})")
        ax.axhline(0, color=INK_2, linewidth=0.7)
        ax.set_ylabel(f"{name}\nr")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax)
    axes[-1].set_xlabel("genes in genomic order")
    fig.suptitle(f"{chromosome} per-gene correlation", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_phase_correlation_heatmap(
    bundle: VisualizationBundle,
    chromosome: str,
    path: Path,
    dpi: int,
    max_genes: int | None = None,
) -> Path:
    genes = _limit_genes(
        bundle, bundle.genes_for_chromosome(chromosome), max_genes
    )
    observed = bundle.observed_mask[genes]
    truth_m = correlation_matrix(bundle.maternal[genes].to_numpy())
    pred_m = correlation_matrix(
        bundle.pred_maternal[genes].where(observed).to_numpy()
    )
    truth_p = correlation_matrix(bundle.paternal[genes].to_numpy())
    pred_p = correlation_matrix(
        bundle.pred_paternal[genes].where(observed).to_numpy()
    )
    panels = [
        (truth_m, f"Truth {bundle.phase_a_label} corr"),
        (pred_m, f"Pred {bundle.phase_a_label} corr"),
        (pred_m - truth_m, f"{bundle.phase_a_label} residual"),
        (truth_p, f"Truth {bundle.phase_b_label} corr"),
        (pred_p, f"Pred {bundle.phase_b_label} corr"),
        (pred_p - truth_p, f"{bundle.phase_b_label} residual"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (matrix, title) in zip(axes.flat, panels):
        limit = 1.0 if "residual" not in title.lower() else max(1e-6, np.max(np.abs(matrix)))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{chromosome} phase correlation", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def _delta_correlation_matrices(
    bundle: VisualizationBundle, genes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = bundle.observed_mask[list(genes)]
    truth = correlation_matrix(bundle.maternal[list(genes)].to_numpy()) - correlation_matrix(
        bundle.paternal[list(genes)].to_numpy()
    )
    predicted = correlation_matrix(
        bundle.pred_maternal[list(genes)].where(observed).to_numpy()
    ) - correlation_matrix(
        bundle.pred_paternal[list(genes)].where(observed).to_numpy()
    )
    return truth, predicted, predicted - truth


def plot_delta_correlation(
    bundle: VisualizationBundle,
    chromosome: str,
    path: Path,
    dpi: int,
    max_genes: int | None = None,
) -> Path:
    genes = _limit_genes(
        bundle, bundle.genes_for_chromosome(chromosome), max_genes
    )
    panels = zip(
        _delta_correlation_matrices(bundle, genes),
        ("Truth Δcorr", "Predicted Δcorr", "Residual"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (matrix, title) in zip(axes, panels):
        limit = max(1e-6, np.max(np.abs(matrix)))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{chromosome} differential correlation", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_genome_delta_correlation(
    bundle: VisualizationBundle,
    path: Path,
    dpi: int,
    max_genes: int | None = None,
) -> Path:
    genes = _limit_genes(bundle, bundle.ordered_genes(), max_genes)
    matrices = _delta_correlation_matrices(bundle, genes)
    boundaries = []
    offset = 0
    selected = set(genes)
    for chromosome in bundle.chromosomes:
        offset += sum(
            gene in selected
            for gene in bundle.genes_for_chromosome(chromosome)
        )
        if offset:
            boundaries.append(offset - 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, matrix, title in zip(
        axes, matrices, ("Truth Δcorr", "Predicted Δcorr", "Residual")
    ):
        limit = max(1e-6, np.max(np.abs(matrix)))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for boundary in boundaries[:-1]:
            ax.axhline(boundary, color=INK, linewidth=0.5)
            ax.axvline(boundary, color=INK, linewidth=0.5)
        ax.set_title(title, fontsize=9, loc="left")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Genome-wide differential correlation", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)


def plot_gene_detail(
    bundle: VisualizationBundle,
    gene: str,
    path: Path,
    dpi: int,
) -> Path:
    if gene not in bundle.genes:
        raise KeyError(f"requested gene not found after alignment: {gene}")
    truth_m = bundle.maternal[gene].to_numpy()
    truth_p = bundle.paternal[gene].to_numpy()
    pred_m = bundle.pred_maternal[gene].to_numpy()
    pred_p = bundle.pred_paternal[gene].to_numpy()
    truth_imb = (truth_m - truth_p) / (np.abs(truth_m) + np.abs(truth_p) + EPS)
    pred_imb = (pred_m - pred_p) / (np.abs(pred_m) + np.abs(pred_p) + EPS)
    x = np.arange(len(bundle.cells))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for values, label, color, ls in (
        (truth_m, f"truth {bundle.phase_a_label}", INK, "-"),
        (truth_p, f"truth {bundle.phase_b_label}", INK_2, "--"),
        (pred_m, f"pred {bundle.phase_a_label}", POS, "-"),
        (pred_p, f"pred {bundle.phase_b_label}", NEG, "--"),
    ):
        axes[0].plot(x, values, label=label, color=color, linestyle=ls, linewidth=1)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_ylabel("expression")
    axes[1].plot(x, truth_imb, color=INK, label="truth")
    axes[1].plot(x, pred_imb, color=POS, label="prediction")
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("signed imbalance")
    axes[2].plot(x, pred_imb - truth_imb, color=NEG)
    axes[2].axhline(0, color=INK_2, linewidth=0.7)
    axes[2].set_ylabel("imbalance residual")
    axes[2].set_xlabel("cells")
    for ax in axes:
        style_axis(ax)
    fig.suptitle(f"{gene} detail", fontsize=12, fontweight="bold")
    return save_figure(fig, path, dpi)
