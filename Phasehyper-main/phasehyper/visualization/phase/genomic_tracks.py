"""Chromosome-ordered phase-resolution tracks and clustering summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phasehyper.visualization.plot_style import apply_plot_style, POS, NEG


def _longest_run(values: list[str], target: str) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value == target else 0
        best = max(best, current)
    return best


def prepare_genomic_tracks(
    resolution: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    window: int,
    permutations: int,
    seed: int,
):
    if annotation.empty:
        raise ValueError("gene annotation is unavailable")
    annotation = annotation.rename(columns={"gene_id": "gene", "TES": "end"})
    required = {"gene", "chromosome", "TSS"}
    if not required.issubset(annotation):
        raise ValueError(f"gene annotation missing {sorted(required - set(annotation))}")
    annotation_columns = [
        column for column in annotation.columns
        if column == "gene" or column not in resolution.columns
    ]
    data = resolution.merge(annotation[annotation_columns], on="gene", how="inner")
    data = data.dropna(subset=["chromosome", "TSS"]).sort_values(["chromosome", "TSS"])
    if data.empty:
        raise ValueError("no genes have real chromosome and TSS annotations")
    for column in ("mean_signed_contrast", "separation_magnitude", "direction_consistency", "detection_rate", "resolution_score"):
        data[f"{column}_rolling"] = data.groupby("chromosome")[column].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).median()
        )
    summaries = []
    rng = np.random.default_rng(seed)
    null_rows = []
    for chromosome, frame in data.groupby("chromosome", sort=False):
        classes = frame["resolution_class"].astype(str).tolist()
        adjacent = float(np.mean(np.asarray(classes[:-1]) == np.asarray(classes[1:]))) if len(classes) > 1 else np.nan
        summaries.append({
            "chromosome": chromosome,
            "n_genes": len(frame),
            "adjacent_class_agreement": adjacent,
            "well_resolved_fraction": np.mean(np.asarray(classes) == "well_resolved"),
            "ambiguous_fraction": np.mean(np.asarray(classes) == "ambiguous"),
            "longest_well_resolved_run": _longest_run(classes, "well_resolved"),
            "longest_ambiguous_run": _longest_run(classes, "ambiguous"),
        })
        for iteration in range(permutations):
            shuffled = rng.permutation(classes)
            null_rows.append({
                "chromosome": chromosome,
                "permutation": iteration,
                "adjacent_class_agreement": float(np.mean(shuffled[:-1] == shuffled[1:])) if len(shuffled) > 1 else np.nan,
            })
    return data, pd.DataFrame(summaries), pd.DataFrame(null_rows)


def _track_figure(frame: pd.DataFrame, title: str):
    fig, axes = plt.subplots(5, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    x = np.arange(len(frame))
    columns = (
        ("mean_signed_contrast", "Mean signed contrast"),
        ("separation_magnitude", "Median |contrast|"),
        ("direction_consistency", "Direction consistency"),
        ("detection_rate", "Detection rate"),
        ("resolution_score", "Resolution score"),
    )
    for ax, (column, label) in zip(axes, columns):
        raw = frame[column].to_numpy(dtype=float).copy()
        rolling = frame[f"{column}_rolling"].to_numpy(dtype=float).copy()
        if "chromosome" in frame and frame["chromosome"].nunique() > 1:
            boundary = frame["chromosome"].astype(str).ne(
                frame["chromosome"].astype(str).shift()
            ).to_numpy()
            boundary[0] = False
            raw[boundary] = np.nan
            rolling[boundary] = np.nan
        ax.plot(x, raw, color="#aaa", linewidth=0.6, alpha=0.6)
        ax.plot(x, rolling, color=POS if column != "mean_signed_contrast" else NEG, linewidth=1.5)
        ax.set_ylabel(label, fontsize=8)
        if column == "mean_signed_contrast":
            ax.axhline(0, color="#555", linewidth=0.7)
    axes[-1].set_xlabel("Genes ordered by TSS")
    fig.suptitle(title)
    return fig


def plot_genome_resolution_tracks(data: pd.DataFrame, summary: pd.DataFrame):
    apply_plot_style()
    yield "09_genome_phase_resolution_tracks", _track_figure(
        data, "Genome-wide phase-resolution tracks"
    )
    for chromosome, frame in data.groupby("chromosome", sort=False):
        safe = str(chromosome).replace("/", "_").replace("\\", "_")
        yield (
            f"chromosomes/{safe}_phase_resolution_track",
            _track_figure(frame, f"{chromosome} phase-resolution tracks"),
        )
    pivot = summary.set_index("chromosome")[[
        "adjacent_class_agreement", "well_resolved_fraction", "ambiguous_fraction"
    ]]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_title("Chromosome resolution summary")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    yield "09_chromosome_resolution_summary", fig
