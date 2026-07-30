"""Shared publication-style formatting for the Figure 2 workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

BLUE = "#2F74B8"
RED = "#D45959"
TEAL = "#4B9B8E"
GOLD = "#D89A3D"
PURPLE = "#7766A8"
SLATE = "#667085"
LIGHT_SLATE = "#AAB2BF"
INK = "#1F2937"
GRID = "#E6EAF0"
PAPER = "#FFFFFF"
PALE = "#F6F8FB"

TRUTH_A = GOLD
TRUTH_B = PURPLE
PRED_A = BLUE
PRED_B = TEAL

METHOD_COLORS = {
    "PhaseHyper": BLUE,
    "NMF2 preview": TEAL,
    "StateSplit": GOLD,
    "RankSplit": PURPLE,
    "RandomSplit": RED,
    "EqualSplit": SLATE,
}


def _mix_with_white(hex_color: str, amount: float = 0.92) -> tuple[float, float, float]:
    red, green, blue = to_rgb(hex_color)
    return (
        amount + (1.0 - amount) * red,
        amount + (1.0 - amount) * green,
        amount + (1.0 - amount) * blue,
    )


def sequential_cmap(hex_color: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        name,
        [_mix_with_white(hex_color, 0.985), _mix_with_white(hex_color, 0.72), hex_color],
    )


CMAP_TRUTH_A = sequential_cmap(TRUTH_A, "truth_a")
CMAP_TRUTH_B = sequential_cmap(TRUTH_B, "truth_b")
CMAP_PRED_A = sequential_cmap(PRED_A, "pred_a")
CMAP_PRED_B = sequential_cmap(PRED_B, "pred_b")
CMAP_TOTAL = LinearSegmentedColormap.from_list(
    "total", ["#FFFFFF", "#D5D9DF", "#353B45"]
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#B8C0CC",
            "axes.linewidth": 0.7,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def method_color(method: str, index: int = 0) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    fallback = [BLUE, TEAL, GOLD, PURPLE, RED, SLATE, LIGHT_SLATE]
    return fallback[index % len(fallback)]


def save_figure(fig: plt.Figure, output_base: Path, *, dpi: int = 300) -> list[Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in ("png", "pdf"):
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        paths.append(path)
    return paths


def add_caption(fig: plt.Figure, text: str, y: float = 0.012) -> None:
    fig.text(0.04, y, text, ha="left", va="bottom", fontsize=7.7, color="#586174")


def sorted_legend(ax: plt.Axes, order: Iterable[str], **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    mapping = {label: handle for handle, label in zip(handles, labels)}
    labels_out = [label for label in order if label in mapping]
    handles_out = [mapping[label] for label in labels_out]
    ax.legend(handles_out, labels_out, **kwargs)
