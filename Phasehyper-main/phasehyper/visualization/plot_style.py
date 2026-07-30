"""Shared plotting style for simulation visualizations."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
POS = "#2a78d6"
NEG = "#e34948"
MID = "#f0efec"
SERIES = {
    "phasehyper": POS,
    "HyperPhase": POS,
    "RandomSplit": "#008300",
    "NMF2Factor": "#e87ba4",
    "[trivial] combined/2": NEG,
    "combined/2": NEG,
}


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 0,
            "ytick.major.size": 3,
        }
    )


def style_axis(ax, *, grid_axis: str | None = "y", ylabel: str | None = None) -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)


def save_figure(fig, path: Path, dpi: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


apply_plot_style()
