"""File output helpers shared by phase visualization modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_SUBDIRS = (
    "overview",
    "phase",
    "correlation",
    "genome",
    "modules",
    "structure",
    "genes",
    "associations",
    "training",
    "source_data",
)


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {name: root / name for name in OUTPUT_SUBDIRS}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_source_data(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def save_figure_formats(
    fig,
    base_path: Path,
    *,
    dpi: int,
    formats: Iterable[str] = ("png",),
) -> list[Path]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    try:
        for suffix in formats:
            path = base_path.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            saved.append(path)
    finally:
        plt.close(fig)
    return saved
