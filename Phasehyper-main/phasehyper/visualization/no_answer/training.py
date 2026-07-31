"""Training-history diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..plot_style import apply_plot_style


def prepare_training_diagnostics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "epoch" not in history or "loss" not in history:
        raise ValueError("training_history.csv lacks epoch/loss")
    data = history.copy()
    epochs = len(data)
    window = max(3, min(21, epochs // 20)) if epochs >= 6 else 1
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column == "epoch":
            continue
        first = abs(float(data[column].iloc[0])) + 1e-12
        data[f"{column}_relative"] = data[column] / first
        data[f"{column}_smoothed"] = data[column].rolling(window, center=True, min_periods=1).mean()
        data[f"{column}_relative_smoothed"] = data[f"{column}_relative"].rolling(
            window, center=True, min_periods=1
        ).mean()
    return data


def plot_training_diagnostics(data: pd.DataFrame, *, best_epoch: int | None = None):
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    epoch = data["epoch"]
    panels = (
        ("Total objective", ("loss",), False),
        ("Representation objectives", ("cyc_comp", "barlow", "info_nce"), True),
        ("Phase separation objectives", ("compartment", "orthogonality", "phase_cosine"), True),
        ("Gate and asymmetry dynamics", ("gate_regularization", "asym_scale"), True),
    )
    for ax, (title, columns, relative) in zip(axes.flat, panels):
        for column in columns:
            if column not in data:
                continue
            shown = f"{column}_relative" if relative else column
            ax.plot(epoch, data[shown], alpha=0.28, linewidth=0.8)
            ax.plot(epoch, data[f"{shown}_smoothed"] if f"{shown}_smoothed" in data else data[shown], label=column)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    if best_epoch is None:
        best_epoch = int(data.loc[data["loss"].idxmin(), "epoch"])
    axes[0, 0].axvline(best_epoch, linestyle="--", color="#555", linewidth=0.9)
    axes[0, 0].text(best_epoch, axes[0, 0].get_ylim()[1], " Minimum training-loss epoch", va="top", fontsize=8)
    fig.suptitle("Training diagnostics")
    return fig
