from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPRESENTATION_NAMES = {
    "original_expression_embedding": "original",
    "phase_A_expression_embedding": "phaseA",
    "phase_B_expression_embedding": "phaseB",
    "truth_maternal_expression_embedding": "truth_maternal",
    "truth_paternal_expression_embedding": "truth_paternal",
    "truth_total_expression_embedding": "truth_total",
}


def build_prior_ablation_summary(metric_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    summary_dir = Path(output_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = metric_df[metric_df["embedding"].isin(REPRESENTATION_NAMES)].copy()
    frame["representation"] = frame["embedding"].map(REPRESENTATION_NAMES)
    frame["dataset"] = frame["dataset_type"]
    frame["prior_builder"] = frame["prior_name"]
    frame["is_simulation"] = frame["dataset_type"].astype(str).str.startswith(("sim_", "simulation"))
    frame["has_truth"] = frame.groupby(["dataset_type", "prior_name"])["source"].transform(
        lambda values: bool((values == "ground_truth").any())
    )
    result = frame[
        ["dataset", "prior_builder", "cluster_method", "source", "representation", "ari", "nmi", "fmi", "is_simulation", "has_truth", "run_dir"]
    ].rename(columns={"ari": "ARI", "nmi": "NMI", "fmi": "FMI"})
    result.to_csv(summary_dir / "prior_ablation_summary.csv", index=False, encoding="utf-8-sig")

    predicted = result[result["source"] == "predicted"].copy()
    if not predicted.empty:
        grouped = predicted.groupby(["dataset", "prior_builder"], as_index=False)[["ARI", "NMI", "FMI"]].mean()
        labels = grouped["dataset"] + "\n" + grouped["prior_builder"]
        x = np.arange(len(grouped))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(12, len(grouped) * 0.8), 6))
        for idx, metric in enumerate(["ARI", "NMI", "FMI"]):
            ax.bar(x + (idx - 1) * width, grouped[metric], width=width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(summary_dir / "prior_ablation_metric_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return result

