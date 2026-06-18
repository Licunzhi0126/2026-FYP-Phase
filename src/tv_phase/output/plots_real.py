from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .layout import RunOutputLayout
from .plots_common import plot_metric_bars, save_umap, save_umap_overview


CORE = {
    "original_expression_embedding": "original",
    "phase_A_expression_embedding": "phaseA",
    "phase_B_expression_embedding": "phaseB",
}


def _metric_frame(metric_df: pd.DataFrame) -> pd.DataFrame:
    sub = metric_df[(metric_df["source"] == "predicted") & metric_df["embedding"].isin(CORE)].copy()
    return pd.DataFrame(
        {
            "representation": sub["embedding"].map(CORE),
            "ARI": sub["ari"],
            "NMI": sub["nmi"],
            "FMI": sub["fmi"],
        }
    )


def render_real_outputs(
    layout: RunOutputLayout,
    *,
    metric_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    cell_names: List[str],
    labels: List[str],
    seed: int,
) -> None:
    metrics = _metric_frame(metric_df)
    metrics.to_csv(layout.plot_data / "metric_original_phaseA_phaseB.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars(metrics, layout.figures / "metric_original_phaseA_phaseB.png")
    frames = {}
    for embedding_name, display_name in CORE.items():
        frames[display_name] = save_umap(
            embeddings[embedding_name],
            cell_names=cell_names,
            labels=labels,
            title=display_name,
            figure_path=layout.figures / f"umap_{display_name}.png",
            data_path=layout.plot_data / f"umap_{display_name}.csv",
            seed=seed,
        )
    save_umap_overview(frames, layout.figures / "umap_overview.png")

