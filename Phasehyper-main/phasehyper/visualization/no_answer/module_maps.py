"""Pathway and PPI module-level phase maps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..plot_style import apply_plot_style


def compute_module_phase_metrics(bundle, membership: pd.DataFrame, family: str, *, min_genes: int = 3) -> pd.DataFrame:
    columns = [
        "module_family", "module", "label_name", "gene_count", "phase_dominance",
        "gene_agreement", "coherence_A", "coherence_B", "coherence_delta", "genes",
    ]
    if membership.empty or not {"module", "gene"}.issubset(membership):
        return pd.DataFrame(columns=columns)
    pos = {gene: i for i, gene in enumerate(bundle.genes)}
    label_array = np.asarray(bundle.label_names)
    rows = []
    for module, frame in membership.groupby("module"):
        genes = sorted({str(g) for g in frame["gene"] if str(g) in pos})
        if len(genes) < min_genes:
            continue
        indices = [pos[g] for g in genes]
        for label in dict.fromkeys(bundle.label_names):
            mask = label_array == label
            a = bundle.phase_a[np.ix_(mask, indices)]
            b = bundle.phase_b[np.ix_(mask, indices)]
            normalized = (b.mean(axis=0) - a.mean(axis=0)) / (
                np.abs(a).mean(axis=0) + np.abs(b).mean(axis=0) + 1e-12
            )
            if a.shape[0] >= 2:
                valid = (np.std(a, axis=0) > 1e-12) & (np.std(b, axis=0) > 1e-12)
                if valid.sum() >= 3:
                    corr_a = np.corrcoef(a[:, valid], rowvar=False)
                    corr_b = np.corrcoef(b[:, valid], rowvar=False)
                    tri = np.triu_indices(int(valid.sum()), 1)
                    coherence_a = float(corr_a[tri].mean())
                    coherence_b = float(corr_b[tri].mean())
                else:
                    coherence_a = coherence_b = np.nan
            else:
                coherence_a = coherence_b = np.nan
            rows.append({
                "module_family": family,
                "module": module,
                "label_name": label,
                "gene_count": len(genes),
                "phase_dominance": float(normalized.mean()),
                "gene_agreement": float(abs(np.sign(normalized).sum()) / len(normalized)),
                "coherence_A": coherence_a,
                "coherence_B": coherence_b,
                "coherence_delta": coherence_a - coherence_b,
                "genes": ";".join(genes),
            })
    return pd.DataFrame(rows, columns=columns)


def plot_module_phase_maps(data: pd.DataFrame, family: str):
    apply_plot_style()
    if data.empty:
        raise ValueError(f"no {family} modules meet the minimum gene count")
    modules = (
        data.groupby("module")["phase_dominance"].apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False).head(40).index
    )
    labels = list(dict.fromkeys(data["label_name"]))
    subset = data[data["module"].isin(modules)]
    matrices = [
        subset.pivot(index="module", columns="label_name", values=column).reindex(index=modules, columns=labels)
        for column in ("phase_dominance", "gene_agreement", "coherence_delta")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(7, 0.24 * len(modules))), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, ("Phase dominance", "Gene agreement", "Coherence A − B")):
        if title == "Gene agreement":
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        else:
            vmax = max(float(np.nanquantile(np.abs(matrix), 0.98)), 0.05)
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(modules)), modules if ax is axes[0] else [], fontsize=6)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.035)
    fig.suptitle(f"{family.title()} module phase map (model-input annotation consistency)")
    return fig
