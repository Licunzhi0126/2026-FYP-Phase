from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phasehyper.evaluation.clustering import (
    _align_labels_to_cells,
    _evaluate_embedding_metrics,
)
from phasehyper.schemas import DatasetBundle, PhaseTrainingConfig


SABER_CSV_FIELDS = [
    "method", "pcc_global", "pcc_cell", "pcc_A", "pcc_B", "auroc_A", "auroc_B",
    "pcc_cell_A", "pcc_cell_B", "cos_A", "cos_B", "mse_A", "mse_B",
    "imb", "imb_gene", "auroc", "mse", "pearson", "imb_spearman", "imb_pcc",
    "imb_gene_pcc", "imb_auroc", "imb_auprc", "major_r", "minor_r",
    "mean_imb_pred", "mean_imb_true", "orient_level",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_metrics_json(path: Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics), handle, indent=2, ensure_ascii=False)


def save_metric_rows_csv(
    path: Path,
    rows: list[dict],
    *,
    index: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=index)
    return path


def _save_saber_rows(
    path: Path,
    headline_rows: list[dict],
    saber_rows: list[dict],
    metadata: dict | None = None,
) -> Path:
    head = {row["name"]: row for row in headline_rows}
    protocol = {row["name"]: row for row in saber_rows}
    names = list(head) + [name for name in protocol if name not in head]
    extra = metadata or {}
    rows = []
    for name in names:
        row: dict[str, Any] = {"method": name}
        for source in (head.get(name, {}), protocol.get(name, {})):
            for key, value in source.items():
                if key in SABER_CSV_FIELDS and key != "method":
                    row[key] = f"{value:.6f}" if isinstance(value, float) else value
        row.update(extra)
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SABER_CSV_FIELDS + list(extra)).to_csv(path, index=False)
    return path


def save_saber_evaluation(
    *,
    output_dir: Path,
    headline_rows: list[dict],
    saber_rows: list[dict],
    orientation_rows: list[dict],
    metadata: dict | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metrics": _save_saber_rows(
            output_dir / "metrics.csv", headline_rows, saber_rows, metadata
        ),
        "orientation": save_metric_rows_csv(
            output_dir / "orientation_audit.csv", orientation_rows
        ),
    }


def save_grn_evaluation(
    *,
    output_dir: Path,
    headline_rows: list[dict],
    saber_rows: list[dict],
    differential_rows: list[dict],
    metadata: dict | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "saber": _save_saber_rows(
            output_dir / "saber_protocol.csv",
            headline_rows,
            saber_rows,
            metadata,
        ),
        "differential": save_metric_rows_csv(
            output_dir / "differential.csv", differential_rows
        ),
    }


def _metric_text(value: Any) -> str:
    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return "n/a"
    return f"{float(value):.4f}"


def print_orientation_audit(rows, title="Orientation audit (Saber Fig 3E protocol)"):
    print(f"\n  {title}")
    print(f"  {'level':<12}{'GT bits':>9}{'#swapped':>10}{'phaseMSE':>11}{'vs raw':>10}")
    print("  " + "-" * 50)
    for row in rows:
        print(
            f"  {row['level']:<12}{row['bits']:>9}{row['n_swapped']:>10}"
            f"{row['mse']:>11.4f}{row['gain_pct']:>9.1f}%"
        )


def print_headline(rows):
    print("\n  HEADLINE")
    print(
        f"  {'method':<24}{'PCC':>9}{'PCC/cell':>10}{'imbPCC':>9}"
        f"{'imbGene':>9}{'AUROC':>8}"
    )
    print("  " + "-" * 69)
    for row in rows:
        values = [
            _metric_text(row.get(key))
            for key in ("pcc_global", "pcc_cell", "imb", "imb_gene", "auroc")
        ]
        print(
            f"  {row['name']:<24}{values[0]:>9}{values[1]:>10}"
            f"{values[2]:>9}{values[3]:>9}{values[4]:>8}"
        )


def print_saber_table(rows, title="Saber-protocol evaluation"):
    columns = [
        ("mse", "phaseMSE"), ("pearson", "phase_r"),
        ("imb_spearman", "imbSpear"), ("imb_gene_pcc", "imbGeneR"),
        ("imb_auroc", "AUROC"), ("imb_auprc", "AUPRC"),
        ("major_r", "major_r"), ("minor_r", "minor_r"),
    ]
    print(f"\n  {title}")
    print(f"  {'method':<22}" + "".join(f"{label:>10}" for _, label in columns))
    for row in rows:
        print(
            f"  {row['name']:<22}"
            + "".join(f"{_metric_text(row.get(key)):>10}" for key, _ in columns)
        )


def print_differential(rows, title="GRN differential-component metrics"):
    print(f"\n  {title}")
    print(
        f"  {'method':<22}{'PCC(D)':>9}{'percell':>9}{'Spear':>8}"
        f"{'|D|ratio':>10}{'skill':>9}{'alpha':>8}{'skill@a':>9}{'AUROC':>8}"
    )
    for row in rows:
        print(
            f"  {row['name']:<22}{row['pcc']:>9.3f}{row['pcc_cell']:>9.3f}"
            f"{row['spearman']:>8.3f}{row['d_ratio']:>10.3f}"
            f"{row['skill']:>9.3f}{row['alpha']:>8.3f}"
            f"{row['skill_cal']:>9.3f}{row['auroc']:>8.3f}"
        )


def print_final(headline_rows, grn_headline_rows=None):
    rows = [
        row for row in headline_rows if not row["name"].startswith("[floor]")
    ]
    print("\n" + "=" * 72)
    print("  FINAL — expression decomposition (main task)")
    print(
        f"  {'method':<24}{'PCC':>9}{'PCC/cell':>10}{'PCC_A':>9}"
        f"{'PCC_B':>9}{'imb':>9}{'imb/gene':>10}"
    )
    for row in rows:
        values = [
            _metric_text(row.get(key))
            for key in ("pcc_global", "pcc_cell", "pcc_A", "pcc_B", "imb", "imb_gene")
        ]
        print(
            f"  {row['name']:<24}{values[0]:>9}{values[1]:>10}"
            f"{values[2]:>9}{values[3]:>9}{values[4]:>9}{values[5]:>10}"
        )
    if grn_headline_rows:
        print("\n  FINAL — GRN decomposition (vs reference GRNs)")
        for row in grn_headline_rows:
            if not row["name"].startswith("[floor]"):
                print(
                    f"  {row['name']:<24}{_metric_text(row.get('auroc')):>9}"
                    f"{_metric_text(row.get('auroc_A')):>10}"
                    f"{_metric_text(row.get('auroc_B')):>10}"
                )
    print("=" * 72)


def _save_embedding_metrics(
    outputs: Dict[str, np.ndarray],
    *,
    dataset: DatasetBundle,
    cluster_method: str,
    cluster_resolution: Optional[float],
    out_dir: Path,
    report_title: str,
    sample_names: Optional[List[str]] = None,
    gate_g: Optional[np.ndarray] = None,
    edge_type_weights: Optional[Dict[str, float]] = None,
    phase_recon_mse: Optional[float] = None,
    phase_cosine_sim: Optional[float] = None,
    phase_l2_dist: Optional[float] = None,
) -> pd.DataFrame:
    from phasehyper.visualization.umap_plots import _save_embedding_umap

    true_ids, aligned_label_names = _align_labels_to_cells(dataset, sample_names)
    assessment_mode = (
        "aligned_by_exported_sample_names" if sample_names is not None else "dataset_order"
    )
    rows = []
    for name, emb in outputs.items():
        metrics = _evaluate_embedding_metrics(
            emb,
            true_ids,
            dataset_type=dataset.dataset_type,
            cluster_method=cluster_method,
            cluster_resolution=cluster_resolution,
        )
        rows.append(
            {
                "embedding": name,
                "cluster_method": cluster_method,
                "pred_clusters": metrics["pred_clusters"],
                "fmi": metrics["fmi"],
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
            }
        )
        _save_embedding_umap(out_dir / f"{name}_umap.png", emb, aligned_label_names, f"{name} UMAP")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "embedding_metrics_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "metric_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(report_title + "\n")
        f.write("=" * 60 + "\n")
        f.write(f"Assessment mode: {assessment_mode}\n")
        f.write(f"Aligned samples : {len(true_ids)}\n\n")

        f.write("Assessment groups:\n\n")
        f.write("1. Expression-space embeddings:\n")
        f.write("   - original_expression_embedding: total/original cell x gene expression matrix\n")
        f.write("   - phase_A_expression_embedding: phase A-specific cell x gene expression matrix\n")
        f.write("   - phase_B_expression_embedding: phase B-specific cell x gene expression matrix\n\n")
        f.write("2. HGNN-VAE intermediate embeddings (inference pipeline):\n")
        f.write("   - cell_embedding_input_x: Original input node features (learnable embedding initialized)\n")
        f.write("   - cell_embedding_hgnn_h: HGNN output after hypergraph propagation (before VAE)\n")
        f.write("   - cell_embedding_vae_mu: VAE encoder mean output (deterministic latent)\n")
        f.write("   - cell_embedding_vae_z: VAE sampled latent (with reparameterization)\n\n")

        f.write("Important note:\n")
        f.write("Phase A and Phase B are learned unsupervised labels.\n")
        f.write("They should not be directly interpreted as maternal/paternal without biological validation.\n\n")

        if gate_g is not None:
            f.write("-" * 60 + "\n")
            f.write("Gate Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  gate_mean: {float(gate_g.mean()):.4f}\n")
            f.write(f"  gate_std: {float(gate_g.std()):.4f}\n")
            f.write(f"  gate_min: {float(gate_g.min()):.4f}\n")
            f.write(f"  gate_max: {float(gate_g.max()):.4f}\n")
            f.write(f"  percent_gate_lt_0.3: {float((gate_g < 0.3).mean() * 100):.2f}%\n")
            f.write(f"  percent_gate_gt_0.7: {float((gate_g > 0.7).mean() * 100):.2f}%\n")
            f.write(
                f"  percent_gate_between_0.4_0.6: {float(((gate_g >= 0.4) & (gate_g <= 0.6)).mean() * 100):.2f}%\n"
            )
            f.write(f"  phase_A_genes (gate < 0.5): {int((gate_g < 0.5).sum())}\n")
            f.write(f"  phase_B_genes (gate >= 0.5): {int((gate_g >= 0.5).sum())}\n\n")

        if phase_recon_mse is not None:
            f.write("-" * 60 + "\n")
            f.write("Phase Reconstruction Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  phase_recon_mse: {phase_recon_mse:.6f}\n")
            if phase_cosine_sim is not None:
                f.write(f"  phase_A_phase_B_cosine_similarity: {phase_cosine_sim:.4f}\n")
            if phase_l2_dist is not None:
                f.write(f"  phase_A_phase_B_l2_distance: {phase_l2_dist:.4f}\n")
            f.write("\n")

        if edge_type_weights is not None:
            f.write("-" * 60 + "\n")
            f.write("Learned Edge Type Weights:\n")
            f.write("-" * 60 + "\n")
            for et, weight in edge_type_weights.items():
                f.write(f"  {et}: {weight:.4f}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("Clustering Metrics:\n")
        f.write("-" * 60 + "\n")
        for row in rows:
            f.write(f"[{row['embedding']}]\n")
            f.write(f"  Cluster Method: {row['cluster_method']}\n")
            f.write(f"  Pred Clusters : {row['pred_clusters']}\n")
            f.write(f"  FMI: {row['fmi']:.4f}\n")
            f.write(f"  NMI: {row['nmi']:.4f}\n")
            f.write(f"  ARI: {row['ari']:.4f}\n\n")
    return df


def _save_run_metadata_phase(
    out_dir: Path,
    *,
    version_name: str,
    dataset: DatasetBundle,
    config: PhaseTrainingConfig,
    edge_type_weights: Dict[str, float],
) -> None:
    payload = {
        "version": version_name,
        "dataset_type": dataset.dataset_type,
        "view1_name": dataset.view1_name,
        "n_cells": len(dataset.common_cells),
        "n_genes": len(dataset.common_genes),
        "label_names": sorted(set(dataset.label_names)),
        "training_config": {
            "feature_dim": config.feature_dim,
            "hidden_dim": config.hidden_dim,
            "latent_dim": config.latent_dim,
            "prior_dim": config.prior_dim,
            "train_epochs": config.train_epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "vae_recon_weight": config.vae_recon_weight,
            "vae_kl_weight": config.vae_kl_weight,
            "vae_kl_warmup_epochs": config.vae_kl_warmup_epochs,
            "cell_gene_recon_weight": config.cell_gene_recon_weight,
            "phase_sep_weight": config.phase_sep_weight,
            "gate_balance_weight": config.gate_balance_weight,
            "gate_entropy_weight": config.gate_entropy_weight,
            "gene_gate_smoothness_weight": config.gene_gate_smoothness_weight,
        },
        "learned_edge_type_weights": edge_type_weights,
        "training_mode": "end_to_end_unsupervised",
        "note": (
            "Phase A and Phase B are learned unsupervised labels and should not be directly "
            "interpreted as maternal/paternal without biological validation."
        ),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
