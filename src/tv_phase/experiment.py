from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


from . import pipeline as TV_PHASE
from .config import resolve_project_paths
from .output import build_prior_ablation_summary
from .prior import list_prior_builders, prior_builder_labels


# Get dataset types from DATASET_CONFIG
DATASET_ORDER = list(TV_PHASE.DATASET_CONFIG.keys())
METHOD_ORDER = ["kmeans", "leiden", "louvain"]
PRIOR_ORDER = list_prior_builders()
METRIC_ORDER = ["ari", "nmi", "fmi"]
EMBEDDING_ORDER = [
    "original_expression_embedding",
    "phase_A_expression_embedding",
    "phase_B_expression_embedding",
    "cell_embedding_input_x",
    "cell_embedding_hgnn_h",
    "cell_embedding_vae_mu",
    "cell_embedding_phase_aware",
    "cell_embedding_vae_z",
    "truth_maternal_expression_embedding",
    "truth_paternal_expression_embedding",
    "truth_total_expression_embedding",
    "truth_phase_difference_embedding",
]

DATASET_LABELS = {
    "PEA_STA": "PEA_STA",
    "SCoPE2": "SCoPE2",
    "sc_GEM": "sc_GEM",
    "sim_gene100_alpha_1_beta_2": "Sim (alpha=1, beta=2)",
    "sim_gene100_alpha_1_beta_1": "Sim (alpha=1, beta=1)",
    "sim_gene100_alpha_2_beta_1": "Sim (alpha=2, beta=1)",
    "sim_gene100_alpha_2_beta_2": "Sim (alpha=2, beta=2)",
    "simulation0611_gene_position": "Sim 0611 (gene position)",
    "simulation0611_position_kegg": "Sim 0611 (position + KEGG)",
    "simulation0615_gene_position": "Sim 0615 (gene position)",
    "simulation0615_position_kegg": "Sim 0615 (position + KEGG)",
    "simulation0616_expr_position": "Sim 0616 expression (position)",
    "simulation0616_expr_position_kegg": "Sim 0616 expression (position + KEGG)",
    "simulation0616_ratio_position": "Sim 0616 ratio (position)",
    "simulation0616_ratio_position_kegg": "Sim 0616 ratio (position + KEGG)",
}

METHOD_LABELS = {
    "kmeans": "KMeans",
    "leiden": "Leiden",
    "louvain": "Louvain",
}

PRIOR_LABELS = {
    **prior_builder_labels(),
}

EMBEDDING_LABELS = {
    "original_expression_embedding": "Original",
    "phase_A_expression_embedding": "Phase-A",
    "phase_B_expression_embedding": "Phase-B",
    "cell_embedding_input_x": "Input-X",
    "cell_embedding_hgnn_h": "HGNN",
    "cell_embedding_vae_mu": "VAE-Mu",
    "cell_embedding_phase_aware": "Phase-Aware",
    "cell_embedding_vae_z": "VAE-Z",
    "truth_maternal_expression_embedding": "Truth-M",
    "truth_paternal_expression_embedding": "Truth-P",
    "truth_total_expression_embedding": "Truth-Total",
    "truth_phase_difference_embedding": "Truth-Diff",
}


def _run_seed(base_seed: int, dataset_type: str, cluster_method: str) -> int:
    dataset_offset = DATASET_ORDER.index(dataset_type) * 100 if dataset_type in DATASET_ORDER else 0
    method_offset = METHOD_ORDER.index(cluster_method) * 10 if cluster_method in METHOD_ORDER else 0
    return int(base_seed) + dataset_offset + method_offset


def _reset_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TV-PHASE v11 End-to-End Training and Comparison")
    parser.add_argument(
        "--dataset_types", "--dataset-types",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Datasets to include in the comparison run",
    )
    parser.add_argument(
        "--cluster_methods", "--cluster-methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=METHOD_ORDER,
        help="Cluster methods to compare",
    )
    parser.add_argument(
        "--prior-builders",
        nargs="+",
        choices=PRIOR_ORDER,
        default=["dataset"],
        help="Prior builders to compare. Use p_glue/p_denoise for SCoPE2 data-driven priors.",
    )
    parser.add_argument(
        "--output-dir", "--output-root",
        dest="output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--project-root", type=Path, default=None, help="Project root; can also use TV_PHASE_PROJECT_ROOT")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root; can also use TV_PHASE_DATA_ROOT")
    parser.add_argument("--legacy-output", action="store_true", help="Also create the legacy deep output tree")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--feature-dim", type=int, default=64, help="Feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="HGNN hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--prior-dim", type=int, default=16, help="Reduced gene prior dimension")
    parser.add_argument("--prior-top-k", type=int, default=5, help="Top-k edges per feature for p_glue")
    parser.add_argument("--prior-max-features", type=int, default=800, help="Max features per modality for data-driven priors")
    parser.add_argument("--denoise-candidate-top-k", type=int, default=5, help="Candidate top-k edges per feature for p_denoise")
    parser.add_argument("--denoise-epochs", type=int, default=20, help="Edge denoising epochs for p_denoise")
    parser.add_argument("--denoise-top-percent", type=float, default=0.7, help="Fraction of denoised prior edges to keep")
    parser.add_argument("--train-epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    return parser


def _run_single_experiment(args, dataset_type: str, prior_name: str, cluster_method: str, out_dir: Path) -> pd.DataFrame:
    seed = _run_seed(args.seed, dataset_type, cluster_method)
    _reset_run_seed(seed)
    print(f"  Run seed         : {seed}")
    print(f"  Prior builder    : {prior_name}")
    print(f"  Cluster method   : {cluster_method}")

    # Build PhaseTrainingConfig for TV_PHASE_v11
    config = TV_PHASE.PhaseTrainingConfig(
        data_name=dataset_type,
        output_dir=out_dir,
        device=args.device,
        seed=seed,
        project_root=args.project_root,
        data_root=args.data_root,
        output_root=args.output_dir,
        legacy_output=args.legacy_output,
        prior_name=prior_name,
        prior_top_k=args.prior_top_k,
        prior_max_features=args.prior_max_features,
        denoise_candidate_top_k=args.denoise_candidate_top_k,
        denoise_epochs=args.denoise_epochs,
        denoise_top_percent=args.denoise_top_percent,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        prior_dim=args.prior_dim,
        train_epochs=args.train_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cluster_method=cluster_method,
    )

    version_name = f"TV-PHASE_v11"
    result = TV_PHASE.run_hgnn_vae_phase_end2end(config, version_name=version_name)
    
    # Extract metrics from the result
    metric_df = result.get("metric_df", pd.DataFrame())
    if not metric_df.empty:
        metric_df.insert(0, "dataset_type", dataset_type)
        metric_df.insert(1, "cluster_version", version_name)
        metric_df.insert(2, "prior_name", prior_name)
        metric_df["run_dir"] = str(Path(out_dir).resolve())
    
    return metric_df


def _ordered_subset(items: List[str], preferred_order: List[str]) -> List[str]:
    item_set = set(items)
    return [item for item in preferred_order if item in item_set]


def _plot_dataset_metric_bars(df: pd.DataFrame, dataset_key: str, out_dir: Path) -> None:
    dataset_col = "dataset_plot_key" if "dataset_plot_key" in df.columns else "dataset_type"
    sub = df[df[dataset_col] == dataset_key].copy()
    if "source" in sub.columns:
        sub["source"] = sub["source"].fillna("predicted")
        sub = sub[sub["source"] == "predicted"].copy()
    if sub.empty:
        return
    methods = _ordered_subset(sub["cluster_method"].tolist(), METHOD_ORDER)
    embeddings = _ordered_subset(sub["embedding"].tolist(), EMBEDDING_ORDER)
    if not methods or not embeddings:
        return

    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(24, 7), squeeze=False)
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(embeddings), dtype=np.float32)

    for col_idx, metric_name in enumerate(METRIC_ORDER):
        ax = axes[0, col_idx]
        for method_idx, method in enumerate(methods):
            method_df = sub[sub["cluster_method"] == method].set_index("embedding").reindex(embeddings)
            y = method_df[metric_name].fillna(0.0).to_numpy(dtype=np.float32)
            offset = (method_idx - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, label=METHOD_LABELS.get(method, method))
        ax.set_xticks(x)
        ax.set_xticklabels([EMBEDDING_LABELS.get(name, name) for name in embeddings], rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{dataset_key} {metric_name.upper()}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    safe_key = str(dataset_key).replace("[", "_").replace("]", "").replace(" ", "_").replace(":", "_")
    fig.savefig(out_dir / f"{safe_key}_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    if "source" in df.columns:
        df["source"] = df["source"].fillna("predicted")
    else:
        df["source"] = "predicted"
    if df.empty:
        return
    dataset_col = "dataset_plot_key" if "dataset_plot_key" in df.columns else "dataset_type"
    datasets = list(dict.fromkeys(df[dataset_col].tolist()))
    methods = _ordered_subset(df["cluster_method"].tolist(), METHOD_ORDER)
    if not datasets or not methods:
        return

    for dataset_key in datasets:
        sub = df[df[dataset_col] == dataset_key].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, len(METRIC_ORDER), figsize=(24, 12), squeeze=False)
        for row_idx, source in enumerate(["predicted", "ground_truth"]):
            source_df = sub[sub["source"] == source].copy()
            embeddings = _ordered_subset(source_df["embedding"].tolist(), EMBEDDING_ORDER)
            if not embeddings:
                for col_idx, metric_name in enumerate(METRIC_ORDER):
                    axes[row_idx, col_idx].axis("off")
                    axes[row_idx, col_idx].set_title(f"{source} {metric_name.upper()} (no rows)")
                continue

            width = 0.8 / max(1, len(methods))
            x = np.arange(len(embeddings), dtype=np.float32)
            source_label = "Prediction" if source == "predicted" else "Ground Truth"
            for col_idx, metric_name in enumerate(METRIC_ORDER):
                ax = axes[row_idx, col_idx]
                for method_idx, method in enumerate(methods):
                    method_df = source_df[source_df["cluster_method"] == method].set_index("embedding").reindex(embeddings)
                    y = method_df[metric_name].fillna(0.0).to_numpy(dtype=np.float32)
                    offset = (method_idx - (len(methods) - 1) / 2.0) * width
                    ax.bar(x + offset, y, width=width, label=METHOD_LABELS.get(method, method))
                ax.set_xticks(x)
                ax.set_xticklabels([EMBEDDING_LABELS.get(name, name) for name in embeddings], rotation=25, ha="right")
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(metric_name.upper())
                ax.set_title(f"{source_label} {metric_name.upper()}")
                ax.grid(axis="y", linestyle="--", alpha=0.3)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
        fig.suptitle(f"{dataset_key} Prediction vs Ground Truth Metrics", fontsize=16, fontweight="bold", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        safe_key = str(dataset_key).replace("[", "_").replace("]", "").replace(" ", "_").replace(":", "_")
        fig.savefig(out_dir / f"{safe_key}_cluster_method_comparison_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_best_ari_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if df.empty:
        best_df = df.copy()
    else:
        group_cols = ["dataset_type", "embedding"]
        if "prior_name" in df.columns:
            group_cols.insert(1, "prior_name")
        if "source" in df.columns:
            insert_at = group_cols.index("embedding")
            group_cols.insert(insert_at, "source")
        idx = df.groupby(group_cols)["ari"].idxmax()
        best_df = (
            df.loc[idx]
            .sort_values([col for col in ["dataset_type", "prior_name", "source", "embedding"] if col in df.columns])
            .reset_index(drop=True)
        )
    best_df.to_csv(out_dir / "best_ari_by_dataset_embedding.csv", index=False, encoding="utf-8-sig")
    return best_df


def _plot_pred_vs_truth_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "source" not in df.columns:
        return
    sub = df[df["source"].isin(["predicted", "ground_truth"])].copy()
    if sub.empty:
        return
    dataset_col = "dataset_plot_key" if "dataset_plot_key" in sub.columns else "dataset_type"
    sub["plot_label"] = (
        sub[dataset_col].astype(str)
        + "\n"
        + sub["source"].astype(str)
        + ":"
        + sub["embedding"].map(lambda name: EMBEDDING_LABELS.get(name, name))
    )
    labels = sub["plot_label"].tolist()
    x = np.arange(len(sub), dtype=np.float32)
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(max(18, len(sub) * 0.7), 7), squeeze=False)
    for col_idx, metric_name in enumerate(METRIC_ORDER):
        ax = axes[0, col_idx]
        colors = ["#4c78a8" if source == "predicted" else "#f58518" for source in sub["source"]]
        ax.bar(x, sub[metric_name].to_numpy(dtype=np.float32), color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(metric_name.upper())
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#4c78a8", label="Predicted"),
        plt.Line2D([], [], marker="s", ls="", color="#f58518", label="Ground truth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "pred_vs_truth_ari_nmi_fmi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_report(df: pd.DataFrame, best_df: pd.DataFrame, out_dir: Path, args) -> None:
    payload = {
        "dataset_types": list(args.dataset_types),
        "cluster_methods": list(args.cluster_methods),
        "prior_builders": list(args.prior_builders),
        "output_dir": str(Path(out_dir).resolve()),
        "device": args.device,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "prior_dim": args.prior_dim,
        "train_epochs": args.train_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "base_seed": args.seed,
        "seed_policy": "reset_before_each_run_with_offset",
        "main_code": "TV_PHASE_v11.py",
        "runner_code": "experiment_v11.py",
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(out_dir / "experiment_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TV-PHASE v11 End-to-End Experiment Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("[Configuration]\n")
        f.write(f"  Datasets       : {', '.join(args.dataset_types)}\n")
        f.write(f"  Cluster methods: {', '.join(args.cluster_methods)}\n")
        f.write(f"  Device         : {args.device}\n")
        f.write(f"  Epochs         : {args.train_epochs}\n")
        f.write(f"  LR             : {args.lr}\n")
        f.write("\n")
        
        dataset_col = "dataset_plot_key" if "dataset_plot_key" in df.columns else "dataset_type"
        for dataset_key in list(dict.fromkeys(df[dataset_col].tolist())):
            f.write(f"[{dataset_key}]\n")
            dataset_best = best_df[best_df[dataset_col] == dataset_key] if dataset_col in best_df.columns else best_df[best_df["dataset_type"] == dataset_key]
            if dataset_best.empty:
                f.write("  No results.\n\n")
                continue
            for _, row in dataset_best.iterrows():
                f.write(
                    f"  {row.get('source', 'predicted')} | {EMBEDDING_LABELS.get(row['embedding'], row['embedding'])}: "
                    f"{METHOD_LABELS.get(row['cluster_method'], row['cluster_method'])} "
                    f"(ARI={row['ari']:.4f}, NMI={row['nmi']:.4f}, FMI={row['fmi']:.4f})\n"
                )
            f.write("\n")


def run_experiment(args) -> pd.DataFrame:
    _, _, default_output_root = resolve_project_paths(args.project_root, args.data_root, None)
    out_dir = Path(args.output_dir) if args.output_dir is not None else default_output_root / "prior_ablation_v1"
    runs_root = out_dir / "runs"
    summary_dir = out_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    dataset_types = _ordered_subset(list(args.dataset_types), DATASET_ORDER)
    cluster_methods = _ordered_subset(list(args.cluster_methods), METHOD_ORDER)
    prior_builders = _ordered_subset(list(args.prior_builders), PRIOR_ORDER)

    print("=" * 70)
    print("TV-PHASE v11 End-to-End Experiment")
    print("=" * 70)
    print(f"Datasets       : {dataset_types}")
    print(f"Prior builders : {prior_builders}")
    print(f"Cluster methods: {cluster_methods}")
    print(f"Output root    : {out_dir.resolve()}")

    for dataset_type in dataset_types:
        for prior_name in prior_builders:
            for cluster_method in cluster_methods:
                run_dir = runs_root / dataset_type / prior_name / cluster_method
                print("\n" + "-" * 70)
                print(
                    f"Running {DATASET_LABELS.get(dataset_type, dataset_type)} | "
                    f"{PRIOR_LABELS.get(prior_name, prior_name)} | "
                    f"{METHOD_LABELS.get(cluster_method, cluster_method)}"
                )
                print("-" * 70)
                metric_df = _run_single_experiment(args, dataset_type, prior_name, cluster_method, run_dir)
                frames.append(metric_df)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if frames:
        summary_df = pd.concat(frames, ignore_index=True)
    else:
        summary_df = pd.DataFrame(
            columns=[
                "dataset_type",
                "cluster_version",
                "prior_name",
                "source",
                "embedding",
                "cluster_method",
                "pred_clusters",
                "fmi",
                "nmi",
                "ari",
                "n_cells",
                "n_features",
                "run_dir",
            ]
        )
    if "source" not in summary_df.columns:
        summary_df["source"] = "predicted"
    if "prior_name" in summary_df.columns:
        summary_df["dataset_plot_key"] = summary_df.apply(
            lambda row: row["dataset_type"]
            if str(row.get("prior_name", "dataset")) == "dataset"
            else f"{row['dataset_type']} [{row['prior_name']}]",
            axis=1,
        )

    summary_df.to_csv(summary_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    build_prior_ablation_summary(summary_df, out_dir)
    best_df = _save_best_ari_summary(summary_df, summary_dir)
    _plot_overview(summary_df, summary_dir)
    _plot_pred_vs_truth_summary(summary_df, summary_dir)
    dataset_col = "dataset_plot_key" if "dataset_plot_key" in summary_df.columns else "dataset_type"
    for dataset_key in list(dict.fromkeys(summary_df[dataset_col].tolist())):
        _plot_dataset_metric_bars(summary_df, dataset_key, summary_dir)
    _save_report(summary_df, best_df, summary_dir, args)

    print("\n[Done] Experiment outputs saved to:")
    print(f"  {out_dir.resolve()}")
    return summary_df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
