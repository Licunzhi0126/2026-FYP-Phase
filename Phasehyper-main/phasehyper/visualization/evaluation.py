from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _run_evaluation_visualization(
    out_dir: Path,
    dataset,
    dataset_config: dict,
    phase_a_expression: np.ndarray,
    phase_b_expression: np.ndarray,
    gene_names: List[str],
    sample_names: List[str],
    gate_g: np.ndarray,
):
    """Run evaluation visualization for datasets with ground truth (have_answer=True)."""
    import seaborn as sns

    eval_dir = out_dir / "evaluation_visualization"
    eval_dir.mkdir(parents=True, exist_ok=True)

    root_dir = dataset_config["root"]
    gene_pos_file = root_dir / dataset_config["files"]["poswin_prior"]
    kegg_file = root_dir / dataset_config["files"]["kegg_prior"]
    e_m_file = root_dir / "E_M.csv"
    e_p_file = root_dir / "E_P.csv"

    e_m_truth = None
    e_p_truth = None
    if e_m_file.exists() and e_p_file.exists():
        e_m_truth = pd.read_csv(e_m_file, index_col=0)
        e_p_truth = pd.read_csv(e_p_file, index_col=0)

    print(f"  Creating evaluation visualizations in: {eval_dir}")

    # 1. Gene Correlation by Chromosome Position (4-panel comparison)
    print("  [1/3] Gene correlation by chromosome position...")
    if gene_pos_file.exists():
        gene_pos = pd.read_csv(
            gene_pos_file,
            sep="\t",
            header=None,
            names=["Gene", "Chr", "Start", "End", "Strand"],
        )
        gene_order = gene_pos["Gene"].tolist()

        expr_dfs = {
            "Phase_A": pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names),
            "Phase_B": pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names),
        }

        if e_m_truth is not None and e_p_truth is not None:
            expr_dfs["E_M (Truth)"] = e_m_truth
            expr_dfs["E_P (Truth)"] = e_p_truth

        corr_matrices = {}
        for name, expr_df in expr_dfs.items():
            common_genes = [g for g in gene_order if g in expr_df.columns]
            expr_filtered = expr_df[common_genes]
            corr_matrix = expr_filtered.corr(method="pearson")
            corr_matrices[name] = corr_matrix.loc[common_genes, common_genes]

        n_plots = len(corr_matrices)
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)"]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            titles = [
                "Phase_A (Predicted)",
                "Phase_B (Predicted)",
                "E_M (Ground Truth)",
                "E_P (Ground Truth)",
            ]

        for i, (name, corr_sorted) in enumerate(corr_matrices.items()):
            ax = axes[i]
            sns.heatmap(
                corr_sorted,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0,
                xticklabels=False,
                yticklabels=False,
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            ax.set_title(titles[i], fontsize=14, fontweight="bold")

        if n_plots == 2:
            axes[1].axis("off")

        plt.suptitle(
            "Gene Correlation Heatmaps\nOrdered by Chromosome Position",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(eval_dir / "correlation_chromosome.png", dpi=300, bbox_inches="tight")
        plt.close()

        for name, corr_sorted in corr_matrices.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            corr_sorted.to_csv(eval_dir / f"gene_correlation_{safe_name}_chromosome.csv")

        print("    Saved: correlation_chromosome.png")
    else:
        print(f"    Skipped: gene position file not found ({gene_pos_file})")

    # 2. Gene Correlation by KEGG Pathway (4-panel comparison)
    print("  [2/3] Gene correlation by KEGG pathway...")
    if kegg_file.exists():
        anno = pd.read_csv(
            kegg_file, sep="\t", header=None, names=["Gene", "KEGG", "Pathway"]
        )
        anno_sorted = anno.sort_values(by=["KEGG", "Gene"])
        gene_order = anno_sorted["Gene"].tolist()

        expr_dfs = {
            "Phase_A": pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names),
            "Phase_B": pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names),
        }

        if e_m_truth is not None and e_p_truth is not None:
            expr_dfs["E_M (Truth)"] = e_m_truth
            expr_dfs["E_P (Truth)"] = e_p_truth

        corr_matrices = {}
        for name, expr_df in expr_dfs.items():
            common_genes = [g for g in gene_order if g in expr_df.columns]
            expr_filtered = expr_df[common_genes]
            corr_matrix = expr_filtered.corr(method="pearson")
            corr_matrices[name] = corr_matrix.loc[common_genes, common_genes]

        n_plots = len(corr_matrices)
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            axes = axes.flatten()
            titles = ["Phase_A (Predicted)", "Phase_B (Predicted)"]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            titles = [
                "Phase_A (Predicted)",
                "Phase_B (Predicted)",
                "E_M (Ground Truth)",
                "E_P (Ground Truth)",
            ]

        for i, (name, corr_sorted) in enumerate(corr_matrices.items()):
            ax = axes[i]
            sns.heatmap(
                corr_sorted,
                cmap="coolwarm",
                center=0,
                square=True,
                vmin=-1,
                vmax=1,
                xticklabels=False,
                yticklabels=False,
                linewidths=0,
                ax=ax,
            )
            ax.set_title(titles[i], fontsize=14, fontweight="bold")

        if n_plots == 2:
            axes[1].axis("off")

        plt.suptitle(
            "Gene Correlation Heatmaps\nOrdered by KEGG Pathway",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(eval_dir / "correlation_kegg.png", dpi=300, bbox_inches="tight")
        plt.close()

        for name, corr_sorted in corr_matrices.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            corr_sorted.to_csv(eval_dir / f"gene_correlation_{safe_name}_kegg.csv")

        print("    Saved: correlation_kegg.png (4-panel comparison)")
    else:
        print(f"    Skipped: KEGG file not found ({kegg_file})")

    # 3. Phase Separation Evaluation
    print("  [3/3] Phase separation evaluation...")
    if e_m_file.exists() and e_p_file.exists():
        e_m_truth = pd.read_csv(e_m_file, index_col=0)
        e_p_truth = pd.read_csv(e_p_file, index_col=0)

        phase_a_pred = pd.DataFrame(phase_a_expression, index=sample_names, columns=gene_names)
        phase_b_pred = pd.DataFrame(phase_b_expression, index=sample_names, columns=gene_names)

        common_cells = list(set(phase_a_pred.index) & set(e_m_truth.index))
        common_genes = list(set(phase_a_pred.columns) & set(e_m_truth.columns))
        common_cells = sorted(common_cells)
        common_genes = sorted(common_genes)

        phase_a_pred = phase_a_pred.loc[common_cells, common_genes]
        phase_b_pred = phase_b_pred.loc[common_cells, common_genes]
        e_m_truth = e_m_truth.loc[common_cells, common_genes]
        e_p_truth = e_p_truth.loc[common_cells, common_genes]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        data_list = [
            (e_m_truth, "E_M (Ground Truth - Maternal)", axes[0, 0]),
            (e_p_truth, "E_P (Ground Truth - Paternal)", axes[0, 1]),
            (phase_a_pred, "Phase A (Predicted)", axes[1, 0]),
            (phase_b_pred, "Phase B (Predicted)", axes[1, 1]),
        ]
        for df, title, ax in data_list:
            sns.heatmap(df, ax=ax, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Genes")
            ax.set_ylabel("Cells")
        plt.tight_layout()
        plt.savefig(eval_dir / "individual_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        diff_m = (e_m_truth - phase_a_pred).abs()
        diff_p = (e_p_truth - phase_b_pred).abs()
        diff_list = [
            (diff_m, "|E_M - Phase A|", axes[0]),
            (diff_p, "|E_P - Phase B|", axes[1]),
        ]
        for df, title, ax in diff_list:
            sns.heatmap(df, ax=ax, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Genes")
            ax.set_ylabel("Cells")
        plt.tight_layout()
        plt.savefig(eval_dir / "difference_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()

        if gene_pos_file.exists():
            pos_df = pd.read_csv(
                gene_pos_file,
                sep="\t",
                header=None,
                names=["gene", "chromosome", "start", "end", "strand"],
            )
            pos_df["mid_mb"] = (pos_df["start"] + pos_df["end"]) / 2_000_000

            phase_df_pred = pd.DataFrame(
                {
                    "gene": gene_names,
                    "gate_g": gate_g,
                    "phase": ["Phase_B" if g > 0.5 else "Phase_A" for g in gate_g],
                }
            )

            e_m_mean = e_m_truth.mean(axis=0)
            e_p_mean = e_p_truth.mean(axis=0)
            total = e_m_mean + e_p_mean
            gate_g_true = e_p_mean / total
            gate_g_true = gate_g_true.fillna(0.5)

            phase_df_true = pd.DataFrame(
                {
                    "gene": gate_g_true.index.tolist(),
                    "gate_g": gate_g_true.values,
                    "phase": [
                        "Phase_B" if g > 0.5 else "Phase_A" for g in gate_g_true.values
                    ],
                }
            )

            merged_pred = phase_df_pred.merge(pos_df, on="gene")
            counts_pred = (
                merged_pred.groupby(["chromosome", "phase"]).size().unstack(fill_value=0)
            )
            for phase in ["Phase_A", "Phase_B"]:
                if phase not in counts_pred.columns:
                    counts_pred[phase] = 0
            counts_pred["total"] = counts_pred.sum(axis=1)
            counts_pred = counts_pred.sort_values("total", ascending=False)

            fig, ax = plt.subplots(figsize=(14, 8))
            counts_pred[["Phase_A", "Phase_B"]].plot(
                kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"]
            )
            ax.set_title(
                "Predicted Chromosome Phase Distribution", fontsize=16, fontweight="bold"
            )
            ax.set_xlabel("Chromosome", fontsize=12)
            ax.set_ylabel("Number of Genes", fontsize=12)
            ax.legend(title="Phase", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                eval_dir / "chromosome_phase_distribution_predicted.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            merged_true = phase_df_true.merge(pos_df, on="gene")
            counts_true = (
                merged_true.groupby(["chromosome", "phase"]).size().unstack(fill_value=0)
            )
            for phase in ["Phase_A", "Phase_B"]:
                if phase not in counts_true.columns:
                    counts_true[phase] = 0
            counts_true["total"] = counts_true.sum(axis=1)
            counts_true = counts_true.sort_values("total", ascending=False)

            fig, ax = plt.subplots(figsize=(14, 8))
            counts_true[["Phase_A", "Phase_B"]].plot(
                kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"]
            )
            ax.set_title(
                "Ground Truth Chromosome Phase Distribution", fontsize=16, fontweight="bold"
            )
            ax.set_xlabel("Chromosome", fontsize=12)
            ax.set_ylabel("Number of Genes", fontsize=12)
            ax.legend(title="Phase", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                eval_dir / "chromosome_phase_distribution_truth.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            from scipy.ndimage import gaussian_filter1d

            fig, ax = plt.subplots(figsize=(16, 6))
            chrom_order = [str(i) for i in range(1, 23)]
            chrom_colors = plt.cm.tab20(np.linspace(0, 1, 22))
            chrom_color_map = {chrom: chrom_colors[i] for i, chrom in enumerate(chrom_order)}

            for chrom in chrom_order:
                subset = merged_pred[merged_pred["chromosome"] == int(chrom)]
                if len(subset) > 0:
                    ax.scatter(
                        subset["mid_mb"],
                        subset["gate_g"],
                        color=chrom_color_map[chrom],
                        label=f"chr{chrom}",
                        s=50,
                        alpha=0.7,
                    )

            sorted_df = merged_pred.sort_values("mid_mb")
            smoothed_gate = gaussian_filter1d(sorted_df["gate_g"], sigma=2)
            ax.plot(
                sorted_df["mid_mb"],
                smoothed_gate,
                color="red",
                linewidth=2,
                label="Smoothed Gate Trend",
                alpha=0.8,
            )
            ax.set_title("Predicted Gene Position vs Gate Value", fontsize=16, fontweight="bold")
            ax.set_xlabel("Genomic Position (Mb)", fontsize=12)
            ax.set_ylabel("Gate Value (0 = Phase A, 1 = Phase B)", fontsize=12)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=10)
            plt.tight_layout()
            plt.savefig(
                eval_dir / "chromosome_position_scatter_predicted.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            fig, ax = plt.subplots(figsize=(16, 6))

            for chrom in chrom_order:
                subset = merged_true[merged_true["chromosome"] == int(chrom)]
                if len(subset) > 0:
                    ax.scatter(
                        subset["mid_mb"],
                        subset["gate_g"],
                        color=chrom_color_map[chrom],
                        label=f"chr{chrom}",
                        s=50,
                        alpha=0.7,
                    )

            sorted_df = merged_true.sort_values("mid_mb")
            smoothed_gate = gaussian_filter1d(sorted_df["gate_g"], sigma=2)
            ax.plot(
                sorted_df["mid_mb"],
                smoothed_gate,
                color="red",
                linewidth=2,
                label="Smoothed Gate Trend",
                alpha=0.8,
            )

            ax.set_title(
                "Ground Truth Gene Position vs Gate Value", fontsize=16, fontweight="bold"
            )
            ax.set_xlabel("Genomic Position (Mb)", fontsize=12)
            ax.set_ylabel("Gate Value (0 = Phase A, 1 = Phase B)", fontsize=12)
            ax.set_ylim(-0.1, 1.1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=10)
            plt.tight_layout()
            plt.savefig(
                eval_dir / "chromosome_position_scatter_truth.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        phase_a_np = phase_a_pred.values.flatten()
        phase_b_np = phase_b_pred.values.flatten()
        e_m_np = e_m_truth.values.flatten()
        e_p_np = e_p_truth.values.flatten()

        mse_a = np.mean((phase_a_np - e_m_np) ** 2)
        mse_b = np.mean((phase_b_np - e_p_np) ** 2)
        mae_a = np.mean(np.abs(phase_a_np - e_m_np))
        mae_b = np.mean(np.abs(phase_b_np - e_p_np))

        metrics = {
            "mse_phase_a": float(mse_a),
            "mse_phase_b": float(mse_b),
            "mse_mean": float(np.mean([mse_a, mse_b])),
            "mae_phase_a": float(mae_a),
            "mae_phase_b": float(mae_b),
            "mae_mean": float((mae_a + mae_b) / 2),
            "rmse_mean": float(np.sqrt(np.mean([mse_a, mse_b]))),
            "pearson_corr_mean": float(
                (
                    np.corrcoef(phase_a_np, e_m_np)[0, 1]
                    + np.corrcoef(phase_b_np, e_p_np)[0, 1]
                )
                / 2
            ),
        }

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(eval_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")

        report_lines = [
            "=" * 60,
            "PHASE SEPARATION EVALUATION METRICS",
            "=" * 60,
            "",
            "# 误差指标",
            "----------------------------------------",
            f"MSE (Phase A):             {metrics['mse_phase_a']:.4f}",
            f"MSE (Phase B):             {metrics['mse_phase_b']:.4f}",
            f"MSE Mean:                  {metrics['mse_mean']:.4f}",
            "",
            f"MAE (Phase A):             {metrics['mae_phase_a']:.4f}",
            f"MAE (Phase B):             {metrics['mae_phase_b']:.4f}",
            f"MAE Mean:                  {metrics['mae_mean']:.4f}",
            "",
            f"RMSE Mean:                 {metrics['rmse_mean']:.4f}",
            "",
            "# 相关性指标",
            "----------------------------------------",
            f"Pearson Correlation Mean:  {metrics['pearson_corr_mean']:.4f}",
            "",
        ]
        with open(eval_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print("    Saved: individual_heatmaps.png, difference_heatmaps.png")
        print(
            "    Saved: chromosome_phase_distribution_predicted.png, chromosome_phase_distribution_truth.png"
        )
        print(
            "    Saved: chromosome_position_scatter_predicted.png, chromosome_position_scatter_truth.png"
        )
        print("    Saved: evaluation_metrics.csv, evaluation_report.txt")
    else:
        print(f"    Skipped: ground truth files not found ({e_m_file}, {e_p_file})")

    print(f"  Evaluation visualization completed")
