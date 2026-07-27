"""两层级 HGNN cluster-only 入口。

用法:
  python run_two_level.py \
      --clean-root data_clean \
      --dataset CITE_seq \
      --epochs 100 \
      --device cuda \
      --tag cluster_only

输出:
  output/<dataset>_cluster_metrics_<tag>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phage_model.hypergraph.builder import build_two_level_hypergraph
from phage_model.losses.total import compute_total_loss, UncertaintyWeights
from phage_model.models.phase_model import HGNN_DualRAE_Phase_Model
from phage_model.schemas import DatasetBundle, PhaseTrainingConfig
from phage_model.training.trainer import _build_aux


def _save_pca_scatter(matrices, labels, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    label_names = np.asarray(labels).astype(str)
    enc = LabelEncoder()
    y_color = enc.fit_transform(label_names)
    classes = list(enc.classes_)
    cmap = plt.get_cmap("tab10", max(1, len(classes)))
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.4 * len(matrices), 5.0), constrained_layout=True)
    if len(matrices) == 1:
        axes = [axes]

    for ax, (title, X) in zip(axes, matrices.items()):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        X = StandardScaler().fit_transform(X)
        coords = PCA(n_components=2, random_state=0).fit_transform(X)
        ax.scatter(coords[:, 0], coords[:, 1], c=y_color, cmap=cmap, s=34, alpha=0.88, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.18, linewidth=0.6)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=cmap(i), label=name, markersize=7)
        for i, name in enumerate(classes)
    ]
    fig.legend(handles=handles, title="Cell label", loc="center right", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_cluster(args):
    from phage_model.data.loading import load_clean_hetero_bundle
    from phage_model.data.priors import build_prior_bundle
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

    device = torch.device(args.device)

    bundle = load_clean_hetero_bundle(args.clean_root, args.dataset)
    ds = DatasetBundle(
        dataset_type=args.dataset,
        view1_name=args.dataset,
        common_genes=bundle.genes,
        common_cells=bundle.cells,
    )
    prior = build_prior_bundle(Path(args.clean_root), ds, d_prior=16)
    built = build_two_level_hypergraph(
        bundle,
        prior,
        gene_channel_out=args.hidden_dim,
        cell_channel_out=args.hidden_dim,
        ppi_max_members=args.ppi_max_members,
        rna_knn_k=args.rna_knn_k,
        adt_knn_k=args.adt_knn_k,
        pca_dim=args.pca_dim,
        knn_weight=True,
    )

    y = built["true_cell_labels"]
    k = int(len(np.unique(y)))
    print("=" * 78)
    print(f"[task=cluster] {args.dataset}  head=additive  decoder=gnmf")
    print(
        f"  细胞={built['n_cells']} 基因={built['n_genes']}  "
        f"H_cell {built['H_cell'].shape}  H_gene {built['H_gene'].shape}  k={k}"
    )

    config = PhaseTrainingConfig(
        data_name=args.dataset,
        device=args.device,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        hgnn_num_layers=args.hgnn_layers,
        train_epochs=args.epochs,
        lr=args.lr,
        diff_warmup_epochs=args.diff_warmup_epochs,
        nmf_rank=args.nmf_rank,
        phase_structure_weight=args.phase_structure_weight,
    )

    model = HGNN_DualRAE_Phase_Model(built, config).to(device)
    aux = _build_aux(built, device)
    uw = UncertaintyWeights(2).to(device)
    aux["uncertainty_weights"] = uw
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(uw.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, config.train_epochs),
        eta_min=config.lr_min,
    )

    print("\n[训练 additive 头：recon(X_A+X_B) + UW(reg, 正交)]")
    for ep in range(config.train_epochs):
        model.train()
        out = model()
        loss, diag = compute_total_loss(out, aux, ep, config)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"Ep {ep+1:4d} | total={diag['total_loss']:.4f} "
                f"recon={diag['recon_loss']:.4f} var_exp={diag['var_explained']:.3f} "
                f"{diag['extra']}"
            )

    model.eval()
    with torch.no_grad():
        final = model()

    def ev(X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        d = max(1, min(int(args.eval_dim), X.shape[0] - 1, X.shape[1]))
        Z = PCA(n_components=d, svd_solver="randomized", random_state=0).fit_transform(X)
        pred = KMeans(n_clusters=k, n_init=args.n_init, random_state=0).fit_predict(Z)
        return (
            adjusted_rand_score(y, pred),
            normalized_mutual_info_score(y, pred),
            fowlkes_mallows_score(y, pred),
        )

    print("\n" + "=" * 78)
    print(f"{'对象':<26}{'ARI':>9}{'NMI':>9}{'FMI':>9}")
    print("-" * 78)
    rows = []
    for name, X in [
        ("Phase_A", final["phase_a_expr"].cpu().numpy()),
        ("Phase_B", final["phase_b_expr"].cpu().numpy()),
        ("重构 X_A+X_B", final["expr_recon"].cpu().numpy()),
        ("cell_h 嵌入", final["cell_h"].cpu().numpy()),
        ("基线 原始 expr", built["expr"]),
    ]:
        ari, nmi, fmi = ev(X)
        print(f"{name:<26}{ari:>9.4f}{nmi:>9.4f}{fmi:>9.4f}")
        rows.append((name, ari, nmi, fmi))
    print("-" * 78)

    out_dir = Path("output") / f"{args.dataset}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = built.get("cells")
    genes = built.get("genes")
    matrix_outputs = {
        "raw_expr": built["expr"],
        "phase_a": final["phase_a_expr"].cpu().numpy(),
        "phase_b": final["phase_b_expr"].cpu().numpy(),
    }
    for matrix_name, matrix in matrix_outputs.items():
        matrix_path = out_dir / f"{matrix_name}.csv"
        pd.DataFrame(matrix, index=cells, columns=genes).to_csv(matrix_path)
        print(f"{matrix_name} cell-gene CSV -> {matrix_path}")

    pca_plot_path = out_dir / "pca_raw_phase_a_phase_b.png"
    _save_pca_scatter(
        {
            "raw_expr": matrix_outputs["raw_expr"],
            "phase_a": matrix_outputs["phase_a"],
            "phase_b": matrix_outputs["phase_b"],
        },
        y,
        pca_plot_path,
    )
    print(f"PCA cell scatter -> {pca_plot_path}")

    out_path = out_dir / "cluster_metrics.csv"
    pd.DataFrame(rows, columns=["rep", "ari", "nmi", "fmi"]).to_csv(out_path, index=False)
    print(f"指标 → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="两层级 HGNN cluster-only 训练入口")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hgnn-layers", type=int, default=2)
    p.add_argument("--rna-knn-k", type=int, default=15)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--tag", default="gnmf_r32_struct1")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--clean-root", default="sample_data/data_clean")
    p.add_argument("--dataset", default="scNMT")
    p.add_argument("--ppi-max-members", type=int, default=50)
    p.add_argument("--adt-knn-k", type=int, default=15)
    p.add_argument("--eval-dim", type=int, default=256)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--diff-warmup-epochs", type=int, default=0)
    p.add_argument("--nmf-rank", type=int, default=32)
    p.add_argument("--phase-structure-weight", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_cluster(args)


if __name__ == "__main__":
    main()
