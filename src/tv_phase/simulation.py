from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from .config import DATA_ROOT
from .simulation_adapter import adapt_raw_simulation_to_tv_phase


@dataclass
class SimulationConfig:
    n_cells: int = 120
    n_cell_types: int = 4
    n_genes: int = 100
    n_chr: int = 22
    n_kegg: int = 12
    lambda_kegg_position: float = 1e-8
    chr_expr_strength: float = 0.8
    kegg_expr_strength: float = 0.8
    celltype_expr_strength: float = 1.0
    gene_noise: float = 0.2
    marker_fraction: float = 0.1
    gamma_shape: float = 2.0
    kegg_ratio_strength: float = 1.0
    celltype_ratio_strength: float = 1.0
    ratio_variation: float = 0.3
    ratio_noise: float = 0.05
    global_ratio_mean: float = 0.5
    ppi_same_kegg_prob: float = 0.08
    ppi_diff_kegg_prob: float = 0.01
    seed: int = 42

    def legacy_json(self) -> Dict[str, float]:
        return {
            "N_cells": self.n_cells,
            "N_cell_types": self.n_cell_types,
            "N_genes": self.n_genes,
            "N_chr": self.n_chr,
            "N_kegg": self.n_kegg,
            "lambda_kegg_position": self.lambda_kegg_position,
            "chr_expr_strength": self.chr_expr_strength,
            "kegg_expr_strength": self.kegg_expr_strength,
            "celltype_expr_strength": self.celltype_expr_strength,
            "gene_noise": self.gene_noise,
            "marker_fraction": self.marker_fraction,
            "gamma_shape": self.gamma_shape,
            "kegg_ratio_strength": self.kegg_ratio_strength,
            "celltype_ratio_strength": self.celltype_ratio_strength,
            "ratio_variation": self.ratio_variation,
            "ratio_noise": self.ratio_noise,
            "global_ratio_mean": self.global_ratio_mean,
            "ppi_same_kegg_prob": self.ppi_same_kegg_prob,
            "ppi_diff_kegg_prob": self.ppi_diff_kegg_prob,
            "seed": self.seed,
        }


def _format_number_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def simulation_dataset_dir(alpha: float, beta: float, *, root: Path = DATA_ROOT / "sim_gene100_generated") -> Path:
    return Path(root) / f"alpha_{_format_number_token(alpha)}_beta_{_format_number_token(beta)}"


def raw_simulation_dir(alpha: float, beta: float, *, root: Path = DATA_ROOT / "simulation_raw") -> Path:
    return Path(root) / f"alpha_{_format_number_token(alpha)}_beta_{_format_number_token(beta)}"


def _assign_kegg(gene_info: pd.DataFrame, kegg_ids, lam: float) -> pd.DataFrame:
    gene_info = gene_info.copy()
    gene_info["kegg"] = None

    for chr_id in gene_info["chr"].unique():
        sub = gene_info[gene_info["chr"] == chr_id].sort_values("start")
        idx = sub.index.tolist()
        if len(idx) == 0:
            continue

        gene_info.loc[idx[0], "kegg"] = np.random.choice(kegg_ids)
        for i in range(1, len(idx)):
            prev = idx[i - 1]
            cur = idx[i]
            dist = gene_info.loc[cur, "start"] - gene_info.loc[prev, "end"]
            p = np.exp(-lam * dist)
            if np.random.rand() < p:
                gene_info.loc[cur, "kegg"] = gene_info.loc[prev, "kegg"]
            else:
                gene_info.loc[cur, "kegg"] = np.random.choice(kegg_ids)
    return gene_info


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Pass overwrite=True or use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_raw_simulation(config: SimulationConfig, output_dir, *, overwrite: bool = False) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite=overwrite)
    np.random.seed(config.seed)

    genes = [f"G{i}" for i in range(config.n_genes)]
    cells = [f"C{i}" for i in range(config.n_cells)]
    chr_groups = np.array_split(genes, config.n_chr)

    gene_info_rows = []
    chr_length = 1_000_000
    for i, group in enumerate(chr_groups):
        chr_id = i + 1
        n = len(group)
        lengths = np.random.randint(100, 10000, n)
        gaps = np.random.randint(1000, 100000, n)
        current_pos = np.random.randint(1, 100000)
        starts, ends = [], []

        for length, gap in zip(lengths, gaps):
            start = current_pos
            end = start + length
            if end > chr_length:
                break
            starts.append(start)
            ends.append(end)
            current_pos = end + gap

        for gene, start, end in zip(group[: len(starts)], starts, ends):
            gene_info_rows.append([gene, chr_id, start, end, end - start])

    gene_info = pd.DataFrame(gene_info_rows, columns=["gene", "chr", "start", "end", "length"])
    kegg_ids = [f"K{i:02d}" for i in range(config.n_kegg)]
    gene_info = _assign_kegg(gene_info, kegg_ids, config.lambda_kegg_position)

    cell_types = np.random.randint(0, config.n_cell_types, config.n_cells)

    gene_order = gene_info["gene"].tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_order)}
    n_genes = len(gene_order)
    corr = np.eye(n_genes) * config.gene_noise

    for chr_id in gene_info["chr"].unique():
        idx = gene_info[gene_info["chr"] == chr_id]["gene"].map(gene_to_idx).tolist()
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = config.chr_expr_strength

    for kegg in gene_info["kegg"].unique():
        idx = gene_info[gene_info["kegg"] == kegg]["gene"].map(gene_to_idx).tolist()
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = max(corr[i, j], config.kegg_expr_strength)

    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    min_eig = np.min(np.linalg.eigvalsh(corr))
    if min_eig < 0:
        corr += np.eye(n_genes) * (-min_eig + 1e-3)
    chol = np.linalg.cholesky(corr)

    marker_gene_mask = np.zeros((config.n_cell_types, n_genes))
    for cell_type in range(config.n_cell_types):
        markers = np.random.choice(
            n_genes,
            int(config.marker_fraction * n_genes),
            replace=False,
        )
        marker_gene_mask[cell_type, markers] = 1

    celltype_effect = np.random.normal(
        0,
        config.celltype_expr_strength,
        (config.n_cell_types, n_genes),
    )

    expr = np.zeros((n_genes, config.n_cells))
    for cell_idx in range(config.n_cells):
        cell_type = cell_types[cell_idx]
        z = np.random.normal(0, 1, n_genes)
        latent = chol @ z
        cell_base = np.random.gamma(3, 3)
        log_expr = (
            np.log(cell_base)
            + latent
            + marker_gene_mask[cell_type] * celltype_effect[cell_type]
        )
        theta = np.exp(log_expr)
        expr[:, cell_idx] = np.random.gamma(config.gamma_shape, theta)

    expression = pd.DataFrame(expr, index=gene_order, columns=cells)

    kegg_mu = {kegg: np.random.uniform(0.3, 0.7) for kegg in kegg_ids}
    cell_mu = {cell_type: np.random.uniform(0.3, 0.7) for cell_type in range(config.n_cell_types)}
    ratio = np.zeros((n_genes, config.n_cells))

    for gene_idx in range(n_genes):
        kegg = gene_info.loc[gene_idx, "kegg"]
        for cell_idx in range(config.n_cells):
            cell_type = cell_types[cell_idx]
            z = (
                config.global_ratio_mean
                + config.kegg_ratio_strength * (kegg_mu[kegg] - 0.5)
                + config.celltype_ratio_strength * (cell_mu[cell_type] - 0.5)
                + np.random.normal(0, config.ratio_noise)
            )
            ratio[gene_idx, cell_idx] = _sigmoid(z)

    ratio = pd.DataFrame(ratio, index=gene_order, columns=cells)
    e_p = expression * ratio
    e_m = expression * (1 - ratio)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_genes))
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            same_kegg = gene_info.loc[i, "kegg"] == gene_info.loc[j, "kegg"]
            p = config.ppi_same_kegg_prob if same_kegg else config.ppi_diff_kegg_prob
            if np.random.rand() < p:
                graph.add_edge(i, j)

    gene_info.to_csv(output_dir / "gene_metadata.csv", index=False)
    pd.DataFrame({"cell": cells, "cell_type": cell_types}).to_csv(
        output_dir / "cell_metadata.csv",
        index=False,
    )
    expression.to_csv(output_dir / "E_obs.csv")
    e_p.to_csv(output_dir / "E_P.csv")
    e_m.to_csv(output_dir / "E_M.csv")
    ratio.to_csv(output_dir / "ratio.csv")

    edges = nx.to_pandas_edgelist(graph)
    if edges.empty:
        edges = pd.DataFrame(columns=["source", "target"])
    edges["source"] = edges["source"].apply(lambda x: f"G{x}")
    edges["target"] = edges["target"].apply(lambda x: f"G{x}")
    edges.to_csv(output_dir / "ppi.csv", index=False)

    (output_dir / "config.json").write_text(
        json.dumps(config.legacy_json(), indent=4),
        encoding="utf-8",
    )

    return {
        "gene_metadata": output_dir / "gene_metadata.csv",
        "cell_metadata": output_dir / "cell_metadata.csv",
        "expression": output_dir / "E_obs.csv",
        "e_p": output_dir / "E_P.csv",
        "e_m": output_dir / "E_M.csv",
        "ratio": output_dir / "ratio.csv",
        "ppi": output_dir / "ppi.csv",
        "config": output_dir / "config.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simulation_v4 data and adapt it for TV-PHASE")
    parser.add_argument("--alpha", type=float, default=1.0, help="KEGG ratio strength")
    parser.add_argument("--beta", type=float, default=1.0, help="Cell type ratio strength")
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--tv-phase-output-dir", type=Path, default=None)
    parser.add_argument("--skip-tv-phase-format", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cells", type=int, default=120)
    parser.add_argument("--n-cell-types", type=int, default=4)
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-chr", type=int, default=22)
    parser.add_argument("--n-kegg", type=int, default=12)
    return parser


def run_from_args(args) -> Dict[str, Path]:
    config = SimulationConfig(
        n_cells=args.n_cells,
        n_cell_types=args.n_cell_types,
        n_genes=args.n_genes,
        n_chr=args.n_chr,
        n_kegg=args.n_kegg,
        kegg_ratio_strength=args.alpha,
        celltype_ratio_strength=args.beta,
        seed=args.seed,
    )

    raw_dir = Path(args.raw_output_dir) if args.raw_output_dir else raw_simulation_dir(args.alpha, args.beta)
    print(f"Generating raw simulation data: {raw_dir}")
    raw_paths = generate_raw_simulation(config, raw_dir, overwrite=args.overwrite)

    if args.skip_tv_phase_format:
        return raw_paths

    tv_phase_dir = (
        Path(args.tv_phase_output_dir)
        if args.tv_phase_output_dir
        else simulation_dataset_dir(args.alpha, args.beta)
    )
    print(f"Adapting simulation data for TV-PHASE: {tv_phase_dir}")
    adapt_raw_simulation_to_tv_phase(raw_dir, tv_phase_dir, overwrite=args.overwrite)
    return {**raw_paths, "tv_phase_dir": tv_phase_dir}


def main() -> None:
    args = build_parser().parse_args()
    paths = run_from_args(args)
    print("Simulation outputs:")
    for name, path in paths.items():
        print(f"  {name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()


__all__ = [
    "SimulationConfig",
    "adapt_raw_simulation_to_tv_phase",
    "generate_raw_simulation",
    "raw_simulation_dir",
    "simulation_dataset_dir",
]
