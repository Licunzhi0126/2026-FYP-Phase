#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert TF-by-gene GRN matrices into gene-by-gene matrices.

This script reads the existing paternal/maternal GRN outputs from V6 and
re-indexes them so rows are all genes and only TF genes have non-zero
outgoing edges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_gene_ids(data_dir: Path) -> list[str]:
    gene_info_path = data_dir / "gene_info.csv"
    if gene_info_path.exists():
        gene_info = pd.read_csv(gene_info_path, dtype={"gene_id": str})
        return gene_info["gene_id"].astype(str).tolist()

    # Fallback: use gene ids appearing in GRN matrices.
    paternal_path = data_dir / "paternal_GRN.csv"
    if paternal_path.exists():
        df = pd.read_csv(paternal_path, index_col=0, dtype=str)
        return list(df.columns.astype(str))

    raise FileNotFoundError(
        "Cannot infer gene IDs: gene_info.csv and paternal_GRN.csv are both missing."
    )


def load_tf_mapping(data_dir: Path) -> pd.Series:
    tf_info_path = data_dir / "shared_TF_info.csv"
    if not tf_info_path.exists():
        raise FileNotFoundError("shared_TF_info.csv not found in {}".format(data_dir))
    tf_info = pd.read_csv(tf_info_path, dtype={"TF_id": str, "gene_id": str})
    return tf_info.set_index("TF_id")["gene_id"].astype(str)


def load_module_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf_links_path = data_dir / "TF_module_links.csv"
    membership_path = data_dir / "gene_module_membership.csv"
    if not tf_links_path.exists():
        raise FileNotFoundError("TF_module_links.csv not found in {}".format(data_dir))
    if not membership_path.exists():
        raise FileNotFoundError("gene_module_membership.csv not found in {}".format(data_dir))
    tf_links = pd.read_csv(tf_links_path, dtype={"TF_id": str, "module_id": str})
    membership = pd.read_csv(membership_path, dtype={"gene_id": str, "module_id": str})
    return tf_links, membership


def compute_module_contribution(
    tf_to_gene: pd.Series,
    tf_links: pd.DataFrame,
    membership: pd.DataFrame,
    gene_ids: list[str],
) -> pd.DataFrame:
    """Compute static module-mediated TF->gene contributions.

    The result is a gene-by-gene matrix where the row gene is the TF gene
    corresponding to a TF_id and the column gene is the target gene.
    """
    tf_ids = tf_to_gene.index.astype(str).tolist()
    module_ids = sorted(set(tf_links["module_id"]).union(membership["module_id"]))

    # TF->module weight
    tf_module = tf_links.pivot_table(
        index="TF_id", columns="module_id", values="weight", aggfunc="sum", fill_value=0.0
    ).reindex(index=tf_ids, columns=module_ids, fill_value=0.0)
    # gene->module loading
    gene_module = membership.pivot_table(
        index="gene_id", columns="module_id", values="gene_module_loading", aggfunc="sum", fill_value=0.0
    ).reindex(index=gene_ids, columns=module_ids, fill_value=0.0)

    contrib = tf_module.to_numpy(dtype=float) @ gene_module.to_numpy(dtype=float).T
    tf_gene_rows = [tf_to_gene.loc[tf_id] for tf_id in tf_ids]
    result = pd.DataFrame(contrib, index=tf_gene_rows, columns=gene_ids)
    result.index = result.index.astype(str)
    result.columns = result.columns.astype(str)
    return result


def load_tf_module_and_gene_module(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TF->module weights and gene->module loadings as matrices.

    Returns (tf_module, gene_module) where tf_module is TF_id x module_id and
    gene_module is gene_id x module_id (both numeric DataFrames).
    """
    tf_links_path = data_dir / "TF_module_links.csv"
    membership_path = data_dir / "gene_module_membership.csv"
    if not tf_links_path.exists() or not membership_path.exists():
        raise FileNotFoundError("TF_module_links.csv or gene_module_membership.csv missing")

    tf_links = pd.read_csv(tf_links_path, dtype={"TF_id": str, "module_id": str})
    membership = pd.read_csv(membership_path, dtype={"gene_id": str, "module_id": str})
    # ensure numeric columns
    if "weight" in tf_links.columns:
        tf_links["weight"] = pd.to_numeric(tf_links["weight"], errors="coerce").fillna(0.0)
    if "gene_module_loading" in membership.columns:
        membership["gene_module_loading"] = pd.to_numeric(membership["gene_module_loading"], errors="coerce").fillna(0.0)

    tf_info = pd.read_csv(data_dir / "shared_TF_info.csv", dtype={"TF_id": str, "gene_id": str})
    tf_ids = tf_info["TF_id"].astype(str).tolist()
    gene_ids = load_gene_ids(data_dir)
    module_ids = sorted(set(tf_links["module_id"]).union(membership["module_id"]))

    tf_module = tf_links.pivot_table(
        index="TF_id", columns="module_id", values="weight", aggfunc="sum", fill_value=0.0
    ).reindex(index=tf_ids, columns=module_ids, fill_value=0.0)

    gene_module = membership.pivot_table(
        index="gene_id", columns="module_id", values="gene_module_loading", aggfunc="sum", fill_value=0.0
    ).reindex(index=gene_ids, columns=module_ids, fill_value=0.0)

    return tf_module.astype(float), gene_module.astype(float)


def generate_per_cell_gene2gene(
    data_dir: Path,
    out_dir: Path,
    matrix_path: Path,
    allele_name: str,
    tf_to_gene: pd.Series,
    gene_ids: list[str],
    tf_module: pd.DataFrame,
    gene_module: pd.DataFrame,
) -> Path:
    """Generate per-cell gene-by-gene GRN for one allele and save per-cell CSVs.

    Returns the output directory path containing per-cell CSVs.
    """
    # Load static TF->gene matrix (TF_id rows)
    W = pd.read_csv(matrix_path, index_col=0)
    W.index = W.index.astype(str)
    W.columns = W.columns.astype(str)

    # Load per-cell TF activity and module activity
    tf_activity_path = data_dir / "shared_TF_activity.csv"
    module_activity_path = data_dir / "module_activity.csv"
    if not tf_activity_path.exists():
        raise FileNotFoundError("shared_TF_activity.csv not found")
    if not module_activity_path.exists():
        raise FileNotFoundError("module_activity.csv not found")

    tf_activity = pd.read_csv(tf_activity_path, index_col=0)
    module_activity = pd.read_csv(module_activity_path, index_col=0)

    # Prepare indices
    tf_ids = tf_module.index.astype(str).tolist()
    module_ids = tf_module.columns.astype(str).tolist()
    gene_index = pd.Index(gene_ids, dtype=str)

    allele_dir = out_dir / "per_cell" / allele_name
    allele_dir.mkdir(parents=True, exist_ok=True)

    # Ensure W contains all TFs we expect (reindex missing rows/cols with zeros)
    W = W.reindex(index=tf_ids, columns=gene_index, fill_value=0.0)

    # Ensure activity tables have expected columns (reindex silently to avoid KeyError)
    tf_activity = tf_activity.reindex(columns=tf_ids, fill_value=0.0)
    module_activity = module_activity.reindex(columns=module_ids, fill_value=0.0)

    for cell_id in tf_activity.index:
        # TF scaling as in solve_expression: log1p(activity)
        s = np.log1p(tf_activity.loc[cell_id, tf_ids].astype(float)).values  # array aligned to tf_ids

        # scaled direct TF->gene (TF x gene)
        scaled_W = W.values * s[:, None]

        # module-mediated: TF->module scaled by module activity for this cell
        ma = module_activity.loc[cell_id, module_ids].astype(float).values  # (modules,)
        tf_module_cell = tf_module.values * ma[None, :]  # TF x module
        module_contrib_tf_gene = tf_module_cell @ gene_module.values.T  # TF x gene

        effective_tf_gene = scaled_W + module_contrib_tf_gene

        # map TF rows to TF genes (accumulate if multiple TFs map to same gene)
        gene2gene = pd.DataFrame(0.0, index=gene_index, columns=gene_index)
        for i, tf_id in enumerate(tf_ids):
            if tf_id not in tf_to_gene.index:
                continue
            tf_gene = tf_to_gene.loc[tf_id]
            if tf_gene not in gene2gene.index:
                # skip TFs whose mapped gene is not in gene list
                continue
            gene2gene.loc[tf_gene, :] += effective_tf_gene[i, :]

        out_path = allele_dir / f"{cell_id}.csv"
        gene2gene.to_csv(out_path, index=True)

    return allele_dir


def convert_matrix(
    data_dir: Path,
    matrix_name: str,
    tf_to_gene: pd.Series,
    gene_ids: list[str],
    module_contrib: pd.DataFrame | None = None,
) -> pd.DataFrame:
    path = data_dir / matrix_name
    if not path.exists():
        raise FileNotFoundError(f"{matrix_name} not found in {data_dir}")

    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    if not matrix.index.isin(tf_to_gene.index).all():
        missing = sorted(set(matrix.index) - set(tf_to_gene.index))
        raise ValueError(
            f"The following TF ids in {matrix_name} are missing from shared_TF_info.csv: {missing}"
        )

    gene_index = pd.Index(gene_ids, dtype=str)
    gene_cols = pd.Index(gene_ids, dtype=str)
    new_df = pd.DataFrame(
        data=np.zeros((len(gene_index), len(gene_cols))),
        index=gene_index,
        columns=gene_cols,
    )

    for tf_id, row in matrix.iterrows():
        target_gene_weights = row.to_numpy(dtype=float)
        tf_gene = tf_to_gene.loc[tf_id]
        if tf_gene not in gene_index:
            raise ValueError(
                f"Mapped TF gene '{tf_gene}' for TF_id '{tf_id}' is not present in gene list."
            )
        new_df.loc[tf_gene, :] = target_gene_weights

    if module_contrib is not None:
        if not module_contrib.index.isin(new_df.index).all():
            missing = sorted(set(module_contrib.index) - set(new_df.index))
            raise ValueError(
                f"The following TF gene IDs in module contribution are missing from gene list: {missing}"
            )
        new_df = new_df.add(module_contrib, fill_value=0.0)

    return new_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TF-by-gene GRN matrices to gene-by-gene matrices."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing paternal_GRN.csv, maternal_GRN.csv, shared_TF_info.csv, and gene_info.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to save gene-by-gene GRN matrices.",
    )
    args = parser.parse_args()

    data_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_ids = load_gene_ids(data_dir)
    tf_to_gene = load_tf_mapping(data_dir)
    tf_links, membership = load_module_data(data_dir)
    module_contrib = compute_module_contribution(tf_to_gene, tf_links, membership, gene_ids)

    # also load TF-module and gene-module matrices for per-cell generation
    tf_module_df, gene_module_df = load_tf_module_and_gene_module(data_dir)

    # generate per-cell gene->gene GRNs for paternal and maternal alleles
    pat_dir = generate_per_cell_gene2gene(
        data_dir, out_dir, data_dir / "paternal_GRN.csv", "paternal", tf_to_gene, gene_ids, tf_module_df, gene_module_df
    )
    mat_dir = generate_per_cell_gene2gene(
        data_dir, out_dir, data_dir / "maternal_GRN.csv", "maternal", tf_to_gene, gene_ids, tf_module_df, gene_module_df
    )
    print(f"Saved per-cell paternal GRNs to: {pat_dir}")
    print(f"Saved per-cell maternal GRNs to: {mat_dir}")

    # create combined per-cell by summing paternal + maternal per cell
    combined_dir = out_dir / "per_cell" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    tf_activity_path = data_dir / "shared_TF_activity.csv"
    if tf_activity_path.exists():
        cells = pd.read_csv(tf_activity_path, index_col=0).index.astype(str).tolist()
        for cell_id in cells:
            pat_file = Path(pat_dir) / f"{cell_id}.csv"
            mat_file = Path(mat_dir) / f"{cell_id}.csv"
            if pat_file.exists():
                pat_df = pd.read_csv(pat_file, index_col=0)
            else:
                pat_df = pd.DataFrame(0.0, index=gene_ids, columns=gene_ids)
            if mat_file.exists():
                mat_df = pd.read_csv(mat_file, index_col=0)
            else:
                mat_df = pd.DataFrame(0.0, index=gene_ids, columns=gene_ids)
            comb = pat_df.add(mat_df, fill_value=0.0)
            comb.to_csv(combined_dir / f"{cell_id}.csv", index=True)
        print(f"Saved per-cell combined GRNs to: {combined_dir}")

    outputs = []
    for matrix_name, out_name in [
        ("paternal_GRN.csv", "paternal_GRN_gene2gene.csv"),
        ("maternal_GRN.csv", "maternal_GRN_gene2gene.csv"),
    ]:
        out_df = convert_matrix(data_dir, matrix_name, tf_to_gene, gene_ids, module_contrib)
        out_path = out_dir / out_name
        out_df.to_csv(out_path, index=True)
        outputs.append((matrix_name, out_df, out_path))
        print(f"Saved: {out_path}")

    # combined GRN = paternal + maternal
    pat_df = outputs[0][1]
    mat_df = outputs[1][1]
    combined_df = pat_df.add(mat_df, fill_value=0.0)
    combined_path = out_dir / "combined_GRN_gene2gene.csv"
    combined_df.to_csv(combined_path, index=True)
    print(f"Saved: {combined_path}")

    print("Done.")


if __name__ == "__main__":
    main()
