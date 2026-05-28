from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


TV_PHASE_FILES = (
    "expression_data.csv",
    "cell_stage.csv",
    "kegg_prior.txt",
    "poswin_prior.txt",
    "gene_positions_pea.txt",
    "ppi_prior.csv",
    "E_P.csv",
    "E_M.csv",
)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass overwrite=True or use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_raw_matrix(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing raw simulation file: {path}")
    return pd.read_csv(path, index_col=0)


def _write_cell_stage(raw_dir: Path, output_dir: Path, cells) -> None:
    meta_path = raw_dir / "cell_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing raw simulation file: {meta_path}")
    cell_meta = pd.read_csv(meta_path)
    if "cell" not in cell_meta.columns or "cell_type" not in cell_meta.columns:
        raise ValueError("cell_metadata.csv must contain 'cell' and 'cell_type' columns")

    stage_by_cell = dict(zip(cell_meta["cell"].astype(str), cell_meta["cell_type"].astype(str)))
    stages = [stage_by_cell[str(cell)] for cell in cells]
    (output_dir / "cell_stage.csv").write_text(",".join(stages), encoding="utf-8")


def _write_position_prior(gene_meta: pd.DataFrame, output_dir: Path) -> None:
    required = {"gene", "chr", "start", "end"}
    missing = required - set(gene_meta.columns)
    if missing:
        raise ValueError(f"gene_metadata.csv is missing columns: {sorted(missing)}")

    pos = gene_meta[["gene", "chr", "start", "end"]].copy()
    pos["strand"] = "+"
    pos.to_csv(output_dir / "poswin_prior.txt", sep="\t", header=False, index=False)
    pos.to_csv(output_dir / "gene_positions_pea.txt", sep="\t", header=False, index=False)


def _write_kegg_prior(gene_meta: pd.DataFrame, output_dir: Path) -> None:
    required = {"gene", "kegg"}
    missing = required - set(gene_meta.columns)
    if missing:
        raise ValueError(f"gene_metadata.csv is missing columns: {sorted(missing)}")

    kegg = pd.DataFrame(
        {
            "gene": gene_meta["gene"].astype(str),
            "pathway_a": gene_meta["kegg"].astype(str),
            "pathway_b": gene_meta["kegg"].astype(str),
        }
    )
    kegg.to_csv(output_dir / "kegg_prior.txt", sep="\t", header=False, index=False)


def _write_ppi_prior(raw_dir: Path, output_dir: Path, genes) -> None:
    ppi_path = raw_dir / "ppi.csv"
    if not ppi_path.exists():
        raise FileNotFoundError(f"Missing raw simulation file: {ppi_path}")

    genes = [str(gene) for gene in genes]
    adjacency = pd.DataFrame(0.0, index=genes, columns=genes)
    edges = pd.read_csv(ppi_path)
    if not edges.empty:
        if "source" not in edges.columns or "target" not in edges.columns:
            raise ValueError("ppi.csv must contain 'source' and 'target' columns")
        for source, target in zip(edges["source"].astype(str), edges["target"].astype(str)):
            if source in adjacency.index and target in adjacency.columns and source != target:
                adjacency.loc[source, target] = 1.0
                adjacency.loc[target, source] = 1.0
    np.fill_diagonal(adjacency.values, 0.0)
    adjacency.to_csv(output_dir / "ppi_prior.csv")


def adapt_raw_simulation_to_tv_phase(raw_dir, output_dir, *, overwrite: bool = False) -> Dict[str, Path]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw simulation directory does not exist: {raw_dir}")

    _prepare_output_dir(output_dir, overwrite=overwrite)

    expression_raw = _read_raw_matrix(raw_dir, "E_obs.csv")
    e_p_raw = _read_raw_matrix(raw_dir, "E_P.csv")
    e_m_raw = _read_raw_matrix(raw_dir, "E_M.csv")

    expression = expression_raw.T
    e_p = e_p_raw.T
    e_m = e_m_raw.T

    expression.to_csv(output_dir / "expression_data.csv")
    e_p.to_csv(output_dir / "E_P.csv")
    e_m.to_csv(output_dir / "E_M.csv")

    _write_cell_stage(raw_dir, output_dir, expression.index.tolist())

    gene_meta_path = raw_dir / "gene_metadata.csv"
    if not gene_meta_path.exists():
        raise FileNotFoundError(f"Missing raw simulation file: {gene_meta_path}")
    gene_meta = pd.read_csv(gene_meta_path)
    _write_position_prior(gene_meta, output_dir)
    _write_kegg_prior(gene_meta, output_dir)
    _write_ppi_prior(raw_dir, output_dir, expression.columns.tolist())

    config_path = raw_dir / "config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "simulation_config.json")

    manifest = {
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "files": list(TV_PHASE_FILES),
    }
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {name: output_dir / name for name in TV_PHASE_FILES}


__all__ = [
    "TV_PHASE_FILES",
    "adapt_raw_simulation_to_tv_phase",
]
