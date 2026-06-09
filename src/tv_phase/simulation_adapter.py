from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

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

SIMULATION69_FILES = TV_PHASE_FILES + ("simulation69_source_summary.csv",)


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


def _prepare_output_dir_no_delete(output_dir: Path, *, overwrite: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Pass overwrite=True or use --overwrite to replace known adapter outputs in place."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_raw_matrix(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing raw simulation file: {path}")
    return pd.read_csv(path, index_col=0)


def _read_simulation69_matrix(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing simulation_6.9 file: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = [str(col).strip() for col in df.columns]
    if df.index.duplicated().any():
        raise ValueError(f"{path} contains duplicated cell ids")
    if df.columns.duplicated().any():
        raise ValueError(f"{path} contains duplicated gene ids")
    return df


def _read_simulation69_gene_info(raw_dir: Path) -> pd.DataFrame:
    candidates = [raw_dir / "gene_info_with_pathway.csv", raw_dir / "gene_info.csv"]
    gene_path = next((path for path in candidates if path.exists()), None)
    if gene_path is None:
        raise FileNotFoundError(
            f"Missing simulation_6.9 gene info file. Expected one of: {candidates}"
        )

    gene_info = pd.read_csv(gene_path)
    required = {"gene_id", "chromosome", "start_pos", "end_pos", "pathway"}
    missing = required - set(gene_info.columns)
    if missing:
        raise ValueError(f"{gene_path} is missing columns: {sorted(missing)}")
    gene_info["gene_id"] = gene_info["gene_id"].astype(str).str.strip()
    if gene_info["gene_id"].duplicated().any():
        raise ValueError(f"{gene_path} contains duplicated gene_id values")
    return gene_info


def _write_simulation69_cell_stage(
    output_dir: Path,
    cells,
    *,
    cell_info_path: Optional[Path],
    missing_stage_label: Optional[str],
) -> Dict[str, object]:
    cells = [str(cell).strip() for cell in cells]
    if cell_info_path is not None:
        cell_info_path = Path(cell_info_path)
    if cell_info_path is not None and cell_info_path.exists():
        cell_info = pd.read_csv(cell_info_path)
        required = {"cell_id", "cell_type"}
        missing = required - set(cell_info.columns)
        if missing:
            raise ValueError(f"{cell_info_path} is missing columns: {sorted(missing)}")
        stage_by_cell = dict(
            zip(
                cell_info["cell_id"].astype(str).str.strip(),
                cell_info["cell_type"].astype(str).str.strip(),
            )
        )
        missing_cells = [cell for cell in cells if cell not in stage_by_cell]
        if missing_cells:
            raise ValueError(
                f"{cell_info_path} does not contain stage labels for cells: {missing_cells[:5]}"
            )
        stages = [stage_by_cell[cell] for cell in cells]
        source = str(cell_info_path.resolve())
        mode = "cell_info"
    elif missing_stage_label is not None:
        stages = [str(missing_stage_label)] * len(cells)
        source = ""
        mode = "constant_missing_stage_label"
    else:
        expected = cell_info_path if cell_info_path is not None else output_dir / "cell_info.csv"
        raise FileNotFoundError(
            f"Missing cell info file for simulation_6.9 adapter: {expected}. "
            "Pass missing_stage_label to create a smoke-test-only cell_stage.csv."
        )

    (output_dir / "cell_stage.csv").write_text(",".join(stages), encoding="utf-8")
    return {
        "cell_stage_mode": mode,
        "cell_stage_source": source,
        "cell_stage_unique_labels": sorted(set(stages)),
    }


def _write_simulation69_position_prior(gene_info: pd.DataFrame, output_dir: Path, genes) -> None:
    genes = [str(gene).strip() for gene in genes]
    gene_meta = gene_info.set_index("gene_id").loc[genes].reset_index()
    pos = pd.DataFrame(
        {
            "gene": gene_meta["gene_id"].astype(str),
            "chr": gene_meta["chromosome"].astype(str),
            "start": gene_meta["start_pos"].astype(int),
            "end": gene_meta["end_pos"].astype(int),
            "strand": "+",
        }
    )
    pos.to_csv(output_dir / "poswin_prior.txt", sep="\t", header=False, index=False)
    pos.to_csv(output_dir / "gene_positions_pea.txt", sep="\t", header=False, index=False)


def _write_simulation69_kegg_prior(gene_info: pd.DataFrame, output_dir: Path, genes) -> None:
    genes = [str(gene).strip() for gene in genes]
    gene_meta = gene_info.set_index("gene_id").loc[genes].reset_index()
    kegg = pd.DataFrame(
        {
            "gene": gene_meta["gene_id"].astype(str),
            "pathway_a": gene_meta["pathway"].astype(str),
            "pathway_b": gene_meta["pathway"].astype(str),
        }
    )
    kegg.to_csv(output_dir / "kegg_prior.txt", sep="\t", header=False, index=False)


def _write_simulation69_ppi_prior(raw_dir: Path, output_dir: Path, genes) -> Dict[str, object]:
    genes = [str(gene).strip() for gene in genes]
    adjacency = pd.DataFrame(0.0, index=genes, columns=genes)
    ppi_path = raw_dir / "synthetic_expression_ppi.csv"
    edge_count = 0
    if ppi_path.exists():
        edges = pd.read_csv(ppi_path)
        required = {"gene1", "gene2"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"{ppi_path} is missing columns: {sorted(missing)}")
        for _, row in edges.iterrows():
            source = str(row["gene1"]).strip()
            target = str(row["gene2"]).strip()
            if source not in adjacency.index or target not in adjacency.columns or source == target:
                continue
            weight = 1.0
            if "weight" in edges.columns and pd.notna(row["weight"]):
                weight = float(abs(row["weight"]))
            adjacency.loc[source, target] = weight
            adjacency.loc[target, source] = weight
            edge_count += 1
    np.fill_diagonal(adjacency.values, 0.0)
    adjacency.to_csv(output_dir / "ppi_prior.csv")
    return {
        "ppi_source": str(ppi_path.resolve()) if ppi_path.exists() else "",
        "ppi_edges_written": edge_count,
    }


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


def adapt_simulation69_to_tv_phase(
    raw_dir,
    output_dir,
    *,
    cell_info_path=None,
    missing_stage_label: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"simulation_6.9 directory does not exist: {raw_dir}")

    _prepare_output_dir_no_delete(output_dir, overwrite=overwrite)

    expression = _read_simulation69_matrix(raw_dir, "mixed_expression.csv")
    e_p = _read_simulation69_matrix(raw_dir, "paternal_expression.csv")
    e_m = _read_simulation69_matrix(raw_dir, "maternal_expression.csv")

    if not expression.index.equals(e_p.index) or not expression.index.equals(e_m.index):
        raise ValueError("mixed/paternal/maternal expression cell orders do not match")
    if not expression.columns.equals(e_p.columns) or not expression.columns.equals(e_m.columns):
        raise ValueError("mixed/paternal/maternal expression gene orders do not match")

    gene_info = _read_simulation69_gene_info(raw_dir)
    missing_genes = [gene for gene in expression.columns if gene not in set(gene_info["gene_id"])]
    if missing_genes:
        raise ValueError(f"gene_info is missing expression genes: {missing_genes[:5]}")

    expression.to_csv(output_dir / "expression_data.csv")
    e_p.to_csv(output_dir / "E_P.csv")
    e_m.to_csv(output_dir / "E_M.csv")

    stage_info = _write_simulation69_cell_stage(
        output_dir,
        expression.index.tolist(),
        cell_info_path=Path(cell_info_path) if cell_info_path is not None else raw_dir / "cell_info.csv",
        missing_stage_label=missing_stage_label,
    )
    _write_simulation69_position_prior(gene_info, output_dir, expression.columns.tolist())
    _write_simulation69_kegg_prior(gene_info, output_dir, expression.columns.tolist())
    ppi_info = _write_simulation69_ppi_prior(raw_dir, output_dir, expression.columns.tolist())

    summary = pd.DataFrame(
        [
            {
                "raw_dir": str(raw_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "n_cells": expression.shape[0],
                "n_genes": expression.shape[1],
                "n_pathways": int(gene_info["pathway"].nunique()),
                **stage_info,
                **ppi_info,
            }
        ]
    )
    summary.to_csv(output_dir / "simulation69_source_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "schema": "simulation_6.9",
        "files": list(SIMULATION69_FILES),
        "stage_info": stage_info,
        "ppi_info": ppi_info,
    }
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {name: output_dir / name for name in SIMULATION69_FILES}


__all__ = [
    "SIMULATION69_FILES",
    "TV_PHASE_FILES",
    "adapt_raw_simulation_to_tv_phase",
    "adapt_simulation69_to_tv_phase",
]
