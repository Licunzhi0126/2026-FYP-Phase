"""Single-run HyperPhase experiment for real multi-omic data.

The model and training objective come from ``phasehyper.model``, which is also
used by ``run_simulation.py``. Dataset labels are loaded for final evaluation
only and never enter graph construction, features, losses, phase naming, or
checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from phasehyper.config import DATASET_CONFIG, REAL_PHASE_DATASETS
from phasehyper.data import load_and_align_labels, resolve_label_path
from phasehyper.evaluation.metrics_io import save_metrics_json
from phasehyper.evaluation.phase import evaluate_phase_model
from phasehyper.model import (
    HyperPhaseModel,
    SetCriterion,
    build_criterion,
    build_model,
    build_optimizer,
)


# =============================================================================
# 1. IMPORTS AND CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parent
EXAMPLE_DATA_ROOT = ROOT / "example_data"

DEFAULT_DC = 64
DEFAULT_HIDDEN = 256
DEFAULT_DROPOUT = 0.2
DEFAULT_RNA_TOP_K = 15
DEFAULT_PROTEIN_TOP_K = 15
DEFAULT_GRN_POSITIVE_TOP_K = 10
DEFAULT_GRN_NEGATIVE_TOP_K = 10
DEFAULT_ACTIVE_TF_TOP_K = 32
DEFAULT_RNA_KNN_K = 15
DEFAULT_PROTEIN_KNN_K = 15
DEFAULT_W_COMP = 8.0
DEFAULT_W_ORTHO = 4.0
DEFAULT_W_GATE = 0.05
DEFAULT_W_NCE = 1.0
DEFAULT_GRAD_CLIP = 5.0
EPS = 1e-8

DIRECTED_EDGE_TYPES = (
    "rna_inject", "rna_readout", "prot_inject", "prot_readout", "translation",
    "tf_activation", "reg_cascade", "module_coop", "chromatin_region",
    "compartment_A", "compartment_B", "proximity_200kb", "same_strand_adj",
    "grn_stim", "grn_inhib", "grn_activate", "grn_activate_readout",
    "grn_repress", "grn_repress_readout",
)
UNDIRECTED_EDGE_TYPES = (
    "RNA_obs", "prot_obs", "view2_obs", "chromatin_region", "compartment_A",
    "compartment_B", "proximity_200kb", "proximity_500kb", "same_strand_adj",
    "pathway_module", "ppi_module", "mech_bridge", "mvmod", "rna_knn",
    "adt_knn", "view2_knn", "cci",
)


# =============================================================================
# 2. DATACLASSES
# =============================================================================

@dataclass
class RealDataset:
    name: str
    rna: np.ndarray
    views: dict[str, np.ndarray]
    genes: list[str]
    cell_ids: list[str]
    labels: np.ndarray
    n_clusters: int
    metadata: dict[str, Any]


@dataclass
class PriorBundle:
    gene_info: pd.DataFrame
    pathways: dict[str, list[str]]
    ppi_modules: dict[str, list[str]]
    static_grn: pd.DataFrame
    ligand_receptor: pd.DataFrame
    modality_gene_map: dict[str, list[str]]
    metadata: dict[str, Any]


@dataclass
class NodeIndex:
    cell: dict[str, int]
    gene: dict[str, int]
    protein: dict[str, int]
    tf: dict[str, int]
    n_nodes: int


@dataclass(frozen=True)
class Hyperedge:
    name: str
    members: tuple[int, ...] | None
    tail: tuple[int, ...] | None
    head: tuple[int, ...] | None
    weight: float


@dataclass
class EdgeAudit:
    candidate_count: int = 0
    dropped_duplicate_count: int = 0
    dropped_invalid_count: int = 0
    reason: str | None = None


class DataValidationError(RuntimeError):
    pass


class PriorValidationError(RuntimeError):
    pass


class HyperedgeValidationError(RuntimeError):
    pass


# =============================================================================
# 3. ARGUMENT PARSING
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one HyperPhase real-data experiment from example_data."
    )
    parser.add_argument(
        "--dataset", default="PEA_STA",
        choices=REAL_PHASE_DATASETS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--strict-priors", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("./result_phase"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    return args


# =============================================================================
# 4. GENERAL UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_phase")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {"rows": int(value.shape[0]), "columns": value.columns.tolist()}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > EPS), sd, 1.0)
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fill_columns(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        good = np.isfinite(col)
        if not good.any():
            fill = 0.0
        elif mode == "min":
            fill = float(np.min(col[good]))
        elif mode == "median":
            fill = float(np.median(col[good]))
        else:
            fill = float(np.mean(col[good]))
        col[~good] = fill
        x[:, j] = col
    return x


def read_numeric_frame(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"data file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, na_values=["NA", ""])
    if df.shape[1] < 2:
        raise DataValidationError(f"expected cell ID plus feature columns in {path}")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id")
    df.columns = [str(c).strip() for c in df.columns]
    # ``read_csv`` already parses these matrices as numeric.  Converting the
    # complete block at once is crucial for CITE-seq (20,400 HUMAN columns);
    # assigning one pandas Series at a time creates multi-gigabyte transient
    # copies and turns a seconds-long conversion into minutes.
    try:
        df = df.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    if df.index.duplicated().any():
        df = df.groupby(level=0, sort=False).mean()
    return df


def load_hgnc_maps(root: Path) -> dict[str, Any]:
    path = root / "hgnc_complete_set.txt"
    if not path.exists():
        return {"alias": {}, "uniprot": {}, "source": str(path)}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    alias: dict[str, str] = {}
    uniprot: dict[str, str] = {}
    for row in df.itertuples(index=False):
        record = row._asdict()
        symbol = str(record.get("symbol", "")).strip()
        if not symbol:
            continue
        alias.setdefault(symbol.upper(), symbol)
        for field_name in ("alias_symbol", "prev_symbol", "cd"):
            for token in str(record.get(field_name, "")).split("|"):
                token = token.strip()
                if token:
                    alias.setdefault(token.upper(), symbol)
        for token in str(record.get("uniprot_ids", "")).split("|"):
            token = token.strip()
            if token:
                uniprot.setdefault(token, symbol)
    alias.update({
        "CXCL8.IL.8": "CXCL8", "IGFBP.2": "IGFBP2", "TENASCIN.C": "TNC",
        "SNAL1": "SNAI1", "CASPR1": "CNTNAP1", "HS1": "HCLS1",
        "OCT4": "POU5F1", "NESTIN": "NES", "LEFTY": "LEFTY1",
        "TMEM173": "STING1",
    })
    return {"alias": alias, "uniprot": uniprot, "source": str(path)}


def canonicalize_frame(
    frame: pd.DataFrame,
    mapper: dict[str, str],
    *,
    uniprot: bool = False,
    drop_unmapped: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original = frame.columns.tolist()
    rename: dict[str, str] = {}
    dropped: list[str] = []
    for column in original:
        key = str(column).strip()
        mapped = mapper.get(key if uniprot else key.upper())
        if mapped:
            rename[column] = mapped
        elif drop_unmapped:
            dropped.append(key)
        else:
            rename[column] = key
    kept = [column for column in original if column in rename]
    out = frame.loc[:, kept].rename(columns=rename)
    if out.columns.duplicated().any():
        out = out.T.groupby(level=0, sort=False).mean().T
    return out, {
        "input_features": len(original),
        "output_features": int(out.shape[1]),
        "mapped": sum(1 for c in original if c in rename and rename[c] != str(c).strip()),
        "dropped_unmapped": dropped,
    }


# =============================================================================
# 5. REAL DATA LOADING
# =============================================================================

def load_real_dataset(name: str, logger: logging.Logger) -> RealDataset:
    if name not in REAL_PHASE_DATASETS:
        raise DataValidationError(f"unsupported real dataset: {name}")
    data_dir = EXAMPLE_DATA_ROOT / name
    config = DATASET_CONFIG[name]
    expr_name = config["files"]["expression"]
    label_config = config["labels"]
    modality_specs = [
        spec for spec in config["modalities"] if spec["name"] != "rna"
    ]
    if not modality_specs:
        raise DataValidationError(f"dataset={name}: no auxiliary modalities configured")
    maps = load_hgnc_maps(EXAMPLE_DATA_ROOT)

    if name == "CITE_seq":
        header = pd.read_csv(data_dir / expr_name, nrows=0).columns.tolist()
        selected = [header[0]] + [c for c in header[1:] if str(c).startswith("HUMAN_")]
        rna_df = read_numeric_frame(data_dir / expr_name, usecols=selected)
        rna_df.columns = [str(c)[6:] for c in rna_df.columns]
        rna_df, rna_map_audit = canonicalize_frame(rna_df, maps["alias"])
    else:
        rna_df = read_numeric_frame(data_dir / expr_name)
        if name == "SCoPE2":
            rna_df, rna_map_audit = canonicalize_frame(
                rna_df, maps["uniprot"], uniprot=True, drop_unmapped=True
            )
        else:
            rna_df, rna_map_audit = canonicalize_frame(rna_df, maps["alias"])

    source_rna_cells = rna_df.index.astype(str).tolist()
    view_frames: dict[str, pd.DataFrame] = {}
    view_original_columns_by_view: dict[str, list[str]] = {}
    view_mapping_by_view: dict[str, dict[str, Any]] = {}
    view_paths: dict[str, str] = {}
    for spec in modality_specs:
        modality = str(spec["name"])
        view_path = data_dir / str(spec["file"])
        view_df = read_numeric_frame(view_path)
        view_original_columns_by_view[modality] = [
            str(value) for value in view_df.columns
        ]
        if name == "SCoPE2":
            view_df, view_map_audit = canonicalize_frame(
                view_df, maps["uniprot"], uniprot=True, drop_unmapped=True
            )
        elif name == "CITE_seq" and modality == "protein":
            view_map_audit = {
                "input_features": int(view_df.shape[1]),
                "output_features": int(view_df.shape[1]),
                "mapped": 0,
                "dropped_unmapped": [],
            }
        else:
            view_df, view_map_audit = canonicalize_frame(view_df, maps["alias"])
        view_frames[modality] = view_df
        view_mapping_by_view[modality] = view_map_audit
        view_paths[modality] = str(view_path)

    common_cells = [
        cell
        for cell in source_rna_cells
        if all(cell in frame.index for frame in view_frames.values())
    ]
    if not common_cells:
        raise DataValidationError(
            f"dataset={name}: no common cell IDs across configured modalities"
        )
    rna_df = rna_df.reindex(common_cells)
    view_frames = {
        modality: frame.reindex(common_cells)
        for modality, frame in view_frames.items()
    }
    try:
        label_path = resolve_label_path(data_dir, label_config)
        loaded_labels = load_and_align_labels(
            dataset_name=name,
            label_path=label_path,
            source_cell_ids=source_rna_cells,
            target_cell_ids=common_cells,
            optional_header_tokens=label_config.get("optional_header_tokens", []),
            expected_names=label_config.get("expected_names", []),
        )
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc

    raw_rna = rna_df.to_numpy(dtype=np.float32)
    rna_missing = ~np.isfinite(raw_rna)
    rna = fill_columns(raw_rna, "median")
    preprocessing = "as_loaded"
    if name == "CITE_seq":
        rna = np.maximum(rna, 0.0)
        library = rna.sum(axis=1, keepdims=True)
        rna = np.log1p(rna / np.maximum(library, 1.0) * 1.0e4).astype(np.float32)
        preprocessing = "CP10K_log1p"

    views = {
        modality: frame.to_numpy(dtype=np.float32)
        for modality, frame in view_frames.items()
    }
    view_feature_names_by_view = {
        modality: frame.columns.astype(str).tolist()
        for modality, frame in view_frames.items()
    }
    first_view_name = next(iter(views))
    metadata = {
        "data_dir": str(data_dir),
        "rna_path": str(data_dir / expr_name),
        "view_paths": view_paths,
        "label_path": str(loaded_labels.source_path),
        "label_source": str(loaded_labels.source_path),
        "label_rule": loaded_labels.rule_name,
        "label_map": loaded_labels.id_to_name,
        "label_counts": loaded_labels.counts,
        "n_clusters": loaded_labels.n_clusters,
        "view_feature_names_by_view": view_feature_names_by_view,
        "view_original_feature_names_by_view": view_original_columns_by_view,
        # Compatibility fields for callers that still consume a single view.
        "view_name": first_view_name,
        "view_path": view_paths[first_view_name],
        "view_feature_names": view_feature_names_by_view[first_view_name],
        "view_original_feature_names": view_original_columns_by_view[first_view_name],
        "rna_mapping": rna_map_audit,
        "view_mapping_by_view": view_mapping_by_view,
        "view_mapping": view_mapping_by_view[first_view_name],
        "rna_missing_count": int(rna_missing.sum()),
        "view_missing_count_by_view": {
            modality: int((~np.isfinite(values)).sum())
            for modality, values in views.items()
        },
        "rna_preprocessing": preprocessing,
        "_hgnc_maps": maps,
        "_view_frames": view_frames,
        "_rna_missing_mask": rna_missing,
        "_view_missing_masks": {
            modality: ~np.isfinite(values) for modality, values in views.items()
        },
    }
    dataset = RealDataset(
        name=name,
        rna=rna,
        views=views,
        genes=rna_df.columns.astype(str).tolist(),
        cell_ids=[str(x) for x in common_cells],
        labels=loaded_labels.ids,
        n_clusters=loaded_labels.n_clusters,
        metadata=metadata,
    )
    validate_dataset(dataset)
    logger.info(
        "Loaded %s: RNA=%s views=%s cells=%d genes=%d label_rule=%s "
        "label_counts=%s n_clusters=%d",
        name, dataset.rna.shape,
        {modality: values.shape for modality, values in views.items()},
        len(common_cells), len(dataset.genes), loaded_labels.rule_name,
        loaded_labels.counts, dataset.n_clusters,
    )
    return dataset


def load_pea_sta(data_root: Path | None = None) -> RealDataset:
    del data_root
    return load_real_dataset("PEA_STA", logging.getLogger("run_phase"))


def validate_dataset(dataset: RealDataset) -> None:
    if dataset.rna.ndim != 2 or not dataset.views:
        raise DataValidationError(f"dataset={dataset.name}: expected 2-D RNA and a second view")
    for view_name, view in dataset.views.items():
        if view.ndim != 2 or view.shape[0] != dataset.rna.shape[0]:
            raise DataValidationError(
                f"dataset={dataset.name}: RNA/{view_name} cell dimensions differ"
            )
    if dataset.rna.shape[0] != len(dataset.cell_ids) or dataset.rna.shape[0] != len(dataset.labels):
        raise DataValidationError(f"dataset={dataset.name}: cell/label length mismatch")
    if dataset.rna.shape[1] != len(dataset.genes):
        raise DataValidationError(f"dataset={dataset.name}: gene dimension mismatch")
    if len(set(dataset.cell_ids)) != len(dataset.cell_ids):
        raise DataValidationError(f"dataset={dataset.name}: duplicate cell IDs")
    if len(set(dataset.genes)) != len(dataset.genes):
        raise DataValidationError(f"dataset={dataset.name}: duplicate canonical genes")
    if not np.isfinite(dataset.rna).all():
        raise DataValidationError(f"dataset={dataset.name}: RNA remains non-finite after preprocessing")
    if dataset.rna.shape[0] < 3 or dataset.rna.shape[1] < 2:
        raise DataValidationError(f"dataset={dataset.name}: insufficient cells or genes")
    derived_clusters = int(np.unique(dataset.labels).size)
    if dataset.n_clusters != derived_clusters:
        raise DataValidationError(
            f"dataset={dataset.name}: n_clusters={dataset.n_clusters} does not "
            f"match unique labels={derived_clusters}"
        )
    if dataset.n_clusters < 2:
        raise DataValidationError(f"dataset={dataset.name}: at least two labels are required")
    label_map = dataset.metadata.get("label_map", {})
    if set(label_map) != set(range(dataset.n_clusters)):
        raise DataValidationError(f"dataset={dataset.name}: label_map IDs are not contiguous")
    if dataset.name == "PEA_STA":
        expected = set(DATASET_CONFIG["PEA_STA"]["labels"]["expected_names"])
        actual = set(label_map.values())
        if actual != expected:
            raise DataValidationError(
                f"dataset=PEA_STA: expected labels {sorted(expected)}, got {sorted(actual)}"
            )


def get_view_feature_names(dataset: RealDataset, view_name: str) -> list[str]:
    try:
        return list(dataset.metadata["view_feature_names_by_view"][view_name])
    except KeyError as exc:
        raise DataValidationError(
            f"dataset={dataset.name}: missing feature names for view {view_name!r}"
        ) from exc


# =============================================================================
# 6. PRIOR LOADING AND STRICT VALIDATION
# =============================================================================

def _warn_skip(
    logger: logging.Logger,
    skipped: dict[str, Any],
    key: str,
    reason: str,
    path: Path | str,
    detail: str = "",
) -> None:
    logger.warning("%s prior unavailable: reason=%s path=%s %s", key, reason, path, detail)
    skipped[key] = {"reason": reason, "path": str(path), "detail": detail}


def _load_gene_info(genes: Sequence[str], logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    gz_path = EXAMPLE_DATA_ROOT / "gencode.v38.basic.annotation.gtf.gz"
    plain_path = EXAMPLE_DATA_ROOT / "gencode.v38.basic.annotation.gtf"
    path = gz_path if gz_path.exists() else plain_path
    columns = [
        "gene_id", "chromosome", "TSS", "TES", "strand",
        "chromatin_region_id", "compartment", "local_gene_density", "is_TF",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns), {
            "reason": "file_not_found", "path": str(path), "matched": 0
        }
    wanted = set(genes)
    found: dict[str, dict[str, Any]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = re.search(r'gene_name "([^"]+)"', fields[8])
            if not match:
                continue
            gene = match.group(1)
            if gene not in wanted or gene in found:
                continue
            start, end = int(fields[3]), int(fields[4])
            found[gene] = {
                "gene_id": gene, "chromosome": fields[0],
                "TSS": start if fields[6] == "+" else end,
                "TES": end if fields[6] == "+" else start,
                "strand": fields[6], "chromatin_region_id": np.nan,
                "compartment": np.nan, "local_gene_density": 0.0, "is_TF": 0,
            }
    rows = [found[g] for g in genes if g in found]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        density: dict[str, float] = {}
        for _, group in frame.groupby("chromosome"):
            positions = group["TSS"].to_numpy(dtype=float)
            ids = group["gene_id"].tolist()
            order = np.argsort(positions)
            pos = positions[order]
            for rank, index in enumerate(order):
                left = np.searchsorted(pos, pos[rank] - 500_000, side="left")
                right = np.searchsorted(pos, pos[rank] + 500_000, side="right")
                density[ids[index]] = float(max(0, right - left - 1))
        frame["local_gene_density"] = frame["gene_id"].map(density).fillna(0.0)
    logger.info("Gene annotation matched %d/%d genes from %s", len(frame), len(genes), path)
    return frame, {"path": str(path), "matched": len(frame), "total": len(genes)}


def _load_pathways(genes: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    gene_set = set(genes)
    groups: dict[str, list[str]] = {}
    sources: list[str] = []
    reg_dir = EXAMPLE_DATA_ROOT / "regulons_pathways"
    for path in sorted(reg_dir.glob("*.gmt")):
        sources.append(str(path))
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                members = sorted(set(fields[2:]) & gene_set)
                if len(members) >= 2:
                    groups[f"{path.stem}::{fields[0]}"] = members
    kegg = EXAMPLE_DATA_ROOT / "hsa00001.txt"
    if kegg.exists():
        sources.append(str(kegg))
        by_path: dict[str, set[str]] = defaultdict(set)
        with kegg.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3 and fields[0] in gene_set:
                    by_path[fields[2]].add(fields[0])
        for name, members in by_path.items():
            if len(members) >= 2:
                groups[f"kegg::{name}"] = sorted(members)
    unique: dict[tuple[str, ...], str] = {}
    deduped: dict[str, list[str]] = {}
    for name, members in groups.items():
        key = tuple(sorted(set(members)))
        if key not in unique:
            unique[key] = name
            deduped[name] = list(key)
    return deduped, {"sources": sources, "n_groups": len(deduped)}


def _load_ppi_modules(genes: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    gene_set = set(genes)
    modules: dict[str, list[str]] = {}
    sources: list[str] = []
    corum = EXAMPLE_DATA_ROOT / "corum_humanComplexes.txt"
    if corum.exists():
        sources.append(str(corum))
        frame = pd.read_csv(corum, sep="\t", dtype=str, usecols=lambda c: c in {
            "complex_id", "complex_name", "subunits_gene_name"
        })
        for row in frame.itertuples(index=False):
            record = row._asdict()
            members = sorted(set(str(record.get("subunits_gene_name", "")).split(";")) & gene_set)
            if len(members) >= 2:
                modules[f"corum::{record.get('complex_id', '')}::{record.get('complex_name', '')}"] = members
    ppi = EXAMPLE_DATA_ROOT / "human_ppi.csv"
    ppi_components = 0
    if ppi.exists():
        sources.append(str(ppi))
        header = pd.read_csv(ppi, nrows=0).columns.tolist()
        selected = [header[0]] + [g for g in genes if g in set(header[1:])]
        if len(selected) >= 3:
            sub = pd.read_csv(ppi, usecols=selected, index_col=0).fillna(0.0)
            shared = [g for g in selected[1:] if g in sub.index]
            if shared:
                matrix = sp.csr_matrix(sub.reindex(index=shared, columns=shared, fill_value=0).to_numpy() != 0)
                n_comp, labels = sp.csgraph.connected_components(matrix, directed=False)
                for component in range(n_comp):
                    members = [shared[i] for i in np.where(labels == component)[0]]
                    if len(members) >= 2:
                        modules[f"ppi_component::{component}"] = sorted(members)
                        ppi_components += 1
    unique: dict[tuple[str, ...], str] = {}
    deduped: dict[str, list[str]] = {}
    for name, members in modules.items():
        key = tuple(sorted(set(members)))
        if key not in unique:
            unique[key] = name
            deduped[name] = list(key)
    return deduped, {
        "sources": sources, "n_groups": len(deduped), "ppi_components": ppi_components
    }


def _load_static_grn(genes: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = EXAMPLE_DATA_ROOT / "regulons_pathways" / "collectri_human.tsv"
    required = ["tf", "target", "sign", "confidence"]
    if not path.exists():
        return pd.DataFrame(columns=required), {"path": str(path), "reason": "file_not_found"}
    frame = pd.read_csv(path, sep="\t")
    missing = {"source", "target", "weight"} - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=required), {
            "path": str(path), "reason": "missing_columns", "missing": sorted(missing)
        }
    gene_set = set(genes)
    frame = frame[frame["source"].isin(gene_set) & frame["target"].isin(gene_set)].copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame[np.isfinite(frame["weight"]) & (frame["weight"] != 0)]
    frame["abs_weight"] = frame["weight"].abs()
    if not frame.empty:
        frame = frame.sort_values("abs_weight", ascending=False).drop_duplicates(["source", "target"])
    out = pd.DataFrame({
        "tf": frame["source"].astype(str),
        "target": frame["target"].astype(str),
        "sign": np.sign(frame["weight"]).astype(np.int8),
        "confidence": frame["abs_weight"].astype(np.float32),
    })
    return out.reset_index(drop=True), {
        "path": str(path), "n_edges": len(out),
        "n_positive": int((out["sign"] > 0).sum()),
        "n_negative": int((out["sign"] < 0).sum()),
    }


def _protein_mapping(dataset: RealDataset) -> dict[str, list[str]]:
    if "protein" not in dataset.views:
        return {}
    genes = set(dataset.genes)
    alias = dataset.metadata["_hgnc_maps"]["alias"]
    mapping: dict[str, list[str]] = {}
    cite_multi = {
        "CD3": ["CD3D", "CD3E", "CD3G"], "CD8": ["CD8A", "CD8B"],
        "CD16": ["FCGR3A"], "CD45RA": ["PTPRC"], "CD56": ["NCAM1"],
        "CD11C": ["ITGAX"], "CD10": ["MME"],
    }
    for protein in get_view_feature_names(dataset, "protein"):
        candidates = cite_multi.get(protein.upper())
        if candidates is None:
            canonical = alias.get(protein.upper(), protein)
            candidates = [canonical]
        valid = [g for g in candidates if g in genes]
        if valid:
            mapping[protein] = valid
    return mapping


def load_priors(
    dataset: RealDataset,
    prior_root: Path | None = None,
    *,
    strict: bool,
    logger: logging.Logger | None = None,
) -> PriorBundle:
    del prior_root
    logger = logger or logging.getLogger("run_phase")
    skipped: dict[str, Any] = {}
    gene_info, gene_meta = _load_gene_info(dataset.genes, logger)
    if gene_info.empty:
        _warn_skip(logger, skipped, "gene_annotation", "no_panel_genes_matched",
                   gene_meta.get("path", EXAMPLE_DATA_ROOT))
    pathways, pathway_meta = _load_pathways(dataset.genes)
    if not pathways:
        _warn_skip(logger, skipped, "pathway", "no_valid_members_after_filtering",
                   EXAMPLE_DATA_ROOT / "regulons_pathways")
    ppi_modules, ppi_meta = _load_ppi_modules(dataset.genes)
    if not ppi_modules:
        _warn_skip(logger, skipped, "ppi_corum", "no_valid_members_after_filtering",
                   EXAMPLE_DATA_ROOT / "corum_humanComplexes.txt")
    static_grn, grn_meta = _load_static_grn(dataset.genes)
    if static_grn.empty:
        _warn_skip(logger, skipped, "signed_static_grn",
                   grn_meta.get("reason", "no_panel_edges"),
                   grn_meta.get("path", EXAMPLE_DATA_ROOT))
    else:
        tf_set = set(static_grn["tf"])
        if not gene_info.empty:
            gene_info["is_TF"] = gene_info["gene_id"].isin(tf_set).astype(int)
    lr_candidates = [
        EXAMPLE_DATA_ROOT / "ligand_receptor.csv",
        EXAMPLE_DATA_ROOT / "regulons_pathways" / "ligand_receptor.csv",
    ]
    lr_path = next((p for p in lr_candidates if p.exists()), None)
    if lr_path is None:
        ligand_receptor = pd.DataFrame(columns=["ligand", "receptor"])
        _warn_skip(logger, skipped, "ligand_receptor", "file_not_found",
                   ";".join(str(p) for p in lr_candidates))
    else:
        ligand_receptor = pd.read_csv(lr_path)
        missing = {"ligand", "receptor"} - set(ligand_receptor.columns)
        if missing:
            _warn_skip(logger, skipped, "ligand_receptor", "missing_columns", lr_path,
                       f"missing={sorted(missing)}")
            ligand_receptor = pd.DataFrame(columns=["ligand", "receptor"])

    if gene_info.empty or gene_info["chromatin_region_id"].notna().sum() == 0:
        _warn_skip(logger, skipped, "chromatin_region", "field_unavailable",
                   gene_meta.get("path", EXAMPLE_DATA_ROOT), "field=chromatin_region_id")
    if gene_info.empty or gene_info["compartment"].notna().sum() == 0:
        _warn_skip(logger, skipped, "compartment", "field_unavailable",
                   gene_meta.get("path", EXAMPLE_DATA_ROOT), "field=compartment")

    mapping = _protein_mapping(dataset)
    if "protein" in dataset.views and not mapping:
        _warn_skip(logger, skipped, "protein_to_gene", "no_valid_mapping",
                   EXAMPLE_DATA_ROOT / "hgnc_complete_set.txt")
    metadata = {
        "strict_priors": bool(strict),
        "prior_policy": "validate_warn_and_skip",
        "gene_annotation": gene_meta,
        "pathway": pathway_meta,
        "ppi_corum": ppi_meta,
        "static_grn": grn_meta,
        "protein_mapping_count": len(mapping),
        "skipped_priors": skipped,
    }
    return PriorBundle(
        gene_info=gene_info,
        pathways=pathways,
        ppi_modules=ppi_modules,
        static_grn=static_grn,
        ligand_receptor=ligand_receptor,
        modality_gene_map=mapping,
        metadata=metadata,
    )


# =============================================================================
# 7. PER-CELL GRN
# =============================================================================

def load_or_build_percell_grn(
    dataset: RealDataset,
    priors: PriorBundle,
    cache_path: Path,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("run_phase")
    columns = ["cell_idx", "tf_gene", "target_gene", "weight", "sign"]
    if priors.static_grn.empty:
        logger.warning("Per-cell GRN skipped: signed static GRN is empty")
        return pd.DataFrame(columns=columns)
    gene_idx = {g: i for i, g in enumerate(dataset.genes)}
    rna_z = safe_standardize(dataset.rna)
    tf_names = sorted(set(priors.static_grn["tf"]) & set(dataset.genes))
    activity = np.zeros((len(dataset.cell_ids), len(tf_names)), dtype=np.float32)
    for j, tf in enumerate(tf_names):
        activity[:, j] = np.maximum(rna_z[:, gene_idx[tf]], 0.0)

    if "protein" in dataset.views and priors.modality_gene_map:
        protein = dataset.views["protein"]
        pnames = get_view_feature_names(dataset, "protein")
        for j, tf in enumerate(tf_names):
            cols = [i for i, p in enumerate(pnames) if tf in priors.modality_gene_map.get(p, [])]
            if not cols:
                continue
            values = protein[:, cols]
            observed = np.isfinite(values)
            means = np.divide(
                np.nansum(values, axis=1), np.maximum(observed.sum(axis=1), 1),
                where=np.ones((values.shape[0],), dtype=bool),
            )
            pz = safe_standardize(means[:, None])[:, 0]
            has = observed.any(axis=1)
            activity[has, j] = np.maximum(pz[has], 0.0)

    groups = {tf: frame for tf, frame in priors.static_grn.groupby("tf")}
    rows: list[tuple[int, str, str, float, int]] = []
    cap_tfs = dataset.name in {"SCoPE2", "CITE_seq"} and len(tf_names) > DEFAULT_ACTIVE_TF_TOP_K
    for cell in range(len(dataset.cell_ids)):
        if cap_tfs:
            selected_idx = np.argsort(-activity[cell])[:DEFAULT_ACTIVE_TF_TOP_K]
        else:
            selected_idx = np.arange(len(tf_names))
        for tf_j in selected_idx:
            tf = tf_names[int(tf_j)]
            tf_activity = float(activity[cell, tf_j])
            if tf_activity <= 0 or tf not in groups:
                continue
            frame = groups[tf]
            candidates: list[tuple[float, str, int]] = []
            for row in frame.itertuples(index=False):
                target = str(row.target)
                if target not in gene_idx:
                    continue
                availability = max(float(rna_z[cell, gene_idx[target]]), 0.0)
                magnitude = float(row.confidence) * tf_activity * availability
                if magnitude > 0:
                    candidates.append((magnitude, target, int(row.sign)))
            for sign, top_k in ((1, DEFAULT_GRN_POSITIVE_TOP_K), (-1, DEFAULT_GRN_NEGATIVE_TOP_K)):
                signed = sorted((x for x in candidates if x[2] == sign), reverse=True)[:top_k]
                rows.extend((cell, tf, target, sign * magnitude, sign)
                            for magnitude, target, sign in signed)
    result = pd.DataFrame(rows, columns=columns)
    logger.info(
        "Per-cell GRN: rows=%d positive=%d negative=%d TF_policy=%s",
        len(result), int((result["sign"] > 0).sum()) if len(result) else 0,
        int((result["sign"] < 0).sum()) if len(result) else 0,
        "top32_activity" if cap_tfs else "all_valid_tfs",
    )
    if result.empty:
        logger.warning("No positive or negative per-cell GRN edges survived filtering")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_pickle(cache_path)
    return result


# =============================================================================
# 8. NODE INDEX AND FEATURES
# =============================================================================

def build_node_index(dataset: RealDataset, priors: PriorBundle) -> NodeIndex:
    offset = 0
    cell = {name: offset + i for i, name in enumerate(dataset.cell_ids)}
    offset += len(cell)
    gene = {name: offset + i for i, name in enumerate(dataset.genes)}
    offset += len(gene)
    proteins = (
        get_view_feature_names(dataset, "protein")
        if "protein" in dataset.views
        else []
    )
    protein = {name: offset + i for i, name in enumerate(proteins)}
    offset += len(protein)
    tf_names = sorted(set(priors.static_grn["tf"]) & set(dataset.genes))
    tf = {name: offset + i for i, name in enumerate(tf_names)}
    offset += len(tf)
    return NodeIndex(cell=cell, gene=gene, protein=protein, tf=tf, n_nodes=offset)


def build_node_features(
    dataset: RealDataset,
    priors: PriorBundle,
    node_index: NodeIndex,
    *,
    dc: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rna = dataset.rna.astype(np.float32)
    mu = rna.mean(axis=0).astype(np.float32)
    sigma = (rna.std(axis=0) + EPS).astype(np.float32)
    m_rna = ((rna - mu) / sigma).astype(np.float32)
    pca = PCA(n_components=dc, random_state=seed)
    cell_rna = pca.fit_transform(m_rna).astype(np.float32)
    w = pca.components_.astype(np.float32)
    m_graph = m_rna.copy()

    gene_pos = {g: i for i, g in enumerate(dataset.genes)}
    aligned_count = np.zeros(len(dataset.genes), dtype=np.float32)
    view_z_by_view: dict[str, np.ndarray] = {}
    for view_name, raw_view in dataset.views.items():
        view_mode = "min" if view_name == "protein" else "mean"
        view_z = safe_standardize(fill_columns(raw_view, view_mode))
        view_z_by_view[view_name] = view_z
        vnames = get_view_feature_names(dataset, view_name)
        if view_name == "protein":
            for feature_idx, protein in enumerate(vnames):
                for gene in priors.modality_gene_map.get(protein, []):
                    if gene in gene_pos:
                        m_graph[:, gene_pos[gene]] += view_z[:, feature_idx]
                        aligned_count[gene_pos[gene]] += 1
        else:
            for feature_idx, gene in enumerate(vnames):
                if gene in gene_pos:
                    m_graph[:, gene_pos[gene]] += view_z[:, feature_idx]
                    aligned_count[gene_pos[gene]] += 1
    used = aligned_count > 0
    m_graph[:, used] /= (1.0 + aligned_count[used])[None, :]

    features = np.zeros((node_index.n_nodes, dc), dtype=np.float32)
    features[:len(dataset.cell_ids)] = cell_rna
    gene_start = len(dataset.cell_ids)
    features[gene_start:gene_start + len(dataset.genes)] = w.T

    if not priors.gene_info.empty:
        info = priors.gene_info.set_index("gene_id")
        annotations = np.zeros((len(dataset.genes), 8), dtype=np.float32)
        chrom_tokens = {c: i + 1 for i, c in enumerate(sorted(info["chromosome"].dropna().unique()))}
        for i, gene in enumerate(dataset.genes):
            if gene not in info.index:
                continue
            row = info.loc[gene]
            annotations[i] = [
                chrom_tokens.get(row["chromosome"], 0) / max(len(chrom_tokens), 1),
                np.log1p(abs(float(row["TSS"]))) / 25.0,
                np.log1p(abs(float(row["TES"]))) / 25.0,
                1.0 if row["strand"] == "+" else -1.0,
                float(row.get("local_gene_density", 0.0)),
                float(row.get("is_TF", 0)),
                1.0 if row.get("compartment") == "A" else 0.0,
                1.0 if row.get("compartment") == "B" else 0.0,
            ]
        annotations = safe_standardize(annotations)
        features[gene_start:gene_start + len(dataset.genes), :min(8, dc)] += annotations[:, :min(8, dc)]

    cell_basis = cell_rna
    protein_names = (
        get_view_feature_names(dataset, "protein")
        if "protein" in dataset.views
        else []
    )
    protein_z = view_z_by_view.get("protein")
    for p_idx, protein in enumerate(protein_names):
        node = node_index.protein.get(protein)
        if node is None:
            continue
        profile = protein_z[:, p_idx]
        features[node] = (profile[:, None] * cell_basis).mean(axis=0)
        mapped = priors.modality_gene_map.get(protein, [])
        if mapped:
            features[node] += np.mean([features[node_index.gene[g]] for g in mapped], axis=0)

    for tf, node in node_index.tf.items():
        value = features[node_index.gene[tf]].copy()
        targets = priors.static_grn.loc[priors.static_grn["tf"] == tf, "target"].tolist()
        valid = [features[node_index.gene[g]] for g in targets if g in node_index.gene]
        if valid:
            value += np.mean(valid, axis=0)
        mapped_proteins = [p for p, gs in priors.modality_gene_map.items() if tf in gs and p in node_index.protein]
        if mapped_proteins:
            value += np.mean([features[node_index.protein[p]] for p in mapped_proteins], axis=0)
        features[node] = value

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if features.shape != (node_index.n_nodes, dc):
        raise DataValidationError(f"node feature shape {features.shape} != {(node_index.n_nodes, dc)}")
    return features, {
        "M_graph": m_graph.astype(np.float32), "M_rna": m_rna,
        "pca_init": w, "ch_target": (m_rna @ w.T).astype(np.float32),
        "mu": mu, "sigma": sigma,
        "view_z_by_view": view_z_by_view,
        "view_z": next(iter(view_z_by_view.values())),
    }


# =============================================================================
# 9. DIRECTED HYPEREDGES
# =============================================================================

def _edge_add(
    edges: list[Hyperedge],
    audits: dict[tuple[str, str], EdgeAudit],
    channel: str,
    name: str,
    *,
    members: Iterable[int] | None = None,
    tail: Iterable[int] | None = None,
    head: Iterable[int] | None = None,
    weight: float = 1.0,
) -> None:
    audit = audits.setdefault((channel, name), EdgeAudit())
    audit.candidate_count += 1
    mem = tuple(sorted(set(int(x) for x in members))) if members is not None else None
    tl = tuple(sorted(set(int(x) for x in tail))) if tail is not None else None
    hd = tuple(sorted(set(int(x) for x in head))) if head is not None else None
    valid = np.isfinite(weight)
    if channel == "directed":
        valid = valid and bool(tl) and bool(hd) and (len(tl) + len(hd) >= 2)
    else:
        valid = valid and mem is not None and len(mem) >= 2
    if not valid:
        audit.dropped_invalid_count += 1
        return
    edges.append(Hyperedge(name, mem, tl, hd, float(weight)))


def _position_groups(priors: PriorBundle, window: float) -> list[list[str]]:
    if priors.gene_info.empty:
        return []
    groups: list[list[str]] = []
    for _, frame in priors.gene_info.groupby("chromosome"):
        frame = frame.sort_values("TSS")
        genes = frame["gene_id"].tolist()
        pos = frame["TSS"].to_numpy(dtype=float)
        for i in range(len(genes)):
            right = int(np.searchsorted(pos, pos[i] + window, side="right"))
            members = genes[i:right]
            if len(members) >= 2:
                groups.append(members)
    return groups


def build_directed_hyperedges(
    dataset: RealDataset,
    priors: PriorBundle,
    percell_grn: pd.DataFrame,
    node_index: NodeIndex,
    args: argparse.Namespace,
    audits: dict[tuple[str, str], EdgeAudit] | None = None,
) -> list[Hyperedge]:
    del args
    audits = audits if audits is not None else {}
    edges: list[Hyperedge] = []
    z = safe_standardize(dataset.rna)
    for cell in range(len(dataset.cell_ids)):
        top = [i for i in np.argsort(-z[cell])[:DEFAULT_RNA_TOP_K] if z[cell, i] > 0]
        genes = [node_index.gene[dataset.genes[i]] for i in top]
        _edge_add(edges, audits, "directed", "rna_inject", tail=[cell], head=genes)
        _edge_add(edges, audits, "directed", "rna_readout", tail=genes, head=[cell])

    if "protein" in dataset.views:
        raw = dataset.views["protein"]
        view_z = safe_standardize(fill_columns(raw, "min"))
        pnames = get_view_feature_names(dataset, "protein")
        for cell in range(len(dataset.cell_ids)):
            top = [i for i in np.argsort(-view_z[cell])[:DEFAULT_PROTEIN_TOP_K]
                   if np.isfinite(raw[cell, i]) and view_z[cell, i] > 0]
            proteins = [node_index.protein[pnames[i]] for i in top if pnames[i] in node_index.protein]
            _edge_add(edges, audits, "directed", "prot_inject", tail=[cell], head=proteins)
            _edge_add(edges, audits, "directed", "prot_readout", tail=proteins, head=[cell])
        for protein, genes in priors.modality_gene_map.items():
            if protein not in node_index.protein:
                continue
            for gene in genes:
                if gene in node_index.gene:
                    _edge_add(edges, audits, "directed", "translation",
                              tail=[node_index.gene[gene]], head=[node_index.protein[protein]])

    for tf, tf_node in node_index.tf.items():
        tail = [node_index.gene[tf]]
        tail.extend(node_index.protein[p] for p, gs in priors.modality_gene_map.items()
                    if tf in gs and p in node_index.protein)
        _edge_add(edges, audits, "directed", "tf_activation", tail=tail, head=[tf_node])
        frame = priors.static_grn[priors.static_grn["tf"] == tf]
        targets = [node_index.gene[g] for g in frame["target"] if g in node_index.gene]
        _edge_add(edges, audits, "directed", "reg_cascade", tail=[tf_node], head=targets,
                  weight=math.sqrt(max(len(targets), 1)))
        for sign, name in ((1, "grn_stim"), (-1, "grn_inhib")):
            sub = frame[frame["sign"] == sign]
            heads = [node_index.gene[g] for g in sub["target"] if g in node_index.gene]
            confidence = float(sub["confidence"].mean()) if len(sub) else 1.0
            _edge_add(edges, audits, "directed", name, tail=[tf_node], head=heads,
                      weight=sign * confidence)

    for members in priors.ppi_modules.values():
        nodes = [node_index.gene[g] for g in members if g in node_index.gene]
        _edge_add(edges, audits, "directed", "module_coop", tail=nodes, head=nodes,
                  weight=math.sqrt(max(len(nodes), 1)))

    if not priors.gene_info.empty:
        for _, frame in priors.gene_info.dropna(subset=["chromatin_region_id"]).groupby("chromatin_region_id"):
            nodes = [node_index.gene[g] for g in frame["gene_id"] if g in node_index.gene]
            _edge_add(edges, audits, "directed", "chromatin_region", tail=nodes, head=nodes)
        for comp in ("A", "B"):
            frame = priors.gene_info[priors.gene_info["compartment"] == comp]
            nodes = [node_index.gene[g] for g in frame["gene_id"] if g in node_index.gene]
            _edge_add(edges, audits, "directed", f"compartment_{comp}", tail=nodes, head=nodes)
        for members in _position_groups(priors, 200_000):
            nodes = [node_index.gene[g] for g in members if g in node_index.gene]
            _edge_add(edges, audits, "directed", "proximity_200kb", tail=nodes, head=nodes)
        for (_, strand), frame in priors.gene_info.groupby(["chromosome", "strand"]):
            genes = frame.sort_values("TSS")["gene_id"].tolist()
            for a, b in zip(genes, genes[1:]):
                if a in node_index.gene and b in node_index.gene:
                    nodes = [node_index.gene[a], node_index.gene[b]]
                    _edge_add(edges, audits, "directed", "same_strand_adj", tail=nodes, head=nodes)

    if not percell_grn.empty:
        for (cell, tf, sign), frame in percell_grn.groupby(["cell_idx", "tf_gene", "sign"]):
            if tf not in node_index.tf:
                continue
            heads = [node_index.gene[g] for g in frame["target_gene"] if g in node_index.gene]
            magnitude = float(frame["weight"].abs().mean())
            if int(sign) > 0:
                name, readout, weight = "grn_activate", "grn_activate_readout", magnitude
            else:
                name, readout, weight = "grn_repress", "grn_repress_readout", -magnitude
            _edge_add(edges, audits, "directed", name,
                      tail=[int(cell), node_index.tf[tf]], head=heads, weight=weight)
            _edge_add(edges, audits, "directed", readout,
                      tail=heads, head=[int(cell)], weight=weight)
    return edges


# =============================================================================
# 10. UNDIRECTED HYPEREDGES
# =============================================================================

def _add_knn_edges(
    edges: list[Hyperedge],
    audits: dict[tuple[str, str], EdgeAudit],
    features: np.ndarray,
    name: str,
    k: int,
) -> None:
    n = features.shape[0]
    if n < 2:
        return
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    indices = nn_model.fit(features).kneighbors(features, return_distance=False)
    for cell in range(n):
        _edge_add(edges, audits, "undirected", name, members=indices[cell].tolist())


def build_undirected_hyperedges(
    dataset: RealDataset,
    priors: PriorBundle,
    node_index: NodeIndex,
    args: argparse.Namespace,
    audits: dict[tuple[str, str], EdgeAudit] | None = None,
) -> list[Hyperedge]:
    del args
    audits = audits if audits is not None else {}
    edges: list[Hyperedge] = []
    z = safe_standardize(dataset.rna)
    for cell in range(len(dataset.cell_ids)):
        top = [i for i in np.argsort(-z[cell])[:DEFAULT_RNA_TOP_K] if z[cell, i] > 0]
        members = [cell] + [node_index.gene[dataset.genes[i]] for i in top]
        _edge_add(edges, audits, "undirected", "RNA_obs", members=members)

    standardized_views: dict[str, np.ndarray] = {}
    for view_name, raw in dataset.views.items():
        view_z = safe_standardize(
            fill_columns(raw, "min" if view_name == "protein" else "mean")
        )
        standardized_views[view_name] = view_z
        vnames = get_view_feature_names(dataset, view_name)
        obs_name = "prot_obs" if view_name == "protein" else "view2_obs"
        for cell in range(len(dataset.cell_ids)):
            top = [
                i
                for i in np.argsort(-np.abs(view_z[cell]))[:DEFAULT_PROTEIN_TOP_K]
                if np.isfinite(raw[cell, i])
            ]
            if view_name == "protein":
                nodes = [
                    node_index.protein[vnames[i]]
                    for i in top
                    if vnames[i] in node_index.protein
                ]
            else:
                nodes = [
                    node_index.gene[vnames[i]]
                    for i in top
                    if vnames[i] in node_index.gene
                ]
            _edge_add(
                edges, audits, "undirected", obs_name, members=[cell] + nodes
            )

    if not priors.gene_info.empty:
        for _, frame in priors.gene_info.dropna(subset=["chromatin_region_id"]).groupby("chromatin_region_id"):
            nodes = [node_index.gene[g] for g in frame["gene_id"] if g in node_index.gene]
            _edge_add(edges, audits, "undirected", "chromatin_region", members=nodes)
        for comp in ("A", "B"):
            frame = priors.gene_info[priors.gene_info["compartment"] == comp]
            nodes = [node_index.gene[g] for g in frame["gene_id"] if g in node_index.gene]
            _edge_add(edges, audits, "undirected", f"compartment_{comp}", members=nodes)
        for window, name in ((200_000, "proximity_200kb"), (500_000, "proximity_500kb")):
            for members in _position_groups(priors, window):
                nodes = [node_index.gene[g] for g in members if g in node_index.gene]
                _edge_add(edges, audits, "undirected", name, members=nodes)
        for (_, strand), frame in priors.gene_info.groupby(["chromosome", "strand"]):
            genes = frame.sort_values("TSS")["gene_id"].tolist()
            for a, b in zip(genes, genes[1:]):
                if a in node_index.gene and b in node_index.gene:
                    _edge_add(edges, audits, "undirected", "same_strand_adj",
                              members=[node_index.gene[a], node_index.gene[b]])

    for members in priors.pathways.values():
        _edge_add(edges, audits, "undirected", "pathway_module",
                  members=[node_index.gene[g] for g in members if g in node_index.gene])
    for members in priors.ppi_modules.values():
        _edge_add(edges, audits, "undirected", "ppi_module",
                  members=[node_index.gene[g] for g in members if g in node_index.gene])
    for protein, genes in priors.modality_gene_map.items():
        if protein not in node_index.protein:
            continue
        mapped = [node_index.gene[g] for g in genes if g in node_index.gene]
        if len(mapped) >= 2:
            _edge_add(edges, audits, "undirected", "mech_bridge",
                      members=[node_index.protein[protein]] + mapped)
        if mapped:
            _edge_add(edges, audits, "undirected", "mvmod",
                      members=[node_index.protein[protein]] + mapped)

    dc = max(2, min(50, dataset.rna.shape[0] - 1, dataset.rna.shape[1]))
    rna_space = PCA(dc, random_state=0).fit_transform(safe_standardize(dataset.rna))
    _add_knn_edges(edges, audits, rna_space, "rna_knn", DEFAULT_RNA_KNN_K)
    for view_name, raw in dataset.views.items():
        view_z = standardized_views[view_name]
        view_dc = max(2, min(50, raw.shape[0] - 1, raw.shape[1]))
        view_space = PCA(view_dc, random_state=0).fit_transform(view_z)
        _add_knn_edges(
            edges, audits, view_space,
            "adt_knn" if view_name == "protein" else "view2_knn",
            DEFAULT_PROTEIN_KNN_K,
        )

    if not priors.ligand_receptor.empty:
        gene_pos = {g: i for i, g in enumerate(dataset.genes)}
        for row in priors.ligand_receptor.itertuples(index=False):
            ligand, receptor = str(row.ligand), str(row.receptor)
            if ligand not in gene_pos or receptor not in gene_pos:
                continue
            send = np.argsort(-z[:, gene_pos[ligand]])[:DEFAULT_RNA_TOP_K]
            receive = np.argsort(-z[:, gene_pos[receptor]])[:DEFAULT_RNA_TOP_K]
            members = sorted(set(send.tolist() + receive.tolist()))
            weight = float(
                np.maximum(z[send, gene_pos[ligand]], 0).mean()
                * np.maximum(z[receive, gene_pos[receptor]], 0).mean()
            )
            _edge_add(edges, audits, "undirected", "cci", members=members, weight=weight)
    return edges


# =============================================================================
# 11. HYPERGRAPH COMPILATION
# =============================================================================

def deduplicate_hyperedges(
    edges: list[Hyperedge],
    audits: dict[tuple[str, str], EdgeAudit] | None = None,
    channel: str | None = None,
) -> list[Hyperedge]:
    unique: dict[tuple[Any, ...], Hyperedge] = {}
    for edge in edges:
        key = (edge.name, edge.members, edge.tail, edge.head)
        previous = unique.get(key)
        if previous is None:
            unique[key] = edge
        else:
            if audits is not None and channel is not None:
                audits.setdefault((channel, edge.name), EdgeAudit()).dropped_duplicate_count += 1
            if abs(edge.weight) > abs(previous.weight):
                unique[key] = edge
    return list(unique.values())


def validate_required_edges(
    directed: Sequence[Hyperedge],
    undirected: Sequence[Hyperedge],
    audits: dict[tuple[str, str], EdgeAudit],
    logger: logging.Logger,
) -> None:
    for name in DIRECTED_EDGE_TYPES:
        count = sum(edge.name == name for edge in directed)
        if count == 0:
            audit = audits.setdefault(("directed", name), EdgeAudit(reason="no_valid_candidates"))
            audit.reason = audit.reason or "no_valid_candidates"
            logger.warning("Directed hyperedge type %s has zero valid edges: %s", name, audit.reason)
    for name in UNDIRECTED_EDGE_TYPES:
        count = sum(edge.name == name for edge in undirected)
        if count == 0:
            audit = audits.setdefault(("undirected", name), EdgeAudit(reason="no_valid_candidates"))
            audit.reason = audit.reason or "no_valid_candidates"
            logger.warning("Undirected hyperedge type %s has zero valid edges: %s", name, audit.reason)
    if not directed:
        raise HyperedgeValidationError("no valid directed hyperedges remain")
    if not undirected:
        raise HyperedgeValidationError("no valid undirected hyperedges remain")


def compile_hypergraph(
    node_features: np.ndarray,
    directed_edges: list[Hyperedge],
    undirected_edges: list[Hyperedge],
    node_index: NodeIndex,
    device: torch.device,
) -> dict[str, Any]:
    del device
    n = node_index.n_nodes
    if node_features.shape[0] != n or not np.isfinite(node_features).all():
        raise HyperedgeValidationError("node features are invalid or do not match n_nodes")

    def compile_directed() -> dict[str, Any]:
        tr: list[int] = []
        tc: list[int] = []
        hr: list[int] = []
        hc: list[int] = []
        weights: list[float] = []
        etypes: list[int] = []
        names: list[str] = []
        name_to_id: dict[str, int] = {}
        counts: Counter[str] = Counter()
        for e, edge in enumerate(directed_edges):
            if edge.name not in name_to_id:
                name_to_id[edge.name] = len(names)
                names.append(edge.name)
            tr.extend(edge.tail or ())
            tc.extend([e] * len(edge.tail or ()))
            hr.extend(edge.head or ())
            hc.extend([e] * len(edge.head or ()))
            weights.append(edge.weight)
            etypes.append(name_to_id[edge.name])
            counts[edge.name] += 1
        h_tail = sp.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=(n, len(directed_edges)))
        h_head = sp.csr_matrix((np.ones(len(hr), np.float32), (hr, hc)), shape=(n, len(directed_edges)))
        return {
            "H_tail": h_tail, "H_head": h_head, "W": np.asarray(weights, np.float32),
            "etype": np.asarray(etypes, np.int64), "et_names": names,
            "n_types": len(names), "e": len(directed_edges), "cnt": dict(counts),
        }

    def compile_undirected() -> dict[str, Any]:
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        etypes: list[int] = []
        names: list[str] = []
        name_to_id: dict[str, int] = {}
        counts: Counter[str] = Counter()
        for e, edge in enumerate(undirected_edges):
            if edge.name not in name_to_id:
                name_to_id[edge.name] = len(names)
                names.append(edge.name)
            rows.extend(edge.members or ())
            cols.extend([e] * len(edge.members or ()))
            weights.append(edge.weight)
            etypes.append(name_to_id[edge.name])
            counts[edge.name] += 1
        h = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(undirected_edges)))
        return {
            "H": h, "W": np.asarray(weights, np.float32),
            "etype": np.asarray(etypes, np.int64), "et_names": names,
            "n_types": len(names), "e": len(undirected_edges), "cnt": dict(counts),
        }

    return {"directed": compile_directed(), "undirected": compile_undirected()}


def build_edge_summary(
    directed: Sequence[Hyperedge],
    undirected: Sequence[Hyperedge],
    audits: dict[tuple[str, str], EdgeAudit],
    n_nodes: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for channel, registry, edges in (
        ("directed", DIRECTED_EDGE_TYPES, directed),
        ("undirected", UNDIRECTED_EDGE_TYPES, undirected),
    ):
        for name in registry:
            selected = [edge for edge in edges if edge.name == name]
            sizes = np.asarray([
                len(edge.members or ()) if channel == "undirected"
                else len(set((edge.tail or ()) + (edge.head or ())))
                for edge in selected
            ], dtype=float)
            weights = np.asarray([edge.weight for edge in selected], dtype=float)
            covered: set[int] = set()
            for edge in selected:
                covered.update(edge.members or ())
                covered.update(edge.tail or ())
                covered.update(edge.head or ())
            audit = audits.get((channel, name), EdgeAudit())
            rows.append({
                "edge_type": name, "channel": channel, "n_edges": len(selected),
                "min_size": float(sizes.min()) if len(sizes) else 0,
                "mean_size": float(sizes.mean()) if len(sizes) else 0,
                "median_size": float(np.median(sizes)) if len(sizes) else 0,
                "max_size": float(sizes.max()) if len(sizes) else 0,
                "min_weight": float(weights.min()) if len(weights) else 0,
                "mean_weight": float(weights.mean()) if len(weights) else 0,
                "max_weight": float(weights.max()) if len(weights) else 0,
                "node_coverage": len(covered) / max(n_nodes, 1),
                "candidate_count": audit.candidate_count,
                "dropped_duplicate_count": audit.dropped_duplicate_count,
                "dropped_invalid_count": audit.dropped_invalid_count,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 12. SHARED HYPERPHASE MODEL API
# =============================================================================

# The model, criterion, and optimizer groups are defined in phasehyper/model.py.

# =============================================================================
# 13. TRAINING
# =============================================================================

def train_model(
    model: HyperPhaseModel,
    criterion: SetCriterion,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    comp_indicator: torch.Tensor,
    epochs: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor], int, float]:
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        model_output = model(
            tensors["M_graph"], tensors["gf"], tensors["ch_target"]
        )
        loss, loss_terms = criterion(
            model=model,
            model_output=model_output,
            gene_projection=tensors["W"],
            compartment_indicator=comp_indicator,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
        optimizer.step()
        row = {
            "epoch": epoch,
            "loss": float(loss_terms["total"].item()),
            "cyc_comp": float(loss_terms["cyc_comp"].item()),
            "barlow": float(loss_terms["barlow"].item()),
            "compartment": float(loss_terms["compartment"].item()),
            "orthogonality": float(loss_terms["orthogonality"].item()),
            "info_nce": float(loss_terms["info_nce"].item()),
            "gate_regularization": float(
                loss_terms["gate_regularization"].item()
            ),
            "phase_cosine": float(loss_terms["phase_cosine"].item()),
            "asym_scale": float(model.asym_scale.item()),
        }
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            logger.info(
                "epoch=%d loss=%.6f cyc=%.6f barlow=%.6f comp=%.6f "
                "ortho=%.6f nce=%.6f",
                epoch, row["loss"], row["cyc_comp"], row["barlow"],
                row["compartment"], row["orthogonality"], row["info_nce"],
            )
    return pd.DataFrame(history), best_state, best_epoch, best_loss


# =============================================================================
# 14. EVALUATION AND OUTPUT
# =============================================================================

def evaluate_and_save(
    model: HyperPhaseModel,
    tensors: dict[str, torch.Tensor],
    feature_data: dict[str, np.ndarray],
    dataset: RealDataset,
    output_dir: Path,
    best_epoch: int,
    best_loss: float,
    edge_names: Sequence[str],
    evaluation_seed: int = 0,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, _, xa_dc, xb_dc = model(
            tensors["M_graph"], tensors["gf"], tensors["ch_target"]
        )
    xa = xa_dc.cpu().numpy()
    xb = xb_dc.cpu().numpy()
    w = feature_data["pca_init"]
    mu, sigma = feature_data["mu"], feature_data["sigma"]
    phase_a = (xa @ w) * sigma + mu / 2.0
    phase_b = (xb @ w) * sigma + mu / 2.0
    correction = 0.5 * (dataset.rna - phase_a - phase_b)
    phase_a = (phase_a + correction).astype(np.float32)
    phase_b = (phase_b + correction).astype(np.float32)
    if np.linalg.norm(phase_a, axis=1).mean() > np.linalg.norm(phase_b, axis=1).mean():
        phase_a, phase_b = phase_b, phase_a
    relative_error = float(
        np.linalg.norm(phase_a + phase_b - dataset.rna)
        / max(np.linalg.norm(dataset.rna), 1e-12)
    )
    if relative_error >= 1e-6:
        raise RuntimeError(f"phase-sum relative error {relative_error} >= 1e-6")

    axes = {"index": pd.Index(dataset.cell_ids, name="cell_id"), "columns": dataset.genes}
    pd.DataFrame(phase_a, **axes).to_csv(output_dir / "phase_A.csv", float_format="%.8g")
    pd.DataFrame(phase_b, **axes).to_csv(output_dir / "phase_B.csv", float_format="%.8g")
    cell_h = model.last_fused.cpu().numpy()
    pd.DataFrame(
        cell_h, index=axes["index"], columns=[f"h_{i}" for i in range(cell_h.shape[1])]
    ).to_csv(output_dir / "cell_h.csv", float_format="%.8g")

    metrics: dict[str, Any] = evaluate_phase_model(
        raw_rna=dataset.rna,
        cell_embedding=cell_h,
        phase_a=phase_a,
        phase_b=phase_b,
        labels=dataset.labels,
        n_clusters=dataset.n_clusters,
        seed=evaluation_seed,
    )
    metrics["best_epoch"] = best_epoch
    metrics["final_loss"] = best_loss
    save_metrics_json(output_dir / "metrics.json", metrics)
    gates = torch.sigmoid(model.causal_ch.type_logit).detach().cpu().numpy()
    pd.DataFrame({
        "channel": "directed", "edge_type": list(edge_names),
        "gate": gates[:len(edge_names)],
    }).to_csv(output_dir / "edge_gates.csv", index=False)
    return metrics


def save_visualization_inputs(
    *,
    dataset: RealDataset,
    priors: PriorBundle,
    directed_edges: Sequence[Hyperedge],
    undirected_edges: Sequence[Hyperedge],
    node_index: NodeIndex,
    output_dir: Path,
) -> None:
    """Persist aligned metadata and real structural inputs needed for redraws."""
    label_map = dataset.metadata.get("label_map", {})
    pd.DataFrame({
        "cell_id": dataset.cell_ids,
        "label_id": dataset.labels,
        "label_name": [str(label_map.get(int(label), label)) for label in dataset.labels],
    }).to_csv(output_dir / "cell_metadata.csv", index=False)

    view_rows: list[dict[str, Any]] = []
    view_paths = dataset.metadata.get("view_paths", {})
    for view_name, values in dataset.views.items():
        missing_count = int((~np.isfinite(values)).sum())
        view_rows.append({
            "view_name": view_name,
            "path": view_paths.get(view_name, ""),
            "n_cells": int(values.shape[0]),
            "n_features": int(values.shape[1]),
            "missing_count": missing_count,
            "missing_rate": missing_count / max(int(values.size), 1),
            "feature_names": json.dumps(
                get_view_feature_names(dataset, view_name), ensure_ascii=False
            ),
        })
    pd.DataFrame(view_rows).to_csv(output_dir / "view_metadata.csv", index=False)

    annotation = priors.gene_info.copy()
    if annotation.empty:
        annotation = pd.DataFrame({"gene_id": dataset.genes})
    annotation.to_csv(output_dir / "gene_annotation.csv", index=False)

    def module_frame(modules: dict[str, list[str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"module": module, "gene": gene}
            for module, members in modules.items()
            for gene in members
        ], columns=["module", "gene"])

    module_frame(priors.pathways).to_csv(
        output_dir / "pathway_membership.csv", index=False
    )
    module_frame(priors.ppi_modules).to_csv(
        output_dir / "ppi_membership.csv", index=False
    )

    gene_by_node = {node: gene for gene, node in node_index.gene.items()}
    membership_rows: list[dict[str, Any]] = []
    for channel, edges in (
        ("directed", directed_edges),
        ("undirected", undirected_edges),
    ):
        for edge_index, edge in enumerate(edges):
            nodes = set(edge.members or ())
            nodes.update(edge.tail or ())
            nodes.update(edge.head or ())
            for node in nodes:
                gene = gene_by_node.get(int(node))
                if gene is None:
                    continue
                membership_rows.append({
                    "gene": gene,
                    "channel": channel,
                    "edge_type": edge.name,
                    "edge_id": f"{channel}:{edge_index}",
                    "edge_weight": float(edge.weight),
                })
    pd.DataFrame(
        membership_rows,
        columns=["gene", "channel", "edge_type", "edge_id", "edge_weight"],
    ).to_csv(output_dir / "hyperedge_membership.csv", index=False)


# =============================================================================
# 15. MAIN
# =============================================================================

def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_root / args.dataset / f"seed_{args.seed}"
    logger = setup_logging(output_dir)
    logger.info(
        "dataset=%s seed=%d epochs=%d device=%s example_data=%s",
        args.dataset, args.seed, args.epochs, device, EXAMPLE_DATA_ROOT,
    )
    if not EXAMPLE_DATA_ROOT.exists():
        raise DataValidationError(f"example_data root not found: {EXAMPLE_DATA_ROOT}")

    config = {
        "dataset": args.dataset, "seed": args.seed, "epochs": args.epochs,
        "strict_priors": args.strict_priors, "example_data_root": str(EXAMPLE_DATA_ROOT),
        "output_root": str(args.output_root), "device": str(device),
        "dc": DEFAULT_DC, "hidden": DEFAULT_HIDDEN, "dropout": DEFAULT_DROPOUT,
        "rna_top_k": DEFAULT_RNA_TOP_K, "protein_top_k": DEFAULT_PROTEIN_TOP_K,
        "grn_positive_top_k": DEFAULT_GRN_POSITIVE_TOP_K,
        "grn_negative_top_k": DEFAULT_GRN_NEGATIVE_TOP_K,
        "active_tf_top_k": DEFAULT_ACTIVE_TF_TOP_K,
        "rna_knn_k": DEFAULT_RNA_KNN_K, "protein_knn_k": DEFAULT_PROTEIN_KNN_K,
        "w_comp": DEFAULT_W_COMP, "w_ortho": DEFAULT_W_ORTHO,
        "w_gate": DEFAULT_W_GATE, "w_nce": DEFAULT_W_NCE,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "prior_policy": "validate_warn_and_skip",
    }
    save_json(output_dir / "config.json", config)

    dataset = load_real_dataset(args.dataset, logger)
    save_json(output_dir / "data_summary.json", {
        "dataset": dataset.name, "rna_shape": dataset.rna.shape,
        "view_shapes": {k: v.shape for k, v in dataset.views.items()},
        "n_cells": len(dataset.cell_ids), "n_genes": len(dataset.genes),
        "label_source": dataset.metadata["label_source"],
        "label_rule": dataset.metadata["label_rule"],
        "label_counts": dataset.metadata["label_counts"],
        "label_map": dataset.metadata["label_map"],
        "n_clusters": dataset.n_clusters, "metadata": dataset.metadata,
    })
    priors = load_priors(dataset, strict=args.strict_priors, logger=logger)
    save_json(output_dir / "prior_summary.json", priors.metadata)
    save_json(output_dir / "skipped_priors.json", priors.metadata["skipped_priors"])

    percell = load_or_build_percell_grn(
        dataset, priors, output_dir / "percell_grn_cache.pkl", logger
    )
    save_json(output_dir / "percell_grn_summary.json", {
        "n_rows": len(percell),
        "positive": int((percell["sign"] > 0).sum()) if len(percell) else 0,
        "negative": int((percell["sign"] < 0).sum()) if len(percell) else 0,
        "tf_policy": "top32_activity" if args.dataset in {"SCoPE2", "CITE_seq"} else "all",
    })

    node_index = build_node_index(dataset, priors)
    save_json(output_dir / "node_index.json", asdict(node_index))
    dc = max(2, min(DEFAULT_DC, len(dataset.cell_ids) - 1, len(dataset.genes)))
    config.update({
        "resolved_dc": dc,
        "n_cells": len(node_index.cell),
        "n_genes": len(node_index.gene),
        "n_proteins": len(node_index.protein),
        "n_tfs": len(node_index.tf),
        "n_nodes": node_index.n_nodes,
        "cross_attention_dropout": 0.1,
        "rae_dropout": 0.1,
        "functional_dropout": 0.2,
    })
    save_json(output_dir / "config.json", config)
    node_features, feature_data = build_node_features(
        dataset, priors, node_index, dc=dc, seed=args.seed
    )
    audits: dict[tuple[str, str], EdgeAudit] = {}
    directed = deduplicate_hyperedges(
        build_directed_hyperedges(dataset, priors, percell, node_index, args, audits),
        audits, "directed",
    )
    undirected = deduplicate_hyperedges(
        build_undirected_hyperedges(dataset, priors, node_index, args, audits),
        audits, "undirected",
    )
    reason_by_type: dict[tuple[str, str], str] = {}
    if "chromatin_region" in priors.metadata["skipped_priors"]:
        reason_by_type.update({
            ("directed", "chromatin_region"): "chromatin_region_prior_unavailable",
            ("undirected", "chromatin_region"): "chromatin_region_prior_unavailable",
        })
    if "compartment" in priors.metadata["skipped_priors"]:
        for channel in ("directed", "undirected"):
            reason_by_type[(channel, "compartment_A")] = "compartment_prior_unavailable"
            reason_by_type[(channel, "compartment_B")] = "compartment_prior_unavailable"
    if priors.ligand_receptor.empty:
        reason_by_type[("undirected", "cci")] = "ligand_receptor_prior_unavailable"
    if "protein" not in dataset.views:
        for name in ("prot_inject", "prot_readout", "translation"):
            reason_by_type[("directed", name)] = "protein_modality_not_applicable"
        for name in ("prot_obs", "adt_knn", "mech_bridge", "mvmod"):
            reason_by_type[("undirected", name)] = "protein_modality_not_applicable"
    else:
        reason_by_type[("undirected", "view2_obs")] = "nonprotein_view_not_applicable"
        reason_by_type[("undirected", "view2_knn")] = "nonprotein_view_not_applicable"
        if not any(len(genes) >= 2 for genes in priors.modality_gene_map.values()):
            reason_by_type[("undirected", "mech_bridge")] = "no_multigene_protein_mapping"
    for key, reason in reason_by_type.items():
        selected = directed if key[0] == "directed" else undirected
        if not any(edge.name == key[1] for edge in selected):
            audits.setdefault(key, EdgeAudit()).reason = reason
    validate_required_edges(directed, undirected, audits, logger)
    edge_summary = build_edge_summary(directed, undirected, audits, node_index.n_nodes)
    edge_summary.to_csv(output_dir / "edge_summary.csv", index=False)
    skipped_edges = {
        f"{channel}:{name}": asdict(audit)
        for (channel, name), audit in audits.items()
        if not any(
            edge.name == name for edge in (directed if channel == "directed" else undirected)
        )
    }
    save_json(output_dir / "skipped_hyperedges.json", skipped_edges)
    compiled = compile_hypergraph(node_features, directed, undirected, node_index, device)
    logger.info(
        "nodes=%d cells=%d genes=%d proteins=%d TF=%d directed=%d undirected=%d",
        node_index.n_nodes, len(node_index.cell), len(node_index.gene),
        len(node_index.protein), len(node_index.tf),
        compiled["directed"]["e"], compiled["undirected"]["e"],
    )

    directed_data = compiled["directed"]
    undirected_data = compiled["undirected"]
    model = build_model(
        directed_data=directed_data,
        undirected_data=undirected_data,
        n_cells=len(dataset.cell_ids),
        n_genes=len(dataset.genes),
        dc=dc,
        pca_init=feature_data["pca_init"], hidden=DEFAULT_HIDDEN,
        latent=dc,
        use_asym=True,
        device=device,
    )
    criterion = build_criterion(
        w_comp=DEFAULT_W_COMP,
        w_ortho=DEFAULT_W_ORTHO,
        w_nce=DEFAULT_W_NCE,
        w_gate=DEFAULT_W_GATE,
    )
    optimizer = build_optimizer(model)
    n_cells = len(dataset.cell_ids)
    gf = node_features[n_cells:]
    tensors = {
        "M_graph": torch.from_numpy(feature_data["M_graph"]).to(device),
        "gf": torch.from_numpy(gf).to(device),
        "ch_target": torch.from_numpy(feature_data["ch_target"]).to(device),
        "W": torch.from_numpy(feature_data["pca_init"]).to(device),
    }
    comp_indicator = np.zeros(len(dataset.genes), dtype=np.float32)
    if not priors.gene_info.empty:
        gene_pos = {g: i for i, g in enumerate(dataset.genes)}
        for comp, sign in (("A", 1.0), ("B", -1.0)):
            for gene in priors.gene_info.loc[
                priors.gene_info["compartment"] == comp, "gene_id"
            ]:
                if gene in gene_pos:
                    comp_indicator[gene_pos[gene]] = sign
    comp_tensor = torch.from_numpy(comp_indicator).to(device)

    history, best_state, best_epoch, best_loss = train_model(
        model, criterion, optimizer, tensors, comp_tensor, args.epochs, logger
    )
    history.to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state, "best_epoch": best_epoch,
        "best_loss": best_loss, "config": config,
    }, output_dir / "best_model.pt")
    metrics = evaluate_and_save(
        model, tensors, feature_data, dataset, output_dir,
        best_epoch, best_loss, directed_data["et_names"], args.seed,
    )
    save_visualization_inputs(
        dataset=dataset,
        priors=priors,
        directed_edges=directed,
        undirected_edges=undirected,
        node_index=node_index,
        output_dir=output_dir,
    )
    logger.info("best_epoch=%d best_loss=%.6f metrics=%s", best_epoch, best_loss, metrics)
    from phasehyper.visualization.phase import run_phase_visualization

    label_map = dataset.metadata.get("label_map", {})
    visualization_result = run_phase_visualization(
        result_dir=output_dir,
        dataset_name=dataset.name,
        raw_rna=dataset.rna,
        labels=dataset.labels,
        label_names=[
            str(label_map.get(int(label), label)) for label in dataset.labels
        ],
        cell_ids=dataset.cell_ids,
        genes=dataset.genes,
        dpi=300,
        top_genes=40,
        projection_seed=args.seed,
        cluster_seed=args.seed,
    )
    logger.info(
        "visualization status=%s generated=%d skipped=%d failed=%d output_dir=%s",
        visualization_result["status"],
        len(visualization_result["generated"]),
        len(visualization_result["skipped"]),
        len(visualization_result["failed"]),
        visualization_result["output_dir"],
    )
    for skipped in visualization_result["skipped"]:
        logger.warning(
            "visualization skipped name=%s reason=%s",
            skipped["name"], skipped["reason"],
        )
    for name, reason in visualization_result["failed"].items():
        logger.error("visualization failed name=%s reason=%s", name, reason)
    logger.info("output_directory=%s", output_dir.resolve())


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
