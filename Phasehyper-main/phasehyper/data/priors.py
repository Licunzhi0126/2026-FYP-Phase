from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from phasehyper.config import DATASET_CONFIG, GENE_POSITION_ALIASES
from phasehyper.schemas import DatasetBundle, PriorBundle
from phasehyper.utils import _safe_standardize

_POSITION_PRIOR_AUDIT: Dict[str, Dict[str, object]] = {}
_POSITION_ALIAS_AUDIT_BY_DATASET: Dict[str, List[Dict[str, object]]] = {}


def build_gene_prior_features(common_genes, kegg_txt_path, d_prior: int = 16):
    gene_set = set(common_genes)
    gene_to_pathways: Dict[str, List[str]] = {gene: [] for gene in common_genes}
    kegg_hyperedge: Dict[str, List[str]] = {}

    kegg_df = pd.read_csv(kegg_txt_path, sep="\t", header=None, na_values=["NA", ""])
    for _, row in kegg_df.iterrows():
        gene = str(row.iloc[0]).strip()
        pathway = str(row.iloc[2]).strip() if len(row) > 2 and pd.notna(row.iloc[2]) else ""

        if gene in gene_set and pathway and pathway.lower() != "nan":
            if pathway not in gene_to_pathways[gene]:
                gene_to_pathways[gene].append(pathway)

            kegg_hyperedge.setdefault(pathway, [])
            if gene not in kegg_hyperedge[pathway]:
                kegg_hyperedge[pathway].append(gene)

    pathway_values = sorted({pathway for pathways in gene_to_pathways.values() for pathway in pathways})
    pathway_index = {pathway: idx for idx, pathway in enumerate(pathway_values)}

    pathway_multihot = np.zeros((len(common_genes), len(pathway_values)), dtype=np.float32)
    for gene_idx, gene in enumerate(common_genes):
        for pathway in gene_to_pathways.get(gene, []):
            pathway_multihot[gene_idx, pathway_index[pathway]] = 1.0
    prior_matrix = pathway_multihot
    if prior_matrix.shape[1] == 0:
        gene_prior_matrix = np.zeros((len(common_genes), d_prior), dtype=np.float32)

    elif prior_matrix.shape[1] == 1 or len(common_genes) <= 1:
        gene_prior_matrix = np.concatenate(
            [
                prior_matrix.astype(np.float32),
                np.zeros((len(common_genes), max(0, d_prior - prior_matrix.shape[1])), dtype=np.float32),
            ],
            axis=1,
        )[:, :d_prior]

    else:
        n_components = min(d_prior, prior_matrix.shape[1] - 1, len(common_genes) - 1)
        n_components = max(1, n_components)

        gene_prior_matrix = TruncatedSVD(
            n_components=n_components,
            random_state=42,
        ).fit_transform(prior_matrix)

        if gene_prior_matrix.shape[1] < d_prior:
            pad = np.zeros((len(common_genes), d_prior - gene_prior_matrix.shape[1]), dtype=np.float32)
            gene_prior_matrix = np.concatenate(
                [gene_prior_matrix.astype(np.float32), pad],
                axis=1,
            )
        else:
            gene_prior_matrix = gene_prior_matrix[:, :d_prior].astype(np.float32)

    gene_prior_matrix = np.nan_to_num(
        gene_prior_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    print(f"  gene prior feature shape: {gene_prior_matrix.shape}")
    return gene_prior_matrix, gene_to_pathways, kegg_hyperedge


def _position_file_for_dataset(base_dir: Path, dataset_type: str) -> Path:
    if dataset_type == "simulation":
        sim_path = base_dir / "gene_positions_sim.txt"
        if sim_path.exists():
            return sim_path
    if dataset_type == "sc_GEM":
        expected_name = "gene_positions_sc.txt"
    elif dataset_type == "CITE_seq":
        expected_name = "gene_positions_cite.txt"
    elif dataset_type == "SCoPE2":
        expected_name = "gene_positions_scope2.txt"
    else:
        expected_name = "gene_positions_pea.txt"
    expected_path = Path(base_dir) / expected_name
    return expected_path


def _position_candidate_files(
    base_dir: Path, dataset_type: str, allow_fallback: bool = False
) -> List[Path]:
    primary_path = _position_file_for_dataset(base_dir, dataset_type)
    candidates = [primary_path]
    if allow_fallback:
        input_root = Path(base_dir).parent
        if input_root.exists():
            candidates.extend(sorted(input_root.glob("*/gene_positions*.txt")))

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _chromosome_code(chromosome: str) -> float:
    token = str(chromosome).strip().lower().replace("chr", "")
    if token == "x":
        return 23.0
    if token == "y":
        return 24.0
    try:
        return float(token)
    except ValueError:
        return 0.0


def _load_position_prior(
    base_dir: Path,
    dataset_type: str,
    common_genes: List[str],
    *,
    allow_fallback: bool = False,
) -> Dict[str, object]:
    paths = _position_candidate_files(base_dir, dataset_type, allow_fallback=allow_fallback)
    primary_path = paths[0]
    gene_index = {str(gene).strip(): idx for idx, gene in enumerate(common_genes)}
    candidate_to_gene: Dict[str, str] = {}
    for gene in common_genes:
        gene_key = str(gene).strip()
        candidates = [gene_key] + GENE_POSITION_ALIASES.get(gene_key.upper(), [])
        for candidate in candidates:
            candidate_key = str(candidate).strip().upper()
            if candidate_key and candidate_key not in candidate_to_gene:
                candidate_to_gene[candidate_key] = gene_key

    rows: Dict[str, Dict[str, object]] = {}
    alias_matches: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                fields = raw_line.strip().split("\t")
                if len(fields) < 5:
                    continue
                raw_gene, chrom, start, end, strand = fields[:5]
                raw_gene = str(raw_gene).strip()
                gene = candidate_to_gene.get(raw_gene.upper())
                if gene is None or gene not in gene_index or gene in rows:
                    continue
                try:
                    start_i = int(float(start))
                    end_i = int(float(end))
                except ValueError:
                    continue
                if end_i < start_i:
                    start_i, end_i = end_i, start_i
                rows[gene] = {
                    "chrom": str(chrom).strip(),
                    "start": start_i,
                    "end": end_i,
                    "strand": str(strand).strip(),
                    "source_file": str(path.resolve()),
                }
                if raw_gene.upper() != str(gene).strip().upper():
                    alias_matches.append(
                        {
                            "dataset_type": dataset_type,
                            "common_gene": gene,
                            "matched_position_gene": raw_gene,
                            "position_file": str(path.resolve()),
                        }
                    )

    n_genes = len(common_genes)
    raw_features = np.zeros((n_genes, 7), dtype=np.float32)
    raw_features[:, 6] = 1.0
    max_midpoint = 1.0
    max_length = 1.0
    for info in rows.values():
        midpoint = (int(info["start"]) + int(info["end"])) / 2.0
        length = max(1.0, float(int(info["end"]) - int(info["start"]) + 1))
        max_midpoint = max(max_midpoint, midpoint)
        max_length = max(max_length, length)

    chrom_groups: Dict[str, List[str]] = {}
    chrom_position_rows: Dict[str, List[Tuple[float, str]]] = {}
    for gene_idx, gene in enumerate(common_genes):
        info = rows.get(gene)
        if info is None:
            continue
        chrom = str(info["chrom"])
        start_i = int(info["start"])
        end_i = int(info["end"])
        midpoint = (start_i + end_i) / 2.0
        length = max(1.0, float(end_i - start_i + 1))
        strand = str(info["strand"])
        raw_features[gene_idx, 0] = 1.0
        raw_features[gene_idx, 1] = _chromosome_code(chrom) / 24.0
        raw_features[gene_idx, 2] = np.log1p(midpoint) / np.log1p(max_midpoint)
        raw_features[gene_idx, 3] = np.log1p(length) / np.log1p(max_length)
        raw_features[gene_idx, 4] = 1.0 if strand == "+" else 0.0
        raw_features[gene_idx, 5] = 1.0 if strand == "-" else 0.0
        raw_features[gene_idx, 6] = 0.0
        chrom_groups.setdefault(chrom, []).append(gene)
        chrom_position_rows.setdefault(chrom, []).append((midpoint, gene))

    nearby_groups: Dict[str, List[str]] = {}
    window = 2
    for chrom, chrom_rows in chrom_position_rows.items():
        ordered = [gene for _, gene in sorted(chrom_rows)]
        for idx, gene in enumerate(ordered):
            members = ordered[max(0, idx - window): min(len(ordered), idx + window + 1)]
            if len(members) >= 2:
                nearby_groups[f"{chrom}:{gene}"] = members

    position_features = _safe_standardize(raw_features)
    matched_genes = sorted(rows.keys())
    missing_genes = [gene for gene in common_genes if gene not in rows]
    audit = {
        "dataset_type": dataset_type,
        "position_file": str(primary_path.resolve()) if primary_path.exists() else str(primary_path),
        "position_files_scanned": ";".join(str(path.resolve()) for path in paths if path.exists()),
        "allow_position_file_fallback": int(bool(allow_fallback)),
        "n_common_genes": len(common_genes),
        "n_position_matched": len(matched_genes),
        "n_position_missing": len(missing_genes),
        "missing_genes": ";".join(missing_genes),
        "n_alias_matched": len(alias_matches),
        "n_chrom_groups": sum(1 for genes in chrom_groups.values() if len(genes) > 1),
        "n_nearby_groups": len(nearby_groups),
        "position_feature_columns": "has_position;chromosome_code;midpoint;length;strand_plus;strand_minus;missing_position",
    }
    _POSITION_ALIAS_AUDIT_BY_DATASET[dataset_type] = alias_matches
    return {
        "features": position_features.astype(np.float32),
        "chrom_groups": chrom_groups,
        "nearby_groups": nearby_groups,
        "audit": audit,
        "rows": rows,
    }


def _save_position_prior_audit(out_dir: Path, dataset_type: str) -> None:
    audit = _POSITION_PRIOR_AUDIT.get(dataset_type)
    if audit:
        pd.DataFrame([audit]).to_csv(
            Path(out_dir) / "position_prior_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    alias_rows = _POSITION_ALIAS_AUDIT_BY_DATASET.get(dataset_type, [])
    if alias_rows:
        pd.DataFrame(alias_rows).drop_duplicates().to_csv(
            Path(out_dir) / "position_alias_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _build_prefixed_groups(prefix: str, raw_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name, genes in raw_groups.items():
        unique_genes = [str(g).strip() for g in dict.fromkeys(genes) if str(g).strip()]
        if len(unique_genes) > 1:
            groups[f"{prefix}{name}"] = unique_genes
    return groups


def _build_base_prior_bundle(base_dir: Path, dataset: DatasetBundle, d_prior: int = 16) -> PriorBundle:
    base_dir = Path(base_dir)

    config = DATASET_CONFIG.get(dataset.dataset_type)
    if config is None:
        raise ValueError(f"Unknown dataset type: {dataset.dataset_type}")

    files = config["files"]
    dataset_root = base_dir / config["root"]

    kegg_file = dataset_root / files["kegg_prior"]
    _, _, kegg_hypergraph = build_gene_prior_features(
        dataset.common_genes,
        kegg_file,
        d_prior,
    )
    kegg_groups = _build_prefixed_groups("path::", kegg_hypergraph)

    has_ppi = config.get("has_ppi", False)
    ppi_groups = None
    if has_ppi and files.get("ppi_prior"):
        ppi_file = dataset_root / files["ppi_prior"]

        # PPI 原始矩阵约 11746×11746（~276MB）。这里只按列读入 common_genes 对应的
        # 子集（usecols），再裁掉无关行，得到 n×n 小矩阵（n=len(common_genes)），
        # 避免把整张稠密矩阵读进内存（~1GB+）。
        common_genes = list(dict.fromkeys(str(g).strip() for g in dataset.common_genes))
        common_set = set(common_genes)

        header_cols = list(pd.read_csv(ppi_file, nrows=0, sep=",").columns)
        pos_by_gene: Dict[str, int] = {}
        for pos, col in enumerate(header_cols):
            if pos == 0:  # 第 0 列是行索引（基因名），单独处理
                continue
            pos_by_gene.setdefault(str(col).strip(), pos)
        needed_pos = sorted(pos_by_gene[g] for g in common_genes if g in pos_by_gene)

        ppi_df = pd.read_csv(ppi_file, usecols=[0] + needed_pos, sep=",")
        ppi_df = ppi_df.set_index(ppi_df.columns[0]).fillna(0)
        ppi_df.index = ppi_df.index.astype(str).str.strip()
        ppi_df.columns = [str(col).strip() for col in ppi_df.columns]
        ppi_df = ppi_df.loc[ppi_df.index.isin(common_set)]
        if ppi_df.index.duplicated().any():
            ppi_df = ppi_df[~ppi_df.index.duplicated(keep="first")]

        col_genes = list(ppi_df.columns)
        ppi_groups_raw: Dict[str, List[str]] = {}
        for gene_i in common_genes:
            if gene_i not in ppi_df.index:
                continue
            row = ppi_df.loc[gene_i].to_numpy()
            neighbors = [
                gene_j
                for gene_j, value in zip(col_genes, row)
                if gene_j != gene_i and value != 0.0
            ]
            members = sorted(dict.fromkeys([gene_i] + neighbors))
            if len(members) > 1:
                ppi_groups_raw[gene_i] = members

        ppi_groups = _build_prefixed_groups("ppi::", ppi_groups_raw)

    return PriorBundle(
        kegg_groups=kegg_groups,
        poswin_groups={},
        ppi_groups=ppi_groups,
        gene_prior_matrix=None,
    )


def build_prior_bundle(
    base_dir,
    dataset,
    d_prior=16,
    *,
    allow_position_file_fallback=False,
    genomic_window_bp=200000,
    include_window_groups=True,
):
    base_dir = Path(base_dir)
    config = DATASET_CONFIG.get(dataset.dataset_type)
    dataset_root = base_dir / config["root"] if config else base_dir

    prior = _build_base_prior_bundle(base_dir, dataset, d_prior)

    position = _load_position_prior(
        dataset_root,
        dataset.dataset_type,
        dataset.common_genes,
        allow_fallback=allow_position_file_fallback,
    )
    rows = position.get("rows", {}) or {}

    gene_positions = {}
    for gene in dataset.common_genes:
        info = rows.get(gene)
        if info is None:
            continue
        chrom = str(info.get("chrom", "")).strip()
        if not chrom:
            continue
        try:
            start_i = float(info.get("start"))
            end_i = float(info.get("end"))
        except Exception:
            continue
        if not np.isfinite(start_i) or not np.isfinite(end_i):
            continue
        gene_positions[str(gene).strip()] = {
            "chrom": chrom,
            "start": start_i,
            "end": end_i,
        }

    poswin_groups: Dict[str, List[str]] = {}
    if include_window_groups:
        n_genes = len(dataset.common_genes)
        chroms = []
        mids = []
        for gene in dataset.common_genes:
            info = gene_positions.get(str(gene).strip())
            if info is None:
                chroms.append(None)
                mids.append(None)
            else:
                chroms.append(str(info["chrom"]).strip())
                mids.append((float(info["start"]) + float(info["end"])) / 2.0)

        for i in range(n_genes):
            if chroms[i] is None or mids[i] is None:
                continue
            members = []
            for j in range(n_genes):
                if chroms[j] is None or mids[j] is None:
                    continue
                if chroms[i] != chroms[j]:
                    continue
                dist = abs(mids[i] - mids[j])
                if dist < genomic_window_bp:
                    members.append(dataset.common_genes[j])
            members = [str(g).strip() for g in dict.fromkeys(members) if str(g).strip()]
            if len(members) > 1:
                poswin_groups[f"poswin::{chroms[i]}:{dataset.common_genes[i]}"] = members

    n_matched = sum(1 for gene in dataset.common_genes if str(gene).strip() in gene_positions)
    n_missing = len(dataset.common_genes) - n_matched
    n_window_groups = len(poswin_groups)

    audit = dict(position.get("audit", {}))
    audit["dataset_type"] = dataset.dataset_type
    audit["n_common_genes"] = len(dataset.common_genes)
    audit["n_position_matched"] = n_matched
    audit["n_position_missing"] = n_missing
    audit["uses_position_features"] = 0
    audit["uses_chrom_groups"] = 0
    audit["uses_nearby_groups"] = 0
    audit["uses_genomic_window"] = 1
    audit["genomic_window_bp"] = float(genomic_window_bp)
    audit["n_window_groups"] = n_window_groups
    audit["include_window_groups"] = int(bool(include_window_groups))

    _POSITION_PRIOR_AUDIT[dataset.dataset_type] = audit

    print(
        f"  Position prior (full/genomic_window): matched {n_matched}/{len(dataset.common_genes)} genes, "
        f"window_bp={float(genomic_window_bp):.0f}, window_groups={n_window_groups}, "
        f"fallback={bool(allow_position_file_fallback)}"
    )

    return PriorBundle(
        kegg_groups=prior.kegg_groups,
        poswin_groups=poswin_groups,
        ppi_groups=prior.ppi_groups,
        gene_prior_matrix=None,
    )
