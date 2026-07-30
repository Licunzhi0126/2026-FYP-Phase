"""Input adapters for the Figure 2 expression and GRN benchmarks.

The adapters deliberately keep observation, ground-truth channels, and scoring
masks separate.  This makes low-read allelic measurements explicit instead of
silently treating them as balanced ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionContext:
    """One expression benchmark context with aligned cell-by-gene matrices."""

    name: str
    dataset: str
    total: pd.DataFrame
    truth_a: pd.DataFrame
    truth_b: pd.DataFrame
    score_mask: pd.DataFrame
    cell_metadata: pd.DataFrame
    truth_a_label: str
    truth_b_label: str


@dataclass(frozen=True)
class GRNBundle:
    """Aligned thresholded GRN matrices represented on a selected edge axis."""

    cells: list[str]
    genes: list[str]
    edge_names: list[str]
    edge_index: np.ndarray
    combined: np.ndarray
    maternal: np.ndarray
    paternal: np.ndarray
    combined_union_mismatch_count: int
    combined_union_match_rate: float


def _normalise_text(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def _canonical_sample(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalise_text(value).lower())


def _canonical_gene(value: object) -> str:
    text = _normalise_text(value)
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def _make_unique(values: Iterable[object]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for value in values:
        label = _normalise_text(value) or "unnamed"
        count = counts.get(label, 0)
        counts[label] = count + 1
        result.append(label if count == 0 else f"{label}__{count + 1}")
    return result


def _classify_stage(sample_name: str) -> str:
    text = sample_name.lower().replace("_", "").replace("-", "")
    if "zygote" in text or "1cell" in text or "1c" in text:
        return "zygote"
    if "64cell" in text or "64c" in text:
        return "64-cell"
    if "32cell" in text or "32c" in text:
        return "32-cell"
    if "16cell" in text or "16c" in text:
        return "16-cell"
    if "8cell" in text or "8c" in text:
        return "8-cell"
    if "4cell" in text or "4c" in text:
        return "4-cell"
    if "2cell" in text or "twocell" in text or "2c" in text:
        return "2-cell"
    if "blast" in text:
        return "blastocyst"
    if "liver" in text:
        return "liver"
    return "other"


def _development_group(stage: str) -> str:
    if stage in {"2-cell", "4-cell"}:
        return "cleavage"
    if stage in {"8-cell", "16-cell"}:
        return "morula"
    if stage in {"32-cell", "64-cell", "blastocyst"}:
        return "blastocyst"
    return "other"


def _select_expression_genes(
    total: pd.DataFrame,
    n_genes: int,
    min_detected_cells: int = 5,
) -> list[str]:
    """Select variable genes from observed total expression only.

    The held-out allelic channels and their read-support mask are intentionally
    absent from this function so they cannot influence the axes used for model
    fitting.
    """

    detected = (total > 0).sum(axis=0)
    candidates = detected.index[detected >= min_detected_cells]
    if len(candidates) < min(n_genes, total.shape[1]):
        candidates = detected.index[detected > 0]
    if len(candidates) == 0:
        raise ValueError("No genes have positive observed total expression.")

    logged = np.log1p(total.loc[:, candidates].clip(lower=0.0))
    score = logged.var(axis=0) + 0.05 * logged.mean(axis=0)
    return score.sort_values(ascending=False).head(n_genes).index.tolist()


def _subset_expression_context(
    context: ExpressionContext,
    name: str,
    cells: Iterable[str],
) -> ExpressionContext:
    cell_list = [cell for cell in cells if cell in context.total.index]
    return ExpressionContext(
        name=name,
        dataset=context.dataset,
        total=context.total.loc[cell_list],
        truth_a=context.truth_a.loc[cell_list],
        truth_b=context.truth_b.loc[cell_list],
        score_mask=context.score_mask.loc[cell_list],
        cell_metadata=context.cell_metadata.loc[cell_list],
        truth_a_label=context.truth_a_label,
        truth_b_label=context.truth_b_label,
    )


def load_gse45719(
    dataset_root: str | Path,
    n_genes: int = 120,
    min_allelic_reads: int = 2,
    min_scoreable_genes: int = 8,
) -> list[ExpressionContext]:
    """Load GSE45719 per-sample text files.

    ``RPKM`` is the observed total expression.  ``C57_hits`` and ``CAST_hits``
    determine the two reference-strain truth channels where they have adequate
    read support.
    """

    root = Path(dataset_root)
    files = sorted(root.glob("*_expression.txt"))
    if not files:
        raise FileNotFoundError(f"No GSE45719 expression files found in {root}")

    total_series: dict[str, pd.Series] = {}
    c57_series: dict[str, pd.Series] = {}
    cast_series: dict[str, pd.Series] = {}

    required = {"#Gene_symbol", "RPKM", "CAST_hits", "C57_hits"}
    for path in files:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        if not required.issubset(frame.columns):
            missing = sorted(required.difference(frame.columns))
            raise ValueError(f"{path.name} is missing columns: {missing}")

        sample = path.name.removesuffix("_expression.txt")
        clean = pd.DataFrame(
            {
                "gene": frame["#Gene_symbol"].map(_canonical_gene),
                "total": pd.to_numeric(frame["RPKM"], errors="coerce"),
                "c57": pd.to_numeric(frame["C57_hits"], errors="coerce"),
                "cast": pd.to_numeric(frame["CAST_hits"], errors="coerce"),
            }
        )
        clean = clean.loc[clean["gene"] != ""].fillna(0.0)
        clean = clean.groupby("gene", sort=False)[["total", "c57", "cast"]].sum()
        total_series[sample] = clean["total"].clip(lower=0.0)
        c57_series[sample] = clean["c57"].clip(lower=0.0)
        cast_series[sample] = clean["cast"].clip(lower=0.0)

    total = pd.DataFrame.from_dict(total_series, orient="index").fillna(0.0)
    c57 = pd.DataFrame.from_dict(c57_series, orient="index").reindex_like(total).fillna(0.0)
    cast = pd.DataFrame.from_dict(cast_series, orient="index").reindex_like(total).fillna(0.0)

    allelic_reads = c57 + cast
    score_mask = allelic_reads >= min_allelic_reads
    retained_cells = total.index[(total > 0).sum(axis=1) > 0]
    if len(retained_cells) == 0:
        raise ValueError("GSE45719 has no cells with positive total expression.")
    total = total.loc[retained_cells]
    c57 = c57.loc[retained_cells]
    cast = cast.loc[retained_cells]
    score_mask = score_mask.loc[retained_cells]

    genes = _select_expression_genes(
        total,
        n_genes=n_genes,
        min_detected_cells=max(3, min(8, len(retained_cells) // 4)),
    )
    total = total.loc[:, genes]
    c57 = c57.loc[:, genes]
    cast = cast.loc[:, genes]
    score_mask = score_mask.loc[:, genes]

    fractions_c57 = c57.divide(c57 + cast).where(score_mask, 0.5).fillna(0.5)
    truth_c57 = total * fractions_c57
    truth_cast = total - truth_c57

    metadata = pd.DataFrame(index=total.index)
    metadata["sample"] = metadata.index
    metadata["stage"] = [_classify_stage(cell) for cell in metadata.index]
    metadata["development_group"] = metadata["stage"].map(_development_group)

    base = ExpressionContext(
        name="GSE45719 embryo",
        dataset="GSE45719",
        total=total,
        truth_a=truth_c57,
        truth_b=truth_cast,
        score_mask=score_mask,
        cell_metadata=metadata,
        truth_a_label="C57",
        truth_b_label="CAST",
    )

    embryo_cells = metadata.index[metadata["development_group"] != "other"]
    if len(embryo_cells) >= 4:
        base = _subset_expression_context(base, "GSE45719 embryo", embryo_cells)

    contexts = [base]
    for group, label in (
        ("cleavage", "GSE45719 cleavage"),
        ("morula", "GSE45719 morula"),
        ("blastocyst", "GSE45719 blastocyst"),
    ):
        cells = base.cell_metadata.index[
            base.cell_metadata["development_group"] == group
        ]
        if len(cells) >= 4:
            contexts.append(_subset_expression_context(base, label, cells))
    return contexts


def _read_largest_xls_table(path: Path) -> pd.DataFrame:
    try:
        workbook = pd.ExcelFile(path, engine="xlrd")
    except ImportError as exc:
        raise ImportError(
            "Reading GSE80810 .xls files requires xlrd in the active environment. "
            "Install it with: conda install -c conda-forge xlrd"
        ) from exc

    best: pd.DataFrame | None = None
    best_score = -1
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
        raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if raw.empty:
            continue

        header_limit = min(20, len(raw))
        header_scores: list[tuple[int, int]] = []
        for row_index in range(header_limit):
            values = raw.iloc[row_index]
            nonempty = int(values.notna().sum())
            text_values = values.dropna().map(str)
            text_count = int(
                text_values.str.contains(r"[A-Za-z_]", regex=True).sum()
            )
            header_scores.append((3 * text_count + nonempty, row_index))
        _, header_row = max(header_scores)

        table = raw.iloc[header_row + 1 :].copy()
        table.columns = _make_unique(raw.iloc[header_row].tolist())
        table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
        score = int(table.shape[0] * table.shape[1])
        if score > best_score:
            best = table
            best_score = score

    if best is None or best.empty:
        raise ValueError(f"No tabular data could be read from {path}")
    return best


def _table_to_gene_sample_matrix(table: pd.DataFrame) -> pd.DataFrame:
    normalised_columns = {
        column: re.sub(r"[^a-z0-9]+", "", str(column).lower())
        for column in table.columns
    }
    preferred_gene_headers = {
        "genesymbol",
        "genesymbols",
        "gene",
        "genes",
        "symbol",
        "geneid",
        "ensemblgeneid",
        "entrezgeneid",
        "idref",
    }
    gene_column: object | None = next(
        (
            column
            for column, normalised in normalised_columns.items()
            if normalised in preferred_gene_headers
        ),
        None,
    )

    sample_columns = [
        column
        for column, normalised in normalised_columns.items()
        if "sample" in normalised or "cell" in normalised
    ]
    identifier_column = (
        gene_column
        if gene_column is not None
        else (sample_columns[0] if sample_columns else table.columns[0])
    )
    numeric = table.drop(columns=[identifier_column]).apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.dropna(axis=1, how="all")

    genes_are_rows = gene_column is not None
    if gene_column is None and numeric.shape[1] > 0:
        genes_are_rows = table.shape[0] >= numeric.shape[1]

    if genes_are_rows:
        gene_labels = table[identifier_column].map(_canonical_gene)
        numeric.index = gene_labels
        numeric = numeric.loc[numeric.index != ""]
        numeric = numeric.groupby(level=0, sort=False).mean()
        numeric.columns = _make_unique(numeric.columns)
        return numeric.transpose()

    sample_labels = table[identifier_column].map(_normalise_text)
    numeric.index = sample_labels
    numeric = numeric.loc[numeric.index != ""]
    numeric.columns = [_canonical_gene(column) for column in numeric.columns]
    numeric = numeric.loc[:, numeric.columns != ""]
    return numeric


def _canonicalise_matrix_axes(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    sample_display: dict[str, str] = {}
    sample_keys = []
    for sample in matrix.index:
        key = _canonical_sample(sample)
        sample_keys.append(key)
        sample_display.setdefault(key, _normalise_text(sample))

    canonical = matrix.copy()
    canonical.index = sample_keys
    canonical = canonical.loc[canonical.index != ""]
    canonical = canonical.groupby(level=0, sort=False).mean()
    canonical.columns = [_canonical_gene(gene) for gene in canonical.columns]
    canonical = canonical.loc[:, canonical.columns != ""]
    canonical = canonical.transpose().groupby(level=0, sort=False).mean().transpose()
    return canonical, sample_display


def load_gse80810(
    dataset_root: str | Path,
    n_genes: int = 120,
    min_count_reads: int = 8,
    min_scoreable_genes: int = 8,
) -> list[ExpressionContext]:
    """Load GSE80810 total-expression, allelic-ratio, and count workbooks."""

    root = Path(dataset_root)
    paths = {
        "ratio": root / "GSE80810_AllelicRatio.xls",
        "count": root / "GSE80810_CountTable.xls",
        "total": root / "GSE80810_RPRT.xls",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GSE80810 workbooks: {missing}")

    matrices: dict[str, pd.DataFrame] = {}
    display_maps: dict[str, dict[str, str]] = {}
    for key, path in paths.items():
        raw_matrix = _table_to_gene_sample_matrix(_read_largest_xls_table(path))
        matrices[key], display_maps[key] = _canonicalise_matrix_axes(raw_matrix)

    common_samples = sorted(
        set(matrices["total"].index)
        & set(matrices["ratio"].index)
        & set(matrices["count"].index)
    )
    common_genes = sorted(
        set(matrices["total"].columns)
        & set(matrices["ratio"].columns)
        & set(matrices["count"].columns)
    )
    if not common_samples or not common_genes:
        raise ValueError(
            "The three GSE80810 workbooks do not share aligned sample and gene labels."
        )

    total = matrices["total"].loc[common_samples, common_genes].clip(lower=0.0)
    ratio = matrices["ratio"].loc[common_samples, common_genes]
    count = matrices["count"].loc[common_samples, common_genes].clip(lower=0.0)
    score_mask = (
        ratio.notna()
        & ratio.ge(0.0)
        & ratio.le(1.0)
        & count.ge(min_count_reads)
        & total.gt(0.0)
    )
    ratio = ratio.clip(lower=0.0, upper=1.0).where(score_mask, 0.5).fillna(0.5)

    retained_samples = total.index[(total > 0).sum(axis=1) > 0]
    if len(retained_samples) == 0:
        raise ValueError("GSE80810 has no samples with positive total expression.")
    total = total.loc[retained_samples]
    ratio = ratio.loc[retained_samples]
    score_mask = score_mask.loc[retained_samples]

    genes = _select_expression_genes(
        total,
        n_genes=n_genes,
        min_detected_cells=max(3, min(8, len(retained_samples) // 4)),
    )
    total = total.loc[:, genes]
    ratio = ratio.loc[:, genes]
    score_mask = score_mask.loc[:, genes]

    truth_paternal = total * ratio
    truth_maternal = total - truth_paternal

    display_map = display_maps["total"]
    display_names = [display_map.get(sample, sample) for sample in total.index]
    total.index = display_names
    truth_paternal.index = display_names
    truth_maternal.index = display_names
    score_mask.index = display_names

    metadata = pd.DataFrame(index=total.index)
    metadata["sample"] = metadata.index
    metadata["stage"] = [_classify_stage(cell) for cell in metadata.index]
    metadata["development_group"] = metadata["stage"].map(_development_group)
    metadata["genotype"] = [
        "KO" if re.search(r"(^|[^a-z])ko([^a-z]|$)", cell.lower()) else "WT"
        for cell in metadata.index
    ]

    all_context = ExpressionContext(
        name="GSE80810 all",
        dataset="GSE80810",
        total=total,
        truth_a=truth_maternal,
        truth_b=truth_paternal,
        score_mask=score_mask,
        cell_metadata=metadata,
        truth_a_label="maternal",
        truth_b_label="paternal",
    )

    wt_cells = metadata.index[metadata["genotype"] == "WT"]
    if len(wt_cells) >= 4:
        primary = _subset_expression_context(
            all_context, "GSE80810 WT embryo", wt_cells
        )
    else:
        primary = all_context

    contexts = [all_context]
    if primary.name != all_context.name:
        contexts.append(primary)
    for group, label in (
        ("cleavage", "GSE80810 cleavage"),
        ("morula", "GSE80810 morula"),
        ("blastocyst", "GSE80810 blastocyst"),
    ):
        cells = primary.cell_metadata.index[
            primary.cell_metadata["development_group"] == group
        ]
        if len(cells) >= 4:
            contexts.append(_subset_expression_context(primary, label, cells))

    ko_cells = metadata.index[metadata["genotype"] == "KO"]
    if len(ko_cells) >= 4:
        contexts.append(
            _subset_expression_context(all_context, "GSE80810 KO embryo", ko_cells)
        )
    return contexts


def load_answerdata_contexts(
    answer_root: str | Path,
    n_genes: int = 120,
    min_allelic_reads: int = 2,
    min_gse80810_reads: int = 8,
    min_scoreable_genes: int = 8,
) -> list[ExpressionContext]:
    """Load the two approved real-data sources from ``data/answerdata``."""

    root = Path(answer_root)
    contexts = load_gse45719(
        root / "GSE45719",
        n_genes=n_genes,
        min_allelic_reads=min_allelic_reads,
        min_scoreable_genes=min_scoreable_genes,
    )
    contexts.extend(
        load_gse80810(
            root / "GSE80810",
            n_genes=n_genes,
            min_count_reads=min_gse80810_reads,
            min_scoreable_genes=min_scoreable_genes,
        )
    )
    return contexts


def read_phase_matrix(
    path: str | Path,
    cells: Iterable[str],
    genes: Iterable[str],
) -> np.ndarray:
    """Read a cell-by-gene prediction matrix from CSV, NPY, or NPZ."""

    source = Path(path)
    cell_list = list(cells)
    gene_list = list(genes)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(source, index_col=0)
        frame.index = frame.index.map(str)
        frame.columns = frame.columns.map(str)
        if set(cell_list).issubset(frame.index) and set(gene_list).issubset(
            frame.columns
        ):
            array = frame.loc[cell_list, gene_list].to_numpy(dtype=float)
        elif set(gene_list).issubset(frame.index) and set(cell_list).issubset(
            frame.columns
        ):
            array = frame.loc[gene_list, cell_list].transpose().to_numpy(dtype=float)
        else:
            array = frame.to_numpy(dtype=float)
    elif suffix == ".npy":
        array = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        preferred = [
            key for key in ("prediction", "pred", "array", "arr_0") if key in archive
        ]
        if not preferred and len(archive.files) != 1:
            raise ValueError(
                f"{source} contains multiple arrays; use one of "
                "prediction/pred/array/arr_0."
            )
        array = archive[preferred[0] if preferred else archive.files[0]]
    else:
        raise ValueError(f"Unsupported prediction format: {source.suffix}")

    array = np.asarray(array, dtype=float)
    expected = (len(cell_list), len(gene_list))
    if array.shape == expected[::-1]:
        array = array.transpose()
    if array.shape != expected:
        raise ValueError(
            f"{source} has shape {array.shape}; expected {expected} (cell-by-gene)."
        )
    return array


def _read_square_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"{path} is not a square adjacency matrix: {frame.shape}")
    if set(frame.columns).issubset(frame.index):
        frame = frame.loc[frame.columns, frame.columns]
    return frame


def load_grn_bundle(
    per_cell_root: str | Path,
    max_edges: int = 1200,
    min_prevalence: float = 0.05,
    max_prevalence: float = 0.95,
) -> GRNBundle:
    """Load independently thresholded combined/maternal/paternal GRN matrices."""

    root = Path(per_cell_root)
    directories = {name: root / name for name in ("combined", "maternal", "paternal")}
    missing_directories = [
        str(path) for path in directories.values() if not path.is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"Missing per-cell GRN directories: {missing_directories}"
        )

    combined_files = sorted(directories["combined"].glob("*.csv"))
    if not combined_files:
        raise FileNotFoundError(f"No combined GRN CSV files found in {root}")

    cells = [path.stem for path in combined_files]
    for channel in ("maternal", "paternal"):
        channel_cells = {path.stem for path in directories[channel].glob("*.csv")}
        absent = [cell for cell in cells if cell not in channel_cells]
        if absent:
            raise FileNotFoundError(
                f"{channel} is missing {len(absent)} cells, including {absent[:3]}"
            )

    combined_frames: list[pd.DataFrame] = []
    maternal_frames: list[pd.DataFrame] = []
    paternal_frames: list[pd.DataFrame] = []
    reference_genes: list[str] | None = None

    for cell in cells:
        frames = {
            "combined": _read_square_csv(directories["combined"] / f"{cell}.csv"),
            "maternal": _read_square_csv(directories["maternal"] / f"{cell}.csv"),
            "paternal": _read_square_csv(directories["paternal"] / f"{cell}.csv"),
        }
        if reference_genes is None:
            reference_genes = frames["combined"].columns.tolist()
        for channel, frame in frames.items():
            if not set(reference_genes).issubset(frame.index) or not set(
                reference_genes
            ).issubset(frame.columns):
                raise ValueError(f"{cell}/{channel} has incompatible gene labels.")
            frames[channel] = frame.loc[reference_genes, reference_genes]
        combined_frames.append(frames["combined"])
        maternal_frames.append(frames["maternal"])
        paternal_frames.append(frames["paternal"])

    assert reference_genes is not None
    combined_full = np.stack(
        [frame.to_numpy(dtype=float) for frame in combined_frames]
    )
    maternal_full = np.stack(
        [frame.to_numpy(dtype=float) for frame in maternal_frames]
    )
    paternal_full = np.stack(
        [frame.to_numpy(dtype=float) for frame in paternal_frames]
    )

    combined_full = (combined_full > 0).astype(float)
    maternal_full = (maternal_full > 0).astype(float)
    paternal_full = (paternal_full > 0).astype(float)

    n_genes = len(reference_genes)
    source_indices, target_indices = np.where(~np.eye(n_genes, dtype=bool))
    combined_edges = combined_full[:, source_indices, target_indices]
    maternal_edges = maternal_full[:, source_indices, target_indices]
    paternal_edges = paternal_full[:, source_indices, target_indices]

    truth_union_full = np.maximum(maternal_edges, paternal_edges)
    mismatch_count = int(np.not_equal(combined_edges, truth_union_full).sum())
    match_rate = float(np.equal(combined_edges, truth_union_full).mean())

    prevalence = combined_edges.mean(axis=0)
    candidate = np.flatnonzero(
        (prevalence >= min_prevalence) & (prevalence <= max_prevalence)
    )
    if candidate.size == 0:
        candidate = np.arange(combined_edges.shape[1])
    variability = prevalence[candidate] * (1.0 - prevalence[candidate])
    order = np.argsort(-variability, kind="stable")
    selected = candidate[order[: min(max_edges, len(order))]]

    edge_index = np.column_stack(
        (source_indices[selected], target_indices[selected])
    ).astype(int)
    edge_names = [
        f"{reference_genes[source]}->{reference_genes[target]}"
        for source, target in edge_index
    ]
    return GRNBundle(
        cells=cells,
        genes=reference_genes,
        edge_names=edge_names,
        edge_index=edge_index,
        combined=combined_edges[:, selected],
        maternal=maternal_edges[:, selected],
        paternal=paternal_edges[:, selected],
        combined_union_mismatch_count=mismatch_count,
        combined_union_match_rate=match_rate,
    )


def load_external_grn_prediction(
    prediction_path: str | Path,
    bundle: GRNBundle,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``pred_A``/``pred_B`` GRN predictions and align selected edges."""

    source = Path(prediction_path)
    archive = np.load(source, allow_pickle=False)
    if "pred_A" not in archive or "pred_B" not in archive:
        raise ValueError(f"{source} must contain arrays named pred_A and pred_B.")

    pred_a = np.asarray(archive["pred_A"], dtype=float)
    pred_b = np.asarray(archive["pred_B"], dtype=float)
    if pred_a.shape != pred_b.shape:
        raise ValueError("pred_A and pred_B must have the same shape.")

    expected = bundle.maternal.shape
    if pred_a.shape == expected:
        return pred_a, pred_b

    if pred_a.ndim == 3 and pred_a.shape[0] == len(bundle.cells):
        rows = bundle.edge_index[:, 0]
        columns = bundle.edge_index[:, 1]
        return pred_a[:, rows, columns], pred_b[:, rows, columns]

    if "edge_index" in archive and pred_a.ndim == 2:
        supplied_edges = np.asarray(archive["edge_index"], dtype=int)
        if supplied_edges.ndim != 2 or supplied_edges.shape[1] != 2:
            raise ValueError("edge_index must have shape (n_edges, 2).")
        lookup = {tuple(edge): index for index, edge in enumerate(supplied_edges)}
        try:
            positions = [lookup[tuple(edge)] for edge in bundle.edge_index]
        except KeyError as exc:
            raise ValueError(
                "The external prediction does not contain every selected benchmark edge."
            ) from exc
        aligned_a = pred_a[:, positions]
        aligned_b = pred_b[:, positions]
        if aligned_a.shape == expected:
            return aligned_a, aligned_b

    raise ValueError(
        f"{source} has shape {pred_a.shape}; expected {expected}, a full "
        "cell-by-gene-by-gene array, or arrays with a matching edge_index."
    )
