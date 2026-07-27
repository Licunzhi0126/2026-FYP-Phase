"""Convert per-cell weighted GRN matrices to binary adjacency matrices.

The script preserves CSV row/column labels, removes diagonal self-loops, and
uses a shared absolute-weight threshold for every cell in a version:

    binary[i, j] = 1 if i != j and abs(weight[i, j]) >= threshold else 0

By default it reads ``data/per_cell`` and creates the two non-overwriting
versions ``data/per_cell_threshold_0.1`` and
``data/per_cell_threshold_0.01``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ("combined", "maternal", "paternal")


@dataclass(frozen=True)
class GrnMatrix:
    group: str
    file_name: str
    index: pd.Index
    columns: pd.Index
    values: np.ndarray


def _project_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _threshold_label(threshold: float) -> str:
    """Return a stable, human-readable directory suffix."""
    return format(threshold, ".15g")


def _validate_thresholds(thresholds: list[float]) -> list[float]:
    validated: list[float] = []
    seen: set[float] = set()
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"Threshold must be finite and non-negative: {threshold}")
        numeric = float(threshold)
        if numeric not in seen:
            validated.append(numeric)
            seen.add(numeric)
    if not validated:
        raise ValueError("At least one threshold is required.")
    return validated


def _load_inputs(
    input_dir: Path,
    groups: tuple[str, ...],
    expected_size: int,
) -> list[GrnMatrix]:
    """Load and validate every source CSV before creating output files."""
    matrices: list[GrnMatrix] = []
    for group in groups:
        group_dir = input_dir / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Missing input group directory: {group_dir}")

        csv_paths = sorted(group_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in: {group_dir}")

        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path, index_col=0)
            if frame.shape != (expected_size, expected_size):
                raise ValueError(
                    f"{csv_path} has shape {frame.shape}; "
                    f"expected ({expected_size}, {expected_size})."
                )
            if not frame.index.is_unique or not frame.columns.is_unique:
                raise ValueError(f"Duplicate row or column labels in: {csv_path}")

            row_labels = tuple(str(value) for value in frame.index)
            column_labels = tuple(str(value) for value in frame.columns)
            if row_labels != column_labels:
                raise ValueError(f"Row and column labels do not match in: {csv_path}")

            values = frame.to_numpy(dtype=np.float64, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"NaN or infinite weight found in: {csv_path}")

            matrices.append(
                GrnMatrix(
                    group=group,
                    file_name=csv_path.name,
                    index=frame.index.copy(),
                    columns=frame.columns.copy(),
                    values=values,
                )
            )

    return matrices


def _write_threshold_version(
    matrices: list[GrnMatrix],
    output_root: Path,
    threshold: float,
) -> Path:
    """Write one threshold version via a staging directory, then rename it."""
    label = _threshold_label(threshold)
    final_dir = output_root / f"per_cell_threshold_{label}"
    staging_dir = output_root / f".per_cell_threshold_{label}.building"

    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"Staging directory already exists; inspect it before retrying: {staging_dir}"
        )

    for group in sorted({matrix.group for matrix in matrices}):
        (staging_dir / group).mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, object]] = []
    for matrix in matrices:
        binary = (np.abs(matrix.values) >= threshold).astype(np.uint8)
        np.fill_diagonal(binary, 0)

        output_path = staging_dir / matrix.group / matrix.file_name
        pd.DataFrame(
            binary,
            index=matrix.index,
            columns=matrix.columns,
        ).to_csv(output_path, encoding="utf-8", lineterminator="\n")

        offdiag = ~np.eye(binary.shape[0], dtype=bool)
        kept = binary.astype(bool)
        row_degree = np.count_nonzero(kept, axis=1)
        column_degree = np.count_nonzero(kept, axis=0)
        retained_edges = int(np.count_nonzero(kept))
        source_nonzero = int(np.count_nonzero(matrix.values[offdiag]))

        summary_rows.append(
            {
                "threshold": threshold,
                "group": matrix.group,
                "cell_file": matrix.file_name,
                "retained_edges": retained_edges,
                "possible_offdiag_edges": int(np.count_nonzero(offdiag)),
                "density": retained_edges / int(np.count_nonzero(offdiag)),
                "retained_positive_edges": int(
                    np.count_nonzero(kept & (matrix.values > 0))
                ),
                "retained_negative_edges": int(
                    np.count_nonzero(kept & (matrix.values < 0))
                ),
                "source_nonzero_offdiag_edges": source_nonzero,
                "removed_nonzero_edges": source_nonzero - retained_edges,
                "isolated_genes": int(
                    np.count_nonzero((row_degree == 0) & (column_degree == 0))
                ),
                "zero_row_degree_genes": int(np.count_nonzero(row_degree == 0)),
                "zero_column_degree_genes": int(np.count_nonzero(column_degree == 0)),
                "output_file": str(
                    Path(f"per_cell_threshold_{label}")
                    / matrix.group
                    / matrix.file_name
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["group", "cell_file"])
    summary.to_csv(
        staging_dir / "threshold_summary.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    staging_dir.rename(final_dir)
    return final_dir


def _print_version_summary(version_dir: Path) -> None:
    summary = pd.read_csv(version_dir / "threshold_summary.csv")
    print(f"\nCreated: {version_dir}")
    grouped = (
        summary.groupby("group", sort=True)["retained_edges"]
        .agg(["count", "min", "median", "mean", "max"])
        .round({"mean": 2})
    )
    print(grouped.to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Binarize per-cell GRN matrices using absolute-weight thresholds. "
            "Existing output directories are never overwritten."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/per_cell"),
        help="Input directory containing combined/maternal/paternal folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Parent directory for per_cell_threshold_<value> versions.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.01],
        help="One or more non-negative absolute-weight thresholds.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Input group folders to process.",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=100,
        help="Expected square matrix size.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = _validate_thresholds(args.thresholds)
    input_dir = _project_path(args.input_dir)
    output_root = _project_path(args.output_root)
    groups = tuple(args.groups)

    if args.expected_size <= 0:
        raise ValueError("--expected-size must be positive.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    target_dirs = [
        output_root / f"per_cell_threshold_{_threshold_label(value)}"
        for value in thresholds
    ]
    existing_targets = [path for path in target_dirs if path.exists()]
    if existing_targets:
        joined = "\n".join(str(path) for path in existing_targets)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined}")

    matrices = _load_inputs(input_dir, groups, args.expected_size)
    print(
        f"Validated {len(matrices)} matrices from {input_dir} "
        f"across {len(groups)} groups."
    )

    for threshold in thresholds:
        version_dir = _write_threshold_version(matrices, output_root, threshold)
        _print_version_summary(version_dir)


if __name__ == "__main__":
    main()
