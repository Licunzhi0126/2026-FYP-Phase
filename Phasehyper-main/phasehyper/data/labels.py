from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LoadedLabels:
    names: list[str]
    ids: np.ndarray
    id_to_name: dict[int, str]
    counts: dict[str, int]
    n_clusters: int
    source_path: Path
    rule_name: str


def resolve_label_path(data_dir: Path, label_config: Mapping[str, object]) -> Path:
    """Resolve a configured label file without silently choosing an ambiguity."""
    if label_config.get("file"):
        path = data_dir / str(label_config["file"])
        if not path.exists():
            raise ValueError(f"Label file not found: {path}")
        return path

    candidates = [
        data_dir / str(name)
        for name in label_config.get("file_candidates", [])
    ]
    existing = [path for path in candidates if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            "Multiple configured label files exist; configure one explicit file: "
            + ", ".join(str(path) for path in existing)
        )
    raise ValueError(
        "None of the configured label files exist: "
        + ", ".join(str(path) for path in candidates)
    )


def read_label_tokens(
    path: Path,
    *,
    optional_header_tokens: Sequence[str] = (),
) -> list[str]:
    """Read either one horizontal CSV record or a one-column CSV file."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Label file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Label file is empty: {path}")

    if len(rows) == 1:
        tokens = rows[0]
    elif all(len(row) == 1 for row in rows):
        tokens = [row[0] for row in rows]
    else:
        raise ValueError(
            f"Label file must contain one horizontal row or one vertical column: {path}"
        )

    tokens = [str(token).strip() for token in tokens]
    if any(not token for token in tokens):
        empty_positions = [i for i, token in enumerate(tokens) if not token]
        raise ValueError(
            f"Empty label token(s) at zero-based position(s) {empty_positions}: {path}"
        )
    headers = {str(token).strip().lower() for token in optional_header_tokens}
    if tokens and tokens[0].lower() in headers:
        tokens = tokens[1:]
    if not tokens:
        raise ValueError(f"Label file contains no labels after header removal: {path}")
    return tokens


def normalize_pea_sta_label(token: str) -> str:
    low = str(token).strip().lower().replace("-", "_").replace(" ", "_")
    if "6d" in low and "bmp4" in low:
        return "6d_BMP4"
    if "6d" in low and ("control" in low or "contol" in low):
        return "6d_control"
    if "0h" in low and ("control" in low or "contol" in low):
        return "0h_control"
    raise ValueError(f"Unrecognized PEA_STA label: {token!r}")


def normalize_dataset_labels(
    dataset_name: str,
    raw_labels: Sequence[str],
) -> list[str]:
    if dataset_name == "PEA_STA":
        return [normalize_pea_sta_label(value) for value in raw_labels]
    if dataset_name in {"sc_GEM", "CITE_seq", "SCoPE2", "scNMT"}:
        normalized = [str(value).strip() for value in raw_labels]
        if any(not value for value in normalized):
            raise ValueError(f"Empty label is not allowed for dataset {dataset_name}")
        return normalized
    raise ValueError(f"Unsupported real dataset: {dataset_name}")


def encode_label_names(
    names: Sequence[str],
) -> tuple[np.ndarray, dict[int, str]]:
    cleaned = [str(name).strip() for name in names]
    if not cleaned or any(not name for name in cleaned):
        raise ValueError("Labels must be non-empty strings")
    ordered = list(dict.fromkeys(cleaned))
    if len(ordered) < 2:
        raise ValueError(f"At least two label classes are required, got {ordered}")
    name_to_id = {name: idx for idx, name in enumerate(ordered)}
    ids = np.asarray([name_to_id[name] for name in cleaned], dtype=np.int64)
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    if set(np.unique(ids).tolist()) != set(range(len(ordered))):
        raise ValueError("Encoded label IDs are not contiguous")
    return ids, id_to_name


def load_and_align_labels(
    *,
    dataset_name: str,
    label_path: Path,
    source_cell_ids: Sequence[str],
    target_cell_ids: Sequence[str],
    optional_header_tokens: Sequence[str] = (),
    expected_names: Sequence[str] = (),
) -> LoadedLabels:
    source = [str(cell).strip() for cell in source_cell_ids]
    target = [str(cell).strip() for cell in target_cell_ids]
    if len(set(source)) != len(source):
        raise ValueError(f"Duplicate source cell IDs for dataset {dataset_name}")
    if len(set(target)) != len(target):
        raise ValueError(f"Duplicate target cell IDs for dataset {dataset_name}")

    raw = read_label_tokens(
        Path(label_path), optional_header_tokens=optional_header_tokens
    )
    if len(raw) != len(source):
        raise ValueError(
            f"dataset={dataset_name}, path={label_path}: label count {len(raw)} "
            f"!= source RNA cell count {len(source)}"
        )
    label_by_cell = dict(zip(source, raw))
    missing = [cell for cell in target if cell not in label_by_cell]
    if missing:
        raise ValueError(
            f"dataset={dataset_name}: target cells missing from RNA label axis: {missing[:10]}"
        )
    aligned_raw = [label_by_cell[cell] for cell in target]
    names = normalize_dataset_labels(dataset_name, aligned_raw)
    ids, id_to_name = encode_label_names(names)
    counts = dict(Counter(names))
    if expected_names and set(counts) != set(expected_names):
        raise ValueError(
            f"dataset={dataset_name}: expected labels {sorted(expected_names)}, "
            f"got {sorted(counts)}"
        )
    return LoadedLabels(
        names=names,
        ids=ids,
        id_to_name=id_to_name,
        counts=counts,
        n_clusters=int(np.unique(ids).size),
        source_path=Path(label_path),
        rule_name="pea_sta" if dataset_name == "PEA_STA" else "identity",
    )
