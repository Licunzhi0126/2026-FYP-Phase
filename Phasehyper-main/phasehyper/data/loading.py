from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from phasehyper.config import DATASET_CONFIG, REAL_PHASE_DATASETS
from phasehyper.data.labels import load_and_align_labels, resolve_label_path


def _load_feature_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id").fillna(0)
    df.columns = [str(col).strip() for col in df.columns]

    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    return df


def _read_stage_tokens(stage_path: Path) -> List[str]:
    if not stage_path.exists():
        return []
    try:
        text = stage_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        line = text.splitlines()[0].strip()
        return [tok.strip() for tok in line.split(",") if tok.strip()]
    except Exception as exc:
        warnings.warn(f"Failed to read stage tokens from {stage_path}: {exc}")
        return []


def _labels_to_ids(label_names: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    ordered = []
    seen = set()

    for name in label_names:
        name = str(name).strip()
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    mapping = {name: idx for idx, name in enumerate(ordered)}
    inverse = {idx: name for name, idx in mapping.items()}

    labels = np.asarray(
        [mapping[str(name).strip()] for name in label_names],
        dtype=np.int64,
    )

    return labels, inverse


def _load_stage_names_simple(stage_path: Path, ordered_cells: List[str]) -> Dict[str, str]:
    tokens = _read_stage_tokens(stage_path)

    if not tokens:
        raise ValueError("Stage file is empty or missing")

    if len(tokens) != len(ordered_cells):
        raise ValueError(
            "Stage count does not match number of cells. "
            "Please ensure one-to-one alignment."
        )

    return {cell: tok for cell, tok in zip(ordered_cells, tokens)}


def _resolve_dataset_type(base_dir: Path, dataset_type: str) -> str:
    if dataset_type in DATASET_CONFIG:
        return dataset_type
    raise ValueError(
        f"Unknown dataset type: {dataset_type}. Valid types: {list(DATASET_CONFIG.keys())}"
    )


def load_dataset(base_dir, dataset_type):
    base_dir = Path(base_dir)
    dataset_type = _resolve_dataset_type(base_dir, dataset_type)

    config = DATASET_CONFIG[dataset_type]
    files = config["files"]
    root = config["root"]

    base_dir = base_dir / root

    view1_dfs = []
    view1_name = dataset_type

    expression_file = base_dir / files["expression"]
    expression_df = _load_feature_frame(expression_file)

    view1_file_names = files.get("view", [])
    if view1_file_names:
        for view1_file_name in view1_file_names:
            view1_dfs.append(_load_feature_frame(base_dir / view1_file_name))
        if len(view1_file_names) > 1:
            view1_name = f"{dataset_type}_multi"

    source_cells = expression_df.index.tolist()
    common_cells = [
        cell
        for cell in source_cells
        if all(cell in view.index for view in view1_dfs)
    ]
    expression_df = expression_df.reindex(common_cells)
    view1_dfs = [view.reindex(common_cells) for view in view1_dfs]

    if dataset_type in REAL_PHASE_DATASETS:
        label_config = config["labels"]
        label_path = resolve_label_path(base_dir, label_config)
        loaded = load_and_align_labels(
            dataset_name=dataset_type,
            label_path=label_path,
            source_cell_ids=source_cells,
            target_cell_ids=common_cells,
            optional_header_tokens=label_config.get("optional_header_tokens", []),
            expected_names=label_config.get("expected_names", []),
        )
        stage_name_by_cell = dict(zip(common_cells, loaded.names))
    else:
        stage_name_by_cell = _load_stage_names_simple(
            base_dir / files["stage"], common_cells
        )

    return view1_dfs, expression_df, stage_name_by_cell, view1_name


# ---------------------------------------------------------------------------
# Clean-data path: 读 data_clean/<ds>/，保留 NaN（不 fillna(0)），并一并读缺失掩码。
# 不改动上面的 load_dataset/_load_feature_frame，原训练链路行为完全不变。
# ---------------------------------------------------------------------------


def _load_feature_frame_keepnan(csv_path: Path) -> pd.DataFrame:
    """与 _load_feature_frame 相同的去重/对齐，但**保留 NaN**（不填 0）。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id")
    df.columns = [str(col).strip() for col in df.columns]

    # groupby(...).mean() 默认跳过 NaN：全 NaN 列/行仍为 NaN，缺失语义保留。
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    return df


def _coerce_bool_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.values.dtype == bool:
        return df
    truthy = {"true", "1", "1.0", "yes", "t"}
    return df.apply(
        lambda col: col.map(lambda v: str(v).strip().lower() in truthy)
    ).astype(bool)


def _load_mask_frame(csv_path: Path) -> pd.DataFrame:
    """读缺失掩码 CSV（True=原始缺失），index=cell_id、columns=基因。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id")
    df.columns = [str(col).strip() for col in df.columns]
    return _coerce_bool_frame(df)


def _clean_mask_filename(feature_filename: str) -> str:
    """expression_data.csv -> expression_missing_mask.csv（与清洗脚本产物命名一致）。"""
    if feature_filename.endswith("_data.csv"):
        return feature_filename[: -len("_data.csv")] + "_missing_mask.csv"
    return feature_filename[: -len(".csv")] + "_missing_mask.csv"


def load_clean_dataset(clean_root, dataset_type):
    """从 data_clean/<ds>/ 读清洗后数据 + 缺失掩码（保留 NaN）。

    返回比 load_dataset 多两项：expression_mask, view1_masks（与 view1_dfs 一一对应）。
    掩码未做 common-gene 对齐，交给上层在构 DatasetBundle 时按需 reindex。
    """
    clean_root = Path(clean_root)
    dataset_type = _resolve_dataset_type(clean_root, dataset_type)

    config = DATASET_CONFIG[dataset_type]
    files = config["files"]
    ds_dir = clean_root / dataset_type

    view1_dfs: List[pd.DataFrame] = []
    view1_masks: List[pd.DataFrame] = []
    view1_name = dataset_type

    expression_df = _load_feature_frame_keepnan(ds_dir / files["expression"])
    expression_mask = _load_mask_frame(ds_dir / _clean_mask_filename(files["expression"]))

    view1_file_names = files.get("view", [])
    for view1_file_name in view1_file_names:
        view1_dfs.append(_load_feature_frame_keepnan(ds_dir / view1_file_name))
        view1_masks.append(_load_mask_frame(ds_dir / _clean_mask_filename(view1_file_name)))
    if len(view1_file_names) > 1:
        view1_name = f"{dataset_type}_multi"

    source_cells = expression_df.index.tolist()
    common_cells = [
        cell
        for cell in source_cells
        if all(cell in view.index for view in view1_dfs)
    ]
    expression_df = expression_df.reindex(common_cells)
    expression_mask = expression_mask.reindex(common_cells)
    view1_dfs = [view.reindex(common_cells) for view in view1_dfs]
    view1_masks = [mask.reindex(common_cells) for mask in view1_masks]

    if dataset_type in REAL_PHASE_DATASETS:
        label_config = config["labels"]
        label_path = resolve_label_path(ds_dir, label_config)
        loaded = load_and_align_labels(
            dataset_name=dataset_type,
            label_path=label_path,
            source_cell_ids=source_cells,
            target_cell_ids=common_cells,
            optional_header_tokens=label_config.get("optional_header_tokens", []),
            expected_names=label_config.get("expected_names", []),
        )
        stage_name_by_cell = dict(zip(common_cells, loaded.names))
    else:
        stage_name_by_cell = _load_stage_names_simple(
            ds_dir / files["stage"], common_cells
        )

    return (
        view1_dfs,
        expression_df,
        stage_name_by_cell,
        view1_name,
        expression_mask,
        view1_masks,
    )


# ---------------------------------------------------------------------------
# 异构多模态路径：读 data_clean/<ds>/ -> HeteroBundle（cell 为共享轴，
# 每个模态各带自己的特征节点集 + 到基因的桥）。供 CITE_seq 等真·多模态使用。
# 不改动上面任何函数，原链路行为不变。
# ---------------------------------------------------------------------------


def _load_bridge_map(csv_path: Path) -> Dict[str, List[str]]:
    """读 long-form 桥接表（列 0=特征名，列 1=基因名）-> {feature: [gene, ...]}。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    fcol, gcol = df.columns[0], df.columns[1]
    bridge: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        feat = str(row[fcol]).strip()
        gene = str(row[gcol]).strip()
        if feat and gene:
            bridge.setdefault(feat, [])
            if gene not in bridge[feat]:
                bridge[feat].append(gene)
    return bridge


def load_clean_hetero_bundle(clean_root, dataset_type):
    """从 data_clean/<ds>/ 读异构多模态数据 -> HeteroBundle。

    模态清单来自 DATASET_CONFIG[ds]["modalities"]，每项形如：
      {"name","file","node_type","bridge"(可空),"binary"(可空)}
    - node_type=="gene" 且无 bridge：identity 模态（特征即基因节点，如 RNA）。
    - 有 bridge：读 long-form 桥接表（如 protein_gene_map.csv）。
    所有模态按第一个模态的细胞序对齐；gene 集 = identity 模态列 ∪ 所有桥目标基因。
    labels 来自 cell_stage.csv（与细胞序对齐，仅评估用）。
    """
    from phasehyper.schemas import HeteroBundle, ModalitySpec

    clean_root = Path(clean_root)
    dataset_type = _resolve_dataset_type(clean_root, dataset_type)
    config = DATASET_CONFIG[dataset_type]
    ds_dir = clean_root / dataset_type

    mod_specs = config.get("modalities")
    if not mod_specs:
        raise ValueError(
            f"DATASET_CONFIG[{dataset_type!r}] 缺少 'modalities'，无法走异构路径。"
        )

    modalities: List = []
    source_cells: List[str] = []
    gene_set: set = set()

    for spec in mod_specs:
        ft = _load_feature_frame_keepnan(ds_dir / spec["file"])
        mask_name = _clean_mask_filename(spec["file"])
        mask = _load_mask_frame(ds_dir / mask_name) if (ds_dir / mask_name).exists() else None
        bridge = None
        if spec.get("bridge"):
            bridge = _load_bridge_map(ds_dir / spec["bridge"])

        if spec["name"] == "rna":
            source_cells = ft.index.tolist()

        node_type = spec["node_type"]
        if node_type == "gene" and not bridge:
            gene_set.update(str(c).strip() for c in ft.columns)  # identity：特征即基因
        if bridge:
            for targets in bridge.values():
                gene_set.update(str(g).strip() for g in targets)

        modalities.append(
            ModalitySpec(
                name=spec["name"],
                node_type=node_type,
                feature_table=ft,
                mask=mask,
                bridge_to_gene=bridge,
                binary=bool(spec.get("binary", False)),
            )
        )

    if not source_cells:
        raise ValueError(f"DATASET_CONFIG[{dataset_type!r}] has no RNA modality")
    cells = [
        cell
        for cell in source_cells
        if all(cell in modality.feature_table.index for modality in modalities)
    ]
    if not cells:
        raise ValueError(f"dataset={dataset_type}: no cells shared by all modalities")

    # 对齐所有模态到统一的 RNA 细胞序
    for m in modalities:
        m.feature_table = m.feature_table.reindex(index=cells)
        if m.mask is not None:
            m.mask = m.mask.reindex(index=cells)

    genes = sorted(gene_set)

    if dataset_type in REAL_PHASE_DATASETS:
        label_config = config["labels"]
        label_path = resolve_label_path(ds_dir, label_config)
        loaded = load_and_align_labels(
            dataset_name=dataset_type,
            label_path=label_path,
            source_cell_ids=source_cells,
            target_cell_ids=cells,
            optional_header_tokens=label_config.get("optional_header_tokens", []),
            expected_names=label_config.get("expected_names", []),
        )
        stage_tokens = loaded.names
        labels = loaded.ids
        label_map = loaded.id_to_name
    else:
        stage_tokens = _read_stage_tokens(ds_dir / config["files"]["stage"])
        if len(stage_tokens) != len(cells):
            raise ValueError(
                f"cell_stage token 数 {len(stage_tokens)} != 细胞数 {len(cells)}"
            )
        labels, label_map = _labels_to_ids(stage_tokens)

    return HeteroBundle(
        cells=cells,
        genes=genes,
        modalities=modalities,
        dataset_type=dataset_type,
        labels=labels,
        label_names=stage_tokens,
        label_map=label_map,
    )
