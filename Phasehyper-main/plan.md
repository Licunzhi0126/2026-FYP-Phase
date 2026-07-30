# Phase 基础数据集聚类标签规则修改计划

## 1. 目标

统一 `run_phase.py` 和 `phasehyper/data/` 中真实基础数据集的标签读取与聚类数规则，使以下数据集遵循同一套明确规则：

| 数据集 | 标签来源 | 标签处理 | 聚类数规则 |
|---|---|---|---|
| `PEA_STA` | 配置指定的标签文件 | 仅该数据集将标签归一化为 `0h_control`、`6d_control`、`6d_BMP4` | 归一化后不同标签数 |
| `sc_GEM` | `cell_stage.csv` | 保留原始 stage，只清理空白 | 文件中实际不同 stage 数 |
| `CITE_seq` | 实际 cell-type 标签文件 | 保留原始 cell type，只清理空白 | 文件中实际不同 cell type 数 |
| `SCoPE2` | `cell_stage.csv` | 保留 `sc_m0`、`sc_u` 等原始标签；允许移除明确的表头 token | 文件中实际不同标签数 |
| `scNMT` | `cell_stage.csv` | 保留 `E5.5`、`E6.5`、`E7.5` 等原始 stage | 文件中实际不同 stage 数 |

统一公式：

```python
normalized_labels = apply_dataset_label_rule(dataset_name, raw_labels)
labels, label_map = encode_labels(normalized_labels)
n_clusters = len(np.unique(labels))
```

KMeans 评估必须使用：

```python
expected_clusters = n_clusters
```

这里的 `n_clusters` 只由最终标签向量中的不同类别数决定，不手工写死为 2、3 或其他固定数。

---

## 2. 当前代码检查结论

### 2.1 当前符合和不符合情况

| 数据集 | 当前实现 | 是否遵循目标规则 | 主要问题 |
|---|---|---|---|
| `PEA_STA` | `run_phase.py::_pea_labels()` 根据 cell ID 字符串猜测标签 | **不完全符合** | 忽略了已经读取的标签文件；无法识别的 cell ID 会生成额外的 `unknown` 类 |
| `sc_GEM` | 从 `cell_stage.csv` 读取 token，直接编码 | **基本符合** | 标签与细胞的对齐依赖“交集后细胞数刚好不变”，缺少按 cell ID 重排 |
| `CITE_seq` | 从 `cell_type.csv` 读取 token，直接编码 | **基本符合** | `run_phase.py` 和 `phasehyper/config.py` 对标签文件名不一致；标签对齐较脆弱 |
| `SCoPE2` | 从 `cell_stage.csv` 读取；特殊移除开头的 `celltype` | **基本符合** | 标签对齐较脆弱；旧加载路径可能错误执行 PEA 标签归一化 |
| `scNMT` | 配置文件中存在，但 `run_phase.py` 不支持 | **不符合** | 不在 CLI choices 和 `DATA_FILES` 中；当前 `run_phase.py` 只处理一个辅助视图，而 scNMT 有两个 |
| 聚类评估 | `dataset.n_clusters = len(unique(labels))`，评估时传给 KMeans | **符合** | 建议增加一致性断言，防止未来传入错误值 |

### 2.2 当前最重要的两个错误

#### 错误 A：PEA_STA 从 cell ID 猜标签

当前：

```python
if name == "PEA_STA":
    raw_labels = _pea_labels(common_cells)
```

问题：

- `cell_stage.csv` 被读取，但实际没有用于标签生成；
- 依赖 cell ID 命名格式；
- 无法识别的 cell 会被标记为 `unknown`；
- `unknown` 会进入 `np.unique(labels)`，导致聚类数可能从 3 变成 4；
- 标签文件与评估标签可能不一致。

应改为：

```python
raw_labels = labels_aligned_to_common_cells
normalized_labels = normalize_pea_sta_labels(raw_labels)
```

#### 错误 B：旧数据加载路径用 `has_ppi` 决定标签归一化

`phasehyper/data/loading.py` 当前存在：

```python
if config.get("has_ppi", False):
    stage_name_by_cell = {
        cell: _normalize_ppi_stage_token(stage)
        for cell, stage in stage_name_by_cell.items()
    }
```

这意味着所有 `has_ppi=True` 的数据集都可能被套用 PEA_STA 的三阶段规则，包括：

- `CITE_seq`
- `SCoPE2`

这是错误的。是否有 PPI prior 与如何处理 cell label 没有关系。

必须改成按数据集显式选择标签规则：

```python
rule = config["label_rule"]
normalized = apply_dataset_label_rule(dataset_type, raw_labels)
```

---

## 3. 需要修改的文件

| 文件 | 修改类型 | 目的 |
|---|---|---|
| `phasehyper/config.py` | 修改 | 为每个真实数据集增加明确的标签配置，作为唯一规则来源 |
| `phasehyper/data/labels.py` | **新增** | 集中实现标签读取、对齐、归一化、编码和类别数验证 |
| `phasehyper/data/loading.py` | 修改 | 删除按 `has_ppi` 处理标签的逻辑，统一调用新标签模块 |
| `phasehyper/data/__init__.py` | 修改 | 导出公共标签加载接口 |
| `run_phase.py` | 修改 | 删除内联标签规则，使用统一加载器；加入 scNMT；保证评估使用派生的 `n_clusters` |
| `phasehyper/evaluation/phase.py` | 小幅修改 | 增加 `n_clusters` 与 labels 唯一类别数一致性检查 |
| `phasehyper/evaluation/clustering.py` | 小幅修改或保持 | 保留 KMeans 使用 `expected_clusters`，增加更清晰的错误信息 |
| `tests/test_real_label_rules.py` | **新增** | 测试五个数据集的标签规则 |
| `tests/test_real_label_alignment.py` | **新增** | 测试标签和细胞交集后的重排对齐 |
| `tests/test_phase_cluster_count.py` | **新增** | 测试评估端聚类数传递 |
| `tests/test_scnmt_loading.py` | **新增** | 测试 scNMT 多视图和标签接入 |

---

# 4. 修改 `phasehyper/config.py`

## 4.1 增加显式标签配置

每个基础数据集增加：

```python
"labels": {
    "file": "cell_stage.csv",
    "kind": "stage",
    "normalizer": "identity",
    "optional_header_tokens": [],
}
```

推荐配置：

```python
"PEA_STA": {
    ...
    "labels": {
        "file": "cell_stage.csv",
        "kind": "stage_or_treatment",
        "normalizer": "pea_sta",
        "optional_header_tokens": [],
        "expected_names": [
            "0h_control",
            "6d_control",
            "6d_BMP4",
        ],
    },
},
```

```python
"sc_GEM": {
    ...
    "labels": {
        "file": "cell_stage.csv",
        "kind": "stage",
        "normalizer": "identity",
        "optional_header_tokens": [],
    },
},
```

```python
"CITE_seq": {
    ...
    "labels": {
        "file": "<实际存在的 cell-type 文件>",
        "kind": "cell_type",
        "normalizer": "identity",
        "optional_header_tokens": ["celltype", "cell_type"],
    },
},
```

```python
"SCoPE2": {
    ...
    "labels": {
        "file": "cell_stage.csv",
        "kind": "cell_type",
        "normalizer": "identity",
        "optional_header_tokens": ["celltype", "cell_type"],
    },
},
```

```python
"scNMT": {
    ...
    "labels": {
        "file": "cell_stage.csv",
        "kind": "stage",
        "normalizer": "identity",
        "optional_header_tokens": [],
    },
},
```

## 4.2 解决 CITE_seq 文件名不一致

当前代码存在冲突：

```text
run_phase.py            -> cell_type.csv
phasehyper/config.py    -> cell_stage.csv
```

上传的压缩包不包含 `example_data` 本体，因此无法从该压缩包确认实际文件名。

实施时必须检查：

```text
example_data/CITE_seq/
```

并将 `DATASET_CONFIG["CITE_seq"]["labels"]["file"]` 设置为真实存在的唯一文件。

不要继续在不同文件中分别硬编码两个文件名。

如果迁移期需要兼容两种目录，可临时配置：

```python
"file_candidates": ["cell_type.csv", "cell_stage.csv"]
```

加载时：

1. 只允许一个候选文件存在；
2. 两个都存在时直接报错，要求明确选择；
3. 两个都不存在时报告完整路径。

最终应收敛到单一 `file` 配置。

## 4.3 不在配置中硬编码 `n_clusters`

可以保留：

```python
"expected_names": [...]
```

作为 PEA_STA 数据完整性检查，但不能写：

```python
"n_clusters": 3
```

作为 KMeans 的实际输入。

实际值始终由：

```python
len(np.unique(labels))
```

计算。

---

# 5. 新增 `phasehyper/data/labels.py`

这是本次修改的核心文件。

## 5.1 建议的数据结构

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
```

## 5.2 标签读取函数

```python
def read_label_tokens(
    path: Path,
    *,
    optional_header_tokens: Sequence[str] = (),
) -> list[str]:
    ...
```

要求：

- 使用 CSV parser，不直接 `line.split(",")`；
- 支持 UTF-8 BOM；
- 支持当前项目的一行横向标签格式；
- 可选支持单列纵向格式；
- 去除 token 两端空白；
- 空标签直接报错，不静默删除后继续；
- 只在首 token 明确匹配配置中的 header 时移除表头；
- 保存实际读取 token 数。

## 5.3 PEA_STA 专用归一化

```python
def normalize_pea_sta_label(token: str) -> str:
    low = token.strip().lower().replace("-", "_").replace(" ", "_")

    if "6d" in low and "bmp4" in low:
        return "6d_BMP4"
    if "6d" in low and ("control" in low or "contol" in low):
        return "6d_control"
    if "0h" in low and ("control" in low or "contol" in low):
        return "0h_control"

    raise ValueError(f"Unrecognized PEA_STA label: {token!r}")
```

重要要求：

- 不再从 cell ID 推断标签；
- 不再默认把无法识别的 token 归为 `6d_BMP4`；
- 不再生成 `unknown` 类；
- 无法识别时立即报错，并显示原始 token。

## 5.4 按数据集应用规则

```python
def normalize_dataset_labels(
    dataset_name: str,
    raw_labels: Sequence[str],
) -> list[str]:
    if dataset_name == "PEA_STA":
        return [normalize_pea_sta_label(x) for x in raw_labels]

    if dataset_name in {
        "sc_GEM",
        "CITE_seq",
        "SCoPE2",
        "scNMT",
    }:
        return [str(x).strip() for x in raw_labels]

    raise ValueError(f"Unsupported real dataset: {dataset_name}")
```

不要使用：

```python
has_ppi
have_answer
modality
```

决定标签规则。

## 5.5 标签编码

```python
def encode_label_names(
    names: Sequence[str],
) -> tuple[np.ndarray, dict[int, str]]:
    ordered = list(dict.fromkeys(names))
    name_to_id = {name: idx for idx, name in enumerate(ordered)}
    ids = np.asarray([name_to_id[name] for name in names], dtype=np.int64)
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return ids, id_to_name
```

继续使用“第一次出现顺序”编码即可，但必须：

- 检查没有空字符串；
- 检查至少存在 2 个类别；
- 检查 ID 连续为 `0..K-1`；
- 输出每类 cell 数量。

## 5.6 按 cell ID 对齐标签

这是对当前代码的重要加强。

当前代码在 RNA 和第二视图求交集后，只检查：

```python
len(tokens) == len(common_cells)
```

这不能保证标签顺序正确。

正确流程：

```python
rna_cell_ids = rna_df.index.tolist()

raw_tokens = read_label_tokens(label_path)

if len(raw_tokens) != len(rna_cell_ids):
    raise DataValidationError(...)

label_by_cell = dict(zip(rna_cell_ids, raw_tokens))

common_cells = intersect_cells_in_rna_order(...)

aligned_raw_labels = [label_by_cell[cell] for cell in common_cells]
```

这样即使某个辅助视图缺少部分 RNA cell，标签仍能跟随 cell ID 正确重排。

建议公共函数：

```python
def load_and_align_labels(
    *,
    dataset_name: str,
    label_path: Path,
    source_cell_ids: Sequence[str],
    target_cell_ids: Sequence[str],
    optional_header_tokens: Sequence[str] = (),
) -> LoadedLabels:
    ...
```

检查：

- `source_cell_ids` 唯一；
- `target_cell_ids` 全部存在于 source；
- label token 数与 source cell 数一致；
- 每个 target cell 只映射一次；
- 最终标签数与 target cell 数一致。

---

# 6. 修改 `phasehyper/data/loading.py`

## 6.1 删除错误的通用 PPI 标签归一化

删除或停止调用：

```python
_normalize_ppi_stage_token()
```

删除两处：

```python
if config.get("has_ppi", False):
    ...
```

位置包括：

- `load_dataset()`
- `load_clean_dataset()`

替换为统一调用：

```python
loaded = load_and_align_labels(
    dataset_name=dataset_type,
    label_path=...,
    source_cell_ids=...,
    target_cell_ids=...,
    optional_header_tokens=config["labels"].get(
        "optional_header_tokens", []
    ),
)
```

## 6.2 修改 `load_dataset()`

当前使用第一个辅助视图的 cell 顺序：

```python
cells = view1_dfs[0].index.tolist()
```

应改为：

```python
source_cells = expression_df.index.tolist()
common_cells = [
    cell
    for cell in source_cells
    if all(cell in view.index for view in view1_dfs)
]
```

然后：

- 所有表达和视图按 `common_cells` 重排；
- 标签先映射到 RNA source cells；
- 再按 `common_cells` 提取。

## 6.3 修改 `load_clean_dataset()`

使用和 `load_dataset()` 完全相同的标签函数，不复制规则。

## 6.4 修改 `load_clean_hetero_bundle()`

当前异构路径只是直接编码：

```python
labels, label_map = _labels_to_ids(stage_tokens)
```

必须改成：

```python
loaded = load_and_align_labels(...)
```

这样 PEA_STA 在所有加载路径中都会得到同样的三类标签，CITE_seq 和 SCoPE2 也不会被误归一化。

## 6.5 保留兼容性

如果项目中还有代码依赖：

```python
stage_name_by_cell
```

继续返回该结构，但其内容来自：

```python
dict(zip(common_cells, loaded.names))
```

而不是由独立的旧规则生成。

---

# 7. 修改 `phasehyper/data/__init__.py`

导出稳定接口：

```python
from .labels import (
    LoadedLabels,
    encode_label_names,
    load_and_align_labels,
    normalize_dataset_labels,
    normalize_pea_sta_label,
)

__all__ = [
    "LoadedLabels",
    "encode_label_names",
    "load_and_align_labels",
    "normalize_dataset_labels",
    "normalize_pea_sta_label",
]
```

`run_phase.py` 只从这个公共接口导入，不调用下划线开头的内部函数。

---

# 8. 修改 `run_phase.py`

## 8.1 删除重复的数据集标签定义

删除：

```python
DATA_FILES = {
    ...
}
```

或者至少不再用它决定标签文件。

数据文件、辅助视图和标签文件全部从：

```python
phasehyper.config.DATASET_CONFIG
```

读取。

这样 `run_phase.py` 和 `phasehyper/data/loading.py` 不会各维护一套冲突规则。

## 8.2 修改 CLI 数据集列表

当前：

```python
choices=["PEA_STA", "SCoPE2", "CITE_seq", "sc_GEM"]
```

改为明确的真实基础数据集：

```python
REAL_PHASE_DATASETS = (
    "PEA_STA",
    "sc_GEM",
    "CITE_seq",
    "SCoPE2",
    "scNMT",
)
```

```python
choices=REAL_PHASE_DATASETS
```

不要直接使用 `DATASET_CONFIG.keys()`，因为其中还包含 simulation 和 ratio 数据集。

建议将 `REAL_PHASE_DATASETS` 放在 `phasehyper/config.py`：

```python
REAL_PHASE_DATASETS = (
    "PEA_STA",
    "sc_GEM",
    "CITE_seq",
    "SCoPE2",
    "scNMT",
)
```

## 8.3 删除 PEA cell-ID 推断函数

删除：

```python
def _pea_labels(cell_ids):
    ...
```

在 `load_real_dataset()` 中改为：

```python
loaded_labels = load_and_align_labels(
    dataset_name=name,
    label_path=label_path,
    source_cell_ids=rna_source_cells,
    target_cell_ids=common_cells,
    optional_header_tokens=label_config.get(
        "optional_header_tokens", []
    ),
)
```

然后：

```python
labels = loaded_labels.ids
label_map = loaded_labels.id_to_name
n_clusters = loaded_labels.n_clusters
```

## 8.4 保存标签审计信息

在 `dataset.metadata` 中增加：

```python
"label_source": str(loaded_labels.source_path),
"label_rule": loaded_labels.rule_name,
"label_map": loaded_labels.id_to_name,
"label_counts": loaded_labels.counts,
"n_clusters": loaded_labels.n_clusters,
```

在日志中输出：

```text
dataset=PEA_STA label_rule=pea_sta
label_counts={'0h_control': ..., '6d_control': ..., '6d_BMP4': ...}
n_clusters=3
```

在 `data_summary.json` 中保存同样信息。

## 8.5 加强 `validate_dataset()`

增加：

```python
derived_clusters = int(np.unique(dataset.labels).size)

if dataset.n_clusters != derived_clusters:
    raise DataValidationError(...)

if dataset.n_clusters < 2:
    raise DataValidationError(...)

if set(dataset.metadata["label_map"]) != set(range(dataset.n_clusters)):
    raise DataValidationError(...)
```

PEA_STA 额外检查：

```python
expected = {"0h_control", "6d_control", "6d_BMP4"}
actual = set(dataset.metadata["label_map"].values())

if actual != expected:
    raise DataValidationError(...)
```

此处是数据完整性验证，不是用固定的 3 作为聚类输入。

---

# 9. scNMT 接入计划

## 9.1 当前障碍

`phasehyper/config.py` 已定义 scNMT：

```text
RNA
methylation
accessibility
```

但当前 `run_phase.py`：

- CLI 不包含 `scNMT`；
- `DATA_FILES` 不包含 `scNMT`；
- `load_real_dataset()` 只加载一个辅助视图；
- 多处使用 `next(iter(dataset.views.items()))`；
- metadata 只保存一个 `view_feature_names` 列表。

因此不能只增加：

```python
choices += ["scNMT"]
```

否则加载和模型输入仍不完整。

## 9.2 改为按配置加载全部视图

在 `load_real_dataset()` 中：

```python
modality_specs = [
    spec
    for spec in config["modalities"]
    if spec["name"] != "rna"
]
```

逐个加载：

```python
views = {}
view_feature_names = {}

for spec in modality_specs:
    frame = read_numeric_frame(data_dir / spec["file"])
    ...
    views[spec["name"]] = frame
    view_feature_names[spec["name"]] = frame.columns.tolist()
```

统一细胞交集：

```python
common_cells = [
    cell
    for cell in rna_df.index
    if all(cell in frame.index for frame in view_frames.values())
]
```

标签按 RNA 原始 cell ID 映射后再提取。

## 9.3 修改 metadata

由：

```python
"view_feature_names": [...]
```

改为：

```python
"view_feature_names_by_view": {
    "protein": [...],
    "methylation": [...],
    "accessibility": [...],
}
```

添加兼容 helper：

```python
def get_view_feature_names(dataset, view_name):
    return dataset.metadata[
        "view_feature_names_by_view"
    ][view_name]
```

## 9.4 修改使用单一视图的函数

需要修改 `run_phase.py` 中以下函数：

| 函数 | 当前问题 | 修改方式 |
|---|---|---|
| `validate_dataset()` | 只验证第一个 view | 遍历全部 `dataset.views` |
| `_protein_mapping()` | 使用单个 `view_feature_names` | 从 protein 对应的 feature-name 字典读取 |
| `build_node_index()` | 使用单个 feature-name 列表 | 只有 protein view 创建 protein node；其他 gene-aligned view 不新建 node |
| `build_node_features()` | `next(iter(dataset.views.items()))` | 遍历全部辅助视图并聚合到 gene-aligned graph matrix |
| `build_directed_hyperedges()` | 只处理 protein 特殊边 | protein 保持现有逻辑；scNMT gene-aligned view 无需 translation |
| `build_undirected_hyperedges()` | 只为第一个 view 建边 | 对 methylation 和 accessibility 分别创建 observation edges |
| `save_visualization_inputs()` | 假设一个 view | 输出每个 view 的 feature 信息和缺失率 |

## 9.5 scNMT 第一版的多视图聚合规则

为了控制修改范围，建议第一版：

- methylation、accessibility 都按 gene 名与 RNA gene 对齐；
- 每个 view 标准化后分别累加；
- 对每个 gene 按实际参与视图数取平均；
- 两个 view 都参与 `M_graph`；
- 不创建额外 methylation/accessibility node；
- observation hyperedges 可共享现有 `view2_obs` 类型，但在 edge metadata 中记录 modality；
- 后续若需要独立 gate，再拆成：
  - `methylation_obs`
  - `accessibility_obs`
  - `methylation_knn`
  - `accessibility_knn`

这样可以先完成基础数据集支持，不在本次标签规则修改中扩大模型结构变化。

---

# 10. 修改评估代码

## 10.1 `phasehyper/evaluation/phase.py`

在 `evaluate_embedding()` 开头增加：

```python
derived_clusters = int(np.unique(labels).size)

if int(n_clusters) != derived_clusters:
    raise ValueError(
        "n_clusters does not match unique label count: "
        f"provided={n_clusters}, derived={derived_clusters}"
    )
```

这样未来即使数据集对象中的 `n_clusters` 被错误写入，也不会静默进入 KMeans。

## 10.2 `phasehyper/evaluation/clustering.py`

现有核心逻辑已经正确：

```python
if expected_clusters is None:
    expected_clusters = np.unique(labels).size
```

并且 KMeans 使用：

```python
KMeans(n_clusters=expected_clusters)
```

保留该逻辑。

建议把合法范围改得更清楚：

```python
if expected_clusters > n_samples:
    raise ValueError(...)
```

并增加日志或返回字段：

```python
"expected_clusters": expected_clusters
```

最终每种表示的指标结构包含：

```json
{
  "NMI": 0.0,
  "FMI": 0.0,
  "ARI": 0.0,
  "ASW": 0.0,
  "ExpectedClusters": 3,
  "PredClusters": 3
}
```

这样图中可以明确区分：

- 根据标签提供给 KMeans 的类别数；
- KMeans 实际产生的类别数。

---

# 11. 测试计划

## 11.1 `tests/test_real_label_rules.py`

### PEA_STA

输入：

```python
[
    "0h control",
    "sample_0h_control",
    "6d contol",
    "6d_control",
    "6d BMP4",
]
```

期望：

```python
[
    "0h_control",
    "0h_control",
    "6d_control",
    "6d_control",
    "6d_BMP4",
]
```

并且：

```python
n_clusters == 3
```

无法识别标签：

```python
"day_unknown"
```

必须抛错，不能生成 `unknown` 或默认归入 BMP4。

### sc_GEM

输入不同 stage token，期望完全保留，仅去除空白。

### CITE_seq

输入不同 cell type，期望不合并、不执行 PEA 规则。

### SCoPE2

输入：

```python
["celltype", "sc_m0", "sc_u", "sc_m0"]
```

在配置允许表头时去掉 `celltype`，最终两类。

### scNMT

输入：

```python
["E5.5", "E6.5", "E7.5", "E5.5"]
```

最终三类，不修改名称。

## 11.2 `tests/test_real_label_alignment.py`

构造：

```text
RNA cells:   c1, c2, c3, c4
view cells:  c4, c2, c1
labels:      A,  B,  C,  D
```

目标 common-cell 顺序使用 RNA 顺序：

```text
c1, c2, c4
```

最终标签必须是：

```text
A, B, D
```

不能只是截取前三个 token。

还要测试：

- view 缺少 cell；
- view 顺序不同；
- 重复 cell ID；
- 标签数与 RNA cell 数不一致；
- 标签文件为空；
- target cell 不在 source cell 中。

## 11.3 `tests/test_phase_cluster_count.py`

为五个数据集构造标签向量，检查：

```python
n_clusters == len(np.unique(labels))
```

检查 `evaluate_phase_model()`：

- 传入正确 `n_clusters` 时成功；
- 传入错误值时抛错；
- metrics 中 `ExpectedClusters` 正确；
- NMI/FMI/ARI/ASW 都存在。

## 11.4 `tests/test_scnmt_loading.py`

构造最小数据：

```text
RNA            4 cells × 5 genes
methylation    4 cells × 5 genes
accessibility  4 cells × 5 genes
labels         E5.5/E6.5/E7.5
```

检查：

- `dataset.views` 同时包含两个辅助视图；
- 三个矩阵 cell 顺序一致；
- label 长度为 4；
- `n_clusters == 3`；
- `build_node_features()` 不只使用第一个 view；
- 数据可以进入模型构建阶段。

---

# 12. 实施顺序

## Phase 1：集中标签规则

- [ ] 在 `phasehyper/config.py` 增加 `REAL_PHASE_DATASETS`。
- [ ] 为五个基础数据集增加 `labels` 配置。
- [ ] 确认 CITE_seq 实际标签文件名。
- [ ] 新建 `phasehyper/data/labels.py`。
- [ ] 实现读取、归一化、编码和 cell-ID 对齐。
- [ ] 添加 PEA_STA 严格三类检查。

## Phase 2：统一所有数据加载路径

- [ ] 修改 `phasehyper/data/loading.py`。
- [ ] 删除 `has_ppi` 驱动的标签归一化。
- [ ] 修改普通、clean 和 hetero 三条加载路径。
- [ ] 更新 `phasehyper/data/__init__.py`。
- [ ] 添加标签规则和对齐测试。

## Phase 3：修改 `run_phase.py`

- [ ] 使用 `DATASET_CONFIG` 替代本地 `DATA_FILES`。
- [ ] 删除 `_pea_labels()`。
- [ ] 使用 `load_and_align_labels()`。
- [ ] 保存 label source、rule、counts 和 map。
- [ ] 加强 `validate_dataset()`。
- [ ] CLI 使用 `REAL_PHASE_DATASETS`。

## Phase 4：接入 scNMT

- [ ] 按配置加载所有辅助视图。
- [ ] 改造 metadata 为 per-view feature names。
- [ ] 修改所有 `next(iter(dataset.views...))` 的位置。
- [ ] 聚合 methylation 与 accessibility。
- [ ] 增加 scNMT 多视图测试。

## Phase 5：锁定评估契约

- [ ] 在 `evaluation/phase.py` 检查 `n_clusters` 一致性。
- [ ] metrics 增加 `ExpectedClusters`。
- [ ] 验证 KMeans 得到正确 `n_clusters` 参数。
- [ ] 验证可视化读取新 metrics 不受影响。

## Phase 6：端到端验证

分别运行：

```bash
python run_phase.py --dataset PEA_STA
python run_phase.py --dataset sc_GEM
python run_phase.py --dataset CITE_seq
python run_phase.py --dataset SCoPE2
python run_phase.py --dataset scNMT
```

检查每个输出目录：

```text
data_summary.json
metrics.json
cell_metadata.csv
```

必须记录：

```text
label_source
label_rule
label_counts
label_map
n_clusters
```

---

# 13. 最终验收标准

- [ ] `PEA_STA` 标签只来自标签文件，不从 cell ID 猜测。
- [ ] `PEA_STA` 无法识别的标签会报错，不生成 `unknown`。
- [ ] `sc_GEM` 原始 stage 不被合并。
- [ ] `CITE_seq` cell type 不被 PEA 规则改写。
- [ ] `SCoPE2` 保留原始两类或文件中实际类别。
- [ ] `scNMT` 可以通过 `run_phase.py` 正常加载。
- [ ] `scNMT` 的 methylation 和 accessibility 都进入数据对象。
- [ ] 不存在 `has_ppi -> label normalization` 逻辑。
- [ ] 所有标签先按 RNA cell ID 建映射，再按共同 cell 重排。
- [ ] `dataset.n_clusters == len(np.unique(dataset.labels))`。
- [ ] KMeans 使用 `dataset.n_clusters`。
- [ ] metrics 同时保存 `ExpectedClusters` 和 `PredClusters`。
- [ ] 五个数据集的 label source、rule、counts 和 map 均写入输出。
- [ ] 所有新增测试通过。
- [ ] simulation 和 ratio 数据集逻辑不在本次修改范围内。

---

## 14. 建议的最小修改优先级

若希望先快速修正最危险的问题，优先级如下：

1. **立即修复 `phasehyper/data/loading.py` 中 `has_ppi` 驱动的标签归一化。**
2. **让 PEA_STA 使用标签文件，而不是 cell ID。**
3. **统一 CITE_seq 标签文件配置。**
4. **增加标签与 cell ID 的映射对齐。**
5. **增加评估端 `n_clusters` 一致性断言。**
6. **最后完成 scNMT 多视图接入。**

前五项属于标签正确性修复；第六项属于完整数据集支持扩展。
