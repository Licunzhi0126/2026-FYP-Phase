"""共享数据清洗纯函数（多组学相分解模型 · 数据阶段）。

设计原则：
- 每个函数无副作用，输入/输出都是 pandas / numpy，可单独 import 测试。
- 方向约定与现有加载一致：DataFrame 为 (n_cells, n_genes)，行=细胞、列=基因。
- 关键区别于 loading._load_feature_frame：本模块**不无差别 fillna(0)**，
  缺失语义交由各模态自行决定（见 fill_missing 的 strategy）。

与现有代码的衔接：
- 复用 utils._safe_standardize 做 z-score（内部已处理 NaN/Inf）。
- 复用 config.GENE_ALIASES 做基因名 canonical 化；数据列名规范后，
  下游 priors.py 的精确匹配（priors.py:28）即可命中 KEGG / 位置 / PPI。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phasehyper.utils import _safe_standardize


# ======================================================================
# 1. 读取（保留 NaN，不填 0）
# ======================================================================

def load_raw_frame(csv_path) -> pd.DataFrame:
    """读入 (n_cells, n_genes) 表，第一列为 cell_id。

    与 loading._load_feature_frame 行为一致（strip 列名、重名列/行取均值去重），
    但**保留 NaN**——不在此处填 0，缺失处理留给 fill_missing。
    非数值单元统一被 coerce 成 NaN。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df = df.set_index("cell_id")  # 注意：不 fillna
    df.columns = [str(col).strip() for col in df.columns]

    # 保证数值型；非数值（脏字符串）→ NaN，而不是悄悄保留为 object
    df = df.apply(pd.to_numeric, errors="coerce")

    # 重名列/行：取均值合并（skipna，沿用原加载语义，但 NaN 不被当作 0）
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()

    return df


# ======================================================================
# 2. 基因名 canonical 化
# ======================================================================

def standardize_gene_names(
    df: pd.DataFrame,
    alias_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """把基因列名规范成 canonical（行=细胞列=基因）。

    规则：
    - 列名（去空白后大写）若命中 alias_map，则改名为 alias_map[KEY][0]（首选 canonical）。
    - 改名后若产生重名列，取均值合并（skipna）。
    - **不做通用点号猜测**：alias 表之外仍含 "." 的列名只在日志里标为 suspicious，
      保持原名不动，避免错误合并。

    返回 (new_df, log)；log 字段：
      renamed   : [(old, new), ...]
      merged    : {canonical: [old1, old2, ...]}  改名后真正发生合并的列
      suspicious: [col, ...]  仍含点号、未在别名表内、需人工确认
    """
    alias_map = {str(k).strip().upper(): v for k, v in alias_map.items()}

    rename: Dict[str, str] = {}
    renamed_log: List[Tuple[str, str]] = []
    suspicious: List[str] = []

    for col in df.columns:
        col_s = str(col).strip()
        key = col_s.upper()
        if key in alias_map and alias_map[key]:
            canonical = str(alias_map[key][0]).strip()
            rename[col] = canonical
            if canonical != col_s:
                renamed_log.append((col_s, canonical))
        else:
            rename[col] = col_s
            if "." in col_s:
                suspicious.append(col_s)

    new_df = df.rename(columns=rename)

    # 改名后可能出现重名（多个旧名指向同一 canonical），取均值合并
    merged: Dict[str, List[str]] = {}
    if new_df.columns.duplicated().any():
        # 记录哪些 canonical 由多个旧名合并而来
        new_to_olds: Dict[str, List[str]] = {}
        for old, new in rename.items():
            new_to_olds.setdefault(new, []).append(str(old).strip())
        merged = {new: olds for new, olds in new_to_olds.items() if len(olds) > 1}
        new_df = new_df.T.groupby(level=0).mean().T

    log = {
        "n_input_genes": int(df.shape[1]),
        "n_output_genes": int(new_df.shape[1]),
        "renamed": renamed_log,
        "merged": merged,
        "suspicious": sorted(set(suspicious)),
    }
    return new_df, log


# ======================================================================
# 3. 缺失掩码 + 缺失填充
# ======================================================================

def build_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """布尔掩码：True = 原始缺失（NaN）。必须在任何填充之前调用。"""
    return df.isna()


def fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """按策略填充缺失。

    strategy:
      "zero"         表达：NaN -> 0（视为未表达）。
      "min_observed" 蛋白：NaN -> 该列最小观测值（≈检测限 LOD）；整列全缺失 -> 0。
      "keep_nan"     甲基化：NaN = 无 CpG 覆盖 ≠ 0，**保留 NaN 不填**。
    """
    if strategy == "zero":
        return df.fillna(0.0)

    if strategy == "keep_nan":
        return df.copy()

    if strategy == "min_observed":
        filled = df.copy()
        col_min = filled.min(axis=0, skipna=True)
        fill_values = col_min.where(col_min.notna(), 0.0)
        # 逐列用各自的近似 LOD 填充
        filled = filled.fillna(fill_values)
        return filled

    raise ValueError(
        f"Unknown fill strategy: {strategy!r} (expected zero / min_observed / keep_nan)"
    )


# ======================================================================
# 4. 过滤无信息基因列 / 死细胞行
# ======================================================================

def _has_signal(series: pd.Series) -> bool:
    """该列/行是否含有效信号：存在至少一个有限且非零的值。"""
    vals = series.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    return finite.size > 0 and bool(np.any(finite != 0.0))


def filter_uninformative(
    expr_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """删除无信息基因列与死细胞行（两模态对齐处理）。

    - 基因（列）：在 expression **或** 第二模态任一有信号即保留；两者都无信号才删。
    - 细胞（行）：在两模态都无信号（全 0 / 全 NaN）才删。

    要求两个 DataFrame 的细胞集合一致；基因列取并集后按上述规则裁剪，
    返回的两个 DataFrame 拥有完全一致、同序的行索引与列。
    """
    # 先对齐细胞（行）：取交集，保持 other_df（第二模态）的原始顺序——
    # 与现有 loader 用 view1 的 index 决定细胞顺序的语义一致。
    common_cells = [c for c in other_df.index if c in set(expr_df.index)]
    expr_df = expr_df.loc[common_cells]
    other_df = other_df.loc[common_cells]

    # 基因列并集
    all_genes = list(dict.fromkeys(list(expr_df.columns) + list(other_df.columns)))
    expr_df = expr_df.reindex(columns=all_genes)
    other_df = other_df.reindex(columns=all_genes)

    # 基因过滤
    keep_genes: List[str] = []
    dropped_genes: List[str] = []
    for g in all_genes:
        if _has_signal(expr_df[g]) or _has_signal(other_df[g]):
            keep_genes.append(g)
        else:
            dropped_genes.append(g)
    expr_df = expr_df[keep_genes]
    other_df = other_df[keep_genes]

    # 细胞过滤：两模态都无信号才删
    keep_cells: List[str] = []
    dropped_cells: List[str] = []
    for c in common_cells:
        if _has_signal(expr_df.loc[c]) or _has_signal(other_df.loc[c]):
            keep_cells.append(c)
        else:
            dropped_cells.append(c)
    expr_df = expr_df.loc[keep_cells]
    other_df = other_df.loc[keep_cells]

    log = {
        "n_genes_in": len(all_genes),
        "n_genes_kept": len(keep_genes),
        "dropped_genes": dropped_genes,
        "n_cells_in": len(common_cells),
        "n_cells_kept": len(keep_cells),
        "dropped_cells": dropped_cells,
    }
    return expr_df, other_df, log


# ======================================================================
# 5. 数值变换
# ======================================================================

def transform_values(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """按模态做数值变换。

    method:
      "none"         二值甲基化：保持原值（含 NaN），不标准化、不 log。
      "zscore"       已是 log 尺度的连续模态：仅按列 z-score。
      "log1p_zscore" 右偏/计数样连续模态：先 log1p 再按列 z-score。

    注意：zscore / log1p_zscore 复用 utils._safe_standardize，其内部会把 NaN/Inf
    视作 0，故应在 fill_missing 之后、对无 NaN 的连续模态调用。
    """
    if method == "none":
        return df.copy()

    if method == "zscore":
        arr = _safe_standardize(df.to_numpy())
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    if method == "log1p_zscore":
        arr = np.log1p(np.nan_to_num(df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
        arr = _safe_standardize(arr)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    raise ValueError(
        f"Unknown transform method: {method!r} (expected none / zscore / log1p_zscore)"
    )


def zscore_observed(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """仅按**观测值**（非 NaN）逐列 z-score，NaN 保留不动（无覆盖 ≠ 0）。

    用于 keep_nan 模态（甲基化 / 可及性）：`transform_values("zscore")` 经 _safe_standardize
    会把 NaN 当 0、污染列均值方差，并把无覆盖伪装成"正好平均"。本函数只用每列的观测值算
    mean/std 标准化观测值，无覆盖处仍为 NaN——下游 builder.gene_features 的 nan_to_num(nan=0)
    会把它置中性 0，而观测值已是相对自身分布的真实 z 值。
    """
    arr = df.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        obs = np.isfinite(col)
        x = col[obs]
        if x.size >= 2 and x.std() > eps:
            out[obs, j] = (x - x.mean()) / x.std()
        elif x.size >= 1:
            out[obs, j] = 0.0  # 观测全相同 → 中性 0（仍区别于 NaN 无覆盖）
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def describe_distribution(df: pd.DataFrame) -> Dict[str, object]:
    """对一个模态做分布画像，并给出变换建议（仅建议，最终由驱动配置决定）。"""
    vals = df.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"empty": True, "suggest": "none"}

    nonzero = finite[finite != 0.0]
    uniq = np.unique(finite)
    looks_binary = uniq.size <= 2 and set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0})

    # 简单偏度（基于非零值，避免大量 0 拉低）
    base = nonzero if nonzero.size > 1 else finite
    mean = float(base.mean())
    std = float(base.std())
    skew = float(((base - mean) ** 3).mean() / (std ** 3)) if std > 0 else 0.0

    vmax = float(finite.max())
    if looks_binary:
        suggest = "none"
    elif vmax > 50.0 or skew > 2.0:
        suggest = "log1p_zscore"
    else:
        suggest = "zscore"

    return {
        "empty": False,
        "min": float(finite.min()),
        "max": vmax,
        "q50": float(np.quantile(finite, 0.50)),
        "q99": float(np.quantile(finite, 0.99)),
        "frac_zero": float((finite == 0.0).mean()),
        "n_unique": int(uniq.size),
        "skew_nonzero": skew,
        "looks_binary": bool(looks_binary),
        "suggest": suggest,
    }


# ======================================================================
# 5b. scRNA 计数专用：物种前缀过滤 / 文库归一+log1p / HVG 选择
#     （CITE_seq 等原始 UMI counts 数据用；sc_GEM/PEA 已 log 尺度，不走这条）
# ======================================================================

def filter_prefixed_genes(df: pd.DataFrame, keep_prefix: str = "HUMAN_") -> pd.DataFrame:
    """只保留列名以 keep_prefix 开头的基因列并剥掉前缀（如 HUMAN_A1BG -> A1BG）。

    barnyard 数据（CITE_seq）列名前缀有 HUMAN_/MOUSE_/ERCC_，这里只取 HUMAN_。
    剥前缀后若产生重名列，取均值合并（skipna）。
    """
    keep_cols = [c for c in df.columns if str(c).startswith(keep_prefix)]
    sub = df[keep_cols].copy()
    sub.columns = [str(c)[len(keep_prefix):].strip() for c in keep_cols]
    if sub.columns.duplicated().any():
        sub = sub.T.groupby(level=0).mean().T
    return sub


def library_normalize_log1p(df: pd.DataFrame, target_sum: float = 1e4) -> pd.DataFrame:
    """原始 UMI counts → 每细胞（行）文库大小归一到 target_sum，再 log1p。

    行=细胞、列=基因。NaN 视作 0 计入；零文库细胞除数兜底为 1（结果仍全 0）。
    这是 scRNA 标准 CP10K(+log1p)；区别于 sc_GEM/PEA「已 log，直接 z-score」。
    """
    mat = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    lib = mat.sum(axis=1, keepdims=True)
    lib = np.where(lib <= 0.0, 1.0, lib)
    normed = np.log1p(mat / lib * float(target_sum))
    return pd.DataFrame(normed, index=df.index, columns=df.columns)


def select_hvg(
    df: pd.DataFrame,
    n_top: int = 2500,
    force_include: Optional[List[str]] = None,
) -> Dict[str, object]:
    """按列方差选 top-N 高变基因（HVG），强制并入 force_include 中存在的基因。

    约定 df 已是 log 归一尺度（先 library_normalize_log1p）。返回：
      genes        : 选中的基因名列表（方差降序；强制基因若不在 top-N 则补到末尾）
      n_var_selected : 纯按方差选中的数量（不含强制补入）
      forced_added : 因强制并入而额外补入的基因（原本不在 top-N）
      forced_missing : force_include 里在数据中根本不存在的基因（断桥风险，需上报）
    """
    force_include = list(force_include or [])
    genes = list(df.columns)
    var = np.nanvar(df.to_numpy(dtype=float), axis=0)
    order = np.argsort(-var)  # 方差降序
    ranked = [genes[i] for i in order]

    n_top = max(0, min(int(n_top), len(ranked)))
    selected = list(ranked[:n_top])
    sel_set = set(selected)

    present = set(genes)
    forced_added = [g for g in force_include if g in present and g not in sel_set]
    forced_missing = [g for g in force_include if g not in present]
    selected.extend(forced_added)

    return {
        "genes": selected,
        "n_var_selected": len(sel_set),
        "forced_added": forced_added,
        "forced_missing": forced_missing,
    }


# ======================================================================
# 6. 先验命中统计（清洗前后对比）
# ======================================================================

def _read_prior_gene_set(path, *, upper: bool) -> set:
    """读取先验文件第 0 列（tab 分隔）作为基因集合。"""
    path = Path(path)
    if not path.exists():
        return set()
    genes = set()
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0], na_values=["NA", ""])
    for g in df.iloc[:, 0].tolist():
        g = str(g).strip()
        if not g or g.lower() == "nan":
            continue
        genes.add(g.upper() if upper else g)
    return genes


def prior_hit_report(
    genes: List[str],
    *,
    kegg_path=None,
    position_path=None,
) -> Dict[str, object]:
    """统计 genes 对 KEGG / 位置先验的命中（精确匹配，复刻 priors.py 的口径）。

    - KEGG：priors.py:28 用大小写敏感的精确匹配。
    - 位置：_load_position_prior 以大写候选匹配，这里同样按大写比较。
    返回命中数、总数、未命中名单。
    """
    genes = [str(g).strip() for g in genes]
    out: Dict[str, object] = {"n_genes": len(genes)}

    if kegg_path is not None:
        kegg = _read_prior_gene_set(kegg_path, upper=False)
        miss = [g for g in genes if g not in kegg]
        out["kegg"] = {"hit": len(genes) - len(miss), "total": len(genes), "missing": miss}

    if position_path is not None:
        pos = _read_prior_gene_set(position_path, upper=True)
        miss = [g for g in genes if g.upper() not in pos]
        out["position"] = {"hit": len(genes) - len(miss), "total": len(genes), "missing": miss}

    return out


# ======================================================================
# 7. 对齐校验
# ======================================================================

def align_check(
    expr_df: pd.DataFrame,
    other_df: pd.DataFrame,
    stage_tokens: Optional[List[str]],
) -> Dict[str, object]:
    """断言两模态同细胞索引、同基因列，且 stage token 数 == 细胞数。"""
    issues: List[str] = []
    if list(expr_df.index) != list(other_df.index):
        issues.append("expression / second-modality cell index mismatch")
    if list(expr_df.columns) != list(other_df.columns):
        issues.append("expression / second-modality gene columns mismatch")
    if stage_tokens is not None and len(stage_tokens) != len(expr_df.index):
        issues.append(
            f"stage token count ({len(stage_tokens)}) != n_cells ({len(expr_df.index)})"
        )
    return {
        "ok": not issues,
        "issues": issues,
        "n_cells": int(expr_df.shape[0]),
        "n_genes": int(expr_df.shape[1]),
    }


# ======================================================================
# 8. stage 读取 / 写出 + 产物落盘
# ======================================================================

def load_stage_tokens(stage_path) -> List[str]:
    """读取单行、逗号分隔的 stage token（与 loading._read_stage_tokens 同口径）。"""
    stage_path = Path(stage_path)
    if not stage_path.exists():
        return []
    text = stage_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    line = text.splitlines()[0].strip()
    return [tok.strip() for tok in line.split(",") if tok.strip()]


def write_clean_outputs(
    out_dir,
    *,
    expr_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_filename: str,
    masks: Dict[str, pd.DataFrame],
    stage_tokens: List[str],
    report_md: str,
    prior_files: Optional[Dict[str, Path]] = None,
    ppi_df: Optional[pd.DataFrame] = None,
) -> Path:
    """把清洗产物写入 data_clean/<dataset>/。

    产出文件（文件名沿用 DATASET_CONFIG，便于直接当数据根使用）：
      expression_data.csv, <other_filename>, cell_stage.csv,
      *_missing_mask.csv（每个模态一份）,
      hsa00001.txt / gene_positions_*.txt（从 prior_files 原样复制）,
      human_ppi.csv（若提供 ppi_df，裁剪后的子矩阵）,
      cleaning_report.md
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清洗后两模态（index 即 cell_id，回读时 loader 会把首列重命名为 cell_id）
    expr_df.to_csv(out_dir / "expression_data.csv")
    other_df.to_csv(out_dir / other_filename)

    # 缺失掩码
    for fname, mask in masks.items():
        mask.to_csv(out_dir / fname)

    # cell_stage：单行逗号分隔，顺序与清洗后 index 一致
    (out_dir / "cell_stage.csv").write_text(",".join(stage_tokens), encoding="utf-8")

    # 先验文件原样复制（基因名已在数据侧 canonical 化，精确匹配可命中）
    if prior_files:
        for dst_name, src_path in prior_files.items():
            src_path = Path(src_path)
            if src_path.exists():
                (out_dir / dst_name).write_text(
                    src_path.read_text(encoding="utf-8", errors="ignore"),
                    encoding="utf-8",
                )

    # 裁剪后的 PPI 子矩阵
    if ppi_df is not None:
        ppi_df.to_csv(out_dir / "human_ppi.csv")

    (out_dir / "cleaning_report.md").write_text(report_md, encoding="utf-8")
    return out_dir


def trim_ppi_submatrix(ppi_path, genes: List[str]) -> pd.DataFrame:
    """从大 PPI 邻接矩阵中只裁出 genes×genes 子矩阵（不整张读入内存）。

    复刻 priors.py:310-322 的 usecols 取列法：先读表头定位需要的列，再
    usecols 读子集、按行筛 genes，得到 n×n（n=命中基因数）的小矩阵。
    """
    ppi_path = Path(ppi_path)
    genes = list(dict.fromkeys(str(g).strip() for g in genes))
    gene_set = set(genes)

    header_cols = list(pd.read_csv(ppi_path, nrows=0, sep=",").columns)
    pos_by_gene: Dict[str, int] = {}
    for pos, col in enumerate(header_cols):
        if pos == 0:  # 第 0 列是行索引（基因名）
            continue
        pos_by_gene.setdefault(str(col).strip(), pos)
    needed_pos = sorted(pos_by_gene[g] for g in genes if g in pos_by_gene)

    ppi_df = pd.read_csv(ppi_path, usecols=[0] + needed_pos, sep=",")
    ppi_df = ppi_df.set_index(ppi_df.columns[0]).fillna(0)
    ppi_df.index = ppi_df.index.astype(str).str.strip()
    ppi_df.columns = [str(col).strip() for col in ppi_df.columns]
    ppi_df = ppi_df.loc[ppi_df.index.isin(gene_set)]
    if ppi_df.index.duplicated().any():
        ppi_df = ppi_df[~ppi_df.index.duplicated(keep="first")]
    return ppi_df
