# B 模块数据约定与版本差异

## DatasetView 的统一格式

`DatasetView` 在 `core/schema.py`：

- `p_true, m_true`：二维 `DataFrame`，**行=cell_id，列=gene_id**，数值为 *latent true expression*。
- `p_grn, m_grn`：**行=TF_id，列=gene_id** 的权重矩阵；CSV 的 `Unnamed: 0` 被视为 TF 行名。
- `genes`：`gene_id` 唯一；V7 `genes` 提供 `imprinting_status` 和 `imprinting_strength`。
- `elements`：`element_id` 唯一；含 `element_type`、`candidate_target_gene`。
- `p_element_state, m_element_state`：`element_id` 唯一，含 `accessibility` / `methylation`。
- `regions`：预留区域结构；当前 B1-B4 没有完全消费区域字段。
- `capabilities`：`expression`, `chromatin`, `grn`；只有 V7 才包含 `explicit_imprinting`。

## 原始 CSV 文件位置

| 语义 | V6 | V7 |
|---|---|---|
| 父本真实表达 | `input/omics/paternal_true_expression.csv` | `ground_truth/allele_expression/paternal_true_expression.csv` |
| 母本真实表达 | `input/omics/maternal_true_expression.csv` | `ground_truth/allele_expression/maternal_true_expression.csv` |
| 元件 | `input/omics/regulatory_elements.csv` | `input/shared_priors/regulatory_elements.csv` |
| 父本元件状态 | `input/omics/paternal_element_state.csv` | `ground_truth/chromatin/paternal_element_state.csv` |
| 母本元件状态 | `input/omics/maternal_element_state.csv` | `ground_truth/chromatin/maternal_element_state.csv` |
| 父母 GRN | `input/omics/{paternal,maternal}_GRN.csv` | `ground_truth/grn/{paternal,maternal}_GRN.csv` |
| 基因注释 | `input/omics/gene_info.csv` | `ground_truth/imprinting/gene_info_full.csv` |
| 区域注释 | `input/chromatin_regions.csv` | `input/chromatin_regions.csv` |

## 为什么需要适配器

不能对任意 `combined_counts.csv` 名字做匹配：V6 顶层同名 CSV 是连续表达，与 `input/omics/combined_counts.csv` 的整数 counts 语义不同。本阶段 B 故意不使用这两个矩阵，避免混淆不同观察层级。

`core/validation.py` 保障：cell/gene 顺序不一致要抛错，不能直接拿 numpy 按位置相除；重复 cell、gene、element ID 抛错；NaN/Inf 抛错。

**适配器本身不“补全”不存在的 truth**：V6 没有同语义的 imprinting truth，就保留 N/A。以后扩展 V8，优先写一个 `adapters/v8.py`，不复制 B 模块。
