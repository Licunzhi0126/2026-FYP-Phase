# B/C 指标定义、含义与不能推导的结论

## B1 — Expression

每个基因跨细胞平均：`P_g = mean_c(E^p_cg)`、`M_g = mean_c(E^m_cg)`。

- `log2_p_over_m = log2((P_g+1e-12)/(M_g+1e-12))`
- `fold_change = 2**abs(log2_p_over_m)`，所以它必定 ≥1，不表示偏好方向。
- `expression_genes_gt3 / gt10`：严格大于 3 / 10 倍的基因数。
- `max_gene_fold`、`median_gene_fold`：按 gene 统计，不是按 cell-gene 统计。
- `cell_gene_balanced_40_60`：对于每个 cell-gene，计算 `E_p/(E_p+E_m+1e-12)` 在 `[.4,.6]` 的占比。保留与旧审计结果一致的全部 cell-gene 口径。
- `cell_gene_balanced_40_60_covered`：额外对 `E_p+E_m>1e-8` 的 cell-gene 计算，作为低表达敏感性检查。
- `median_gene_paternal_maternal_cell_corr`：每个 gene 在 160 个 cell 的父本、母本真实表达 Pearson 相关系数，中位数；常数向量跳过。

这些指标告诉我们偏好程度与分布的形态，但不能直接判定什么强度才符合真实印记生物学。

## B2 — Explicit Genomic Imprinting (V7 only)

- `n_explicit_imprinted`：`imprinting_status!='none'` 的基因数。
- `imprint_direction_accuracy`：`paternal_expressed` 需 `log2_p_over_m>0`；`maternal_expressed` 需小于 0。
- `imprint_median_preferred_ratio`：已标记 imprinting genes 的偏好倍数中位数。此处假定方向基本正确；方向正确率另报。
- `imprint_strength_vs_fold_spearman`：`imprinting_strength` 对表达偏好倍数的秩相关；n=6 时只做探索性指标。
- `imprint_promoter_direction_accuracy`：偏好亲本与非偏好亲本的 promoter accessibility 差方向一致率。
- V6 对应指标设为 `N/A`，绝非 0 或 Fail。

## B3 — Chromatin / promoter

- `element_accessibility_corr` / `element_methylation_corr`：按 element 对齐后的父母 Pearson 相关。
- `promoter_accessibility_corr`：仅 promoter elements 的父母 accessibility Pearson 相关。
- `promoter_vs_expression_spearman`：跨基因比较 `log2((access_p+1e-3)/(access_m+1e-3))` 与 `log2(P_g/M_g)` 的 Spearman。
- 该 Spearman 是**静态关联**。即使非常高，也无法说明 promoter 造成 expression 差异；因果需要 C 的受控干预。

## B4 — GRN

- 这里计算 `paternal_GRN.csv` 与 `maternal_GRN.csv`，**TF→gene 的权重矩阵**；绝不是 per-cell 或 gene2gene GRN。
- 当 `abs(weight)>1e-8` 视为有边（可使用 `--grn-threshold` 修改）。
- `grn_shared_fraction = |E_p∩E_m|/|E_p∪E_m|`。
- `grn_edges_paternal_only`、`grn_edges_maternal_only`：父/母独有边数量。
- `grn_weight_corr_on_union`：并集有边位置上，两个权重数组的 Pearson 相关系数。
- **极重要**：曾经引用的 99.13% 可能来自另一个 GRN 表示；不能和本程序的 TF→gene 指标交叉替换。论文中必须写具体 GRN 层级与阈值。

## C — Generator intervention

- `preferred_expression_ratio`：生成器提供的 imprinting 验证记录中的偏好亲本 / 非偏好亲本的平均 latent expression 比值。
- `C_gene_detail.csv` 以一个 seed 内的每一个 imprinting gene 为观察；不是独立模拟重复。
- `C_per_seed.csv` 首先聚合成每个 seed × condition 的 `mean_log2_ratio`，用此作为配对比较的统计重复。
- `C_paired_seed_effects.csv` 把同一个 seed 的 original 减去干预条件的 mean_log2_ratio，对 seed 的差值 bootstrap 重采样 5000 次，取 2.5%/97.5% 分位数。seed=1 时 CI 没有统计意义，主要用于 smoke test。
- `C_summary.csv` 的 `ratio_median` 是所有 (seed,gene) 的中位数，和“每 seed 中位数再求中位数”**不是同一个数**。

**不应使用这些统计去断言**“真实 imprinting 不靠 promoter”“某个模块无效”或“生成器不符合真实生物学”。正确表述：对目前这个生成器的反事实删除/替换，某一输出指标有多大响应。
