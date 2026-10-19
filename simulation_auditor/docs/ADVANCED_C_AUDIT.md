# Causal Audit 扩展：四项新增实验 (v0.3)

## 目标与范围
新增 C1 完整无印记负对照、C2 分层机制干预、C3 印记强度剂量扫描、C4 非印记邻近基因溢出测试。这四项属于 **simulator 内部受控干预**，不能直接推断真实生物学因果。原 `simulation_generation/` 完全不修改。B 模块、原 7 组消融和历史数据导入保持兼容。

## C1：完整无印记 (`fully_no_imprinting`)
保留 6 个标记基因的 ID、偏好方向和 `imprinting_strength` 用于配对评估，同时关闭 imprint 驱动的 5 条干预路径：region、promoter、GRN 缩放、表达内部乘数、表达末端 log gate。这样不是把印记标签删掉——否则无法评估原来同一组目标基因的自然差异。`imprinting_validation` 在负对照仍报告相同目标基因的 P/M 倍数。

## C2：机制分层
`region_off`: 关闭 imprint-specific region state override（仍保留原有自然等位染色质偏好）；`promoter_imprint_off`: 关闭 imprint-specific promoter open-state/strength/accessibility override（仍保留非印记 promoter 调控）；`region_only`: 仅保留印记对 region 的显式影响；`promoter_only`: 仅保留印记对 promoter 的显式影响；`region_promoter_only`: 保留两种上游机制而去除 GRN/inner/final（与旧 `upstream_only` 等价）。原 7 个条件保留。

为了避免源生成器单条 RNG 导致的整个下游随机背景错位，region / promoter counterfactual 使用复制 RNG 状态生成干预元件状态，并让原始调用在主 RNG 上消耗抽样。Region 层仅替换印记相关区域，其余 region 状态保留原值。主 RNG 仍会生成与 baseline 一致的后续 motif / TF 活动，配对检查会对其进行验证。

**注意**：干预后的 reg state/element state 当然允许改变。配对检查不要求干预变量完全相同，而要求基因身份、imprint strength、motif 等前置不相关对象一致。模型依赖的下游 contact / GRN / expression 可以变化。

## C3：strength 扫描
`strength-scan` 在 `assign_imprinting` 之后只覆盖被指定的印记基因 `imprinting_strength`；**不**改变 `imprinting_nonpreferred_leak=0.03`，也不改变基因 ID、表达偏好方向。因此这是「strength 参数在固定强抑制背景下是否真正控制表达」的诊断，不是总体印记效应强度标定。`strength=0` 只表示这个参数为 0，**不等于 no-imprinting**。由于不同 strength 的 region override 会改变采样分支，除身份和原 seed 外，不保证所有 downstream random draws 相同；适合通过多种子比较总体剂量趋势。

`C_strength_gene_detail.csv`：每个 seed × strength × gene 的原始效应；`C_strength_per_seed.csv`：每 seed 平均 log2 preference 等；`C_strength_summary.csv`：各 strength 汇总；`C_strength_scan.png`：每个 seed 的趋势及跨 seed 平均线。

## C4：Region 扩散与邻近非印记基因
`neighbors` 比较原版和 region_off。**邻近**在这里严格定义为与印记基因相关调控元件共享 `chromatin_region_id` 的非印记基因，或其本身位于该 region。其他非印记基因作为内部对照。计算每个基因在原版和 region_off 中的 log2(P/M) 差与 log2(total expression) 差，然后对每个 seed 与分组汇总绝对差异。

`C_neighbors_regions.csv` 会记录每个 region 在两组的状态是否改变；预期非印记相关 region 的状态保持一致。`C_neighbors_gene_detail.csv` 给出每 seed 每 gene 的原始差异；`C_neighbors_per_seed.csv` 和 `C_neighbors_summary.csv` 汇总三组；`C_neighbor_spillover.png` 画种子级组间分布。**两 seed 仅为 smoke/integration test，不能用于发表级推断**。

## 运行（项目根目录）
```bash
python -m pytest -q simulation_auditor/tests
python -m simulation_auditor b --v6 /path/SIM_V6.zip --v7 /path/simulation.zip --out outputs/simulation_auditor/B
python -m simulation_auditor c --generator simulation_generation --start 42 --seeds 12 --out outputs/simulation_auditor/C
python -m simulation_auditor strength-scan --generator simulation_generation --start 42 --seeds 12 --strengths 0 .25 .5 .75 .95 --out outputs/simulation_auditor/strength
python -m simulation_auditor neighbors --generator simulation_generation --start 42 --seeds 12 --out outputs/simulation_auditor/neighbors
```

## 组会汇报建议
先 B 介绍 V6/V7 的表型差异，再 C1 指定印记基因的原始与全关闭倍数，再 C2 说明强偏好主要出在哪一层，然后 C3 展示 strength 是否有剂量响应，最后 C4 说明 imprint-region 是否影响周边非印记基因。不可宣称「biologically invalid」而应表述为「对生成机制的敏感性揭示某一假设还需校准」。
