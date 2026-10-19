# 下一步开发计划（按优先级）

## M0–M3 本版已完成

- M0：`CsvStore` 支持 ZIP/目录；V6/V7 adapters 读取统一 `DatasetView` 并校验 cell/gene/element 键。
- M1：B1 表达、B2 印记、B3 chromatin/promoter、B4 TF→gene GRN 的可复现统计；V6/V7 对比。
- M2：V7 7 个明确的 imprinting 消融臂；成对 seed、motif/element-state 检查、结果输出。
- M3：按 seed bootstrap、PNG、Markdown、来源 SHA256、单元测试。

## 下一步优先级 1：C 的“完全无印记”负对照

**问题：**`upstream_only` 仍有上游 imprinting rule，所以它不是严格负对照。

**任务：**对生成器添加可配置的 upstream imprint开关，设计真实 `no_imprint` 条件。保持其他随机因子和 cell/gene 不变。必须校验非印记基因也不被额外改动。

**交付：**新条件 `no_imprinting_all_layers`、至少 5 个种子的负对照/原版配对结果；含来源与统计口径。

## 优先级 2：机制阶梯和强度扫描

按 region→promoter→GRN→expression 分别干预，检测每种机制对 imprint/preferred ratio 的影响。保持其他机制和随机背景完全一致。单调性扫描 `imprinting_strength` 的多个值；评估其是否在输出中真正可控制，避免以极端固定 `leak=0.03` 掩盖个体 strength。

**交付：**机制敏感性矩阵、seed-level 置信区间、strength-response 曲线，结论明确注明是生成器内敏感性，不是生物学真相。

## 优先级 3：邻近普通基因负效应

审计一个被指定为印记基因之后，它邻近普通基因的 promoter/表达是否被 region-level imprint 连带改变；比较距离、区域共享关系和印记基因邻域非印记负对照。若上游改动本意模拟 ICR cluster，应将 cluster 生物学定义写清，而不是把变化一律视作错误。

## 优先级 4：真实外部校准 + 模型有效性

真实测序的 allelic informative read sparsity、印记方向、甲基化/ICR 结构等。选择符合物种和实验协议的数据。然后才接队友最新 phasing 模型，固定模型 commit/数据 schema/分割策略，比较原版与审计后 simulation 的训练评估表现。

## 代码工程升级

- 将参考配置 YAML 真正接到 CLI，并留存 config SHA256。
- 为不同 V7 commits 建立正式插件或 generation hooks，去掉目前脆弱的源码字符串 patch。
- 统一 richer dataset validation（counts、library、annotation/metadata），为 A/D/E/F 预留插件口。
- 测试中增加 seed-level 回归 fixture 和 GitHub CI。
