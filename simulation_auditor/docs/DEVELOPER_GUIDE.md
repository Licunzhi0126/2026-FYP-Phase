# simulation_auditor v0.2 — 完整代码说明与维护手册

> 这是项目的**逐文件说明**。面向刚开始接触 Simulation 生成代码的维护者。建议从项目总入口开始阅读，再学习 B 的数据流，最后进入 C 的干预源代码。

## 0. 这是一个什么项目？

我们已经有 `simulation_generation/`，它负责造出具有父本和母本差异的基因表达、染色质状态、调控网络等数据。

Auditor 不负责“造数据”。它负责回答两个问题：

1. **B：造出来的数据看上去具有什么生物学表型？** 比如 6 个显式印记基因是否表现出很大的亲本表达差异，其他 gene 是否大致平衡。
2. **C：这些表型主要由生成代码中哪一个规则产生？** 例如 promoter 已经强烈父母不同了，最后 expression gate 是否仍决定大部分观察到的极端比值。

完整调用：

```text
terminal -> cli.py
          |--- B --> biological/pipeline.py
          |           |--- adapters/v6.py 或 v7.py
          |           |        |-- core/io.py
          |           |        |-- core/validation.py
          |           |        \-- core/schema.py
          |           |--- biological/expression.py
          |           |--- biological/chromatin.py
          |           |--- biological/imprinting.py
          |           |--- biological/grn.py
          |           \--- reporting/plots.py
          |
          |--- C --> causal/runner.py
          |           |-- causal/interventions.py
          |           |-- causal/v7_bridge.py (只在内存中执行V7源码)
          |           |-- causal/imprinting.py (配对不变性检查)
          |           \-- causal/statistics.py
          |                       \-- reporting/plots.py
          |
          \--- report --> reporting/report.py
                                 \-- reporting/compare.py
```

最重要的方向约束：**adapters 了解各版 CSV 路径，biological 不应该知道路径，causal/v7_bridge 是唯一处理生成器源码的地方，reporting 不能偷偷重算科研统计量。**

---

## 1. 顶层代码

### `__init__.py`

声明包名与 `__version__="0.2.0"`。其他代码可通过 `import simulation_auditor` 引入。没有任何生成器执行副作用。

### `__main__.py`

`python -m simulation_auditor` 的真正入口。它只调用 `cli.main()`，不会自行读取数据。

### `cli.py`

定义四条已经实现的用户命令：

- `b`：从 V6/V7 数据路径启动一次 B 静态审计；接受其中任意一个或两个版本。
- `c`：针对可信 V7 代码按 `seed × condition` 实际生成反事实数据。
- `import-c`：导入此前生成并保存的 C 实验 CSV，产生和 `c` 同格式的输出。
- `report`：根据当前目录中已经存在的 B/C CSV 生成 Markdown。

`cli.py` 不负责科研数学计算；它只解析命令行参数、调用正确的 pipeline、决定输出目录。

示例：

```powershell
python -m simulation_auditor b --v6 "SIM_V6(1).zip" --v7 "simulation.zip" --out "outputs\audit_BC"
python -m simulation_auditor c --generator "simulation_generation.zip" --start 42 --seeds 2 --out "outputs\audit_BC"
python -m simulation_auditor report --out "outputs\audit_BC"
```

---

## 2. `core/`：所有模块都依赖的基础设施

### `core/io.py` — ZIP/目录读取器

`CsvStore(path)` 可读取一个 ZIP 或者一个已解压的数据目录。`locate(suffix)` 用路径后缀找到所需 CSV（如 `ground_truth/allele_expression/paternal_true_expression.csv`），并排除 macOS 的 `__MACOSX` 等辅助文件；`read(suffix)` 得到 pandas DataFrame；`close()` 关闭 ZIP 句柄。

为什么这里不能直接遍历任意 `combined_counts.csv`？因为 V6 的同名文件不同目录可能含有**完全不同的语义**。B 主体只能通过明确的 adapter 映射读取，不会凭文件名猜。

### `core/schema.py` — 统一数据契约

`DatasetView` 定义所有 B 模块能拿到的数据。主要字段：

- `version`：V6 / V7；
- `p_true/m_true`：cell × gene 的潜在父本/母本真实表达；
- `genes`：基因注释表；
- `elements`：调控元件表（promoter/enhancer/...）；
- `p_element_state/m_element_state`：两亲本元件层的开放/甲基化信息；
- `p_grn/m_grn`：TF × gene 的调控权重矩阵；
- `regions`：染色质区域注释（目前加载并预留更深入分析）；
- `capabilities`：该版本真正提供哪些审计能力；
- `notes`：语义提醒。

`AuditFinding` 是以后引入异常分级与证据追踪的结构类型，当前仅定义，没有参与自动判分。**不能宣称已经做了规则引擎**。

### `core/validation.py` — 校验对齐是否正确

- `matrix_from_csv` 将 CSV 的 `cell_id` 变为 index，并拒绝重复 ID、非法数值。
- `require_pair(p,m)` 要求 paternal/maternal **行列顺序和名字全部一致**。这能避免拿 A 基因的父本表达除以 B 基因的母本表达。
- `require_unique` 确保 gene/element ID 不重复。

这些是工程逻辑错误校验，不是生物学合理性判断。

### `core/provenance.py` — 可追溯性

- `sha256_file()`：对来源 ZIP 或脚本计算摘要。可证明两次实验是否使用相同文件字节。
- `json_write()`：保存 provenance（来源、种子、口径、干预等）为 JSON。

注意：如果传入的是目录，`sha256_file(directory)` 结果为 null，不会自动哈希整个目录；C 会额外记录原始 generator source 的 SHA256。未来应补充 Git commit 记录与整体目录 manifest。

---

## 3. `adapters/`：V6/V7 适配层

### `adapters/base.py`

定义 `Adapter` 基类及统一方法 `load(path)->DatasetView`。未来新增 V8，只需实现一个 `V8Adapter`，B 的算法不应该复制粘贴。

### `adapters/v6.py`

职责：

1. 在 V6 `input/omics/` 路径读取父母真实表达、父母 TF→gene GRN、调控元件和元件状态。
2. 在 `input/chromatin_regions.csv` 加载区域注释。
3. 校验真实表达的 cell/gene 对齐，gene/element ID 唯一。
4. 标记能力为 `expression/chromatin/grn`，**不提供 `explicit_imprinting`**。

为什么没有？V6 的 `layer_sim_imprint.csv` 标签是 sim_regstate，不能和 V7 的 `paternal_expressed/maternal_expressed` 直接等同。

### `adapters/v7.py`

职责类似，不过文件来自 `ground_truth/allele_expression/`、`ground_truth/chromatin/`、`ground_truth/grn/` 等多个文件夹。最重要的是它读取 `ground_truth/imprinting/gene_info_full.csv`，该文件包含真正的 imprinting_status 与 strength 标签；把 `explicit_imprinting` 加到 `capabilities`。

这里允许读取 ground truth，是因为 **Auditor 本身就是做 simulation 内部审计**，而不是训练 phasing 模型。以后给模型的 training input 不能复用 Auditor 的 oracle 字段。

---

## 4. `biological/`：B 模块，绝对不改生成器

### `biological/expression.py` — B1 Allele-specific Expression

输入：`DatasetView.p_true`、`m_true`（都是 cell × gene）；输出 `(summary_dict, gene_detail_df)`。

每一个基因都计算父本平均表达、母本平均表达、`log2(P/M)`、取绝对值后的倍数。再统计 >3× / >10× 基因数量和 40%-60% 近似平衡比例。

`p_true/m_true` 只是生成器的 latent expression；不能把统计写成 RNA-seq reads 的实测结果。

### `biological/chromatin.py` — B3 Chromatin / Promoter

使用 `element_id` 对齐父母元件状态，把 `promoter` 元件的 `candidate_target_gene` 与基因表达统计关联起来。

计算父母全部元件 accessibility 的 Pearson、全部元件 methylation Pearson、promoter accessibility Pearson、以及**跨基因** promoter 偏差对 expression 偏差的 Spearman。

输出的 gene_detail 追加 `promoter_acc_p/m`、`promoter_acc_delta` 与 `promoter_log2_p_over_m`。**这个模块只是关联性分析，不负责推导原因**。

### `biological/imprinting.py` — B2 Explicit Imprinting

V7 提供 imprinting_status/strength，于是对所有显式印记基因检查：

- 父本表达型的基因，真实父本平均表达是否更高；母本表达型反之。
- strong imprint gene 的表达偏好倍数中位数。
- promoter 开放差异与预设方向的一致率。
- imprinting_strength 是否与倍数呈单调关系（Spearman）。
- 普通非印记基因的父母差异中位数。

V6 走 N/A 路径，输出 `imprinting_status='unknown'` 并保留 n_explicit_imprinted=null。不允许把 unknown 改成 none。

### `biological/grn.py` — B4 TF→Gene GRN

把两张 GRN CSV 的 TF ID 设为 index，要求父母 TF 和 gene 完全对齐。

按绝对权重阈值 1e-8 定义有无边，计算：共有边、独占边、并集、共享比例、并集权重相关。

这是 TF→gene 网络，不是 gene→gene 网络。先前某些报告中的“99.13% GRN shared”来自不同口径；任何报告必须明确这一点。

### `biological/pipeline.py` — B 总控制器

`run_b({"v6":path_v6,"v7":path_v7}, out)`：

1. 根据版本调用对应 Adapter；
2. Expression 统计，得到基因指标；
3. Chromatin 统计，给每个基因补充 promoter 数据；
4. Imprinting 统计，给每个基因补充状态；
5. GRN 统计，生成网络汇总；
6. 把各模块 summary 合并为 `B_summary.csv`，基因级表合并为 `B_gene_detail.csv`；
7. 画 B 图，写 provenance。

调用顺序非常关键：B2 印记模块依赖 B3 已生成的 promoter delta 字段，所以 pipeline 先调 Chromatin 再调 Imprinting。

---

## 5. `causal/`：C 模块，检查生成机制作用

### `causal/interventions.py` — 7 个实验的“名称与动作表”

`Intervention` dataclass 包括稳定的实验 ID、patch key 列表、通俗描述。7 条记录通过 `REGISTRY` 查找。

注册表**只描述**具体试验，真正替换代码是在 `v7_bridge.py` 里执行。

### `causal/v7_bridge.py` — V7 代码桥接器

这是 C 里最需要谨慎维护的文件：

1. `read_generator` 从可信的 `.zip`/目录/直接 `.py` 中找到 `simulate_allele_specific_grn_v7_imprinting_fixed.py`。
2. `PATCHES` 明确规定字符串匹配与替换，如删除 `softplus` 前的 `log(imprinting_gate)`。
3. Promoter equalization 通过在 `generate_element_state` 后插入内存干预，把印记基因受抑制亲本的 promoter state 复制成偏好亲本的 state。
4. `patched_module` 先检查替换目标在生成器源码中恰好出现 **1 次**，再 `compile`/`exec` 到当前进程的私有 namespace 中。
5. 不输出覆盖任何原生成器文件，原始 `simulation_generation/` 只读。

这里的 source patch **依赖当前 V7 的具体源码写法**。如果队友修改了生成器，替换条件可能找不到，此时是预期的安全失败，不应靠随意修改 anchor “让它跑起来”。先审查新版本的方程。

### `causal/imprinting.py` — 验证对照确实成对

`check_pairing(original,intervention,patches)` 检查：

- gene_id / imprinting_status / imprinting_strength 不变；
- motif 数据一致；
- 非 promoter 干预下两亲本 element state 应全部相等；
- promoter equalization 下**非 promoter** 的 state 不变，印记 gene 的 promoter accessibility 父母一致。

目的：避免一边删 gate，一边不小心把其他随机参数也重新抽了一遍，让实验失去可解释性。

### `causal/runner.py` — 真正运行实验

`run_c(generator,out,start,seeds,conditions)`：

1. 读取生成器源码；
2. 编译原版和各组干预版本；
3. 对每个 seed 运行所有条件；
4. 每次调用 `run_simulation(params, np.random.default_rng(seed))`；
5. 用原版结果对照检查上游状态一致性；
6. 读取模拟器返回的 `imprinting_validation`；
7. 保存所有 seed × version × gene 到 C statistics 并写 provenance。

`import_c(main,promoter,out)` 是另一条入口，把原来独立保存的 12-seed 实验 CSV 整合进相同统计体系，且明确写 `origin=imported_prior_12_seed_pilot_not_rerun`。

### `causal/statistics.py` — 统计的独立单位

`C_gene_detail.csv` 是 seed × condition × imprint gene 的长表；`C_per_seed.csv` 把每个 seed 的 6 个 gene 聚合，随后才能用 seed 当统计重复。

为什么？一个 seed 的 6 个 gene 来自同一套随机生成体系，不可以假装是 6 次独立 Simulation。用 seed 配对统计可防止 pseudo-replication。

`mean_log2_reduction` 是每个 seed 的 `original mean log2 ratio - intervention mean log2 ratio` 后再求平均；用固定 bootstrap RNG 对 seed 进行 5000 次抽样，输出置信区间。

当 `seeds=1`，程序可以跑通，但 bootstrap CI 没有对总体不确定性的有效解释。

---

## 6. `reporting/`：只读取结构化结果

### `reporting/plots.py`

读入已经算好的 dataframe 绘制两类 PNG：

- B：每个 gene 的表达偏好倍数分布、promoter 与 expression 的散点；显式印记 gene 用空心圆强调。
- C：每个干预条件的跨 seed 平均 log2 偏好箱线图（每个 seed 一个观察值）。

图只负责美观呈现，没有内部重复定义科学阈值。

### `reporting/compare.py`

挑选 V6 和 V7 都有定义的 B 指标，转置形成 `B_common_comparison.csv`。未定义的 V6 imprint 不强行纳入同口径的比较。

### `reporting/report.py`

读取 `B_summary.csv`、存在时读 `C_summary.csv` 以及 `C_paired_seed_effects.csv`，自动生成 `GROUP_MEETING_REPORT.md`；并输出 `B_common_comparison.csv`。

输出强调三类结论的边界：

1. B 是观察性模拟表型；
2. C 是计算生成器内部干预敏感性；
3. 尚需真正的真实数据校准，才能讨论 biological realism。

---

## 7. `tests/`：独立、可执行的基本防线

- `test_io.py`：ZIP/目录读取，重复 cell 拦截，父母 gene 顺序不一致拦截。
- `test_metrics.py`：B1 表达倍数边界、V6 imprint N/A、GRN 共有/独占边、轴错位拒绝。
- `test_causal.py`：patch 遇到不符合的 V7 源码必须报错、实验 seed/gene 配对校验、seed 级 bootstrap 的统计单元。

运行：`python -m pytest -q simulation_auditor/tests`。

这些不是全量集成测试：真正接入新 V7 代码版本时，仍应专门跑 `--seeds 1`，核对 C 的 7 个条件再做正式 12-seed 实验。

---

## 8. 初次维护时的建议操作流程

1. 进入 PROJECT_ROOT，确认 `simulation_generation` 与 `simulation_auditor` 同级。
2. `python -m pytest -q simulation_auditor/tests`。
3. `python -m simulation_auditor b --v6 ... --v7 ... --out outputs/...`。
4. 打开 `B_summary.csv` 和 `B_gene_detail.csv`，逐项检查表型。
5. `python -m simulation_auditor c --generator simulation_generation --seeds 1 --out outputs/...`。
6. 运行 `report` 命令，打开 MD 和 PNG，看是否与 CSV 完全一致。
7. 所有实验保留输出目录、source hash、seed、干预定义，再向组会汇报。

### 常见疑问

- **“为什么 V7 的 GRN shared ≈100%，不是早先说的 99.13%？”** 因为选取 TF→gene 的原始矩阵及 support threshold 与此前的另一层 GRN 口径不同，不能拿两个不同指标对比。这里的确切矩阵文件在 `docs/DATA_CONTRACT.md`。
- **“promoter already 0.0053，为什么关掉最终 gate 只有 1.5 倍？”** B 能揭示强 promoter 差异与最终表达的弱关联，C 能指出强印记表型对最终 gate 的敏感性；具体原因仍需逐项解释表达方程，不能仅凭数值断言 promoter 机制没 work。
- **“结果能证明老师的 Simulation 有 bug 吗？”** 目前证明了需要解释的生成机制设计特点，不是生物学错误的充分证据。
- **“为什么 V6 不做 imprinting causality？”** V6 没有提供和 V7 同等的印记标签以及可用的 V6 生成器源文件，不能不顾语义强行套 V7 patch。

