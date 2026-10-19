# Simulation Auditor (ASimAudit) v0.2.0

这是一个可独立维护的 **Allele-specific Simulation 审计项目**，目录与 `simulation_generation/` 同级。**不修改**原始生成器。当前实现 B（Biological Plausibility）和 C（Causal Mechanism），针对 V6/V7；其他审计模块留待后续。

## 一、放在哪里

```text
PROJECT_ROOT/
├── simulation_generation/      # 老师/队友已有代码，保持不动
├── simulation_auditor/         # 本项目（这个目录）
├── data/                       # 如有本地解压数据，也可放到已有位置
│   ├── SIM_V6/
│   └── simulation/             # V7
└── outputs/
    └── simulation_auditor/    # 各种 CSV、PNG、MD，建议加入 .gitignore
```

**无需移动数据**：`--v6`、`--v7` 同时支持 `.zip` 和解压后的目录；V6 和 V7 的输出不用复制到项目内部。

## 二、安装与命令

在 `PROJECT_ROOT` 里运行（PowerShell / bash 均适用）：

```bash
python -m pip install -r simulation_auditor/requirements.txt
python -m pytest -q simulation_auditor/tests
```

要安装成 Python package（可选）：

```bash
python -m pip install -e ./simulation_auditor
```

### B：检查输出数据的生物学表型

```bash
python -m simulation_auditor b \
  --v6 "data/SIM_V6" \
  --v7 "data/simulation" \
  --out "outputs/simulation_auditor/reports"
```

PowerShell 可写成一行，避免 `\` 换行符差异：

```powershell
python -m simulation_auditor b --v6 "data\SIM_V6" --v7 "data\simulation" --out "outputs\simulation_auditor\reports"
```

如果只有 V7，可以只给 `--v7`；**V6 没有同等的印记真值不会被判 Fail，而是 N/A**。

输出：

- `B_summary.csv`：每个版本一行的摘要，方便 V6/V7 比较。
- `B_gene_detail.csv`：每个 gene 一行，包含表达差异、promoter accessibility、印记标签（有才提供）。
- `B_v6_v7.png`：父母表达差异分布 + promoter 差异/表达差异散点。
- `B_provenance.json`：源路径、zip SHA256、指标口径等。

### C：调用可信的 V7 生成器，执行受控的印记消融

```bash
python -m simulation_auditor c \
  --generator "simulation_generation" \
  --start 42 --seeds 2 \
  --out "outputs/simulation_auditor/reports"
```

使用 `--generator simulation_generation.zip` 也可以。`--seeds 12` 代表 seeds=42..53。若要先跑通，可用 `--seeds 1`。

C 运行在**内存中加载与修改源码字符串**，不会写入/覆盖原始 V7 `.py` 文件。原始代码变动时，预设 patch 不再匹配会显式报错，不会静默应用错误的干预。

**安全提醒**：这条命令会执行本地传入的生成器 Python 源文件，应只使用可信来源。当前 C 仅适用于指定的 V7 `simulate_allele_specific_grn_v7_imprinting_fixed.py`，不能直接对 V6 使用。

输出：

- `C_gene_detail.csv`：每个 seed、condition、imprinting gene 的原始指标。
- `C_per_seed.csv`：每个 seed × condition 一行（seed 是统计重复单位）。
- `C_summary.csv`：每组跨 seed 的描述性统计。
- `C_paired_seed_effects.csv`：original 减去其他条件的每-seed 平均 log2 效应、seed bootstrap 95% CI。
- `C_imprinting_ablation.png`：每个条件的 seed-level log2 偏好分布。
- `C_provenance.json`：版本来源、种子、实验条件及解释边界。

### C：仅导入此前的 12-seed 实验 CSV（不会重新模拟）

```bash
python -m simulation_auditor import-c \
  --main "v7_imprinting_ablation_12seeds.csv" \
  --promoter "v7_promoter_counterfactual_12seeds.csv" \
  --out "outputs/simulation_auditor/historical_12_seeds"
```

这个模式专门用于检查旧实验并统一报告。Provenance 会写 `imported_prior_12_seed_pilot_not_rerun`，**绝不可把导入结果说成重新运行了 12 次**。导入结果也要与对应 B 输出放在同一目录后，才能一起报告。

### 生成组会报告

```bash
python -m simulation_auditor report --out "outputs/simulation_auditor/reports"
```

需要先运行 B。若 C 文件同目录存在，报告会加入 C；若没有 C，则只报告 B。报告文件是 `GROUP_MEETING_REPORT.md`。

## 三、核心科学区分

- B 读的是 **latent true expression**，不是观测 RNA counts 或模型预测。
- B 的 Spearman 是相关性，**不能写成因果效应**。
- GRN 共享比例针对的是 **TF→gene** 细胞静态网络矩阵，默认绝对权重阈值 `1e-8`。不要与 gene→gene 网络的共享比例混为一谈。
- C 是基于 V7 代码内部的受控机制干预，不是对真实细胞实施的实验。
- Promoter equalization 是把印记基因受抑制亲本的 promoter 状态复制为另一亲本状态，不代表实验条件真实存在。
- V6 的 `layer_sim_imprint.csv` 属于 `sim_regstate` 标注，**不等价于 V7 的 genomic imprinting ground truth**。

## 四、代码阅读顺序

先看 `cli.py` → `biological/pipeline.py` → `adapters/v6.py` / `adapters/v7.py` → `biological/expression.py` / `imprinting.py` → `causal/runner.py` → `causal/interventions.py` → `causal/v7_bridge.py` → `reporting/report.py`。

详细逐文件说明见 [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)，指标定义见 [`docs/METRICS.md`](docs/METRICS.md)，干预定义见 [`docs/INTERVENTIONS.md`](docs/INTERVENTIONS.md)。

## 五、目前不包含什么

尚未做真实数据外部校准、完整 no-imprinting 负对照、正交 promoter/region/GRN 全开关、GRN causal truth 验证、下游 phasing 模型敏感性实验。`configs/` 中的 YAML 当前是**参考模板**，还未接成生效的配置读入接口；命令行参数和代码定义才是本版本生效的设置。

## v0.3 — 四项新增的 V7 印记因果审计

新增 `fully_no_imprinting`、`region_off`、`promoter_imprint_off`、`region_only`、`promoter_only`、`region_promoter_only`，可在 `c --conditions ...` 中使用。默认 `c` 会运行注册的全部条件。另新增：

```bash
python -m simulation_auditor strength-scan --generator simulation_generation --start 42 --seeds 12 --out outputs/simulation_auditor/strength
python -m simulation_auditor neighbors --generator simulation_generation --start 42 --seeds 12 --out outputs/simulation_auditor/neighbors
```

输出多个 CSV、来源 JSON 和两张 PNG。全部定义、注意事项和各表字段说明见 `docs/ADVANCED_C_AUDIT.md`。v0.3 不修改生成器原文件。
