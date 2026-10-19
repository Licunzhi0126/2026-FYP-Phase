# 2026 FYP PhaseHyper

等位基因特异基因调控网络（GRN）模拟与审计项目，包含 V7 模拟数据、生成器，以及生物学合理性和因果机制审计工具。

## 项目结构

```text
.
├── simulation/                   # 已整理的 V7 模拟数据
│   ├── input/                    # 拟合侧输入
│   ├── ground_truth/             # 评估与诊断使用的模拟真值
│   ├── MANIFEST.csv              # 数据文件分类及说明
│   └── README.md                 # 数据使用边界与泄漏说明
├── simulation_generation/        # 模拟生成与后处理代码
│   ├── simulation/               # V7 等位基因特异 GRN 生成器
│   ├── postprocess/              # TF→gene 到 gene→gene 的 GRN 转换
│   └── downstream/               # 通路、PPI 和可视化脚本
└── simulation_auditor/            # ASimAudit 审计工具
    ├── adapters/                 # V6/V7 数据读取与格式适配
    ├── biological/               # B：生物学合理性审计
    ├── causal/                   # C：生成器内部机制干预与统计
    ├── core/                    # 输入输出、结构验证与来源记录
    ├── reporting/                # 图表与报告
    ├── configs/                  # 参考配置模板
    ├── docs/                     # 指标、干预与开发说明
    └── tests/                    # 测试
```

## 安装依赖

使用 Python 3.10 或更新版本，在项目根目录运行：

```powershell
python -m pip install -r simulation_auditor/requirements.txt
```

## 快速使用

对仓库内的 V7 数据执行 B 审计：

```powershell
python -m simulation_auditor b --v7 simulation --out outputs/simulation_auditor/v7
```

使用 V7 生成器执行一个随机种子的 C 机制干预实验：

```powershell
python -m simulation_auditor c --generator simulation_generation --start 42 --seeds 1 --out outputs/simulation_auditor/v7
```

B 审计完成后生成报告；同一输出目录存在 C 结果时会一并纳入：

```powershell
python -m simulation_auditor report --out outputs/simulation_auditor/v7
```

查看全部子命令和运行测试：

```powershell
python -m simulation_auditor --help
python -m pytest -q simulation_auditor/tests
```

## 数据与结果说明

- `simulation/` 中的数据随仓库保存，自动生成的审计结果放在 `outputs/`，默认不提交到 Git。
- 训练或拟合应遵守 [数据集说明](simulation/README.md) 中的输入边界。`input/combined_true_expression.csv` 仍属于模拟器潜在真值，严格的无 oracle 基准应使用观测 counts；`ground_truth/` 用于评估和诊断。
- B 的相关性指标不等价于因果效应；C 是模拟器内部的受控干预，不等价于真实细胞实验。
- `simulation_generation/downstream/` 中的脚本保留原有 V6 数据路径假设，运行前需核对各脚本的输入输出路径。
- `simulation_auditor/configs/` 当前是参考模板，生效设置以命令行和代码定义为准。

详细用法见 [审计工具说明](simulation_auditor/README.md)、[开发指南](simulation_auditor/docs/DEVELOPER_GUIDE.md) 和 [指标定义](simulation_auditor/docs/METRICS.md)。
