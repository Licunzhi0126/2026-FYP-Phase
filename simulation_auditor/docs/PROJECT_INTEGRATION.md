# 整合到现有项目的操作方法

## 1. 放到同级目录

将 ZIP 解压后的 `simulation_auditor` 文件夹复制到已经有 `simulation_generation` 的项目根目录，与其**并列**，不要塞进 `simulation_generation/simulation` 之内。

建议项目目录：

```text
PHASE_PROJECT/
  simulation_generation/
  simulation_auditor/
  data/
    SIM_V6/
    simulation/            # V7 生成结果
  outputs/
    simulation_auditor/
```

此处 `data` 是建议写法：如果你已有项目中的数据目录不同，不需移动，直接在命令上指定真实路径。

## 2. Windows PowerShell 快速运行

```powershell
# 先进入项目根目录（含 simulation_auditor/）
cd "D:\your\PHASE_PROJECT"

# 运行测试
python -m pip install -r simulation_auditor\requirements.txt
python -m pytest -q simulation_auditor\tests

# V6/V7 B 静态对比
python -m simulation_auditor b --v6 "data\SIM_V6" --v7 "data\simulation" --out "outputs\simulation_auditor\v6_v7_bc"

# V7 C 快速回归（1个随机种子）
python -m simulation_auditor c --generator "simulation_generation" --start 42 --seeds 1 --out "outputs\simulation_auditor\v6_v7_bc"

# 正式 12 seed （会重新运行全部 V7 生成器）
python -m simulation_auditor c --generator "simulation_generation" --start 42 --seeds 12 --out "outputs\simulation_auditor\v6_v7_bc"

# 生成 Markdown 报告
python -m simulation_auditor report --out "outputs\simulation_auditor\v6_v7_bc"
```

如果数据还是 ZIP，可用 `--v6 "SIM_V6(1).zip" --v7 "simulation.zip"`。你只要替换成自己机器上的路径。

## 3. 和其他队友协作的边界

- 你单独维护 `simulation_auditor/` 和测试、审计输出的报告定义。
- 队友可以更新 `simulation_generation/`，但是 Auditor 不覆盖/修改其源码。
- 队友更新 V7 后，应先固定 Git commit + SHA256，再做 1-seed C smoke test。若 patch anchor 不匹配，要审查最新表达方程后再修改桥接器。
- 所有 outputs 可以放在 `outputs/simulation_auditor/`（建议 `.gitignore` 排除大量自动生成 CSV/PNG）。报告摘要 Markdown 可单独复制到组会资料。

## 4. 开发新 B 指标时

在 `biological/` 增加例如 `region_effect.py`，函数接受 `DatasetView`，返回摘要 dictionary 和可选择的 per-gene table；在 `biological/pipeline.py` 统一调用，并在 `docs/METRICS.md` 记录具体定义。

不要让 B 分析函数直接拼 `simulation/ground_truth/...` 路径，所有路径差异放在 adapter。

## 5. 开发新 C 干预时

把新干预的 ID、目的和 patch keys 放在 `causal/interventions.py`；把具体源代码替换（如果有必要）放在 `causal/v7_bridge.py`；在 `causal/imprinting.py` 增加 invariance 检查；修改文档并添加测试。

长期应该改为生成器提供配置开关，而不是扩充太多源代码替换；特别是完全 no-imprint、region-only 等机制必须协调 RNG 一致性。
