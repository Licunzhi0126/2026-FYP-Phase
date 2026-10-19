# C：7 个 V7 Imprinting 机制消融条件

所有条件从同一个 V7 源码构造，所有 seed 以同样的 `PARAMS` 和 `np.random.default_rng(seed)` 运行，随后核验 imprinting gene label、motif、上游状态的一致性。

| Condition | 替换/干预 | 能回答的问题 |
|---|---|---|
| `original` | 不动 | 原始印记表型多强 |
| `no_final_log_gate` | `softplus(z+log(imprinting_gate))` → `softplus(z)` | 强表型是否依赖最后显式 gate |
| `no_grn_imprint_gate` | 删除 `base *= (1-strength)` 的显式 GRN 缩放 | 单独去掉 GRN-level 抑制影响多大 |
| `no_expression_imprint_gates` | 删除 final + inner expression imprint gate | 两处表达层 gate 的总敏感性 |
| `upstream_only` | 删除 final + inner + GRN gate | 仅保留 region/promoter 相关印记规则能带来什么表型 |
| `promoter_equal_final_on` | 将非偏好亲本的 imprinting promoter 状态替换成偏好亲本，保留 final | final gate 在两亲本 promoter 相同时能否仍造成强偏好 |
| `promoter_equal_final_off` | promoter equalization + 删除 final | 双因素控制后是否仍有偏好 |

说明：

- 干预直接针对模拟器逻辑，非真实实验生物学因果推断。
- `promoter_equal_*` 复制 `open_state/open_strength/methylation/accessibility/activity`；可能改变后续 GRN 和表达过程，不是只改最终一个数。
- `upstream_only` 不等于 no-imprinting。上游 region/promoter imprint 仍然存在。**目前尚缺 fully no-imprint 对照**。
- `causal/v7_bridge.py` 是唯一允许处理生成器源码的文件，patch 必须恰好匹配一处，否则中断。
- `causal/imprinting.py` 比对每个种子的基因身份、强度、motif，以及预期不应改变的状态，确保消融的主要差异来自选择的干预。

## 下一版正交设计建议

长期应请生成器维护者提供显式配置参数，例如 `enable_region_imprint`、`enable_promoter_imprint`、`enable_grn_imprint`、`enable_inner_expression_imprint` 和 `enable_final_expression_imprint`。这可以取代源码字符串 patch，并支持区域效应和因果路径验证。**这些开关目前尚不存在，不要在组会中说已经实现。**
