"""Generate meeting-ready Markdown from existing result tables only."""
from pathlib import Path
import pandas as pd
from .compare import write_common

def _fmt(x,dec=2,suffix=''):
    if pd.isna(x):return 'N/A'
    return f'{float(x):.{dec}f}{suffix}'

def write_report(folder):
    folder=Path(folder)
    bp=folder/'B_summary.csv';cp=folder/'C_summary.csv'
    if not bp.exists():raise FileNotFoundError(f'Run B audit first: {bp}')
    b=pd.read_csv(bp).set_index('version')
    write_common(b.reset_index(),folder/'B_common_comparison.csv')
    text=['# ASimAudit B + C — 审计报告','',
        '> 本项目评估模拟器的内部表型及对代码内干预的敏感性，不等同于真实生物学机制验证。','',
        '## B — Biological Plausibility (observation)','',
        '| Metric | V6 | V7 |','|---|---:|---:|']
    metrics=[('genes','基因数',0,''),('cells','细胞数',0,''),
      ('expression_genes_gt3','表达差异 >3× 基因数',0,''),('expression_genes_gt10','表达差异 >10× 基因数',0,''),
      ('max_gene_fold','最大父母表达差异',2,'×'),('cell_gene_balanced_40_60','cell-gene 接近 50:50 的比例',3,''),
      ('median_gene_paternal_maternal_cell_corr','父母跨细胞表达相关中位数',3,''),
      ('grn_shared_fraction','GRN 共享边 / 并集',3,''),
      ('promoter_vs_expression_spearman','promoter/表达偏差 Spearman',3,''),
      ('n_explicit_imprinted','显式印记基因数',0,'')]
    for key,label,dec,suffix in metrics:
        vals=[_fmt(b.loc[v,key],dec,suffix) if v in b.index and key in b.columns else 'N/A' for v in ['V6','V7']]
        text.append(f'| {label} | {vals[0]} | {vals[1]} |')
    text +=['', 'V6 缺少与 V7 同定义的显式 parent-of-origin imprinting 标签，该项应记 N/A。',
            '两版生成机制和随机参数不一定完全相同，不能将 V6/V7 的全部差异归因于 imprinting。',
            '', '![B audit](B_v6_v7.png)','']
    if cp.exists():
        c=pd.read_csv(cp).set_index('version')
        text +=['## C — Causal Mechanism (generator-level interventions)','',
            '| Condition | Imprinting preference median | Direction accuracy |',
            '|---|---:|---:|']
        for name in ['original','promoter_equal_final_on','no_grn_imprint_gate','no_final_log_gate',
                     'promoter_equal_final_off','no_expression_imprint_gates','upstream_only']:
            if name in c.index:
                r=c.loc[name]
                text.append(f'| `{name}` | {r.ratio_median:.2f}× | {r.correct_direction:.1%} |')
        text+=['','![C audit](C_imprinting_ablation.png)','',
               'C 中的原始量是潜在真实表达，不是 RNA counts 或预测结果。',
               'Promoter equalization 为模拟器内部反事实，不代表真实实验条件。']
        pairs=folder/'C_paired_seed_effects.csv'
        if pairs.exists():
            p=pd.read_csv(pairs)
            diff=p[p.comparison=='original - no_final_log_gate']
            if len(diff):
                r=diff.iloc[0]
                text +=['', f"删除 final gate 的平均 seed-level log2 偏好差异变化：{r.mean_log2_reduction:.3f}；",
                   f"bootstrap 95% CI [{r.bootstrap_95_low:.3f}, {r.bootstrap_95_high:.3f}]，{int(r.n_paired_seeds)} 个独立 seeds。",'']
        text +=['### 现阶段可支持的结论','',
            'V7 强印记表型对最终 expression gate 高度敏感；但无法仅凭该消融证明它不符合真实生物学。',
            '要确认某个上游过程是否足以产生印记表型，还需正交干预和无印记负对照。','']
    else:
        text+=['## C — 尚未运行','', '目前只有 B 观察结果。运行 C 或导入历史实验后再生成报告。','']
    text +=['## 下一步','',
        '1. 全面无印记负对照（现阶段不存在），区分预设标签和实际机制作用。',
        '2. region / promoter / GRN / inner expression / final expression 正交开关，避免通过文本 patch 实验。',
        '3. 额外 seed 重复和独立代码版本复现；与队友固定模型 commit 后接入下游 phasing。',
        '4. 真实印记基因与 ICR 参考数据校准；当前 B 不提供生物真实性总分。','',
        '## 复现及来源','',
        '参见 `B_provenance.json`、`C_provenance.json`，记录数据路径、SHA256、实验来源、seed 和方法。']
    dest=folder/'GROUP_MEETING_REPORT.md'
    dest.write_text('\n'.join(text)+'\n',encoding='utf8')
    print('[REPORT]',dest)
    return dest
