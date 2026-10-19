import pandas as pd
import pytest
from simulation_auditor.causal.v7_bridge import patched_module
from simulation_auditor.causal.statistics import summarize_c

def test_source_patcher_fails_closed():
    with pytest.raises(ValueError,match='generator changed'):
        patched_module('not a real generator','no_final_log_gate')

def test_unpaired_seed_gene_rejected(tmp_path):
    data=pd.DataFrame([
        {'seed':42,'version':'original','gene_id':'a','imprinting_status':'paternal_expressed','preferred_expression_ratio':20.,'direction_correct':1},
        {'seed':42,'version':'no_final_log_gate','gene_id':'b','imprinting_status':'paternal_expressed','preferred_expression_ratio':2.,'direction_correct':1},
    ])
    with pytest.raises(ValueError,match='Unpaired experimental'):
        summarize_c(data,tmp_path)

def test_seed_bootstrap_uses_seed_as_unit(tmp_path):
    rows=[]
    for seed in (42,43,44):
        for version,ratio in (('original',16.),('no_final_log_gate',2.)):
            for gene in ('g1','g2'):
                rows.append(dict(seed=seed,version=version,gene_id=gene,imprinting_status='paternal_expressed',
                    preferred_expression_ratio=ratio,direction_correct=1))
    summary=summarize_c(pd.DataFrame(rows),tmp_path,bootstrap=300)
    paired=pd.read_csv(tmp_path/'C_paired_seed_effects.csv').iloc[0]
    assert paired.n_paired_seeds==3
    assert paired.mean_log2_reduction==pytest.approx(3)
    assert summary.set_index('version').loc['original','ratio_median']==pytest.approx(16)
