from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from simulation_auditor.biological.expression import expression_metrics
from simulation_auditor.biological.imprinting import imprint_metrics
from simulation_auditor.biological.grn import grn_metrics

def test_expression_ratio():
    d=SimpleNamespace(p_true=pd.DataFrame({'g1':[6.,6.],'g2':[1.,1.]}),
                      m_true=pd.DataFrame({'g1':[2.,2.],'g2':[1.,1.]}))
    s,g=expression_metrics(d)
    assert s['genes']==2
    assert s['expression_genes_gt3']==0 # exactly 3x is NOT greater than 3x
    assert g.set_index('gene_id').loc['g1','fold_change']==pytest.approx(3)

def test_v6_imprinting_capability_not_false_failure():
    d=SimpleNamespace(capabilities=set())
    s,g=imprint_metrics(d,pd.DataFrame({'gene_id':['a'],'fold_change':[2]}))
    assert s['n_explicit_imprinted'] is None
    assert g.imprinting_status.iloc[0]=='unknown'

def test_grn_support_stats():
    a=pd.DataFrame({'Unnamed: 0':['tf1','tf2'],'g1':[1,0],'g2':[0,2]})
    b=pd.DataFrame({'Unnamed: 0':['tf1','tf2'],'g1':[1,3],'g2':[0,0]})
    s=grn_metrics(SimpleNamespace(p_grn=a,m_grn=b))
    assert s['grn_edges_union']==3
    assert s['grn_edges_shared']==1
    assert s['grn_shared_fraction']==pytest.approx(1/3)

def test_grn_axis_mismatch_rejected():
    a=pd.DataFrame({'Unnamed: 0':['tf1','tf2'],'g1':[1,2]})
    b=pd.DataFrame({'Unnamed: 0':['tf2','tf1'],'g1':[2,1]})
    with pytest.raises(ValueError,match='axes differ'):
        grn_metrics(SimpleNamespace(p_grn=a,m_grn=b))
