from simulation_auditor.causal.interventions import REGISTRY
from simulation_auditor.causal.v7_bridge import patched_module
from simulation_auditor.causal.advanced import run_strength_scan, run_neighbor_audit
from pathlib import Path
import pandas as pd

GEN = Path(__file__).resolve().parents[2] / 'simulation_generation' / 'simulation' / 'simulate_allele_specific_grn_v7_imprinting_fixed.py'

def test_new_registry_conditions():
    for name in ('fully_no_imprinting','region_off','promoter_imprint_off','region_only','promoter_only'):
        assert name in REGISTRY

def test_new_intervention_source_compile():
    if not GEN.exists():return
    source=GEN.read_text(encoding='utf-8')
    for name in ('fully_no_imprinting','region_off','promoter_imprint_off','region_only','promoter_only'):
        assert callable(patched_module(source,name)['run_simulation'])

def test_advanced_reports_smoke(tmp_path):
    if not GEN.exists():return
    scan=run_strength_scan(GEN,tmp_path/'strength',42,1,(0.0,0.9))
    assert scan.n_seeds.eq(1).all()
    df=pd.read_csv(tmp_path/'strength'/'C_strength_gene_detail.csv')
    assert set(df[df.strength_setting==0].gene_id)==set(df[df.strength_setting==.9].gene_id)
    neighbor=run_neighbor_audit(GEN,tmp_path/'neighbors',42,1)
    assert not neighbor.empty
    rows=pd.read_csv(tmp_path/'neighbors'/'C_neighbors_regions.csv')
    assert not rows.loc[rows.is_imprint_associated==0,'state_changed'].any()
