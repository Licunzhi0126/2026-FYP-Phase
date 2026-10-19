"""Advanced imprint audits: neutral negative control, dose response, neighbor spillover.

All data are fresh re-executions of trusted V7 generator; they are not historical CSVs.
"""
from pathlib import Path
import copy, hashlib
import numpy as np
import pandas as pd
from .v7_bridge import read_generator, patched_module
from .imprinting import check_pairing
from .statistics import summarize_c
from ..core.provenance import json_write


def _run(module, seed):
    params = dict(module['PARAMS']); params['seed'] = seed
    return module['run_simulation'](params, np.random.default_rng(seed))


def run_strength_scan(generator, out, start=42, seeds=3, strengths=(0.0, 0.25, 0.5, 0.75, 0.95)):
    """Hold imprinted genes/status fixed while varying strength after assignment.

    This keeps nonpreferred_leak unchanged! It explicitly tests whether *strength*
    controls expression, not the separate fixed hard gate. Region RNG may change
    because of probabilistic override; compare identity, not full downstream RNG.
    """
    source, origin = read_generator(generator)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in range(start,start+seeds):
        for strength in strengths:
            module=patched_module(source,'original')
            assign=module['assign_imprinting']
            def assign_with_strength(params, rng, gene, strength=strength):
                result=assign(params,rng,gene)
                result.loc[result.imprinting_status!='none','imprinting_strength']=float(strength)
                return result
            module['assign_imprinting']=assign_with_strength
            context=_run(module,seed)
            table=context['imprinting_validation'].copy()
            table['seed']=seed;table['strength_setting']=strength
            records.append(table)
        print(f'[scan] completed seed {seed}',flush=True)
    raw=pd.concat(records,ignore_index=True)
    raw['log2_ratio']=np.log2(raw.preferred_expression_ratio.clip(lower=1e-12))
    raw.to_csv(out/'C_strength_gene_detail.csv',index=False)
    per=raw.groupby(['seed','strength_setting']).agg(mean_log2_ratio=('log2_ratio','mean'),
        median_ratio=('preferred_expression_ratio','median'),direction_accuracy=('direction_correct','mean')).reset_index()
    per.to_csv(out/'C_strength_per_seed.csv',index=False)
    summary=per.groupby('strength_setting').agg(mean_log2_ratio=('mean_log2_ratio','mean'),
        mean_median_ratio=('median_ratio','mean'),sd_log2_ratio=('mean_log2_ratio','std'),n_seeds=('seed','nunique')).reset_index()
    summary.to_csv(out/'C_strength_summary.csv',index=False)
    json_write(out/'C_strength_provenance.json',dict(generator=origin,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        seed_start=start,seeds=seeds,strengths=list(strengths),fixed_nonpreferred_leak=True,
        caution='Changing strength changes region probabilistic overrides and downstream random draws; gene identity is fixed, but exact downstream common random numbers are not guaranteed.'))
    return summary


def run_neighbor_audit(generator,out,start=42,seeds=3):
    """Detect nonimprinted genes whose regulation shares region(s) with imprint targets.

    Comparison is original vs region_off; outcomes are *differential* allele bias
    and total expression. Locality defined via elements.chromatin_region_id, not
    mere proximity or an implied biological imprinting cluster.
    """
    source,origin=read_generator(generator)
    a=patched_module(source,'original');b=patched_module(source,'region_off')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];region_rows=[]
    for seed in range(start,start+seeds):
        baseline=_run(a,seed);alternate=_run(b,seed)
        check_pairing(baseline,alternate,('region_off',))
        genes=baseline['gene']
        imprinted=set(genes.loc[genes.imprinting_status!='none','gene_id'])
        els=baseline['elements']
        implicated=set(els.loc[els.candidate_target_gene.isin(imprinted),'chromatin_region_id'])
        neighbor=set(els.loc[els.chromatin_region_id.isin(implicated),'candidate_target_gene'])-imprinted
        neighbor|=set(genes.loc[genes.chromatin_region_id.isin(implicated),'gene_id'])-imprinted
        r0=baseline['regions'].set_index('region_id')['allelic_state']
        r1=alternate['regions'].set_index('region_id')['allelic_state']
        region_rows += [{'seed':seed,'region_id':reg,'is_imprint_associated':int(reg in implicated),
                         'baseline_state':r0.loc[reg],'region_off_state':r1.loc[reg],
                         'state_changed':int(r0.loc[reg]!=r1.loc[reg])} for reg in r0.index]
        for gene in genes.gene_id:
            eps=1e-9
            pb=float(baseline['p_true'][gene].mean());mb=float(baseline['m_true'][gene].mean())
            pa=float(alternate['p_true'][gene].mean());ma=float(alternate['m_true'][gene].mean())
            rows.append(dict(seed=seed,gene_id=gene,is_imprinted=int(gene in imprinted),
                exposed_neighbor=int(gene in neighbor),
                baseline_log2_pm=float(np.log2((pb+eps)/(mb+eps))),
                region_off_log2_pm=float(np.log2((pa+eps)/(ma+eps))),
                delta_log2_pm=float(np.log2((pb+eps)/(mb+eps))-np.log2((pa+eps)/(ma+eps))),
                delta_log2_total=float(np.log2((pb+mb+eps)/(pa+ma+eps)))))
        print(f'[neighbor] completed seed {seed}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'C_neighbors_gene_detail.csv',index=False)
    pd.DataFrame(region_rows).to_csv(out/'C_neighbors_regions.csv',index=False)
    df['group']=np.select([df.is_imprinted.eq(1),df.exposed_neighbor.eq(1)],['imprinted','exposed_nonimprinted'],default='other_nonimprinted')
    group_seed=df.groupby(['seed','group']).agg(n=('gene_id','size'),mean_abs_bias_shift=('delta_log2_pm',lambda x:np.abs(x).mean()),
        mean_abs_total_shift=('delta_log2_total',lambda x:np.abs(x).mean())).reset_index()
    group_seed.to_csv(out/'C_neighbors_per_seed.csv',index=False)
    group_seed.groupby('group').agg(mean_abs_bias_shift=('mean_abs_bias_shift','mean'),
        mean_abs_total_shift=('mean_abs_total_shift','mean'),n_seeds=('seed','nunique')).to_csv(out/'C_neighbors_summary.csv')
    json_write(out/'C_neighbors_provenance.json',dict(generator=origin,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        seed_start=start,seeds=seeds,control='region_off',
        caveat='Region reassignment is stochastic; reported differences are simulation intervention sensitivity, not evidence that nearby genes are biological imprinting targets.'))
    return group_seed
