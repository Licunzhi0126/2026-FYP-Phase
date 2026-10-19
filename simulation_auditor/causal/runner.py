"""Run paired in-memory V7 counterfactuals or import verifiable pilot CSVs."""
import logging
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from .interventions import REGISTRY
from .v7_bridge import read_generator, patched_module
from .imprinting import check_pairing
from .statistics import summarize_c
from ..core.provenance import sha256_file,json_write

DEFAULT_CONDITIONS=tuple(REGISTRY)

def run_c(generator, out, start=42, seeds=1, conditions=DEFAULT_CONDITIONS):
    if seeds<1:raise ValueError('seeds must be positive')
    if len(set(conditions))!=len(conditions):raise ValueError('Duplicate conditions')
    if 'original' not in conditions:raise ValueError('Must include original for matched controls')
    source,location=read_generator(generator)
    mods={name:patched_module(source,name) for name in conditions}
    records=[]
    logging.getLogger('sim_v7').setLevel(logging.WARNING)
    for seed in range(start,start+seeds):
        base=None
        for name in conditions:
            module=mods[name]
            params=dict(module['PARAMS']);params['seed']=seed
            context=module['run_simulation'](params,np.random.default_rng(seed))
            if base is not None:
                check_pairing(base,context,REGISTRY[name].patch_keys)
            if name=='original':base=context
            if 'imprinting_validation' not in context:
                raise ValueError('V7 generator did not return imprinting_validation')
            records.append(context['imprinting_validation'].assign(seed=seed,version=name))
        print(f'[C] completed paired seed {seed}',flush=True)
    raw=pd.concat(records,ignore_index=True)
    summary=summarize_c(raw,out)
    json_write(Path(out)/'C_provenance.json',{
        'origin':'direct_generator_reexecution','generator':location,
        'generator_zip_sha256':sha256_file(generator),'generator_source_sha256':hashlib.sha256(source.encode('utf8')).hexdigest(),'seed_start':start,'n_seeds':seeds,
        'conditions':list(conditions),'paired_upstream_checks':True,
        'patching':'in-memory only; anchor must match exact original source',
        'warning':'Executes locally supplied generator Python source; use trusted code only',
        'interpretation':'Sensitivity of computational simulator, not evidence of experimental biological causality'})
    return summary

def import_c(main, promoter, out):
    """Legacy 12-seed pilot import; clearly marked as imported instead of rerun."""
    a,b=pd.read_csv(main),pd.read_csv(promoter)
    for label,d in [('main',a),('promoter',b)]:
        if d.duplicated(['seed','version','gene_id']).any():
            raise ValueError(f'Duplicate C experimental unit in {label}')
    aa=a[a.version.isin(['original','no_final_log_gate','no_grn_imprint_gate',
                            'no_expression_imprint_gates','upstream_only'])].copy()
    bb=b[b.version.isin(['promoter_equal_final_on','promoter_equal_final_off'])].copy()
    key=['seed','gene_id']
    amap=aa[aa.version=='original'][key+['imprinting_status']]
    bmap=bb[bb.version=='promoter_equal_final_on'][key+['imprinting_status']]
    chk=amap.merge(bmap,on=key,validate='one_to_one',suffixes=('_a','_b'))
    if len(chk)!=len(amap) or len(chk)!=len(bmap) or not chk.imprinting_status_a.equals(chk.imprinting_status_b):
        raise ValueError('Legacy files do not match imprint labels by gene and seed')
    c=pd.concat([aa,bb],ignore_index=True)
    if c.groupby('version')['seed'].nunique().nunique()!=1:
        raise ValueError('Partial seed coverage between variants')
    summary=summarize_c(c,out)
    json_write(Path(out)/'C_provenance.json',{
        'origin':'imported_prior_12_seed_pilot_not_rerun',
        'main_file':str(Path(main).resolve()),'promoter_file':str(Path(promoter).resolve()),
        'main_sha256':sha256_file(main),'promoter_sha256':sha256_file(promoter),
        'n_seeds':int(c.seed.nunique()),
        'limitation':'Imported files matched on seed and gene identity; entire simulation was not rerun in this command.'})
    return summary
