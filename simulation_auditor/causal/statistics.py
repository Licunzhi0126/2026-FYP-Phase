"""Unit of inference is the independent simulation seed, NOT individual genes."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..reporting.plots import plot_c

REQUIRED={'seed','version','gene_id','preferred_expression_ratio','direction_correct','imprinting_status'}

def summarize_c(data, out, bootstrap=5000, bootstrap_seed=20261013):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=data.copy()
    missing=REQUIRED.difference(c.columns)
    if missing:raise ValueError(f'C input missing fields: {missing}')
    if c.duplicated(['seed','version','gene_id']).any():
        raise ValueError('Duplicate (seed,version,gene) in C')
    # All intervention arms must use the same seed/gene experimental units and imprint labels.
    orig=c.loc[c.version=='original',['seed','gene_id','imprinting_status']].copy()
    if orig.empty:raise ValueError('Original V7 control arm missing')
    for name, arm in c.groupby('version'):
        chk=orig.merge(arm[['seed','gene_id','imprinting_status']], on=['seed','gene_id'],
                       how='outer',validate='one_to_one',suffixes=('_base','_arm'),indicator=True)
        if not (chk['_merge']=='both').all() or not (chk.imprinting_status_base==chk.imprinting_status_arm).all():
            raise ValueError(f'Unpaired experimental units / imprint labels in {name}')
    c['preferred_expression_ratio']=pd.to_numeric(c['preferred_expression_ratio'],errors='raise')
    c['direction_correct']=pd.to_numeric(c['direction_correct'],errors='raise')
    if (c.preferred_expression_ratio<=0).any() or (~np.isfinite(c.preferred_expression_ratio)).any():
        raise ValueError('Expected positive finite expression ratios')
    c['log2_preference']=np.log2(c.preferred_expression_ratio.clip(lower=1e-10))
    c.to_csv(out/'C_gene_detail.csv',index=False)
    ss=c.groupby(['seed','version']).agg(mean_log2_ratio=('log2_preference','mean'),
        ratio_median=('preferred_expression_ratio','median'),
        correct_direction=('direction_correct','mean'), n=('gene_id','size')).reset_index()
    ss.to_csv(out/'C_per_seed.csv',index=False)
    summary=c.groupby('version').agg(n=('gene_id','size'),n_seed=('seed','nunique'),
        ratio_median=('preferred_expression_ratio','median'),mean_log2_ratio=('log2_preference','mean'),
        correct_direction=('direction_correct','mean')).reset_index()
    summary.to_csv(out/'C_summary.csv',index=False)
    base=ss[ss.version=='original'].set_index('seed')['mean_log2_ratio']
    if len(base)==0:raise ValueError('original condition is required')
    rng=np.random.default_rng(bootstrap_seed);pairs=[]
    for name in sorted(ss.version.unique()):
        if name=='original':continue
        v=ss[ss.version==name].set_index('seed')['mean_log2_ratio']
        if not base.index.equals(v.index):
            # Need all seeds in both arms; avoid silent partial matching.
            if set(base.index)!=set(v.index):raise ValueError(f'Unpaired seeds in {name}')
        matched=base.index.intersection(v.index)
        delta=(base.loc[matched]-v.loc[matched]).to_numpy()
        draws=rng.choice(delta,size=(bootstrap,len(delta)),replace=True).mean(axis=1)
        pairs.append({'comparison':'original - '+name,'n_paired_seeds':len(delta),
            'mean_log2_reduction':float(delta.mean()),
            'bootstrap_95_low':float(np.quantile(draws,.025)),
            'bootstrap_95_high':float(np.quantile(draws,.975))})
    pd.DataFrame(pairs).to_csv(out/'C_paired_seed_effects.csv',index=False)
    plot_c(ss,out/'C_imprinting_ablation.png')
    return summary
