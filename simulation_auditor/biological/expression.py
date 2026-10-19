"""B1: allele expression statistics. All ratios based on latent true expression."""
import numpy as np
import pandas as pd
from ..core.validation import require_pair

EPS=1e-12

def expression_metrics(data, balanced_low=.4, balanced_high=.6):
    p,m=data.p_true,data.m_true
    require_pair(p,m,'expression')
    pa,ma=p.mean(axis=0),m.mean(axis=0)
    lr=np.log2((pa+EPS)/(ma+EPS))
    tab=pd.DataFrame({'gene_id':p.columns,'paternal_mean':pa.to_numpy(),
        'maternal_mean':ma.to_numpy(),'log2_p_over_m':lr.to_numpy(),
        'fold_change':np.power(2,np.abs(lr.to_numpy()))})
    pf=p.to_numpy()/(p.to_numpy()+m.to_numpy()+EPS)
    # Maintain historical all-cell metric, and record coverage-aware metric separately
    sum_expr=p.to_numpy()+m.to_numpy()
    covered=sum_expr>1e-8
    balanced=((pf>=balanced_low)&(pf<=balanced_high))
    corr=[]
    for g in p.columns:
        if p[g].std()==0 or m[g].std()==0:
            continue
        c=p[g].corr(m[g])
        if np.isfinite(c): corr.append(float(c))
    summary={
        'cells':int(p.shape[0]),'genes':int(p.shape[1]),
        'expression_genes_gt3':int((tab.fold_change>3).sum()),
        'expression_genes_gt10':int((tab.fold_change>10).sum()),
        'max_gene_fold':float(tab.fold_change.max()),
        'median_gene_fold':float(tab.fold_change.median()),
        'cell_gene_balanced_40_60':float(balanced.mean()),
        'cell_gene_balanced_40_60_covered':float(balanced[covered].mean()) if covered.any() else None,
        'covered_fraction':float(covered.mean()),
        'median_gene_paternal_maternal_cell_corr':float(np.median(corr)) if corr else None,
    }
    return summary,tab
