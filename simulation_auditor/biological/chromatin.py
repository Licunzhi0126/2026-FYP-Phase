"""B3: allele-specific element accessibility and association with expression."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..core.validation import require_unique

def _safe_corr(a,b):
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    v=np.corrcoef(x,y)[0,1]
    return float(v) if np.isfinite(v) else None

def chromatin_metrics(data, genes):
    e=data.elements
    require_unique(e,'element_id','elements')
    ps=data.p_element_state
    ms=data.m_element_state
    states=ps[['element_id','accessibility','methylation']].merge(
        ms[['element_id','accessibility','methylation']],on='element_id',validate='one_to_one',suffixes=('_p','_m'))
    joined=e[['element_id','element_type','candidate_target_gene']].merge(states,on='element_id',validate='one_to_one')
    promoters=joined.loc[joined.element_type=='promoter'].copy()
    if promoters.candidate_target_gene.duplicated().any():
        raise ValueError('Multiple promoters per gene: define explicit aggregation rule')
    promoters=promoters.rename(columns={'candidate_target_gene':'gene_id',
        'accessibility_p':'promoter_acc_p','accessibility_m':'promoter_acc_m'})
    details=genes.merge(promoters[['gene_id','promoter_acc_p','promoter_acc_m']],on='gene_id',how='left',validate='one_to_one')
    details['promoter_acc_delta']=details.promoter_acc_p-details.promoter_acc_m
    details['promoter_log2_p_over_m']=np.log2((details.promoter_acc_p+1e-3)/(details.promoter_acc_m+1e-3))
    x,y=details['promoter_log2_p_over_m'],details['log2_p_over_m']
    v=spearmanr(x,y,nan_policy='omit').statistic
    summary={'element_accessibility_corr':_safe_corr(states.accessibility_p,states.accessibility_m),
        'element_methylation_corr':_safe_corr(states.methylation_p,states.methylation_m),
        'promoter_accessibility_corr':_safe_corr(promoters.promoter_acc_p,promoters.promoter_acc_m),
        'promoter_vs_expression_spearman':float(v) if np.isfinite(v) else None,
        'promoter_matched_genes':int(details.promoter_acc_p.notna().sum())}
    return summary,details
