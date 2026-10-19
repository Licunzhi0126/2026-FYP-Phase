"""B2: Explicit V7 imprint labels; V6 is unsupported, not failed."""
import numpy as np
from scipy.stats import spearmanr

def imprint_metrics(data,genes):
    if 'explicit_imprinting' not in data.capabilities:
        genes=genes.copy()
        genes['imprinting_status']='unknown'
        genes['imprinting_strength']=np.nan
        return {'imprinting_annotation':'N/A:no_equivalent_ground_truth',
                'n_explicit_imprinted':None},genes
    allowed={'none','paternal_expressed','maternal_expressed'}
    if data.genes.imprinting_status.isna().any() or not set(data.genes.imprinting_status).issubset(allowed):
        raise ValueError('Unknown or missing V7 imprinting_status in ground truth')
    labels=data.genes[['gene_id','imprinting_status','imprinting_strength']]
    genes=genes.merge(labels,on='gene_id',validate='one_to_one',how='left')
    pick=genes.loc[genes.imprinting_status!='none'].copy()
    if len(pick)==0:
        return {'imprinting_annotation':'explicit_parent_of_origin','n_explicit_imprinted':0},genes
    sign=np.where(pick.imprinting_status=='paternal_expressed',1,-1)
    expr=np.asarray(pick.log2_p_over_m,dtype=float)
    corr=spearmanr(pick.imprinting_strength,pick.fold_change).statistic if len(pick)>2 else np.nan
    return {'imprinting_annotation':'explicit_parent_of_origin',
        'n_explicit_imprinted':len(pick),
        'imprint_direction_accuracy':float(np.mean(sign*expr>0)),
        'imprint_median_preferred_ratio':float(np.median(pick.fold_change)),
        'imprint_promoter_direction_accuracy':float(np.mean(sign*pick.promoter_acc_delta.to_numpy()>0)),
        'imprint_strength_vs_fold_spearman':float(corr) if np.isfinite(corr) else None,
        'non_imprint_median_gene_fold':float(genes.loc[genes.imprinting_status=='none','fold_change'].median()),
        },genes
