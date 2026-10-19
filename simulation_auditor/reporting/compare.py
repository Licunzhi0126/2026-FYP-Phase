"""Build common-metric comparison; preserve N/A without converting it to zero."""
import pandas as pd

COMMON_B_METRICS=(
 'cells','genes','expression_genes_gt3','expression_genes_gt10','max_gene_fold',
 'median_gene_fold','cell_gene_balanced_40_60',
 'median_gene_paternal_maternal_cell_corr','element_accessibility_corr',
 'promoter_accessibility_corr','promoter_vs_expression_spearman',
 'grn_shared_fraction','grn_weight_corr_on_union')

def common_metrics(summary):
    s=summary.set_index('version')
    return s.loc[:,[x for x in COMMON_B_METRICS if x in s.columns]].T

def write_common(summary,out):
    a=common_metrics(summary)
    a.to_csv(out,index_label='metric')
    return a
