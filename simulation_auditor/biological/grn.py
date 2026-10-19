"""B4: directed TF→gene GRN edge support and weight difference."""
import numpy as np

def _arr(x):
    y=x.copy()
    for col in ('Unnamed: 0','TF_id','tf_id'):
        if col in y.columns:
            y=y.set_index(col)
            break
    return y.astype(float)

def grn_metrics(data, threshold=1e-8):
    p,m=_arr(data.p_grn),_arr(data.m_grn)
    if not p.index.equals(m.index) or not p.columns.equals(m.columns):
        raise ValueError('Parental GRN axes differ; cannot compare positionally')
    pa,ma=p.to_numpy(),m.to_numpy()
    ea=np.abs(pa)>threshold; eb=np.abs(ma)>threshold
    intersection=int((ea&eb).sum());union=int((ea|eb).sum())
    corr=None
    if union>2:
        aa,bb=pa[ea|eb],ma[ea|eb]
        if np.std(aa)>0 and np.std(bb)>0:
            corr=float(np.corrcoef(aa,bb)[0,1])
    return {'grn_edges_union':union,'grn_edges_shared':intersection,
        'grn_edges_paternal_only':int((ea&~eb).sum()),
        'grn_edges_maternal_only':int((~ea&eb).sum()),
        'grn_shared_fraction':intersection/union if union else None,
        'grn_weight_corr_on_union':corr,'grn_support_threshold':threshold}
