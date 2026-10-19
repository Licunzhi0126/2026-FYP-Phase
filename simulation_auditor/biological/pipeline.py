"""Run B audits; adapters are the only V6/V7-aware data loaders."""
from pathlib import Path
import pandas as pd
from ..adapters.v6 import V6Adapter
from ..adapters.v7 import V7Adapter
from ..core.provenance import sha256_file,json_write
from .expression import expression_metrics
from .chromatin import chromatin_metrics
from .imprinting import imprint_metrics
from .grn import grn_metrics
from ..reporting.plots import plot_b

ADAPTERS={'v6':V6Adapter(), 'v7':V7Adapter()}

def run_b(version_paths, out, grn_threshold=1e-8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summaries=[];gene_details=[]
    for version,path in version_paths.items():
        data=ADAPTERS[version.lower()].load(path)
        s,g=expression_metrics(data)
        chrom,g=chromatin_metrics(data,g)
        imp,g=imprint_metrics(data,g)
        grn=grn_metrics(data,threshold=grn_threshold)
        s={'version':data.version,**s,**chrom,**imp,**grn}
        summaries.append(s); g.insert(0,'version',data.version);gene_details.append(g)
        print(f"[B] {data.version}: {s['genes']} genes; >10x = {s['expression_genes_gt10']}; GRN shared = {s['grn_shared_fraction']:.3f}")
    summary=pd.DataFrame(summaries);full=pd.concat(gene_details,ignore_index=True)
    summary.to_csv(out/'B_summary.csv',index=False)
    full.to_csv(out/'B_gene_detail.csv',index=False)
    plot_b(full,out/'B_v6_v7.png')
    json_write(out/'B_provenance.json',{
        'datasets':{str(v):{'path':str(Path(p).resolve()),'sha256':sha256_file(p)} for v,p in version_paths.items()},
        'expression_layer':'latent true expression (not observed RNA counts)',
        'grn_support_threshold':grn_threshold,'ratio_pseudocount':1e-12,
        'imprinting_na_policy':'V6 lacks equivalent explicit parent-of-origin truth',
        'plot':'observational association; not evidence of causality'})
    return summary
