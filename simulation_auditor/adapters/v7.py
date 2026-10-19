"""V7: explicit parent-of-origin imprinting is part of ground_truth."""
from pathlib import Path
from .base import Adapter
from ..core.schema import DatasetView
from ..core.io import CsvStore
from ..core.validation import matrix_from_csv, require_pair, require_unique

class V7Adapter(Adapter):
    version='V7'
    def load(self,path):
        with CsvStore(path) as store:
            p=matrix_from_csv(store.read('ground_truth/allele_expression/paternal_true_expression.csv'))
            m=matrix_from_csv(store.read('ground_truth/allele_expression/maternal_true_expression.csv'))
            genes=store.read('ground_truth/imprinting/gene_info_full.csv')
            elements=store.read('input/shared_priors/regulatory_elements.csv')
            ps=store.read('ground_truth/chromatin/paternal_element_state.csv')
            ms=store.read('ground_truth/chromatin/maternal_element_state.csv')
            pg=store.read('ground_truth/grn/paternal_GRN.csv')
            mg=store.read('ground_truth/grn/maternal_GRN.csv')
            regions=store.read('input/chromatin_regions.csv')
        require_pair(p,m,'V7 true expression')
        for df,key,label in [(genes,'gene_id','gene'), (elements,'element_id','element'),(ps,'element_id','p_state'),(ms,'element_id','m_state')]:
            require_unique(df,key,label)
        if not {'imprinting_status','imprinting_strength'}.issubset(genes.columns):
            raise ValueError('V7 imprinting ground truth fields missing')
        if set(p.columns)!=set(genes.gene_id):
            raise ValueError('Gene annotations differ from expression gene set')
        if set(elements.element_id)!=set(ps.element_id) or set(elements.element_id)!=set(ms.element_id):
            raise ValueError('Parental element state IDs must match annotated regulatory elements')
        return DatasetView('V7',Path(path).resolve(),p,m,genes,elements,ps,ms,pg,mg,regions,
                           {'expression','chromatin','grn','explicit_imprinting'})
