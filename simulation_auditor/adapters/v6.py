"""V6: legacy simulation. Its sim_regstate is NOT a genomic imprint truth."""
from pathlib import Path
from .base import Adapter
from ..core.schema import DatasetView
from ..core.io import CsvStore
from ..core.validation import matrix_from_csv, require_pair, require_unique

class V6Adapter(Adapter):
    version = 'V6'
    def load(self, path):
        with CsvStore(path) as store:
            pre = 'input/omics/'
            p = matrix_from_csv(store.read(pre+'paternal_true_expression.csv'))
            m = matrix_from_csv(store.read(pre+'maternal_true_expression.csv'))
            pg, mg = store.read(pre+'paternal_GRN.csv'), store.read(pre+'maternal_GRN.csv')
            genes = store.read(pre+'gene_info.csv')
            elements = store.read(pre+'regulatory_elements.csv')
            ps, ms = store.read(pre+'paternal_element_state.csv'), store.read(pre+'maternal_element_state.csv')
            regions = store.read('input/chromatin_regions.csv')
        require_pair(p, m, 'V6 true expression')
        for df,key,label in [(genes,'gene_id','gene'), (elements,'element_id','element'),(ps,'element_id','p_state'),(ms,'element_id','m_state')]:
            require_unique(df,key,label)
        if set(p.columns)!=set(genes.gene_id):
            raise ValueError('Gene annotations differ from expression gene set')
        if set(elements.element_id)!=set(ps.element_id) or set(elements.element_id)!=set(ms.element_id):
            raise ValueError('Parental element state IDs must match annotated regulatory elements')
        return DatasetView('V6',Path(path).resolve(),p,m,genes,elements,ps,ms,pg,mg,
                           regions, {'expression','chromatin','grn'},
                           ['No explicit parent-of-origin genomic imprint label; sim_regstate must not be treated as such.'])
