import io
import zipfile
import pandas as pd
import pytest
from simulation_auditor.core.io import CsvStore
from simulation_auditor.core.validation import matrix_from_csv,require_pair

def test_csv_store_zip_and_directory(tmp_path):
    f=tmp_path/'root'/'input'/'sample.csv'
    f.parent.mkdir(parents=True)
    f.write_text('cell_id,gene_A\nc1,1\n',encoding='utf8')
    with CsvStore(tmp_path/'root') as d:
        assert d.read('input/sample.csv').loc[0,'gene_A']==1
    zipped=tmp_path/'data.zip'
    with zipfile.ZipFile(zipped,'w') as z:z.write(f,'any-root/input/sample.csv')
    with CsvStore(zipped) as d:
        assert d.read('input/sample.csv').loc[0,'cell_id']=='c1'

def test_duplicate_cells_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        matrix_from_csv(pd.DataFrame({'cell_id':['c1','c1'],'g':[1,2]}))

def test_misordered_genes_rejected():
    a=pd.DataFrame({'a':[1],'b':[2]},index=['c'])
    with pytest.raises(ValueError,match='index/columns differ'):
        require_pair(a,a[['b','a']],'test')
