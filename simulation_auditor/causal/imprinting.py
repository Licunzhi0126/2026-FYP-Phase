"""C: compare the same imprint labels and unaffected random state across conditions."""

def check_pairing(base, current, patches):
    for col in ('gene_id','imprinting_status','imprinting_strength'):
        if not current['gene'][col].equals(base['gene'][col]):
            raise AssertionError(f'Paired audit changed gene field {col}')
    if not current['motif'].equals(base['motif']):
        raise AssertionError('Paired audit changed motif sampling')
    if 'promoter' not in patches and 'promoter_imprint_off' not in patches and 'region_off' not in patches:
        for k in ('p_state','m_state'):
            if not current[k].equals(base[k]):
                raise AssertionError(f'Paired audit changed {k} unexpectedly')
    elif 'promoter' in patches and 'region_off' not in patches:
        for k in ('p_state','m_state'):
            a=current[k].set_index('element_id')
            b=base[k].set_index('element_id')
            keep=b.element_type.ne('promoter')
            if not a.loc[keep].equals(b.loc[keep]):
                raise AssertionError('Promoter intervention modified non-promoter elements')
        imprinted=current['gene'].query('imprinting_status != "none"')['gene_id']
        promoters=current['elements'].query('element_type == "promoter"')
        promoters=promoters[promoters.candidate_target_gene.isin(imprinted)]
        pacc=current['p_state'].set_index('element_id')['accessibility']
        macc=current['m_state'].set_index('element_id')['accessibility']
        if any(abs(pacc[x]-macc[x])>1e-10 for x in promoters.element_id):
            raise AssertionError('Equalized promoters remain different')
