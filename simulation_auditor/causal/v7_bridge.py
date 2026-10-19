"""Isolated in-memory V7 generator patcher. Fail on unexpected source layout.

Warning: executes the user-supplied local generator as Python code; only use trusted
inputs. Does not overwrite simulation_generation sources.
"""
from pathlib import Path
import copy
import numpy as np
import zipfile
from .interventions import REGISTRY

SOURCE_NAME = 'simulate_allele_specific_grn_v7_imprinting_fixed.py'
PATCHES = {
    'final':('true_expr = softplus(z + np.log(imprinting_gate[:, None]))',
             'true_expr = softplus(z)'),
    'inner':(' * promoter_gate * imprinting_gate[:, None]',
             ' * promoter_gate'),
    'grn':('base *= 1.0 - strength','base *= 1.0'),
}
PROMOTER_NEEDLE = '    m_state = generate_element_state(params, rng, elements, regions, gene, "maternal", shared_latent)'
PROMOTER_INJECTION = '''
    # AUDITOR COUNTERFACTUAL: overwrite only imprinted promoters of nonpreferred allele.
    promoter_rows = elements.loc[elements["element_type"] == "promoter", ["element_id", "candidate_target_gene"]]
    ptmp = p_state.set_index("element_id", drop=False)
    mtmp = m_state.set_index("element_id", drop=False)
    status_by_gene = gene.set_index("gene_id")["imprinting_status"]
    copy_cols = ["open_state", "open_strength", "methylation", "accessibility", "activity"]
    for _, erow in promoter_rows.iterrows():
        eid = erow["element_id"]
        label = status_by_gene.loc[erow["candidate_target_gene"]]
        if label == "paternal_expressed":
            mtmp.loc[eid, copy_cols] = ptmp.loc[eid, copy_cols].to_numpy()
        elif label == "maternal_expressed":
            ptmp.loc[eid, copy_cols] = mtmp.loc[eid, copy_cols].to_numpy()
    p_state=ptmp.reset_index(drop=True)
    m_state=mtmp.reset_index(drop=True)
'''

def read_generator(path):
    p=Path(path)
    if p.is_file() and p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            matches=[s for s in z.namelist() if s.endswith('/'+SOURCE_NAME)]
            if len(matches)!=1:raise ValueError(f'Expected one V7 source in {p}; got {matches}')
            return z.read(matches[0]).decode('utf8'),str(p.resolve())+'::'+matches[0]
    if p.is_file():
        if p.name != SOURCE_NAME: raise ValueError(f'Expected generator file {SOURCE_NAME}')
        return p.read_text(encoding='utf8'),str(p.resolve())
    matches=list(p.rglob(SOURCE_NAME))
    if len(matches)!=1:raise ValueError(f'Expected one V7 generator inside {p}; got {len(matches)}')
    return matches[0].read_text(encoding='utf8'),str(matches[0].resolve())

def patched_module(source,condition):
    if condition not in REGISTRY:raise ValueError(f'Unknown C condition: {condition}')
    src=source
    for key in REGISTRY[condition].patch_keys:
        if key in ('region_off', 'promoter_imprint_off'):
            continue
        if key=='promoter':
            if src.count(PROMOTER_NEEDLE)!=1:
                raise ValueError('V7 generator changed: promoter patch anchor mismatch')
            src=src.replace(PROMOTER_NEEDLE,PROMOTER_NEEDLE+'\n'+PROMOTER_INJECTION,1)
        else:
            a,b=PATCHES[key]
            if src.count(a)!=1:
                raise ValueError(f'V7 generator changed: patch {key} expected once, observed {src.count(a)}')
            src=src.replace(a,b,1)
    context={'__name__':f'__asimaudit_{condition}__','__file__':'<in_memory_v7_generator>'}
    exec(compile(src,f'<asimaudit_v7_{condition}>','exec'),context)
    # Function-level wrappers keep imprint labels for matched comparison, while
    # supplying neutral imprint information to a selected upstream mechanism.
    keys = set(REGISTRY[condition].patch_keys)
    baseline_regions = []
    if 'region_off' in keys:
        original_region = context['assign_allelic_region_states']
        def without_region_imprint(params, rng, regions, elements, gene):
            cloned = np.random.default_rng()
            cloned.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
            base_states = original_region(params, rng, regions, elements, gene)
            baseline_regions.append(base_states)
            neutral = gene.copy()
            neutral['imprinting_status'] = 'none'
            neutral['imprinting_strength'] = 0.0
            alternative = original_region(params, cloned, regions, elements, neutral)
            # Restrict intervention to imprint-associated regions; leave all unrelated
            # region states exactly as in the paired baseline.
            imprint_ids = set(gene.loc[gene.imprinting_status != 'none', 'gene_id'])
            affected = set(elements.loc[elements.candidate_target_gene.isin(imprint_ids), 'chromatin_region_id'])
            alternative.loc[~alternative.region_id.isin(affected), 'allelic_state'] = (
                base_states.loc[~base_states.region_id.isin(affected), 'allelic_state'].to_numpy())
            return alternative
        context['assign_allelic_region_states'] = without_region_imprint
    if 'promoter_imprint_off' in keys or 'region_off' in keys:
        original_state = context['generate_element_state']
        def alternate_element_states(params, rng, elements, regions, gene, allele, shared_latent):
            cloned = np.random.default_rng()
            cloned.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
            baseline_region = baseline_regions[-1] if baseline_regions else regions
            original_state(params, rng, elements, baseline_region, gene, allele, shared_latent)
            neutral = gene.copy()
            if 'promoter_imprint_off' in keys:
                neutral['imprinting_status'] = 'none'
                neutral[f'{allele}_imprinting_gate'] = 1.0
            return original_state(params, cloned, elements, regions, neutral, allele, shared_latent)
        context['generate_element_state'] = alternate_element_states
    if 'PARAMS' not in context or 'run_simulation' not in context:
        raise ValueError('Unsupported generator: PARAMS/run_simulation missing')
    return context
