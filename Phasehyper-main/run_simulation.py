"""Simulation data validation: unsupervised decomposition combined -> X_A + X_B.
Evaluate against ground truth maternal/paternal ratio in dc (PCA) space.

Usage: python run_simulation.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from collections import defaultdict
import torch
from sklearn.decomposition import PCA
from phasehyper.evaluation.metrics_io import (
    print_differential,
    print_final,
    print_headline,
    print_orientation_audit,
    print_saber_table,
    save_grn_evaluation,
    save_saber_evaluation,
)
from phasehyper.evaluation.simulation import (
    evaluate_embedding_quality,
    evaluate_scale_diagnostics,
    evaluate_simulation_clustering,
    evaluate_simulation_expression,
    evaluate_simulation_grn,
)
from phasehyper.model import build_criterion, build_model, build_optimizer

DATA_DIR = "simulation_data"

# Outputs are split by task, because the two are not equally citable: the
# expression tables are the result, the GRN tables are a consistency check on
# it (see RESULTS.md §3.1). Keeping them in one flat namespace is what let the
# GRN numbers get quoted as an independent second result.
# ---------------------------------------------------------------------------
#  1. Data loading
# ---------------------------------------------------------------------------

def load_percell_grn(genes, cell_ids):
    """Per-cell gene->gene GRN from input/grn_combined (allele-blind: mat+pat).

    Returns (n_cells, n_genes, n_genes) with rows=source, cols=target, reindexed
    to `genes` order. Only the 15 shared-TF genes have outgoing edges.
    """
    grn_dir = Path(DATA_DIR) / "input" / "grn_combined"
    if not grn_dir.is_dir():
        return None
    mats = []
    for cid in cell_ids:
        f = grn_dir / f"{cid}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f, index_col=0).reindex(index=genes, columns=genes)
        mats.append(df.values.astype(np.float32))
    return np.nan_to_num(np.stack(mats))


def load_ref_grn(genes):
    """Reference allele GRNs. Scoring only -- never during fitting."""
    meta = pd.read_csv(f"{DATA_DIR}/input/cell_metadata.csv")
    out = []
    for sub in ("groundtruth/grn_maternal", "groundtruth/grn_paternal"):
        p = Path(DATA_DIR) / sub
        if not p.is_dir():
            return None, None
        out.append(np.stack([
            pd.read_csv(p / f"{c}.csv", index_col=0)
              .reindex(index=genes, columns=genes).values.astype(np.float32)
            for c in meta['cell_id']]))
    return np.nan_to_num(out[0]), np.nan_to_num(out[1])


def load_simulation():
    combined = pd.read_csv(f"{DATA_DIR}/input/combined_true_expression.csv", index_col=0)
    maternal = pd.read_csv(f"{DATA_DIR}/groundtruth/maternal_true_expression.csv", index_col=0)
    paternal = pd.read_csv(f"{DATA_DIR}/groundtruth/paternal_true_expression.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/input/cell_metadata.csv")
    cell_ids = combined.index.astype(str).tolist()
    gene_info = pd.read_csv(f"{DATA_DIR}/input/gene_info.csv")
    ppi = pd.read_csv(f"{DATA_DIR}/input/ppi_edges.csv")
    # region attributes: everything EXCEPT `allelic_state`. That column holds
    # maternal_open/paternal_open, i.e. a per-region allele label -- it is the
    # answer key sitting in input/, so it is deliberately not read here.
    regions = pd.read_csv(f"{DATA_DIR}/input/chromatin_regions.csv")
    regions = regions.drop(columns=['allelic_state'], errors='ignore').set_index('region_id')
    tf_info = pd.read_csv(f"{DATA_DIR}/input/shared_TF_info.csv")
    genes = list(combined.columns)
    y = meta['cell_type'].values
    k = len(np.unique(y))
    grn = load_percell_grn(genes, cell_ids)
    return dict(combined=combined.values.astype(np.float32),
                maternal=maternal.values.astype(np.float32),
                paternal=paternal.values.astype(np.float32),
                cell_ids=cell_ids, genes=genes, y=y, k=k,
                gene_info=gene_info, ppi=ppi, grn=grn,
                regions=regions, tf_info=tf_info)


# ---------------------------------------------------------------------------
#  2. Graph construction (simulation-specific priors)
# ---------------------------------------------------------------------------

def build_graph_sim(combined, genes, gene_info, ppi, top_k=15, z_floor=1.0,
                    grn=None, grn_top_k=10, regions=None):
    n_cells, n_genes = combined.shape
    gene2idx = {g: i for i, g in enumerate(genes)}
    tfs = set(gene_info[gene_info['is_TF'] == 1]['gene_id'])

    tf_targets = {}
    for _, row in ppi[ppi['grn_link'] == 1].iterrows():
        a, b = row['protein_a'], row['protein_b']
        if a in tfs and b in gene2idx:
            tf_targets.setdefault(a, set()).add(gene2idx[b])
        if b in tfs and a in gene2idx:
            tf_targets.setdefault(b, set()).add(gene2idx[a])
    tf_targets = {tf: sorted(tgts) for tf, tgts in tf_targets.items() if len(tgts) >= 2}
    tf_list = sorted(tf_targets)
    tf2node = {tf: j for j, tf in enumerate(tf_list)}
    n_tf = len(tf_list)
    tf_base = n_cells + n_genes
    N = n_cells + n_genes + n_tf

    mu = np.nanmean(combined, 0); std = np.nanstd(combined, 0) + 1e-8
    Z_rna = (combined - mu) / std

    # -- causal channel (directed) --
    tr, tc, hr, hc, Wl = [], [], [], [], []
    e_dir = 0; cnt_dir = defaultdict(int)
    et_ids, et_names, et2id = [], [], {}

    def add_dir(tail, head, et, w=1.0):
        nonlocal e_dir
        for nd in tail: tr.append(nd); tc.append(e_dir)
        for nd in head: hr.append(nd); hc.append(e_dir)
        if et not in et2id: et2id[et] = len(et_names); et_names.append(et)
        et_ids.append(et2id[et]); Wl.append(w); e_dir += 1; cnt_dir[et] += 1

    for c in range(n_cells):
        zc = Z_rna[c]
        gi = [int(g) for g in np.argsort(-zc)[:top_k] if zc[g] >= z_floor]
        if len(gi) >= 2:
            gn = [n_cells + g for g in gi]
            add_dir([c], gn, "rna_inject"); add_dir(gn, [c], "rna_readout")
    for tf in tf_list:
        tfn = tf_base + tf2node[tf]
        if tf in gene2idx:
            add_dir([n_cells + gene2idx[tf]], [tfn], "tf_activation")
    for tf, tgts in tf_targets.items():
        tfn = tf_base + tf2node[tf]
        head_genes = [n_cells + i for i in tgts]
        head_cells = set()
        for gi in tgts:
            for c in np.where(Z_rna[:, gi] >= z_floor)[0]:
                head_cells.add(int(c))
        head = head_genes + sorted(head_cells)
        add_dir([tfn], head, "reg_cascade", w=np.sqrt(len(head) + 1))
    for _, row in ppi[ppi['same_module'] == 1].iterrows():
        a, b = row['protein_a'], row['protein_b']
        if a in gene2idx and b in gene2idx:
            add_dir([n_cells + gene2idx[a]], [n_cells + gene2idx[b]], "module_coop")
    for region, grp in gene_info.groupby('chromatin_region_id'):
        gids = [n_cells + gene2idx[g] for g in grp['gene_id'] if g in gene2idx]
        if len(gids) >= 2:
            add_dir(gids, gids, "chromatin_region", w=np.sqrt(len(gids)))
    for comp in ['A', 'B']:
        gids = [n_cells + gene2idx[g] for g in gene_info[gene_info['compartment'] == comp]['gene_id'] if g in gene2idx]
        if len(gids) >= 2:
            add_dir(gids, gids, f"compartment_{comp}", w=np.sqrt(len(gids)))
    for chrom, grp in gene_info.groupby('chromosome'):
        grp_sorted = grp.sort_values('TSS')
        positions = grp_sorted['TSS'].values
        gids_chr = [gene2idx[g] for g in grp_sorted['gene_id'] if g in gene2idx]
        for i in range(len(gids_chr)):
            near = [n_cells + gids_chr[i]]
            for j in range(i + 1, len(gids_chr)):
                if abs(positions[j] - positions[i]) <= 200000:
                    near.append(n_cells + gids_chr[j])
                else:
                    break
            if len(near) >= 2:
                add_dir(near, near, "proximity_200kb", w=2.0)
    for chrom, grp in gene_info.groupby('chromosome'):
        for strand in ['+', '-']:
            sub = grp[grp['strand'] == strand].sort_values('TSS')
            gids_s = [gene2idx[g] for g in sub['gene_id'] if g in gene2idx]
            for i in range(len(gids_s) - 1):
                add_dir([n_cells + gids_s[i], n_cells + gids_s[i+1]],
                        [n_cells + gids_s[i], n_cells + gids_s[i+1]], "same_strand_adj")

    # -- per-cell GRN hyperedges (input/grn_combined) --
    # Static reg_cascade above is cell-invariant; these carry the per-cell TF
    # activity / module-activity modulation (per-edge CV ~0.58 across cells).
    # Activation and repression are separate edge types so the sign is not
    # averaged away, and the cell node sits in the tail so cell state drives
    # the cascade -- plus a readout edge so the modulated targets flow back.
    if grn is not None:
        src_rows = np.where((np.abs(grn).sum(0) != 0).any(1))[0]
        for c in range(n_cells):
            for s in src_rows:
                w_row = grn[c, s]
                for et, cand in (("grn_activate", np.argsort(-w_row)),
                                 ("grn_repress", np.argsort(w_row))):
                    sign = 1.0 if et == "grn_activate" else -1.0
                    tgts = [int(t) for t in cand[:grn_top_k]
                            if sign * w_row[t] > 0 and t != s]
                    if len(tgts) < 2:
                        continue
                    mag = float(np.abs(w_row[tgts]).mean())
                    head = [n_cells + t for t in tgts]
                    add_dir([c, n_cells + int(s)], head, et, w=mag)
                    add_dir(head, [c], f"{et}_readout", w=mag)

    H_tail = sp.csr_matrix((np.ones(len(tr)), (tr, tc)), shape=(N, e_dir))
    H_head = sp.csr_matrix((np.ones(len(hr)), (hr, hc)), shape=(N, e_dir))
    dir_data = dict(H_tail=H_tail, H_head=H_head, W=np.asarray(Wl, float),
                    etype=np.asarray(et_ids, dtype=np.int64),
                    et_names=et_names, n_types=len(et_names), e=e_dir, cnt=dict(cnt_dir))

    # -- functional channel (undirected) --
    rows_u, cols_u, vals_u, Wl_u = [], [], [], []
    e_u = 0; cnt_u = defaultdict(int)

    def add_u(nodes, et, w=1.0):
        nonlocal e_u
        for nd, mw in nodes: rows_u.append(nd); cols_u.append(e_u); vals_u.append(mw)
        Wl_u.append(w); e_u += 1; cnt_u[et] += 1

    for c in range(n_cells):
        zc = Z_rna[c]
        gi = [int(g) for g in np.argsort(-zc)[:top_k] if zc[g] >= z_floor]
        if len(gi) >= 2:
            add_u([(c, 1.0)] + [(n_cells + g, float(zc[g])) for g in gi], "RNA_obs")
    for region, grp in gene_info.groupby('chromatin_region_id'):
        gids = [gene2idx[g] for g in grp['gene_id'] if g in gene2idx]
        if len(gids) >= 2:
            # weight open/active regions higher: a closed heterochromatic region
            # co-regulates its genes far less than an active one.
            w = np.sqrt(len(gids))
            if regions is not None and region in regions.index:
                r = regions.loc[region]
                w *= float(np.clip(r['local_activity_score'], 0.05, 1.0)
                           * (1.0 - 0.5 * float(r['heterochromatin_score'])))
            add_u([(n_cells + g, 1.0) for g in gids], "chromatin_region", w=w)
    for chrom, grp in gene_info.groupby('chromosome'):
        grp_sorted = grp.sort_values('TSS')
        positions = grp_sorted['TSS'].values
        gids_chr = [gene2idx[g] for g in grp_sorted['gene_id'] if g in gene2idx]
        for i in range(len(gids_chr)):
            near = [gids_chr[i]]; mid = [gids_chr[i]]
            for j in range(i + 1, len(gids_chr)):
                dist = abs(positions[j] - positions[i])
                if dist <= 200000: near.append(gids_chr[j])
                if dist <= 500000: mid.append(gids_chr[j])
                else: break
            if len(near) >= 2:
                add_u([(n_cells + g, 1.0) for g in near], "proximity_200kb", w=2.0)
            if len(mid) > len(near) and len(mid) >= 2:
                add_u([(n_cells + g, 1.0) for g in mid], "proximity_500kb")
    for chrom, grp in gene_info.groupby('chromosome'):
        for strand in ['+', '-']:
            sub = grp[grp['strand'] == strand].sort_values('TSS')
            gids_s = [gene2idx[g] for g in sub['gene_id'] if g in gene2idx]
            if len(gids_s) >= 2:
                for i in range(len(gids_s) - 1):
                    add_u([(n_cells + gids_s[i], 1.0), (n_cells + gids_s[i+1], 1.0)], "same_strand_adj")
    for comp in ['A', 'B']:
        gids = [gene2idx[g] for g in gene_info[gene_info['compartment'] == comp]['gene_id'] if g in gene2idx]
        if len(gids) >= 2:
            add_u([(n_cells + g, 1.0) for g in gids], f"compartment_{comp}", w=np.sqrt(len(gids)))
    # Pathways as genuine high-order hyperedges. Shattering them into pairwise
    # edges (231 of them) throws away exactly the structure a hypergraph exists
    # to represent. kegg_pathways.csv gives no gene list, so membership is
    # recovered as connected components of the same_pathway PPI graph -- the
    # largest component (38 genes) matches PI3K-Akt's n_genes exactly.
    adj = defaultdict(set)
    for _, row in ppi[ppi['same_pathway'] == 1].iterrows():
        a, b = row['protein_a'], row['protein_b']
        if a in gene2idx and b in gene2idx:
            adj[gene2idx[a]].add(gene2idx[b]); adj[gene2idx[b]].add(gene2idx[a])
    seen = set()
    for start in sorted(adj):
        if start in seen: continue
        comp_nodes, stack = [], [start]; seen.add(start)
        while stack:
            u = stack.pop(); comp_nodes.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        if len(comp_nodes) >= 2:
            add_u([(n_cells + g, 1.0) for g in sorted(comp_nodes)],
                  "pathway_module", w=np.sqrt(len(comp_nodes)))

    H_undir = sp.csr_matrix((vals_u, (rows_u, cols_u)), shape=(N, e_u))
    undir_data = dict(H=H_undir, W=np.asarray(Wl_u, float), e=e_u, cnt=dict(cnt_u))

    info = dict(N=N, n_tf=n_tf, tf_base=tf_base, tf_list=tf_list,
                tf_targets=tf_targets, tf2node=tf2node, e_dir=e_dir, e_undir=e_u)
    return dir_data, undir_data, info


# ---------------------------------------------------------------------------
#  3. Node features
# ---------------------------------------------------------------------------

def build_features_sim(combined, info, dc, gene_info=None, genes=None):
    n_cells, n_genes = combined.shape
    M = (combined - combined.mean(0)) / (combined.std(0) + 1e-8)
    pca = PCA(dc, random_state=0).fit(M)
    cf_std = np.std(pca.transform(M))
    n_extra = info['N'] - n_cells
    X = np.zeros((n_extra, dc), np.float32)
    for g in range(n_genes):
        X[g] = pca.components_[:, g]
    if gene_info is not None and genes is not None:
        gene2idx = {g: i for i, g in enumerate(genes)}
        gi = gene_info.set_index('gene_id')
        n_annot = 10
        annot = np.zeros((n_genes, n_annot), np.float32)
        chr_map = {'chr1': 0, 'chr2': 1, 'chr3': 2, 'chr4': 3, 'chr5': 4}
        for g in genes:
            if g not in gi.index: continue
            row = gi.loc[g]; idx = gene2idx[g]
            if row['chromosome'] in chr_map: annot[idx, chr_map[row['chromosome']]] = 1.0
            annot[idx, 5] = 1.0 if row['compartment'] == 'A' else -1.0
            annot[idx, 6] = 1.0 if row['strand'] == '+' else -1.0
            annot[idx, 7] = row['TSS'] / 1.5e7
            annot[idx, 8] = row['local_gene_density'] / 11.0
            annot[idx, 9] = float(row['is_TF'])
        proj = np.zeros((n_annot, dc), np.float32)
        proj[:n_annot, :n_annot] = np.eye(n_annot)
        annot_proj = annot @ proj
        annot_std = np.std(annot_proj[annot_proj != 0]) if (annot_proj != 0).any() else 1.0
        if annot_std > 1e-9: annot_proj *= cf_std / annot_std * 0.5
        for g in range(n_genes): X[g] += annot_proj[g]
    for t, tf in enumerate(info['tf_list']):
        li = (info['tf_base'] + t) - n_cells
        tgts = info['tf_targets'].get(tf, [])
        if tgts: X[li] = pca.components_[:, tgts].mean(axis=1)
    x_std = np.std(X[X != 0]) if (X != 0).any() else 1.0
    if x_std > 1e-9: X *= cf_std / x_std
    return X, pca.components_


# ---------------------------------------------------------------------------
#  4. Shared model API (implemented in phasehyper/model.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  5. Training and evaluation
# ---------------------------------------------------------------------------

def _build_gene_priors(gene_info, genes):
    gene2idx = {g: i for i, g in enumerate(genes)}
    comp_A = [gene2idx[g] for g in gene_info[gene_info['compartment'] == 'A']['gene_id'] if g in gene2idx]
    comp_B = [gene2idx[g] for g in gene_info[gene_info['compartment'] == 'B']['gene_id'] if g in gene2idx]
    return dict(comp_A=comp_A, comp_B=comp_B)


def run(epochs=1200, seed=0, w_comp=8.0, w_ortho=4.0, w_gate=0.05, w_nce=1.0,
        use_asym=True, use_grn_cells=True, grn_top_k=10, use_phase_sync=False,
        run_grn=True, output_root: Path = Path("result_simulation"),
        visualize=True, visualization_dpi=300, genes_to_plot=None):
    """Calibrated split: per-cell orthogonality sets the magnitude.

    `w_shrink` (a pull of ||xa-xb|| toward 0) is gone. It bought +0.010 global
    PCC purely by collapsing the split toward the degenerate 50/50 answer --
    cos(P1,P2) 0.987 against a true -0.10, split magnitude 1/15 of truth,
    claimed imbalance 0.028 against a true 0.258 -- and cost the GRN head, which
    consumes A-B and therefore needs that difference correctly scaled:

                        w_shrink=0.25   w_ortho=4.0
        PCC                  0.6886        0.6788
        imb / imb-gene   0.3964/0.5177  0.3988/0.5182
        GRN AUROC            0.5527        0.5721
        GRN AUROC_B          0.6704        0.6827
        cos(P1,P2)           0.9869        0.0098

    The PCC it bought is not meaningful here: a plain 50/50 split already scores
    0.6863. See METHOD_AUDIT.md 6.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    output_root = Path(output_root)
    OUT_EXPR = output_root / "expression"
    OUT_GRN = output_root / "grn"
    for output_dir in (OUT_EXPR, OUT_GRN):
        output_dir.mkdir(parents=True, exist_ok=True)
    d = load_simulation()
    combined = d['combined']; n_cells, n_genes = combined.shape
    y, k = d['y'], d['k']
    M = (combined - combined.mean(0)) / (combined.std(0) + 1e-8)
    dc = max(2, min(64, n_cells - 1, n_genes))
    hidden = 256; latent = dc

    grn = d['grn'] if use_grn_cells else None
    if use_grn_cells and grn is None:
        print("  [warn] use_grn_cells=True but input/grn_combined not loaded -- skipping")
    dir_data, undir_data, info = build_graph_sim(combined, d['genes'], d['gene_info'],
                                                 d['ppi'], grn=grn, grn_top_k=grn_top_k,
                                                 regions=d['regions'])
    n_grn_e = sum(v for kk, v in dir_data['cnt'].items() if kk.startswith('grn_'))
    print(f"  graph: N={info['N']}(cell{n_cells}+gene{n_genes}+TF{info['n_tf']}) "
          f"causal:{info['e_dir']}(grn_percell:{n_grn_e}) func:{info['e_undir']}")

    gp = _build_gene_priors(d['gene_info'], d['genes'])
    gene2idx = {g: i for i, g in enumerate(d['genes'])}

    gf, pca_init = build_features_sim(combined, info, dc, gene_info=d['gene_info'], genes=d['genes'])
    model = build_model(
        directed_data=dir_data,
        undirected_data=undir_data,
        n_cells=n_cells,
        n_genes=n_genes,
        dc=dc,
        pca_init=pca_init,
        hidden=hidden,
        latent=latent,
        use_asym=use_asym,
        device="cpu",
    )
    M_t = torch.from_numpy(M.astype(np.float32)); gf_t = torch.from_numpy(gf)

    sigma = combined.std(0) + 1e-8; mu = combined.mean(0)

    comp_indicator = np.zeros(n_genes, dtype=np.float32)
    comp_indicator[gp['comp_A']] = 1.0; comp_indicator[gp['comp_B']] = -1.0
    comp_ind_t = torch.from_numpy(comp_indicator)
    W = pca_init; W_t = torch.from_numpy(W.astype(np.float32))
    # gt_A_dc + gt_B_dc == M @ W.T exactly, so anchoring ch to it lifts the
    # shared-signal ceiling (was 0.696 with ch unconstrained).
    M_pca_t = M_t @ W_t.T

    criterion = build_criterion(
        w_comp=w_comp,
        w_ortho=w_ortho,
        w_nce=w_nce,
        w_gate=w_gate,
    )
    opt = build_optimizer(model)

    print(f"  training {epochs} epochs ...")
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        model_output = model(M_t, gf_t, M_pca_t)
        ch, z, xa_dc, xb_dc = model_output
        loss, loss_terms = criterion(
            model=model,
            model_output=model_output,
            gene_projection=W_t,
            compartment_indicator=comp_ind_t,
        )
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if (ep + 1) % 100 == 0:
            print(f"    ep {ep+1:3d}  loss={loss.item():.4f}  "
                  f"cyc={loss_terms['cyc_comp'].item():.4f}  "
                  f"barlow={loss_terms['barlow'].item():.4f}  "
                  f"comp={loss_terms['compartment'].item():.4f}  "
                  f"ortho={loss_terms['orthogonality'].item():.4f}  "
                  f"nce={loss_terms['info_nce'].item():.4f}  "
                  f"cos={loss_terms['phase_cosine'].item():.4f}"
                  + (f"  asym={model.asym_scale.item():.4f}" if use_asym else ""))

    # ==================================================================
    #  Evaluation — dc space, Saber-style exchangeable labels
    # ==================================================================
    model.eval()
    with torch.no_grad():
        ch, z, xa_dc, xb_dc = model(M_t, gf_t, M_pca_t)
    xa_np = xa_dc.numpy(); xb_np = xb_dc.numpy()

    offset = (mu / sigma) @ W.T
    gt_A_dc = ((d['maternal'] / sigma) @ W.T - offset / 2).astype(np.float32)
    gt_B_dc = ((d['paternal'] / sigma) @ W.T - offset / 2).astype(np.float32)

    # Step 1 -- canonicalisation. Uses predictions only, no ground truth: the
    # lower-norm channel is named P1 so repeated runs are comparable. (Saber
    # does the same, ordering its two base sets by aggregate intensity.)
    na = np.linalg.norm(xa_np, axis=1).mean()
    nb = np.linalg.norm(xb_np, axis=1).mean()
    p1, p2 = (xa_np, xb_np) if na <= nb else (xb_np, xa_np)

    # Step 2 -- orientation. Decided ONCE in the post-training evaluation layer,
    # then applied to the dc-space pair too, so every metric below reports the
    # same assignment. (Previously dc space and gene space each decided this
    # independently and could in principle disagree.)
    # dc -> gene: x_gene = x_dc @ W * sigma (+ mu/2 each, so the halves sum to
    # `combined`).
    A_raw = (p1 @ W) * sigma + mu / 2.0
    B_raw = (p2 @ W) * sigma + mu / 2.0

    # Cross-gene phase-label synchronisation (Saber Eq 14-15). Reads only the
    # predicted pair -- no reference matrix is in scope here. Audited below at
    # the raw/global levels, which is where a real gain has to show up.
    A_pre_sync, B_pre_sync = A_raw.copy(), B_raw.copy()
    if use_phase_sync:
        import phase_sync
        A_raw, B_raw, sync_info = phase_sync.sync(A_raw, B_raw)
    else:
        sync_info = None

    phase_axes = dict(index=pd.Index(d["cell_ids"], name="cell_id"),
                      columns=d["genes"])
    pd.DataFrame(A_raw, **phase_axes).to_csv(
        OUT_EXPR / "phase_A.csv", float_format="%.8g")
    pd.DataFrame(B_raw, **phase_axes).to_csv(
        OUT_EXPR / "phase_B.csv", float_format="%.8g")

    # same rank-dc bottleneck the model's own output passes through
    def _proj(X):
        return (((X - mu / 2.0) / sigma) @ W.T @ W) * sigma + mu / 2.0
    _to_dc = lambda X: ((X - mu / 2.0) / sigma) @ W.T
    expression_eval = evaluate_simulation_expression(
        phase_a_pred=A_raw,
        phase_b_pred=B_raw,
        maternal_true=d["maternal"],
        paternal_true=d["paternal"],
        combined=combined,
        seed=seed,
        projection=_proj,
        to_dc=_to_dc,
        pre_sync_phase_a=A_pre_sync if sync_info is not None else None,
        pre_sync_phase_b=B_pre_sync if sync_info is not None else None,
    )
    A_gene = expression_eval["phase_a_oriented"]
    B_gene = expression_eval["phase_b_oriented"]
    swapped = bool(expression_eval["orientation"]["n_swapped"])
    summary = expression_eval["summary"]
    assign = summary["assign"]
    pcc_mat, pcc_pat = summary["pcc_mat"], summary["pcc_pat"]
    cell_pcc_mat = summary["cell_pcc_mat"]
    cell_pcc_pat = summary["cell_pcc_pat"]
    imb_pcc, imb_gene_pcc = summary["imb_pcc"], summary["imb_gene_pcc"]
    imb_cell_pcc, imb_mae = summary["imb_cell_pcc"], summary["imb_mae"]
    imb_mag_pred, imb_mag_gt = summary["imb_mag_pred"], summary["imb_mag_true"]
    saber_rows = expression_eval["saber_rows"]
    head_rows = expression_eval["headline_rows"]
    orient_rows = expression_eval["orientation_audit"]
    orient_rows_nosync = expression_eval["pre_sync_orientation_audit"]

    clustering_eval = evaluate_simulation_clustering(
        raw_rna=M,
        cell_embedding=ch.numpy(),
        phase_a_embedding=xa_np,
        phase_b_embedding=xb_np,
        labels=y,
        n_clusters=k,
        seed=seed,
    )
    ar_raw = clustering_eval["raw"]
    ar_ch = clustering_eval["cell_h"]
    ar_a = clustering_eval["phase_a"]
    ar_b = clustering_eval["phase_b"]
    embedding_quality = evaluate_embedding_quality(
        cell_embedding=ch.numpy(),
        phase_a_embedding=xa_np,
        phase_b_embedding=xb_np,
    )
    rec, cos_ab = embedding_quality["rec"], embedding_quality["cos_ab"]
    print(f"\n  PCC ({assign}):")
    print(f"    global:   mat={pcc_mat:.4f}  pat={pcc_pat:.4f}")
    print(f"    per-cell: mat={cell_pcc_mat:.4f}  pat={cell_pcc_pat:.4f}")
    print(f"  Imbalance (A-B)/(A+B):")
    print(f"    PCC: global={imb_pcc:.4f}  per-gene={imb_gene_pcc:.4f}  per-cell={imb_cell_pcc:.4f}")
    print(f"    MAE={imb_mae:.4f}   mean|imb|: pred={imb_mag_pred:.4f} true={imb_mag_gt:.4f}")
    print(f"  recon: {rec:.2e}  cos(P1,P2): {cos_ab:.4f}")
    print_headline(head_rows)
    print_saber_table(saber_rows)
    _grn_rows = None
    # ---- second output: GRN decomposition ----
    if run_grn and d['grn'] is not None:
        from phasehyper.evaluation import decompose as GD
        eidx, gvals = GD.build_edges(d['grn'])
        print(f"\n  GRN head: {len(eidx)} valid (TF,target) edges, "
              f"{gvals.shape[0]} cells")
        # A_raw/B_raw, NOT A_gene/B_gene: the latter were oriented against the
        # reference, so training on their difference would put 1 bit of ground
        # truth into fitting.
        gA, gB, gf, _ = GD.train_head(model.last_causal_genes.numpy(), gvals,
                                      eidx, n_genes, A_raw - B_raw, seed=seed)
        gt_gA, gt_gB = load_ref_grn(d['genes'])
        if gt_gA is not None:
            tA = gt_gA[:, eidx[:, 0], eidx[:, 1]]
            tB = gt_gB[:, eidx[:, 0], eidx[:, 1]]
            grn_eval = evaluate_simulation_grn(
                grn_a_pred=gA,
                grn_b_pred=gB,
                grn_a_true=tA,
                grn_b_true=tB,
                combined_grn=gvals,
                inherited_swap=swapped,
                seed=seed,
            )
            gA = grn_eval["phase_a_oriented"]
            gB = grn_eval["phase_b_oriented"]
            grn_rows = grn_eval["saber_rows"]
            _grn_rows = grn_eval["headline_rows"]
            diff_rows = grn_eval["differential_rows"]
            print_saber_table(
                grn_rows,
                "GRN decomposition vs reference GRNs "
                "[consistency check, NOT an independent result -- "
                "the head is driven by the expression split]",
            )
            save_grn_evaluation(
                output_dir=OUT_GRN,
                headline_rows=_grn_rows,
                saber_rows=grn_rows,
                differential_rows=diff_rows,
                metadata=dict(seed=seed, n_edges=len(eidx)),
            )
            print(f"  written to {OUT_GRN / 'saber_protocol.csv'}  "
                  "[protocol continuity only -- not a result]")
            print_differential(diff_rows)
            print(f"  written to {OUT_GRN / 'differential.csv'}   <- the GRN result")

            # Per-edge arrays for the decomposition figure. gA/gB already carry
            # the inherited orientation, so no further swap here.
            bl = grn_eval["baseline_arrays"]
            np.savez_compressed(
                OUT_GRN / "edges.npz",
                edge_index=eidx, combined=gvals, pred_A=gA, pred_B=gB,
                true_A=tA, true_B=tB,
                genes=np.array(d['genes'], dtype=object),
                cell_type=np.array(d['y'], dtype=object), **bl)
            print(f"  per-edge arrays written to {OUT_GRN / 'edges.npz'}")

    expression_paths = save_saber_evaluation(
        output_dir=OUT_EXPR,
        headline_rows=head_rows,
        saber_rows=saber_rows,
        orientation_rows=orient_rows,
        metadata=dict(
            seed=seed,
            epochs=epochs,
            dc=dc,
            w_ortho=w_ortho,
            w_comp=w_comp,
            w_gate=w_gate,
            w_nce=w_nce,
            use_grn_cells=int(use_grn_cells),
            use_phase_sync=int(use_phase_sync),
            min_support_q=expression_eval["settings"]["min_support_q"],
        ),
    )
    csv_path = expression_paths["metrics"]
    print(f"\n  metrics written to {csv_path}")
    print_final(head_rows, _grn_rows)
    if sync_info is not None:
        import phase_sync
        phase_sync.sync_report(sync_info)
        print_orientation_audit(
            orient_rows_nosync, "Orientation audit -- BEFORE cross-gene sync"
        )
    print_orientation_audit(
        orient_rows,
        "Orientation audit -- AFTER cross-gene sync"
        if sync_info is not None
        else "Orientation audit (Saber Fig 3E protocol)",
    )
    print(f"  ARI: raw={ar_raw:.3f}  cell_h={ar_ch:.3f}  P1={ar_a:.3f}  P2={ar_b:.3f}")
    # Console only. 12 of the 14 gates sit within 0.01 of the 0.90 prior, so a
    # persisted table invites reading structure into numerical noise.
    gates = torch.sigmoid(model.causal_ch.type_logit).detach().numpy()
    order = np.argsort(-gates)
    print("\n  learned edge-type gates (directed channel, prior=0.90):")
    for i in order:
        nm = dir_data['et_names'][i]
        print(f"    {nm:<24}{gates[i]:.4f}   ({dir_data['cnt'].get(nm, 0)} edges)")
    print(f"    [spread] std={gates.std():.4f}  range=[{gates.min():.4f},{gates.max():.4f}]")

    if use_asym:
        b = model.last_bias.numpy()
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        cross = bn @ bn.T
        off = cross[~np.eye(len(bn), dtype=bool)]
        print(f"  per-cell split direction: mean pairwise cos={off.mean():.4f} "
              f"(1.0 => all cells share one direction, i.e. the old scalar model)")
        s_np = np.linalg.norm(b, axis=1)
        print(f"  asym_scale: {model.asym_scale.item():.4f}  "
              f"asym_dir_norm: {model.asym_dir.data.norm().item():.4f}")
        # s_c spread == 0 means the per-cell GRN edges changed nothing and the
        # split is still effectively a single global direction.
        print(f"  ||bias||: mean={s_np.mean():.4f} std={s_np.std():.4f} "
              f"range=[{s_np.min():.4f},{s_np.max():.4f}]")

    # --- diagnostic: decompose PCC into baseline vs difference-signal ---
    scale_diagnostics = evaluate_scale_diagnostics(
        cell_embedding=ch.numpy(),
        canonical_phase_a=p1,
        canonical_phase_b=p2,
        maternal_embedding=gt_A_dc,
        paternal_embedding=gt_B_dc,
        standardized_expression=M,
        projection_components=W,
        assign=assign,
    )
    print(
        f"\n  [diag] trivial ch/2 vs gt_A PCC = "
        f"{scale_diagnostics['pcc_trivial']:.4f}   (model mat={pcc_mat:.4f})"
    )
    print(
        f"  [diag] ch/2 vs (gt_A+gt_B)/2 PCC = "
        f"{scale_diagnostics['pcc_shared']:.4f}  <- shared-signal ceiling"
    )
    print(f"  [diag] NULL (PCA/2, no model) vs gt_A PCC = "
          f"{scale_diagnostics['pcc_null']:.4f}  "
          f"<- model must beat this")
    print(f"  [diag] ||d_pred||/||d_gt|| = {scale_diagnostics['diff_ratio']:.4f}")
    for s, r in scale_diagnostics["scale_rows"]:
        print(f"  [diag] scale={s:<4} -> mat PCC {r:.4f}")
    print(
        f"  [diag] best scale={scale_diagnostics['best_scale']} "
        f"PCC={scale_diagnostics['best_pcc']:.4f}"
    )

    # Visualization is the final workflow step. It reads the numerical outputs
    # written above, so standalone reruns and integrated runs use identical data.
    # A plotting failure must not invalidate the completed simulation evaluation.
    if visualize:
        try:
            from phasehyper.visualization import run_simulation_visualization

            run_simulation_visualization(
                sim_dir=Path(DATA_DIR),
                result_dir=output_root,
                dpi=visualization_dpi,
                genes_to_plot=genes_to_plot,
            )
        except Exception as exc:
            print(f"  [visualization warning] {type(exc).__name__}: {exc}")
    else:
        print("  [visualization] skipped")

    # Fitted arrays, so downstream figures can re-use THIS fit instead of
    # standing up their own training loop. saber_fig3_exact.py used to carry a
    # duplicate 300-epoch copy of the pipeline; it drifted off the real API and
    # broke, and until it broke it was plotting a different model than the one
    # the results tables describe.
    fit = dict(xa_dc=xa_np, xb_dc=xb_np, cell_h=ch.numpy(),
               gt_A_dc=gt_A_dc, gt_B_dc=gt_B_dc,
               A_raw=A_raw, B_raw=B_raw, A_gene=A_gene, B_gene=B_gene,
               W=W, mu=mu, sigma=sigma, dc=dc,
               cell_types=d['y'], genes=d['genes'],
               maternal=d['maternal'], paternal=d['paternal'],
               combined=combined)

    return dict(fit=fit,
                pcc_mat=pcc_mat, pcc_pat=pcc_pat,
                cell_pcc_mat=cell_pcc_mat, cell_pcc_pat=cell_pcc_pat,
                imb_pcc=imb_pcc, imb_gene_pcc=imb_gene_pcc, imb_cell_pcc=imb_cell_pcc,
                imb_mae=imb_mae, imb_mag_pred=imb_mag_pred, imb_mag_gt=imb_mag_gt,
                saber=saber_rows, saber_self=saber_rows[0], orient_audit=orient_rows,
                ar_raw=ar_raw, ar_ch=ar_ch, ar_a=ar_a, ar_b=ar_b,
                cos_ab=cos_ab, diff_ratio=scale_diagnostics["diff_ratio"],
                best_scale=scale_diagnostics["best_scale"],
                best_pcc=scale_diagnostics["best_pcc"])


def sweep_ortho(values=(0.0, 1.0, 2.0, 4.0, 8.0), epochs=1200):
    rows = []
    for v in values:
        r = run(epochs=epochs, w_ortho=v, visualize=False)
        rows.append((v, r))
    print("\n" + "=" * 72)
    print(f"{'ortho':>8} {'mat':>8} {'pat':>8} {'imbPCC':>8} {'imbMAE':>8} {'cos':>8} "
          f"{'|d|ratio':>9} {'bestScl':>8} {'bestPCC':>8}")
    for v, r in rows:
        print(f"{v:>8.2f} {r['pcc_mat']:>8.4f} {r['pcc_pat']:>8.4f} "
              f"{r['imb_pcc']:>8.4f} {r['imb_mae']:>8.4f} {r['cos_ab']:>8.4f} "
              f"{r['diff_ratio']:>9.4f} {r['best_scale']:>8.2f} {r['best_pcc']:>8.4f}")
    return rows


def sweep_seeds(seeds=(0, 1, 2, 3, 4), epochs=1200):
    rows = [(s, run(epochs=epochs, seed=s, visualize=False)) for s in seeds]
    print("\n" + "=" * 52)
    print(f"{'seed':>6} {'mat':>8} {'pat':>8} {'imbPCC':>8} {'ARI_P1':>8}")
    for s, r in rows:
        print(f"{s:>6} {r['pcc_mat']:>8.4f} {r['pcc_pat']:>8.4f} "
              f"{r['imb_pcc']:>8.4f} {r['ar_a']:>8.3f}")
    mats = np.array([r['pcc_mat'] for _, r in rows])
    pats = np.array([r['pcc_pat'] for _, r in rows])
    print(f"{'mean':>6} {mats.mean():>8.4f} {pats.mean():>8.4f}")
    print(f"{'std':>6} {mats.std():>8.4f} {pats.std():>8.4f}")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=("run", "sweep", "seeds"),
        help="main run or a benchmark sweep",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="skip the final simulation visualization pipeline",
    )
    parser.add_argument(
        "--visualization-dpi",
        type=int,
        default=300,
        help="DPI for generated simulation figures",
    )
    parser.add_argument(
        "--gene",
        action="append",
        dest="genes_to_plot",
        help="gene ID for a detail figure; may be supplied more than once",
    )
    args = parser.parse_args()
    mode = args.mode
    if mode == "sweep":
        sweep_ortho()
    elif mode == "seeds":
        sweep_seeds()
    else:
        run(
            visualize=not args.no_visualization,
            visualization_dpi=args.visualization_dpi,
            genes_to_plot=args.genes_to_plot,
        )