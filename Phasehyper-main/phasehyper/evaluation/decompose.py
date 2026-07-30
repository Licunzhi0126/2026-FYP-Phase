"""GRN decomposition head for phasehyper.

Second output of the model: split the combined per-cell gene->gene GRN into two
allele-like channels, alongside the expression split.

Two measured facts fix the design. Both were checked before building, on the
reference GRNs, as an audit (never during fitting):

1. THE SPLIT IS STATIC PER EDGE, NOT PER CELL.
   The allelic ratio (A-B)/(A+B) has std < 1e-6 across cells for 82.3% of the
   1165 valid edges (median 1.25e-15, exactly 0 for 15.3%); the remainder is
   denominator blow-up where A+B ~ 0. This follows from the generator:
       grn_X[c,s,t] = W_X[s,t] * log1p(shared_TF_activity[c,s]) + module term
   W_X is allele-specific but the TF activity is SHARED, so it cancels in the
   ratio. Consistent with combined GRN being rank-1 per source TF (mean rank-1
   explained variance 0.9421; exactly 1.0000 for two TFs).
   => the head emits ONE fraction per edge. A per-cell modulation term was
      present in an earlier version and is removed: it models variance that
      does not exist.

2. THE TWO CHANNELS ARE NEARLY IDENTICAL, UNLIKE EXPRESSION.
       expression   true per-cell cos(A,B) = -0.10,  ||A-B||/||A+B|| = 1.064
       GRN          true per-cell cos(A,B) = +0.966, ||A-B||/||A+B|| = 0.123
   So the orthogonality objective that self-calibrates the expression split is
   INVALID here -- it forces a full-magnitude split where the truth is 12%.
   Transplanted unchecked it scored 0.31 phaseMSE against 0.0065 for a plain
   50/50 split, 47x worse. Objectives must have their premise re-measured per
   task; `w_ortho` therefore defaults to 0 in this module.

Which representation the head reads follows the architecture's own split of
labour: the DIRECTED (causal) channel carries TF->target structure and is fed by
the per-cell GRN hyperedges, so the edge head scores edges from its gene
embeddings. The UNDIRECTED (functional) channel carries co-membership
(chromatin regions, pathways, compartments) and drives the expression phase.

Reference GRNs are used only after fitting, for post-hoc orientation and scoring.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRNPhaseHead(nn.Module):
    """Static per-edge split, scored from the directed channel's gene embeddings.

        logit[s,t] = MLP([h_s, h_t, h_s * h_t])
        f          = sigmoid(logit)
        grn_A      = grn_combined * f      grn_B = grn_combined * (1 - f)

    The pair sums back to `grn_combined` by construction, so no reconstruction
    term is needed. A shared MLP over edges (rather than one free parameter per
    edge) ties the split to the learned graph structure instead of memorising
    1165 independent numbers.

    A chromatin-region-structured variant was built and rejected (2026-07-21).
    The structure it encodes is real: allelic imbalance clusters hard by
    `gene_info.chromatin_region_id`, which is allele-blind and therefore legal to
    use -- ICC +0.582 by target region and +0.619 by source TF region, against
    -0.006 +- 0.103 for a random relabel, and 5-fold CV on HELD-OUT edges reaches
    +0.631 for the source x target partition versus +0.014 for a same-size random
    partition. Wiring it in as `u[R(s)] + v[R(t)] + <P[R(s)],Q[R(t)]>` moved the
    score 0.186 -> 0.182 (AUROC 0.643 -> 0.618). Structure fixes degrees of
    freedom; what limits this head is the anchor, so the extra machinery bought
    nothing. See train_head.
    """

    def __init__(self, edge_index, gene_dim, hidden=64):
        super().__init__()
        self.register_buffer("eidx", torch.tensor(edge_index, dtype=torch.long))
        self.mlp = nn.Sequential(
            nn.Linear(3 * gene_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))
        # start at f = 0.5: claim no imbalance until the data says otherwise
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def edge_logit(self, gene_emb):
        hs = gene_emb[self.eidx[:, 0]]
        ht = gene_emb[self.eidx[:, 1]]
        return self.mlp(torch.cat([hs, ht, hs * ht], -1)).squeeze(-1)

    def forward(self, gene_emb, grn_vals):
        f = torch.sigmoid(self.edge_logit(gene_emb))          # (n_edges,)
        return grn_vals * f[None, :], grn_vals * (1 - f)[None, :], f


def build_edges(grn):
    """Valid (src, tgt) pairs: nonzero somewhere across cells."""
    mag = np.abs(grn).mean(0)
    src, tgt = np.where(mag > 1e-6)
    return np.stack([src, tgt], 1), grn[:, src, tgt]


def train_head(gene_emb, grn_vals, edge_index, n_genes, expr_diff,
               epochs=2000, lr=5e-3, w_prior=1e-4, w_mag=1.0, seed=0,
               verbose=True):
    """Fit the edge split by INVERTING the GRN -> expression map.

    Expression is the main view; the GRN split is recovered as its consequence.
    A gene's allelic expression difference must be explained by the allelic
    difference in the regulatory inflow it receives:

        inflow_diff[c,t] = sum_s grn_combined[c,s,t] * (2*f[s,t] - 1)
        loss             = MSE( scale * inflow_diff + bias[t],  expr_diff[c,t] )

    `expr_diff` is the MODEL's own predicted A_gene - B_gene, never the
    reference. So this stage stays unsupervised; its quality is bounded by the
    expression branch that drives it.

    THIS OBJECTIVE IS WEAK AND THE HEAD'S OUTPUT SHOULD BE READ AS SUCH.
    An earlier version of this docstring justified it with "the true allelic GRN
    inflow difference tracks the true allelic expression difference at PCC 0.6446
    flat / 0.8040 per gene", and called the system over-determined. Both claims
    were re-measured on 2026-07-21 and neither supports the objective:

    1. 0.8040 IS THE WRONG AGGREGATION. The four readings are:
           flat (raveled)                    0.6446
           within-gene across cells          0.1426   <- what this loss needs
           cell-averaged, then across genes  0.8040   <- what was quoted
           within-cell across genes          0.6143
       The loss below solves per-(cell,gene) equations with NO per-gene bias
       (removed as collinear, see comment in the loop), so it rides the 0.1426
       linkage. The 0.8040 lives entirely in the per-gene offset that the deleted
       bias term would have absorbed.

    2. NOT OVER-DETERMINED IN PRACTICE. The per-target design matrix
       A_t[cell, source] = grn_combined[c,s,t] is badly conditioned, because f is
       static and grn[c,s,t] = W[s,t] * s_c(s) makes the cell equations near
       scalar multiples of each other. Identifiable unknowns: 1068/1165 at 1e-6
       tolerance, but only 629 (54%) at 1e-2 and 330 (28%) at 5e-2.

    Oracle audit -- feeding the GROUND-TRUTH expression difference and scoring
    the recovered f, across objective and parameterisation variants:

        region-structured + per-cell (this one)   PCC(f) +0.125
        region-structured + per-gene aggregate    PCC(f) +0.143
        region-structured + both, weighted        PCC(f) +0.107
        plain MLP + per-cell                      PCC(f) +0.097
        plain MLP + per-gene aggregate            PCC(f) +0.093

    Every variant lands at 0.09-0.14 even with a perfect target, so the split is
    not recoverable from expression at any anchoring level -- this is a property
    of the data, not a tuning failure. The deployed head scores 0.182, ABOVE all
    of those, which means most of its signal comes from graph structure in
    `gene_emb` rather than from this inversion.

    Consequence: the GRN decomposition is a downstream consistency check on the
    expression split, not an independent second result. Report it that way.

    `w_mag` adds the one constraint that IS well founded here. The inversion
    constrains the direction of the split but nothing about how far each edge
    should split, and unconstrained it over-splits by 2x (measured ||D_pred|| /
    ||D_true|| = 2.03, which drives the raw skill score to -3.59, i.e. worse than
    not phasing). The missing constraint is a priori, not fitted: for ANY
    decomposition of a quantity into two parts, the relative imbalance
    |A-B|/(|A|+|B|) is driven up as |A+B| approaches zero, because the two parts
    increasingly cancel. So weak edges should carry large relative imbalance and
    strong edges small. This follows from summation, not from this dataset -- the
    reference GRNs merely confirm it at Spearman -0.578, and that number is not
    used to set anything here.

    The term shapes ORDER only (a correlation against the edge-weight ranking),
    not magnitude, since nothing unsupervised fixes the overall scale.
    """
    torch.manual_seed(seed)
    head = GRNPhaseHead(edge_index, gene_emb.shape[1])
    scale = nn.Parameter(torch.tensor(1.0))
    bias = nn.Parameter(torch.zeros(n_genes))
    opt = torch.optim.AdamW(list(head.parameters()) + [scale, bias],
                            lr=lr, weight_decay=1e-4)
    ge = torch.as_tensor(gene_emb, dtype=torch.float32)
    gv = torch.as_tensor(grn_vals, dtype=torch.float32)
    tgt_idx = head.eidx[:, 1]
    y = torch.as_tensor(expr_diff, dtype=torch.float32)
    n_cells = gv.shape[0]

    # Target ordering for |2f-1|: the reverse of the edge-weight ordering.
    # Ranks rather than raw weights, because the relation is monotone but not
    # linear. Built from `grn_vals` alone -- no reference is in scope.
    w_edge = gv.abs().mean(0)
    tgt_rank = torch.empty_like(w_edge)
    tgt_rank[torch.argsort(w_edge, descending=True)] = torch.linspace(
        0.0, 1.0, len(w_edge))
    tgt_rank = tgt_rank - tgt_rank.mean()

    def _corr(a, b):
        a = a - a.mean()
        return (a * b).sum() / (a.norm() * b.norm() + 1e-8)

    for ep in range(epochs):
        opt.zero_grad()
        f = torch.sigmoid(head.edge_logit(ge))
        contrib = gv * (2.0 * f - 1.0)[None, :]
        inflow = torch.zeros(n_cells, n_genes).index_add_(1, tgt_idx, contrib)
        # NO per-gene bias: a free bias[t] is collinear with a static per-edge
        # split (both produce a cell-invariant per-gene offset) and absorbs the
        # entire target, leaving f unidentified.
        pred = scale * inflow
        fit = F.mse_loss(pred, y) / (y.pow(2).mean() + 1e-8)
        mag = -_corr((2.0 * f - 1.0).abs(), tgt_rank)
        loss = (fit + w_mag * mag
                + w_prior * head.edge_logit(ge).pow(2).mean())
        loss.backward(); opt.step()
        if verbose and (ep + 1) % 500 == 0:
            print(f"    grn ep {ep+1:4d}  fit={fit.item():.4f}  "
                  f"mean_f={f.mean().item():.4f}  sd_f={f.std().item():.4f}")

    with torch.no_grad():
        A, B, f = head(ge, gv)
    return A.numpy(), B.numpy(), f.numpy(), head


def scatter_back(vals, edge_index, n_cells, n_genes):
    """(n_cells, n_edges) -> (n_cells, n_genes, n_genes)."""
    out = np.zeros((n_cells, n_genes, n_genes), dtype=vals.dtype)
    out[:, edge_index[:, 0], edge_index[:, 1]] = vals
    return out