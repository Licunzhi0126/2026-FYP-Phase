"""Cross-gene phase-label synchronisation (Saber Methods, Eq 14-15).

Per-gene decomposition does not guarantee globally consistent orientation: gene
g may call its higher-activity channel "Phase1" while gene h calls the opposite
one "Phase1". The audit on this dataset showed 26/100 genes mis-phased relative
to the global majority. This stage repairs that.

Saber's formulation:
    H = [E_X ; E_Y]                 profile matrix, 2G x n_cells
    S                               similarity graph over the 2G profiles
    L = D - S                       graph Laplacian
    L u2 = lambda2 u2               Fiedler vector                        (14)
    E_L = {i | u2_i < 0},  E_R = {i | u2_i >= 0}
    g*(i) = argmax_g (1/|g|) sum_{j in g} rho(i, j),   i in Omega         (15)

where Omega is the mismatch set: genes whose two profiles landed on the same
side, which is structurally forbidden.

=============================== LEAKAGE ===============================
This module NEVER sees maternal/paternal/reference matrices. Its only input is
the model's own predicted channel pair. Enforced by:

  * the public entry point takes (A_pred, B_pred) and nothing else -- there is
    no parameter through which a reference matrix could be passed;
  * the partition comes from the sign of the Fiedler vector, which is fixed by
    the data. No threshold, flip count, or stopping rule is chosen by looking
    at a score;
  * the OVERALL sign of an eigenvector is mathematically arbitrary, so "E_L" vs
    "E_R" is itself an exchangeable label. This module deliberately does NOT
    resolve it -- resolving it needs a reference, and that belongs to the
    post-hoc orientation step in evaluation/saber.py (1 bit, audited there). Calling
    sync() then reporting a number without that orientation step would hide
    where the reference information entered.

Correctness check that costs no ground truth: after synchronisation, per-gene
orientation should swap either ~0 or ~all genes. Anything in between means the
labels are still incoherent. `sync_report` returns that count.
=======================================================================
"""
import numpy as np


def _corr_rows(H):
    """Pearson correlation between rows of H, NaN-safe for constant rows."""
    Hc = H - H.mean(1, keepdims=True)
    sd = np.sqrt((Hc ** 2).sum(1))
    sd[sd < 1e-12] = np.inf                     # constant profile -> zero corr
    Hn = Hc / sd[:, None]
    return np.clip(Hn @ Hn.T, -1.0, 1.0)


def _fiedler(S):
    """Second-smallest eigenvector of L = D - S."""
    d = S.sum(1)
    L = np.diag(d) - S
    # normalised Laplacian is better conditioned when degrees are uneven
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Ln = (L * dinv[:, None]) * dinv[None, :]
    w, V = np.linalg.eigh((Ln + Ln.T) / 2.0)
    return V[:, np.argsort(w)[1]] * dinv        # back to unnormalised space


def sync(A_pred, B_pred):
    """Synchronise phase labels across genes.

    A_pred, B_pred : (n_cells, n_genes) predicted channel pair.
    Returns (A_sync, B_sync, info). Genes are flipped in place; the pair still
    sums to the same total, so downstream reconstruction is unchanged.
    """
    A = np.asarray(A_pred, float); B = np.asarray(B_pred, float)
    n_cells, n_genes = A.shape

    # H = [E_X ; E_Y] : row g is gene g's channel-A profile across cells,
    # row n_genes+g is its channel-B profile.
    H = np.vstack([A.T, B.T])                          # (2G, n_cells)
    rho = _corr_rows(H)
    S = np.clip(rho, 0.0, None)                        # anti-correlated -> no edge
    np.fill_diagonal(S, 0.0)

    u2 = _fiedler(S)
    side = (u2 >= 0).astype(int)                       # 0 = E_L, 1 = E_R
    side_A, side_B = side[:n_genes], side[n_genes:]

    # Omega: both profiles of a gene on the same side -- structurally forbidden
    omega = np.where(side_A == side_B)[0]
    for g in omega:
        # Eq 15: average similarity of each profile to each group, then take
        # the consistent assignment with the higher total score.
        i, j = g, n_genes + g
        sc = np.zeros((2, 2))                          # [profile, group]
        for grp in (0, 1):
            members = np.where(side == grp)[0]
            members = members[(members != i) & (members != j)]
            if len(members) == 0:
                continue
            sc[0, grp] = rho[i, members].mean()
            sc[1, grp] = rho[j, members].mean()
        # assignment 1: A->E_L, B->E_R ; assignment 2: A->E_R, B->E_L
        if sc[0, 0] + sc[1, 1] >= sc[0, 1] + sc[1, 0]:
            side_A[g], side_B[g] = 0, 1
        else:
            side_A[g], side_B[g] = 1, 0

    # Convention: channel A holds the E_L profile. Which of E_L/E_R is
    # "maternal" is NOT decided here -- see the LEAKAGE note above.
    flip = side_A == 1
    A_s = np.where(flip[None, :], B, A)
    B_s = np.where(flip[None, :], A, B)

    info = dict(n_flipped=int(flip.sum()), n_genes=n_genes,
                n_mismatch=int(len(omega)),
                fiedler_gap=float(np.abs(u2).mean()))
    return A_s, B_s, info


def sync_report(info, indent="  "):
    print(f"{indent}cross-gene phase sync: flipped {info['n_flipped']}/{info['n_genes']} genes"
          f"  (mismatch set |Omega|={info['n_mismatch']})")
