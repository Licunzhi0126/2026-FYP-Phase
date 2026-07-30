"""Saber-protocol evaluation suite (Xue, Li, Zhang & Li 2026).

Metric families, following the paper's Methods ("Evaluation metrics and
post-hoc analyses"):

  primary    phase-fit MSE      reconstruction error to the two reference
                                allele matrices after post-hoc orientation
  secondary  phase-fit Pearson  orientation audit
  gene-level mean phase imbalance vs held-out truth (Spearman)
             high-imbalance detection (AUROC / AUPRC)
             major/minor recovery (unordered pair, no parental label needed)

Reference matrices are consumed here and here only -- after prediction, for
scoring and orientation. Nothing in this module may be called during fitting.

Saber's own values, for orientation of magnitude (different data, NOT directly
comparable): phase-fit Pearson 0.846-0.932, imbalance Spearman 0.96,
high-imbalance AUPRC 0.95 / AUROC 0.97, major r 0.96 / minor r 0.78.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import NMF

HIGH_IMB_QUANTILE = 0.75   # top quartile of true mean |imbalance| = positives

# Expression-support filter for ratio-based metrics. (A-B)/(A+B) is unusable
# where A+B is near zero: the bottom 1% of entries alone drag a *perfect*
# rank-dc decomposition from 0.89 to 0.27, and its mean |imbalance| error in the
# bottom decile is 0.295 against ~0.08 everywhere else.
#
# 0.05 was chosen on two properties of the METRIC, not of any model's score:
#   * validity  -- below it the perfect-decomposition floor scores *under* the
#                  model, which is incoherent for an upper bound;
#   * stability -- p5..p50 is a plateau (0.886 / 0.901 / 0.887), so the value is
#                  not sensitive to the exact cut.
# The mask is built from `combined`, which is INPUT data, so it leaks nothing,
# and it is identical for every method.
MIN_SUPPORT_Q = 0.05


def _support_mask(A_true, B_true, signed=False):
    """signed=True for GRN weights, which are ~23% negative. There A+B is not a
    magnitude and can sit near zero for large |A|,|B|, so both the support mask
    and the imbalance ratio must be built on |A|+|B| instead. Using the
    expression form on signed data makes every ratio metric n/a."""
    total = (np.abs(A_true) + np.abs(B_true)) if signed else (A_true + B_true)
    return total >= np.quantile(total, MIN_SUPPORT_Q)


def _imbalance(A, B, signed=False):
    if signed:
        return (np.abs(A) - np.abs(B)) / (np.abs(A) + np.abs(B) + 1e-9)
    return (A - B) / (np.abs(A + B) + 1e-9)


def orient(A_pred, B_pred, A_true, B_true, level="global"):
    """Post-hoc orientation. Phase1/Phase2 are exchangeable labels, so the pair
    must be matched to the references before scoring. This consumes ground
    truth and is therefore scoring, never fitting.

    Ground-truth information consumed:
        raw       0 bits          labels left exchangeable
        global    1 bit           one swap decision for the whole matrix
        per_gene  n_genes bits    an independent swap decision per gene

    Saber reports all three (Fig 3E) rather than folding the strongest one into
    the headline; per-gene orientation gives its best MSEs but costs two orders
    of magnitude more reference information than global.
    """
    if level == "raw":
        return A_pred, B_pred, dict(level="raw", bits=0, n_swapped=0)

    if level == "global":
        e1 = np.mean((A_pred - A_true) ** 2) + np.mean((B_pred - B_true) ** 2)
        e2 = np.mean((A_pred - B_true) ** 2) + np.mean((B_pred - A_true) ** 2)
        swap = e2 < e1
        info = dict(level="global", bits=1, n_swapped=int(swap),
                    assign="P2=A" if swap else "P1=A")
        return (B_pred, A_pred, info) if swap else (A_pred, B_pred, info)

    if level == "per_gene":
        e1 = ((A_pred - A_true) ** 2).mean(0) + ((B_pred - B_true) ** 2).mean(0)
        e2 = ((A_pred - B_true) ** 2).mean(0) + ((B_pred - A_true) ** 2).mean(0)
        swap = e2 < e1
        A_o = np.where(swap[None, :], B_pred, A_pred)
        B_o = np.where(swap[None, :], A_pred, B_pred)
        return A_o, B_o, dict(level="per_gene", bits=int(A_pred.shape[1]),
                              n_swapped=int(swap.sum()))

    raise ValueError(f"unknown orientation level: {level}")


def orientation_audit(A_pred, B_pred, A_true, B_true, name=""):
    """Saber Fig 3E-style table: how much does each orientation level buy, and
    how much reference information does it cost to buy it?"""
    rows = []
    for lvl in ("raw", "global", "per_gene"):
        A_o, B_o, info = orient(A_pred, B_pred, A_true, B_true, lvl)
        mse = 0.5 * (np.mean((A_o - A_true) ** 2) + np.mean((B_o - B_true) ** 2))
        rows.append(dict(name=name, **info, mse=mse))
    base = rows[0]['mse']
    for r in rows:
        r['gain_pct'] = 100.0 * (base - r['mse']) / base if base > 0 else np.nan
    return rows


def phase_metrics(A_pred, B_pred, A_true, B_true, name="", level="global",
                  signed=False):
    A_pred, B_pred, oinfo = orient(A_pred, B_pred, A_true, B_true, level)
    assign = oinfo.get("assign", oinfo["level"])

    mse = 0.5 * (np.mean((A_pred - A_true) ** 2) + np.mean((B_pred - B_true) ** 2))
    pear = 0.5 * (pearsonr(A_pred.ravel(), A_true.ravel())[0]
                  + pearsonr(B_pred.ravel(), B_true.ravel())[0])

    msk = _support_mask(A_true, B_true, signed)
    imb_p = np.where(msk, _imbalance(A_pred, B_pred, signed), np.nan)
    imb_t = np.where(msk, _imbalance(A_true, B_true, signed), np.nan)
    with np.errstate(invalid='ignore'):
        gp, gt = np.nanmean(np.abs(imb_p), 0), np.nanmean(np.abs(imb_t), 0)
        sp_, st_ = np.nanmean(imb_p, 0), np.nanmean(imb_t, 0)
    # A column masked out in EVERY cell yields nanmean -> NaN, which then makes
    # spearmanr / roc_auc_score return NaN for the whole metric. Drop those
    # columns (9/1165 on the GRN) rather than letting them void the metric.
    ok = ~(np.isnan(gp) | np.isnan(gt))
    gp, gt, sp_, st_ = gp[ok], gt[ok], sp_[ok], st_[ok]
    # a degenerate 50/50 split has zero predicted imbalance -> undefined ranking
    degenerate = np.allclose(gp, gp[0]) if gp.size else True

    imb_sp = np.nan if degenerate else spearmanr(gp, gt)[0]
    imb_pcc = pearsonr(imb_p[msk], imb_t[msk])[0]
    imb_gene_pcc = np.nan if degenerate else pearsonr(sp_, st_)[0]

    pos = (gt >= np.quantile(gt, HIGH_IMB_QUANTILE)).astype(int)
    valid = (not degenerate) and 0 < pos.sum() < len(pos)
    auroc = roc_auc_score(pos, gp) if valid else np.nan
    auprc = average_precision_score(pos, gp) if valid else np.nan

    # unordered pair -- needs no parental label at all
    maj_p, min_p = np.maximum(A_pred, B_pred), np.minimum(A_pred, B_pred)
    maj_t, min_t = np.maximum(A_true, B_true), np.minimum(A_true, B_true)
    major_r = pearsonr(maj_p.ravel(), maj_t.ravel())[0]
    minor_r = pearsonr(min_p.ravel(), min_t.ravel())[0]

    return dict(name=name, assign=assign, orient_level=level, mse=mse, pearson=pear,
                imb_spearman=imb_sp, imb_pcc=imb_pcc, imb_gene_pcc=imb_gene_pcc,
                imb_auroc=auroc, imb_auprc=auprc,
                major_r=major_r, minor_r=minor_r,
                mean_imb_pred=float(gp.mean()), mean_imb_true=float(gt.mean()))


# ---------------------------------------------------------------------------
#  Baselines. All receive only `combined` and must sum back to it (except
#  NMF, which reconstructs). Same treatment and same scoring as the model.
# ---------------------------------------------------------------------------

def baseline_random_split(combined, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.random(combined.shape)
    return combined * f, combined * (1 - f)


def baseline_mean_fraction_shrinkage(combined):
    """Every gene split at the global mean fraction -> zero claimed imbalance.
    The trivial estimator any method must beat."""
    return combined * 0.5, combined * 0.5


def baseline_nmf2(combined, seed=0):
    m = NMF(n_components=2, init='nndsvda', random_state=seed, max_iter=1000)
    Wm = m.fit_transform(np.clip(combined, 0, None)); Hm = m.components_
    return Wm[:, :1] @ Hm[:1, :], Wm[:, 1:] @ Hm[1:, :]


def run_baselines(combined, A_true, B_true, seed=0, proj=None, signed=False):
    """proj: optional callable applied to every baseline output, so baselines
    pass through the same representational bottleneck as the model.

    The model emits a rank-dc reconstruction (dc=64 of 100 genes here), which
    costs it phaseMSE that has nothing to do with decomposition quality: even a
    perfect split scores 0.0623 after that projection. Saber applies the same
    scope to its baselines ("projected to the same benchmark scope"), so
    withholding it here would flatter the baselines."""
    p = (lambda x: x) if proj is None else proj
    out = []
    a, b = baseline_random_split(combined, seed)
    out.append(phase_metrics(p(a), p(b), A_true, B_true, "RandomSplit", signed=signed))
    a, b = baseline_mean_fraction_shrinkage(combined)
    out.append(phase_metrics(p(a), p(b), A_true, B_true, "MeanFractionShrinkage", signed=signed))
    a, b = baseline_nmf2(combined, seed)
    out.append(phase_metrics(p(a), p(b), A_true, B_true, "NMF2Factor", signed=signed))
    return out


def differential_metrics(A_pred, B_pred, A_true, B_true, name=""):
    """Score ONLY the differential component. The right metric set for the GRN.

    `A + B == combined` holds exactly and `combined` is an input, so

        A = combined/2 + D,   B = combined/2 - D,   D = (A - B)/2

    means combined/2 is given for free and every recoverable bit lives in D.
    Scoring A and B directly measures mostly the free part: the reference
    maternal and paternal GRNs correlate at 0.966 per cell, so combined/2 scores
    0.992 matrix PCC and beats the model in 160/160 cells. That number says
    nothing about phasing. D has a true zero -- refusing to phase is D = 0 --
    which is what makes a skill score well defined here.

    Fields:
        pcc, pcc_cell, spearman   direction of the split
        d_ratio                   ||D_pred|| / ||D_true||; 1.0 = calibrated
        nmse                      ||D_pred - D_true||^2 / ||D_true||^2
        skill                     1 - nmse; 0 = not phasing, 1 = perfect
        alpha                     shrinkage minimising nmse
        skill_cal                 skill at alpha; costs 1 ground-truth parameter
        auroc                     dominant allele, on |D_true| above its median

    Report `skill` and `skill_cal` together: a method can carry real directional
    signal and still land below zero raw if its magnitude is miscalibrated,
    which is exactly what the GRN head does (alpha ~ 0.06).
    """
    Dp, Dt = (A_pred - B_pred) / 2.0, (A_true - B_true) / 2.0
    flat_p, flat_t = Dp.ravel(), Dt.ravel()
    dead = float(np.std(flat_p)) < 1e-12
    nmse = float(((Dp - Dt) ** 2).sum() / ((Dt ** 2).sum() + 1e-12))
    alpha = 0.0 if dead else float((Dp * Dt).sum() / ((Dp ** 2).sum() + 1e-12))
    nmse_c = float(((alpha * Dp - Dt) ** 2).sum() / ((Dt ** 2).sum() + 1e-12))
    msk = np.abs(Dt) > np.median(np.abs(Dt))
    return dict(
        name=name,
        pcc=0.0 if dead else pearsonr(flat_p, flat_t)[0],
        pcc_cell=0.0 if dead else float(np.nanmean(
            [pearsonr(Dp[c], Dt[c])[0] if np.std(Dp[c]) > 1e-12 else 0.0
             for c in range(Dp.shape[0])])),
        spearman=0.0 if dead else spearmanr(flat_p, flat_t)[0],
        d_ratio=float(np.linalg.norm(Dp) / (np.linalg.norm(Dt) + 1e-12)),
        nmse=nmse, skill=1.0 - nmse,
        alpha=alpha, skill_cal=1.0 - nmse_c,
        auroc=0.5 if dead else float(roc_auc_score(Dt[msk] > 0, Dp[msk])))


def headline(A_pred, B_pred, A_true, B_true, name="", to_dc=None, signed=False,
             level="global"):
    """The four numbers actually being tracked: global PCC, per-cell PCC,
    imbalance PCC, high-imbalance AUROC. Identical treatment for every method.

    to_dc: callable mapping gene space -> dc space, so PCC is measured in the
    same space for model and baselines.

    level: orientation level, as in orient(). Pass "raw" when the caller has
    already fixed the phase assignment upstream and must not spend a second bit
    on it -- the GRN head does this, inheriting the expression branch's decision.
    """
    A_pred, B_pred, _ = orient(A_pred, B_pred, A_true, B_true, level)
    if to_dc is not None:
        pA, pB = to_dc(A_pred), to_dc(B_pred)
        gA, gB = to_dc(A_true), to_dc(B_true)
    else:
        pA, pB, gA, gB = A_pred, B_pred, A_true, B_true

    # per-channel similarity to the reference: Phase A vs maternal, B vs paternal
    pcc_A = pearsonr(pA.ravel(), gA.ravel())[0]
    pcc_B = pearsonr(pB.ravel(), gB.ravel())[0]
    pcc_cA = np.nanmean([pearsonr(pA[c], gA[c])[0] for c in range(pA.shape[0])])
    pcc_cB = np.nanmean([pearsonr(pB[c], gB[c])[0] for c in range(pB.shape[0])])
    _cos = lambda X, Y: float(np.nanmean(
        (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)))
    cos_A, cos_B = _cos(pA, gA), _cos(pB, gB)
    # gene-space MSE per channel (unprojected units)
    mse_A = float(np.mean((A_pred - A_true) ** 2))
    mse_B = float(np.mean((B_pred - B_true) ** 2))
    pcc_g = 0.5 * (pcc_A + pcc_B)
    pcc_c = 0.5 * (pcc_cA + pcc_cB)

    msk = _support_mask(A_true, B_true, signed)
    imb_p = np.where(msk, _imbalance(A_pred, B_pred, signed), np.nan)
    imb_t = np.where(msk, _imbalance(A_true, B_true, signed), np.nan)
    imb = pearsonr(imb_p[msk], imb_t[msk])[0]
    with np.errstate(invalid='ignore'):
        gp, gt = np.nanmean(np.abs(imb_p), 0), np.nanmean(np.abs(imb_t), 0)
        sp_, st_ = np.nanmean(imb_p, 0), np.nanmean(imb_t, 0)
    ok = ~(np.isnan(gp) | np.isnan(gt) | np.isnan(sp_) | np.isnan(st_))
    gp, gt, sp_, st_ = gp[ok], gt[ok], sp_[ok], st_[ok]
    degenerate = np.allclose(gp, gp[0]) if gp.size else True
    imb_gene = np.nan if degenerate else pearsonr(sp_, st_)[0]

    def _auc(score, truth):
        pos = (truth >= np.quantile(truth, HIGH_IMB_QUANTILE)).astype(int)
        if degenerate or not (0 < pos.sum() < len(pos)):
            return np.nan
        return roc_auc_score(pos, score)

    # overall: detect edges/genes with large |imbalance|, either direction
    auroc = _auc(gp, gt)
    # per phase: detect the units this channel actually dominates. The two
    # positive sets differ (A-dominant vs B-dominant), so these are not
    # complements of each other.
    auroc_A = _auc(sp_, st_)
    auroc_B = _auc(-sp_, -st_)
    return dict(name=name, pcc_global=pcc_g, pcc_cell=pcc_c,
                pcc_A=pcc_A, pcc_B=pcc_B, pcc_cell_A=pcc_cA, pcc_cell_B=pcc_cB,
                cos_A=cos_A, cos_B=cos_B, mse_A=mse_A, mse_B=mse_B,
                imb=imb, imb_gene=imb_gene, auroc=auroc,
                auroc_A=auroc_A, auroc_B=auroc_B)

