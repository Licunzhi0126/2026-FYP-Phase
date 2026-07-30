"""Metrics and observation-only preview baselines for Figure 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class PairMetrics:
    mse: float
    nmse: float
    pearson: float
    major_r: float
    minor_r: float
    imbalance_r: float
    imbalance_mae: float
    differential_r: float
    differential_nmse: float
    differential_skill: float
    split_magnitude_ratio: float
    sum_relative_error: float
    swapped: bool
    auroc_a: float
    auroc_b: float
    auprc_a: float
    auprc_b: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_valid = np.asarray(x[valid], dtype=float)
    y_valid = np.asarray(y[valid], dtype=float)
    if np.std(x_valid) <= 1e-12 or np.std(y_valid) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _safe_auroc(truth: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(score)
    labels = truth[valid].astype(int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score[valid]))


def _safe_auprc(truth: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(truth) & np.isfinite(score)
    labels = truth[valid].astype(int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, score[valid]))


def orient_pair(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    truth_a: np.ndarray,
    truth_b: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply one global A/B swap if that better matches the two truth channels."""

    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    truth_a = np.asarray(truth_a, dtype=float)
    truth_b = np.asarray(truth_b, dtype=float)
    valid = (
        np.isfinite(pred_a)
        & np.isfinite(pred_b)
        & np.isfinite(truth_a)
        & np.isfinite(truth_b)
    )
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return pred_a, pred_b, False

    direct = np.mean(
        (pred_a[valid] - truth_a[valid]) ** 2
        + (pred_b[valid] - truth_b[valid]) ** 2
    )
    swapped = np.mean(
        (pred_b[valid] - truth_a[valid]) ** 2
        + (pred_a[valid] - truth_b[valid]) ** 2
    )
    if swapped < direct:
        return pred_b, pred_a, True
    return pred_a, pred_b, False


def pair_metrics(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    truth_a: np.ndarray,
    truth_b: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    total: np.ndarray | None = None,
    binary_truth: bool = False,
) -> tuple[PairMetrics, np.ndarray, np.ndarray]:
    """Score an unordered prediction pair after a single global orientation."""

    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    truth_a = np.asarray(truth_a, dtype=float)
    truth_b = np.asarray(truth_b, dtype=float)
    if not (
        pred_a.shape == pred_b.shape == truth_a.shape == truth_b.shape
    ):
        raise ValueError("Prediction and truth arrays must have identical shapes.")

    oriented_a, oriented_b, swapped = orient_pair(
        pred_a, pred_b, truth_a, truth_b, mask
    )
    valid = (
        np.isfinite(oriented_a)
        & np.isfinite(oriented_b)
        & np.isfinite(truth_a)
        & np.isfinite(truth_b)
    )
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("No finite scoreable entries are available.")

    prediction = np.concatenate((oriented_a[valid], oriented_b[valid]))
    truth = np.concatenate((truth_a[valid], truth_b[valid]))
    mse = float(np.mean((prediction - truth) ** 2))
    truth_energy = float(np.mean(truth**2))
    nmse = mse / max(truth_energy, 1e-12)

    truth_major = np.maximum(truth_a, truth_b)
    truth_minor = np.minimum(truth_a, truth_b)
    pred_major = np.maximum(oriented_a, oriented_b)
    pred_minor = np.minimum(oriented_a, oriented_b)

    truth_sum = truth_a + truth_b
    pred_sum = oriented_a + oriented_b
    truth_imbalance = np.divide(
        truth_a - truth_b,
        truth_sum,
        out=np.zeros_like(truth_sum),
        where=np.abs(truth_sum) > 1e-12,
    )
    pred_imbalance = np.divide(
        oriented_a - oriented_b,
        pred_sum,
        out=np.zeros_like(pred_sum),
        where=np.abs(pred_sum) > 1e-12,
    )

    truth_differential = 0.5 * (truth_a - truth_b)
    pred_differential = 0.5 * (oriented_a - oriented_b)
    differential_mse = float(
        np.mean(
            (pred_differential[valid] - truth_differential[valid]) ** 2
        )
    )
    differential_energy = float(np.mean(truth_differential[valid] ** 2))
    differential_nmse = differential_mse / max(differential_energy, 1e-12)
    split_denominator = float(np.mean(np.abs(truth_differential[valid])))
    split_ratio = float(
        np.mean(np.abs(pred_differential[valid]))
        / max(split_denominator, 1e-12)
    )

    reference_total = truth_sum if total is None else np.asarray(total, dtype=float)
    total_valid = (
        np.isfinite(reference_total)
        & np.isfinite(oriented_a)
        & np.isfinite(oriented_b)
    )
    sum_relative_error = float(
        np.mean(np.abs(pred_sum[total_valid] - reference_total[total_valid]))
        / max(float(np.mean(np.abs(reference_total[total_valid]))), 1e-12)
    )

    if binary_truth:
        auroc_a = _safe_auroc(truth_a[valid], oriented_a[valid])
        auroc_b = _safe_auroc(truth_b[valid], oriented_b[valid])
        auprc_a = _safe_auprc(truth_a[valid], oriented_a[valid])
        auprc_b = _safe_auprc(truth_b[valid], oriented_b[valid])
    else:
        auroc_a = auroc_b = auprc_a = auprc_b = float("nan")

    metrics = PairMetrics(
        mse=mse,
        nmse=nmse,
        pearson=safe_correlation(prediction, truth),
        major_r=safe_correlation(pred_major[valid], truth_major[valid]),
        minor_r=safe_correlation(pred_minor[valid], truth_minor[valid]),
        imbalance_r=safe_correlation(
            pred_imbalance[valid], truth_imbalance[valid]
        ),
        imbalance_mae=float(
            np.mean(np.abs(pred_imbalance[valid] - truth_imbalance[valid]))
        ),
        differential_r=safe_correlation(
            pred_differential[valid], truth_differential[valid]
        ),
        differential_nmse=differential_nmse,
        differential_skill=1.0 - differential_nmse,
        split_magnitude_ratio=split_ratio,
        sum_relative_error=sum_relative_error,
        swapped=swapped,
        auroc_a=auroc_a,
        auroc_b=auroc_b,
        auprc_a=auprc_a,
        auprc_b=auprc_b,
    )
    return metrics, oriented_a, oriented_b


def make_preview_methods(
    total: np.ndarray,
    seed: int = 7,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate deterministic observation-only controls.

    These controls are included to exercise the complete benchmark before a
    trained PhaseHyper checkpoint is connected.  They are not model results.
    """

    observed = np.asarray(total, dtype=float)
    if observed.ndim != 2:
        raise ValueError("Expected a two-dimensional cell-by-feature matrix.")
    observed = np.clip(np.nan_to_num(observed, nan=0.0), 0.0, None)
    rng = np.random.default_rng(seed)
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    methods["EqualSplit"] = (0.5 * observed, 0.5 * observed)

    random_fraction = rng.beta(4.0, 4.0, size=observed.shape)
    random_fraction = np.clip(random_fraction, 0.15, 0.85)
    methods["RandomSplit"] = (
        observed * random_fraction,
        observed * (1.0 - random_fraction),
    )

    order = np.argsort(np.argsort(observed, axis=1), axis=1)
    denominator = max(observed.shape[1] - 1, 1)
    rank_fraction = 0.25 + 0.5 * order / denominator
    methods["RankSplit"] = (
        observed * rank_fraction,
        observed * (1.0 - rank_fraction),
    )

    logged = np.log1p(observed)
    centred = logged - logged.mean(axis=0, keepdims=True)
    if min(centred.shape) >= 2 and np.any(np.abs(centred) > 1e-12):
        left, singular, right = np.linalg.svd(centred, full_matrices=False)
        state_score = singular[0] * np.outer(left[:, 0], right[0])
        scale = max(float(np.std(state_score)), 1e-12)
        state_fraction = 1.0 / (1.0 + np.exp(-state_score / scale))
        state_fraction = np.clip(state_fraction, 0.1, 0.9)
    else:
        state_fraction = np.full_like(observed, 0.5)
    methods["StateSplit"] = (
        observed * state_fraction,
        observed * (1.0 - state_fraction),
    )

    if min(observed.shape) >= 2 and np.any(observed > 0):
        model = NMF(
            n_components=2,
            init="nndsvda",
            random_state=seed,
            max_iter=500,
        )
        weights = model.fit_transform(observed)
        components = model.components_
        contribution_a = np.outer(weights[:, 0], components[0])
        contribution_b = np.outer(weights[:, 1], components[1])
        fraction = np.divide(
            contribution_a,
            contribution_a + contribution_b,
            out=np.full_like(observed, 0.5),
            where=(contribution_a + contribution_b) > 1e-12,
        )
        fraction = np.clip(fraction, 0.05, 0.95)
        methods["NMF2"] = (observed * fraction, observed * (1.0 - fraction))

    return methods
