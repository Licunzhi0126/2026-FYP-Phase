from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr

from phasehyper.evaluation import saber
from phasehyper.evaluation.clustering import _ari, _emb


ArrayTransform = Callable[[np.ndarray], np.ndarray]


def _as_matching_arrays(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    shapes = {name: value.shape for name, value in converted.items()}
    if any(value.ndim != 2 for value in converted.values()):
        raise ValueError(f"all evaluation matrices must be 2D, got {shapes}")
    if len(set(shapes.values())) != 1:
        raise ValueError(f"evaluation matrix shapes must match, got {shapes}")
    return converted


def _rename_trivial(rows: list[dict]) -> list[dict]:
    for row in rows:
        if row["name"] == "MeanFractionShrinkage":
            row["name"] = "[trivial] combined/2"
    return rows


def _expression_summary(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    maternal: np.ndarray,
    paternal: np.ndarray,
    *,
    to_dc: ArrayTransform | None,
    orientation: dict,
) -> dict[str, float | str | bool]:
    pred_a = to_dc(phase_a) if to_dc is not None else phase_a
    pred_b = to_dc(phase_b) if to_dc is not None else phase_b
    true_a = to_dc(maternal) if to_dc is not None else maternal
    true_b = to_dc(paternal) if to_dc is not None else paternal
    imb_pred = (phase_a - phase_b) / (np.abs(phase_a + phase_b) + 1e-8)
    imb_true = (maternal - paternal) / (maternal + paternal + 1e-8)
    return {
        "swapped": bool(orientation["n_swapped"]),
        "assign": "P2=mat" if orientation["n_swapped"] else "P1=mat",
        "pcc_mat": float(pearsonr(pred_a.ravel(), true_a.ravel())[0]),
        "pcc_pat": float(pearsonr(pred_b.ravel(), true_b.ravel())[0]),
        "cell_pcc_mat": float(np.nanmean([
            pearsonr(pred_a[i], true_a[i])[0] for i in range(pred_a.shape[0])
        ])),
        "cell_pcc_pat": float(np.nanmean([
            pearsonr(pred_b[i], true_b[i])[0] for i in range(pred_b.shape[0])
        ])),
        "imb_pcc": float(pearsonr(imb_pred.ravel(), imb_true.ravel())[0]),
        "imb_gene_pcc": float(pearsonr(imb_pred.mean(0), imb_true.mean(0))[0]),
        "imb_cell_pcc": float(np.nanmean([
            pearsonr(imb_pred[i], imb_true[i])[0] for i in range(imb_pred.shape[0])
        ])),
        "imb_mae": float(np.mean(np.abs(imb_pred - imb_true))),
        "imb_mag_pred": float(np.mean(np.abs(imb_pred))),
        "imb_mag_true": float(np.mean(np.abs(imb_true))),
    }


def evaluate_simulation_expression(
    *,
    phase_a_pred: np.ndarray,
    phase_b_pred: np.ndarray,
    maternal_true: np.ndarray,
    paternal_true: np.ndarray,
    combined: np.ndarray,
    seed: int = 0,
    projection: ArrayTransform | None = None,
    to_dc: ArrayTransform | None = None,
    method_name: str = "phasehyper",
    pre_sync_phase_a: np.ndarray | None = None,
    pre_sync_phase_b: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the complete post-training expression decomposition evaluation."""
    values = _as_matching_arrays(
        phase_a_pred=phase_a_pred,
        phase_b_pred=phase_b_pred,
        maternal_true=maternal_true,
        paternal_true=paternal_true,
        combined=combined,
    )
    raw_a = values["phase_a_pred"]
    raw_b = values["phase_b_pred"]
    maternal = values["maternal_true"]
    paternal = values["paternal_true"]
    combined_values = values["combined"]
    oriented_a, oriented_b, orientation = saber.orient(
        raw_a, raw_b, maternal, paternal, "global"
    )

    saber_rows = [
        saber.phase_metrics(raw_a, raw_b, maternal, paternal, method_name)
    ]
    baselines = saber.run_baselines(
        combined_values,
        maternal,
        paternal,
        seed=seed,
        proj=projection,
    )
    saber_rows.extend(_rename_trivial(baselines))

    project = projection if projection is not None else (lambda value: value)
    perfect = (project(maternal), project(paternal))
    saber_rows.append(
        saber.phase_metrics(
            perfect[0], perfect[1], maternal, paternal, "[floor] perfect@rank-dc"
        )
    )

    headline_rows = [
        saber.headline(raw_a, raw_b, maternal, paternal, method_name, to_dc)
    ]
    half = combined_values * 0.5
    for name, pair in (
        ("RandomSplit", saber.baseline_random_split(combined_values, seed)),
        ("NMF2Factor", saber.baseline_nmf2(combined_values, seed)),
        ("[trivial] combined/2", (half, half)),
    ):
        headline_rows.append(
            saber.headline(
                project(pair[0]),
                project(pair[1]),
                maternal,
                paternal,
                name,
                to_dc,
            )
        )
    headline_rows.append(
        saber.headline(
            perfect[0],
            perfect[1],
            maternal,
            paternal,
            "[floor] perfect@rank-dc",
            to_dc,
        )
    )

    orientation_rows = saber.orientation_audit(
        raw_a, raw_b, maternal, paternal, method_name
    )
    pre_sync_rows = None
    if (pre_sync_phase_a is None) != (pre_sync_phase_b is None):
        raise ValueError("both pre-sync phase matrices must be supplied together")
    if pre_sync_phase_a is not None:
        pre = _as_matching_arrays(
            phase_a=np.asarray(pre_sync_phase_a),
            phase_b=np.asarray(pre_sync_phase_b),
            maternal=maternal,
            paternal=paternal,
        )
        pre_sync_rows = saber.orientation_audit(
            pre["phase_a"], pre["phase_b"], maternal, paternal, "no-sync"
        )

    return {
        "phase_a_oriented": oriented_a,
        "phase_b_oriented": oriented_b,
        "orientation": orientation,
        "orientation_audit": orientation_rows,
        "pre_sync_orientation_audit": pre_sync_rows,
        "saber_rows": saber_rows,
        "headline_rows": headline_rows,
        "settings": {"min_support_q": saber.MIN_SUPPORT_Q},
        "summary": _expression_summary(
            oriented_a,
            oriented_b,
            maternal,
            paternal,
            to_dc=to_dc,
            orientation=orientation,
        ),
    }


def evaluate_simulation_clustering(
    *,
    raw_rna: np.ndarray,
    cell_embedding: np.ndarray,
    phase_a_embedding: np.ndarray,
    phase_b_embedding: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> dict[str, float]:
    """Reproduce the legacy five-seed ARI auxiliary report."""
    del seed  # Legacy helper always uses the fixed seed tuple 0..4.
    scores = {
        "raw": _ari(_emb(raw_rna), n_clusters, labels),
        "cell_h": _ari(_emb(cell_embedding), n_clusters, labels),
        "phase_a": _ari(_emb(phase_a_embedding), n_clusters, labels),
        "phase_b": _ari(_emb(phase_b_embedding), n_clusters, labels),
    }
    return {name: float(value) for name, value in scores.items()}


def evaluate_embedding_quality(
    *,
    cell_embedding: np.ndarray,
    phase_a_embedding: np.ndarray,
    phase_b_embedding: np.ndarray,
) -> dict[str, float]:
    arrays = _as_matching_arrays(
        cell_embedding=cell_embedding,
        phase_a_embedding=phase_a_embedding,
        phase_b_embedding=phase_b_embedding,
    )
    cell = arrays["cell_embedding"]
    a = arrays["phase_a_embedding"]
    b = arrays["phase_b_embedding"]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return {
        "rec": float(np.linalg.norm(a + b - cell) / (np.linalg.norm(cell) + 1e-9)),
        "cos_ab": float(np.mean(np.abs(np.sum(a * b, axis=1) / denom))),
    }


def evaluate_scale_diagnostics(
    *,
    cell_embedding: np.ndarray,
    canonical_phase_a: np.ndarray,
    canonical_phase_b: np.ndarray,
    maternal_embedding: np.ndarray,
    paternal_embedding: np.ndarray,
    standardized_expression: np.ndarray,
    projection_components: np.ndarray,
    assign: str,
    scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
) -> dict[str, Any]:
    """Compute the legacy shared-signal and split-magnitude diagnostics."""
    cell = np.asarray(cell_embedding)
    p1 = np.asarray(canonical_phase_a)
    p2 = np.asarray(canonical_phase_b)
    true_a = np.asarray(maternal_embedding)
    true_b = np.asarray(paternal_embedding)
    half = cell / 2.0
    true_half = (true_a + true_b) / 2.0
    null_half = (np.asarray(standardized_expression) @ np.asarray(projection_components).T) / 2.0
    maternal_phase = p1 if assign == "P1=mat" else p2
    diff_pred = maternal_phase - half
    diff_true = true_a - true_half
    scale_rows = [
        (float(scale), float(pearsonr((half + scale * diff_pred).ravel(), true_a.ravel())[0]))
        for scale in scales
    ]
    best_scale, best_pcc = max(scale_rows, key=lambda row: row[1])
    return {
        "pcc_trivial": float(pearsonr(half.ravel(), true_a.ravel())[0]),
        "pcc_shared": float(pearsonr(half.ravel(), true_half.ravel())[0]),
        "pcc_null": float(pearsonr(null_half.ravel(), true_a.ravel())[0]),
        "diff_ratio": float(
            np.linalg.norm(diff_pred) / (np.linalg.norm(diff_true) + 1e-12)
        ),
        "scale_rows": scale_rows,
        "best_scale": best_scale,
        "best_pcc": best_pcc,
    }


def evaluate_simulation_grn(
    *,
    grn_a_pred: np.ndarray,
    grn_b_pred: np.ndarray,
    grn_a_true: np.ndarray,
    grn_b_true: np.ndarray,
    combined_grn: np.ndarray,
    inherited_swap: bool,
    seed: int = 0,
    method_name: str = "phasehyper",
) -> dict[str, Any]:
    """Evaluate a GRN split while inheriting the expression phase assignment."""
    values = _as_matching_arrays(
        grn_a_pred=grn_a_pred,
        grn_b_pred=grn_b_pred,
        grn_a_true=grn_a_true,
        grn_b_true=grn_b_true,
        combined_grn=combined_grn,
    )
    pred_a, pred_b = values["grn_a_pred"], values["grn_b_pred"]
    true_a, true_b = values["grn_a_true"], values["grn_b_true"]
    combined = values["combined_grn"]
    if inherited_swap:
        pred_a, pred_b = pred_b, pred_a

    saber_rows = [
        saber.phase_metrics(
            pred_a, pred_b, true_a, true_b, method_name, level="raw", signed=True
        )
    ]
    saber_rows.extend(
        row
        for row in saber.run_baselines(
            combined, true_a, true_b, seed=seed, signed=True
        )
        if row["name"] != "MeanFractionShrinkage"
    )
    headline_rows = [
        saber.headline(
            pred_a,
            pred_b,
            true_a,
            true_b,
            method_name,
            signed=True,
            level="raw",
        )
    ]
    differential_rows = [
        saber.differential_metrics(pred_a, pred_b, true_a, true_b, method_name)
    ]
    baseline_arrays: dict[str, np.ndarray] = {}
    for name, pair in (
        ("RandomSplit", saber.baseline_random_split(combined, seed)),
        ("NMF2Factor", saber.baseline_nmf2(combined, seed)),
    ):
        headline_rows.append(
            saber.headline(pair[0], pair[1], true_a, true_b, name, signed=True)
        )
        oriented_a, oriented_b, _ = saber.orient(
            pair[0], pair[1], true_a, true_b, "global"
        )
        differential_rows.append(
            saber.differential_metrics(
                oriented_a, oriented_b, true_a, true_b, name
            )
        )
        baseline_arrays[f"{name}_A"] = oriented_a
        baseline_arrays[f"{name}_B"] = oriented_b
    differential_rows.append(
        saber.differential_metrics(
            combined / 2,
            combined / 2,
            true_a,
            true_b,
            "[floor] no phasing",
        )
    )
    return {
        "phase_a_oriented": pred_a,
        "phase_b_oriented": pred_b,
        "saber_rows": saber_rows,
        "headline_rows": headline_rows,
        "differential_rows": differential_rows,
        "baseline_arrays": baseline_arrays,
    }


def evaluate_simulation(
    *,
    expression_inputs: dict,
    clustering_inputs: dict | None = None,
    grn_inputs: dict | None = None,
) -> dict[str, Any]:
    """Convenience entry point when every prediction is already available."""
    expression = evaluate_simulation_expression(**expression_inputs)
    result = {
        "expression": expression,
        "clustering": (
            evaluate_simulation_clustering(**clustering_inputs)
            if clustering_inputs is not None
            else None
        ),
        "grn": None,
    }
    if grn_inputs is not None:
        inputs = dict(grn_inputs)
        inputs.setdefault(
            "inherited_swap", bool(expression["orientation"]["n_swapped"])
        )
        result["grn"] = evaluate_simulation_grn(**inputs)
    return result
