#!/usr/bin/env python3
"""Fit the production HyperPhase model for the Figure 2 benchmarks.

This is an adapter around ``Phasehyper-main/phasehyper/model.py``.  It builds
an observation-only cell-feature hypergraph and never accepts the held-out
allelic or parental truth matrices in the fitting function.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODEL_ROOT = PROJECT_ROOT / "Phasehyper-main"
for import_root in (SCRIPT_DIR, MODEL_ROOT):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from figure2_io import (  # noqa: E402
    ExpressionContext,
    load_answerdata_contexts,
    load_grn_bundle,
)
from phasehyper.model import (  # noqa: E402
    build_criterion,
    build_model,
    build_optimizer,
)


ADAPTER_VERSION = "figure2-hyperphase-adapter-v1"


@dataclass(frozen=True)
class HyperPhaseFit:
    phase_a: np.ndarray
    phase_b: np.ndarray
    history: pd.DataFrame
    metadata: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-root", type=Path, required=True)
    parser.add_argument("--per-cell-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--grn-max-edges", type=int, default=300)
    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--max-prevalence", type=float, default=0.95)
    parser.add_argument("--min-allelic-reads", type=int, default=2)
    parser.add_argument("--min-gse80810-reads", type=int, default=8)
    parser.add_argument("--min-scoreable-genes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--grn-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-answerdata", action="store_true")
    parser.add_argument("--skip-simulationdata", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _standardize(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = (np.nanstd(values, axis=0) + 1e-8).astype(np.float32)
    standardized = np.nan_to_num(
        (values - mean) / scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return standardized.astype(np.float32), mean, scale


def _matrix_fingerprint(
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(matrix, dtype=np.float32)
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    for label in rows:
        digest.update(b"\0row\0")
        digest.update(str(label).encode("utf-8"))
    for label in columns:
        digest.update(b"\0column\0")
        digest.update(str(label).encode("utf-8"))
    return digest.hexdigest()


def _build_observation_hypergraphs(
    matrix: np.ndarray,
    *,
    top_k: int = 10,
    correlation_k: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Build directed and undirected hypergraphs from observation only."""

    values = np.asarray(matrix, dtype=np.float32)
    n_cells, n_features = values.shape
    n_nodes = n_cells + n_features
    standardized, _, _ = _standardize(values)
    top_k = max(2, min(int(top_k), n_features))

    tail_rows: list[int] = []
    tail_columns: list[int] = []
    head_rows: list[int] = []
    head_columns: list[int] = []
    edge_weights: list[float] = []
    edge_types: list[int] = []
    edge_id = 0

    def add_directed(
        tail: list[int],
        head: list[int],
        edge_type: int,
        weight: float,
    ) -> None:
        nonlocal edge_id
        for node in tail:
            tail_rows.append(int(node))
            tail_columns.append(edge_id)
        for node in head:
            head_rows.append(int(node))
            head_columns.append(edge_id)
        edge_weights.append(float(weight))
        edge_types.append(int(edge_type))
        edge_id += 1

    for cell in range(n_cells):
        selected = np.argsort(-standardized[cell])[:top_k]
        feature_nodes = [n_cells + int(index) for index in selected]
        weight = float(np.sqrt(len(feature_nodes) + 1))
        add_directed([cell], feature_nodes, 0, weight)
        add_directed(feature_nodes, [cell], 1, weight)

    if n_features > 1:
        correlation = np.corrcoef(standardized, rowvar=False)
        correlation = np.nan_to_num(
            correlation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.fill_diagonal(correlation, 0.0)
        for source in range(n_features):
            neighbours = np.argsort(-np.abs(correlation[source]))[
                : min(correlation_k, n_features - 1)
            ]
            for target in neighbours:
                strength = float(abs(correlation[source, target]))
                if strength > 1e-6:
                    add_directed(
                        [n_cells + source],
                        [n_cells + int(target)],
                        2,
                        0.1 + strength,
                    )
    else:
        correlation = np.zeros((1, 1), dtype=np.float32)

    tail_incidence = sp.coo_matrix(
        (
            np.ones(len(tail_rows), dtype=np.float32),
            (tail_rows, tail_columns),
        ),
        shape=(n_nodes, edge_id),
    ).tocsr()
    head_incidence = sp.coo_matrix(
        (
            np.ones(len(head_rows), dtype=np.float32),
            (head_rows, head_columns),
        ),
        shape=(n_nodes, edge_id),
    ).tocsr()

    undirected_rows: list[int] = []
    undirected_columns: list[int] = []
    undirected_weights: list[float] = []
    seen: set[tuple[int, ...]] = set()
    undirected_id = 0

    def add_undirected(nodes: list[int], weight: float) -> None:
        nonlocal undirected_id
        key = tuple(sorted(set(int(node) for node in nodes)))
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        for node in key:
            undirected_rows.append(node)
            undirected_columns.append(undirected_id)
        undirected_weights.append(float(weight))
        undirected_id += 1

    for cell in range(n_cells):
        selected = np.argsort(-standardized[cell])[:top_k]
        add_undirected(
            [cell] + [n_cells + int(index) for index in selected],
            float(np.sqrt(top_k + 1)),
        )

    if n_features > 1:
        module_k = min(6, n_features)
        for feature in range(n_features):
            neighbours = np.argsort(-np.abs(correlation[feature]))[
                : max(1, module_k - 1)
            ]
            add_undirected(
                [n_cells + feature]
                + [n_cells + int(index) for index in neighbours],
                1.0,
            )

    undirected_incidence = sp.coo_matrix(
        (
            np.ones(len(undirected_rows), dtype=np.float32),
            (undirected_rows, undirected_columns),
        ),
        shape=(n_nodes, undirected_id),
    ).tocsr()

    directed = {
        "H_tail": tail_incidence,
        "H_head": head_incidence,
        "W": np.asarray(edge_weights, dtype=np.float32),
        "etype": np.asarray(edge_types, dtype=np.int64),
        "n_types": 3,
    }
    undirected = {
        "H": undirected_incidence,
        "W": np.asarray(undirected_weights, dtype=np.float32),
    }
    summary = {
        "n_cells": n_cells,
        "n_features": n_features,
        "n_nodes": n_nodes,
        "n_directed_hyperedges": edge_id,
        "n_undirected_hyperedges": undirected_id,
    }
    return directed, undirected, summary


def fit_hyperphase(
    combined: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
    task: str,
) -> HyperPhaseFit:
    """Fit HyperPhase using only the observed combined matrix."""

    values = np.asarray(combined, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(
            "combined matrix must be two-dimensional with both axes >= 2; "
            f"received {values.shape}"
        )

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    standardized, mean, scale = _standardize(values)
    component_count = max(
        2,
        min(24, values.shape[0] - 1, values.shape[1]),
    )
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=seed,
    )
    cell_target = pca.fit_transform(standardized).astype(np.float32)
    projection = pca.components_.astype(np.float32)
    feature_features = projection.transpose().copy()
    directed, undirected, graph_summary = _build_observation_hypergraphs(values)

    model = build_model(
        directed_data=directed,
        undirected_data=undirected,
        n_cells=values.shape[0],
        n_genes=values.shape[1],
        dc=component_count,
        pca_init=projection,
        hidden=max(64, 4 * component_count),
        latent=component_count,
        use_asym=True,
        device=device,
    )
    criterion = build_criterion(
        w_comp=0.0,
        w_ortho=4.0,
        w_nce=0.2,
        w_gate=0.05,
    )
    optimizer = build_optimizer(model)

    model_graph = torch.from_numpy(standardized).to(device)
    gene_features = torch.from_numpy(feature_features).to(device)
    cell_features = torch.from_numpy(cell_target).to(device)
    gene_projection = torch.from_numpy(projection).to(device)
    compartment = torch.zeros(
        values.shape[1],
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(model_graph, gene_features, cell_features)
        loss, terms = criterion(
            model=model,
            model_output=output,
            gene_projection=gene_projection,
            compartment_indicator=compartment,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite HyperPhase loss at epoch {epoch}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        row = {
            "epoch": float(epoch),
            "loss": float(terms["total"].detach().cpu()),
            "cyc_comp": float(terms["cyc_comp"].detach().cpu()),
            "barlow": float(terms["barlow"].detach().cpu()),
            "orthogonality": float(terms["orthogonality"].detach().cpu()),
            "info_nce": float(terms["info_nce"].detach().cpu()),
            "gate_regularization": float(
                terms["gate_regularization"].detach().cpu()
            ),
            "phase_cosine": float(terms["phase_cosine"].detach().cpu()),
            "asym_scale": float(model.asym_scale.detach().cpu()),
        }
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("HyperPhase training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, phase_a_reduced, phase_b_reduced = model(
            model_graph,
            gene_features,
            cell_features,
        )
    a_reduced = phase_a_reduced.detach().cpu().numpy()
    b_reduced = phase_b_reduced.detach().cpu().numpy()

    # Canonical naming is prediction-only.  Held-out truth is not available here.
    if np.linalg.norm(a_reduced, axis=1).mean() > np.linalg.norm(
        b_reduced, axis=1
    ).mean():
        a_reduced, b_reduced = b_reduced, a_reduced

    a_raw = (a_reduced @ projection) * scale + mean / 2.0
    b_raw = (b_reduced @ projection) * scale + mean / 2.0
    phase_a = values / 2.0 + (a_raw - b_raw) / 2.0
    if np.nanmin(values) >= 0:
        phase_a = np.minimum(np.maximum(phase_a, 0.0), values)
    phase_b = values - phase_a

    metadata: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "task": task,
        "model_class": "phasehyper.model.HyperPhaseModel",
        "criterion_class": "phasehyper.model.SetCriterion",
        "optimizer": "phasehyper.model.build_optimizer / AdamW",
        "truth_used_during_fit": False,
        "adapter": "observation-only cell-feature hypergraph",
        "readout": (
            "prediction-only canonicalisation + total-preserving projection"
        ),
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "seed": int(seed),
        "device": str(device),
        "torch_version": torch.__version__,
        "dc": int(component_count),
        **graph_summary,
    }
    return HyperPhaseFit(
        phase_a=np.asarray(phase_a, dtype=np.float32),
        phase_b=np.asarray(phase_b, dtype=np.float32),
        history=pd.DataFrame(history),
        metadata=metadata,
    )


def _cache_matches(
    metadata_path: Path,
    *,
    input_hash: str,
    epochs: int,
    seed: int,
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("adapter_version") == ADAPTER_VERSION
        and metadata.get("input_hash") == input_hash
        and int(metadata.get("epochs", -1)) == int(epochs)
        and int(metadata.get("seed", -1)) == int(seed)
    )


def _write_expression_fit(
    context: ExpressionContext,
    fit: HyperPhaseFit,
    output_dir: Path,
    input_hash: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = {
        "index": context.total.index,
        "columns": context.total.columns,
    }
    pd.DataFrame(fit.phase_a, **axes).to_csv(output_dir / "phase_A.csv")
    pd.DataFrame(fit.phase_b, **axes).to_csv(output_dir / "phase_B.csv")
    fit.history.to_csv(output_dir / "training_history.csv", index=False)
    metadata = {
        **fit.metadata,
        "input_hash": input_hash,
        "dataset": context.dataset,
        "cells": context.total.index.astype(str).tolist(),
        "genes": context.total.columns.astype(str).tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _fit_expression_datasets(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    contexts = load_answerdata_contexts(
        args.answer_root,
        n_genes=args.n_genes,
        min_allelic_reads=args.min_allelic_reads,
        min_gse80810_reads=args.min_gse80810_reads,
        min_scoreable_genes=args.min_scoreable_genes,
    )
    datasets = sorted({context.dataset for context in contexts})
    for offset, dataset in enumerate(datasets):
        candidates = [
            context for context in contexts if context.dataset == dataset
        ]
        context = max(candidates, key=lambda item: item.total.shape[0])
        values = context.total.to_numpy(dtype=np.float32)
        input_hash = _matrix_fingerprint(
            values,
            context.total.index.astype(str).tolist(),
            context.total.columns.astype(str).tolist(),
        )
        seed = args.seed + offset
        output_dir = (
            args.output_root
            / "answerdata"
            / "hyperphase_outputs"
            / dataset
        )
        target = output_dir / "phase_A.csv"
        companion = output_dir / "phase_B.csv"
        metadata_path = output_dir / "metadata.json"

        if target.exists() and companion.exists() and not args.force:
            if _cache_matches(
                metadata_path,
                input_hash=input_hash,
                epochs=args.epochs,
                seed=seed,
            ):
                print(f"[HyperPhase] Reusing compatible output: {output_dir}")
                continue
            raise RuntimeError(
                f"Stale HyperPhase output exists at {output_dir}. "
                "Use --force-model on the main runner to replace it."
            )

        print(
            f"[HyperPhase] Fitting {dataset}: "
            f"cells={values.shape[0]} features={values.shape[1]}"
        )
        fit = fit_hyperphase(
            values,
            epochs=args.epochs,
            seed=seed,
            device=device,
            task=f"expression:{dataset}",
        )
        _write_expression_fit(context, fit, output_dir, input_hash)
        print(
            f"[HyperPhase] {dataset}: best_epoch={fit.metadata['best_epoch']} "
            f"loss={fit.metadata['best_loss']:.6f}"
        )


def _fit_grn(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    bundle = load_grn_bundle(
        args.per_cell_root,
        max_edges=args.grn_max_edges,
        min_prevalence=args.min_prevalence,
        max_prevalence=args.max_prevalence,
    )
    input_hash = _matrix_fingerprint(
        bundle.combined,
        bundle.cells,
        bundle.edge_names,
    )
    seed = args.seed + 11
    output_dir = (
        args.output_root / "simulationdata" / "hyperphase_outputs"
    )
    target = output_dir / "hyperphase_grn_predictions.npz"
    metadata_path = output_dir / "metadata.json"
    if target.exists() and not args.force:
        if _cache_matches(
            metadata_path,
            input_hash=input_hash,
            epochs=args.grn_epochs,
            seed=seed,
        ):
            print(f"[HyperPhase] Reusing compatible output: {output_dir}")
            return
        raise RuntimeError(
            f"Stale HyperPhase output exists at {output_dir}. "
            "Use --force-model on the main runner to replace it."
        )

    print(
        "[HyperPhase] Fitting GRN edge-space adapter: "
        f"cells={len(bundle.cells)} edges={len(bundle.edge_names)}"
    )
    fit = fit_hyperphase(
        bundle.combined,
        epochs=args.grn_epochs,
        seed=seed,
        device=device,
        task="grn:thresholded-edge-space",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        pred_A=fit.phase_a,
        pred_B=fit.phase_b,
        combined=bundle.combined,
        edge_index=bundle.edge_index,
        genes=np.asarray(bundle.genes, dtype=str),
        cells=np.asarray(bundle.cells, dtype=str),
        edge_names=np.asarray(bundle.edge_names, dtype=str),
    )
    fit.history.to_csv(output_dir / "training_history.csv", index=False)
    metadata = {
        **fit.metadata,
        "input_hash": input_hash,
        "cells": bundle.cells,
        "genes": bundle.genes,
        "edge_names": bundle.edge_names,
        "combined_semantics": "independently thresholded observation",
        "combined_union_mismatch_count": (
            bundle.combined_union_mismatch_count
        ),
        "combined_union_match_rate": bundle.combined_union_match_rate,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        "[HyperPhase] GRN: "
        f"best_epoch={fit.metadata['best_epoch']} "
        f"loss={fit.metadata['best_loss']:.6f}"
    )


def main() -> None:
    args = _parse_args()
    if args.skip_answerdata and args.skip_simulationdata:
        raise ValueError("Both workflows were skipped; there is nothing to fit.")
    device = _resolve_device(args.device)
    print(f"[HyperPhase] Device: {device}")
    if not args.skip_answerdata:
        _fit_expression_datasets(args, device)
    if not args.skip_simulationdata:
        _fit_grn(args, device)


if __name__ == "__main__":
    main()
