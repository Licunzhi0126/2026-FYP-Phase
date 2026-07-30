"""两层级 HGNN + Dual-RAE 分相训练循环。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phasehyper.losses.total import compute_total_loss, UncertaintyWeights
from phasehyper.models.phase_model import HGNN_DualRAE_Phase_Model
from phasehyper.models.two_level_hgnn import scipy_to_torch_sparse
from phasehyper.schemas import PhaseTrainingConfig


def _build_aux(built, device):
    """细胞超图传播算子（对比学习用）。"""
    aux = {}
    H_cell = built["H_cell"]
    W_cell = built["W_cell"]
    if H_cell is not None and H_cell.shape[1] > 0:
        H_c = H_cell.tocsr().astype(np.float64)
        n_nodes, n_edges = H_c.shape
        w = np.asarray(W_cell, dtype=np.float64).ravel()
        if w.size != n_edges:
            w = np.ones(n_edges, dtype=np.float64)
        dv = np.asarray(H_c.multiply(w[np.newaxis, :]).sum(axis=1)).ravel()
        de = np.asarray(H_c.sum(axis=0)).ravel()
        dv_inv_sqrt = np.where(dv > 0, 1.0 / np.sqrt(dv), 0.0)
        edge_scale = np.where(de > 0, 1.0 / de, 0.0) * w

        dv_t = torch.tensor(dv_inv_sqrt, dtype=torch.float32, device=device)
        es_t = torch.tensor(edge_scale, dtype=torch.float32, device=device)
        H_sp = scipy_to_torch_sparse(H_c).to(device)
        Ht_sp = H_sp.transpose(0, 1).coalesce()

        def cell_prop_op(X):
            y = X * dv_t[:, None]
            y = torch.sparse.mm(Ht_sp, y)
            y = y * es_t[:, None]
            y = torch.sparse.mm(H_sp, y)
            y = y * dv_t[:, None]
            return y

        aux["cell_prop_op"] = cell_prop_op
    else:
        aux["cell_prop_op"] = None
    return aux


def train_two_level_phase(built, config: PhaseTrainingConfig, device="cpu"):
    device = torch.device(device)
    model = HGNN_DualRAE_Phase_Model(built, config).to(device)
    aux = _build_aux(built, device)

    uw = UncertaintyWeights(2).to(device)  # reg, diff（recon 硬约束不进 UW）
    aux["uncertainty_weights"] = uw

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(uw.parameters()),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.train_epochs), eta_min=config.lr_min
        )

    loss_history = []
    best_loss = float("inf")
    patience_counter = 0

    print("\n[HGNN-DualRAE-Phase: two independent RAEs for phase decomposition]")
    print(f"  cells={model.num_cells}  genes={model.num_genes}  device={device}")
    print(f"  hidden={config.hidden_dim}  latent={config.latent_dim}  "
          f"layers={config.hgnn_num_layers}  epochs={config.train_epochs}")
    print(f"  H_cell={built['H_cell'].shape}  H_gene={built['H_gene'].shape}")
    print(f"  HGNN → CrossAttn → fused → RAE_A/RAE_B → X_A + X_B ≈ expr")
    print(f"  loss = uncertainty_weighted(recon, reg, contrast, diff)")

    for epoch in range(config.train_epochs):
        model.train()
        out = model()
        loss, diag = compute_total_loss(out, aux, epoch, config)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        diag["epoch"] = epoch + 1
        loss_history.append(diag)

        if config.use_early_stopping:
            if diag["total_loss"] < best_loss - config.early_stopping_min_delta:
                best_loss = diag["total_loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\n早停于 epoch {epoch+1}（best={best_loss:.6f}）")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Ep {epoch+1:4d} | total={diag['total_loss']:.4f} "
                f"recon={diag['recon_loss']:.4f} reg={diag['reg_term']:.4f} "
                f"diff={diag['diff_loss']:.4f} | "
                f"var_exp={diag['var_explained']:.3f} "
                f"cos_ab={diag['cos_ab']:.3f} "
                f"warmup={diag['diff_warmup']:.2f}"
            )

    model.eval()
    with torch.no_grad():
        final = model()

    return {
        "phase_a_expr": final["phase_a_expr"].cpu().numpy(),
        "phase_b_expr": final["phase_b_expr"].cpu().numpy(),
        "expr_recon": final["expr_recon"].cpu().numpy(),
        "z": final["z"].cpu().numpy(),
        "cells": built.get("cells"),
        "genes": built.get("genes"),
        "model": model,
        "loss_history": loss_history,
    }
