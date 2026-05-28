from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import leidenalg
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    umap = None

from .config import *
from .models import *
from .losses import *

def train_hgnn_vae_phase(hg_data, config: PhaseTrainingConfig, device="cpu"):
    H = hg_data["H"].to(device)
    base_edge_weights = hg_data.get("W", torch.ones(H.shape[1], dtype=H.dtype)).to(device)
    hyperedge_type_ids = hg_data["hyperedge_type_ids"].to(device)
    sample_mask = hg_data["sample_mask"].to(device)
    gene_mask = hg_data["gene_mask"].to(device)

    cell_edge_mask = hg_data["cell_edge_mask"].to(device)
    H_prior = H[:, ~cell_edge_mask]

    sample_indices = torch.where(sample_mask)[0]
    gene_indices = torch.where(gene_mask)[0]

    num_cells = len(sample_indices)
    num_genes = len(gene_indices)

    gene_idx_to_local = {int(g): i for i, g in enumerate(gene_indices)}
    sample_idx_to_local = {int(s): i for i, s in enumerate(sample_indices)}

    H_cell_dense = H[:, cell_edge_mask]
    cell_gene_pairs = []
    for edge_idx in range(H_cell_dense.shape[1]):
        edge_nodes = torch.where(H_cell_dense[:, edge_idx] > 0)[0]
        cells_in_edge = [n for n in edge_nodes if int(n) in sample_idx_to_local]
        genes_in_edge = [n for n in edge_nodes if int(n) in gene_idx_to_local]
        for c in cells_in_edge:
            for g in genes_in_edge:
                cell_gene_pairs.append((sample_idx_to_local[int(c)], gene_idx_to_local[int(g)]))

    hg_data["H_prior"] = H_prior
    hg_data["cell_gene_pairs"] = cell_gene_pairs

    model = HGNN_VAE_Phase_Model(
        num_cells=num_cells,
        num_genes=num_genes,
        config=config,
        edge_type_to_id=hg_data["edge_type_to_id"],
        initial_node_features=hg_data["X"].to(device),
        cell_gene_pairs=cell_gene_pairs,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs, eta_min=config.lr_min)

    loss_history = []
    
    # Early stopping initialization
    best_loss = float('inf')
    patience_counter = 0
    early_stop_triggered = False

    print(f"\n[End-to-end HGNN-VAE-Phase Training]")
    print(f"  Epochs: {config.train_epochs}")
    print(f"  LR: {config.lr}, Weight decay: {config.weight_decay}")
    print(f"  LR scheduler: {'CosineAnnealingLR' if config.use_lr_scheduler else 'None'}, LR min: {config.lr_min}")
    print(f"  Early stopping: {'Enabled' if config.use_early_stopping else 'Disabled'}, patience: {config.early_stopping_patience}")
    print(f"  Cells: {num_cells}, Genes: {num_genes}")
    print(f"  Device: {device}")

    for epoch in range(config.train_epochs):
        model.train()

        out = model(
            H=H,
            hyperedge_type_ids=hyperedge_type_ids,
            gene_mask=gene_mask,
            sample_mask=sample_mask,
            base_edge_weights=base_edge_weights,
        )

        loss, diagnostics = compute_total_loss(out, hg_data, epoch, config)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        diagnostics["epoch"] = epoch + 1
        loss_history.append(diagnostics)
        
        # Early stopping check
        if config.use_early_stopping:
            current_loss = diagnostics["total_loss"]
            if current_loss < best_loss - config.early_stopping_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best loss: {best_loss:.6f}, Patience: {patience_counter}")
                    early_stop_triggered = True
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            gate_g = out["gate_g"].detach().cpu()
            mu_genes = out["mu_genes"].detach().cpu()
            
            gate_min = float(gate_g.min().item())
            gate_max = float(gate_g.max().item())
            gate_less_02 = float((gate_g < 0.2).float().mean().item())
            gate_more_08 = float((gate_g > 0.8).float().mean().item())
            
            phase_a_mask = gate_g < 0.5
            phase_b_mask = gate_g >= 0.5
            
            if phase_a_mask.any() and phase_b_mask.any():
                phase_a_mean = mu_genes[phase_a_mask].mean(dim=0)
                phase_b_mean = mu_genes[phase_b_mask].mean(dim=0)
                phase_cosine_sim = float(F.cosine_similarity(phase_a_mean, phase_b_mean, dim=0).item())
            else:
                phase_cosine_sim = 0.0
            
            h_norm = float(out["h"].norm(dim=1).mean().detach().cpu().item())
            mu_cells_norm = float(out["mu_cells"].norm(dim=1).mean().detach().cpu().item())
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | "
                  f"lr={current_lr:.2e} | "
                  f"total={diagnostics['total_loss']:.4f} | "
                  f"vae_recon={diagnostics['vae_recon_loss']:.4f} | "
                  f"raw_kl={diagnostics['raw_kl']:.4f} | "
                  f"cell_gene={diagnostics['cell_gene_recon_loss']:.4f} | "
                  f"phase_sep={diagnostics['phase_sep_loss']:.4f}")
            print(f"         | "
                  f"gate_mean={diagnostics['gate_mean']:.4f} | "
                  f"gate_std={diagnostics['gate_std']:.4f} | "
                  f"gate_min={gate_min:.4f} | "
                  f"gate_max={gate_max:.4f}")
            print(f"         | "
                  f"gate<0.2={gate_less_02:.4f} | "
                  f"gate>0.8={gate_more_08:.4f} | "
                  f"phase_cos_sim={phase_cosine_sim:.4f}")
            print(f"         | "
                  f"mu_norm={diagnostics['mu_norm_mean']:.4f} | "
                  f"h_norm={h_norm:.4f} | "
                  f"mu_cells_norm={mu_cells_norm:.4f} | "
                  f"logvar_max={diagnostics['logvar_max']:.4f}")

    model.eval()
    with torch.no_grad():
        final_out = model(
            H=H,
            hyperedge_type_ids=hyperedge_type_ids,
            gene_mask=gene_mask,
            sample_mask=sample_mask,
            base_edge_weights=base_edge_weights,
        )

    cell_embed = final_out["cell_embed"].detach().cpu().numpy()
    gate_g = final_out["gate_g"].detach().cpu().numpy()
    mu_cells = final_out["mu_cells"].detach().cpu().numpy()
    mu_genes = final_out["mu_genes"].detach().cpu().numpy()
    h_cells = final_out["h"][sample_indices].detach().cpu().numpy()
    x_cells = final_out["x"][sample_indices].detach().cpu().numpy()

    sample_names = [
        hg_data["sample_node_names"].get(int(idx), f"node_{idx}")
        for idx in sample_indices.cpu().numpy()
    ]
    gene_names = [
        hg_data["gene_node_names"].get(int(idx), f"gene_{idx}")
        for idx in gene_indices.cpu().numpy()
    ]

    edge_type_weights = {}
    for et, idx in hg_data["edge_type_to_id"].items():
        edge_type_weights[et] = float(F.softplus(model.edge_type_weights[idx]).detach().cpu().item())

    return {
        "cell_embed": cell_embed,
        "mu_cells": mu_cells,
        "h_cells": h_cells,
        "x_cells": x_cells,
        "mu_genes": mu_genes,
        "gate_g": gate_g,
        "sample_names": sample_names,
        "gene_names": gene_names,
        "model": model,
        "loss_history": loss_history,
        "edge_type_weights": edge_type_weights,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
