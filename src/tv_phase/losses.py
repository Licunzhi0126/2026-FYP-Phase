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

def compute_cell_gene_recon_loss(
    z_cells: torch.Tensor,
    z_genes: torch.Tensor,
    cell_gene_pairs: List[Tuple[int, int]],
    temperature: float = 0.2,
    num_neg: int = 1,
    device: torch.device = None,
):
    if device is None:
        device = z_cells.device
    n_genes = z_genes.shape[0]

    z_cells_norm = F.normalize(z_cells, p=2, dim=1)
    z_genes_norm = F.normalize(z_genes, p=2, dim=1)

    if not cell_gene_pairs:
        return torch.tensor(0.0, device=device)

    pos_scores = []
    neg_scores = []

    for c_idx, g_idx in cell_gene_pairs:
        pos_score = torch.dot(z_cells_norm[c_idx], z_genes_norm[g_idx]) / temperature
        pos_scores.append(pos_score)

        for _ in range(num_neg):
            rand_g = torch.randint(0, n_genes, (1,), device=device).item()
            neg_score = torch.dot(z_cells_norm[c_idx], z_genes_norm[rand_g]) / temperature
            neg_scores.append(neg_score)

    pos_scores = torch.stack(pos_scores)
    neg_scores = torch.stack(neg_scores)

    all_scores = torch.cat([pos_scores, neg_scores])
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    all_labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)

    return loss


def compute_centroid_phase_separation_loss(mu_genes: torch.Tensor, gate_g: torch.Tensor):
    mu_genes_norm = F.normalize(mu_genes, p=2, dim=1)

    gate_g = gate_g.clamp(min=1e-8, max=1 - 1e-8)

    weight_a = 1 - gate_g
    weight_b = gate_g

    weight_a = weight_a / (weight_a.sum() + 1e-8)
    weight_b = weight_b / (weight_b.sum() + 1e-8)

    centroid_a = (mu_genes_norm * weight_a.unsqueeze(1)).sum(dim=0)
    centroid_b = (mu_genes_norm * weight_b.unsqueeze(1)).sum(dim=0)

    centroid_a_norm = F.normalize(centroid_a.unsqueeze(0), p=2, dim=1).squeeze(0)
    centroid_b_norm = F.normalize(centroid_b.unsqueeze(0), p=2, dim=1).squeeze(0)

    cosine_sim = torch.dot(centroid_a_norm, centroid_b_norm)

    return torch.abs(cosine_sim)


def compute_gate_entropy_loss(gate_g: torch.Tensor):
    gate_g = gate_g.clamp(min=1e-8, max=1 - 1e-8)
    entropy = -(gate_g * torch.log(gate_g) + (1 - gate_g) * torch.log(1 - gate_g))
    return entropy.mean()


def compute_cell_structure_loss(z_cells, x_cells, temperature=0.2):
    z = F.normalize(z_cells, p=2, dim=1)
    x = F.normalize(x_cells, p=2, dim=1)

    sim_z = torch.matmul(z, z.T) / temperature
    sim_x = torch.matmul(x, x.T) / temperature

    target = torch.softmax(sim_x.detach(), dim=1)
    log_prob = F.log_softmax(sim_z, dim=1)

    return F.kl_div(log_prob, target, reduction="batchmean")


def compute_gate_variance_loss(gate_g, target_std=0.25):
    return F.relu(target_std - gate_g.std()).pow(2)


def compute_gene_gate_smoothness_loss(
    gate_g: torch.Tensor,
    H_prior: torch.Tensor,
    gene_indices: torch.Tensor,
):
    gene_to_idx = {int(g): i for i, g in enumerate(gene_indices)}

    total_diff = 0.0
    pair_count = 0

    for edge_idx in range(H_prior.shape[1]):
        edge_nodes = torch.where(H_prior[:, edge_idx] > 0)[0]
        edge_genes = [n for n in edge_nodes if int(n) in gene_to_idx]

        if len(edge_genes) >= 2:
            for i in range(len(edge_genes)):
                for j in range(i + 1, len(edge_genes)):
                    g_i = gene_to_idx[int(edge_genes[i])]
                    g_j = gene_to_idx[int(edge_genes[j])]
                    diff = (gate_g[g_i] - gate_g[g_j]) ** 2
                    total_diff += diff
                    pair_count += 1

    if pair_count == 0:
        return torch.tensor(0.0, device=gate_g.device)

    return total_diff / pair_count


def compute_dec_loss(q: torch.Tensor, p: torch.Tensor):
    q = q.clamp(min=1e-8)
    p = p.clamp(min=1e-8)

    kl_div = p * (torch.log(p) - torch.log(q))
    return kl_div.sum(dim=1).mean()


def compute_total_loss(
    out: Dict[str, torch.Tensor],
    hg_data: Dict[str, torch.Tensor],
    epoch: int,
    config: PhaseTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    kl_warmup_beta = config.vae_kl_weight * min(1.0, (epoch + 1) / config.vae_kl_warmup_epochs)

    vae_recon_loss = F.mse_loss(out["recon"], out["h"].detach())

    raw_kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp(), dim=1).mean()

    if config.kl_capacity_max > 0:
        capacity = min(config.kl_capacity_max, (epoch + 1) / config.vae_kl_warmup_epochs * config.kl_capacity_max)
        kl_term = config.vae_kl_weight * torch.abs(raw_kl - capacity)
    else:
        kl_term = kl_warmup_beta * raw_kl

    cell_gene_pairs = hg_data.get("cell_gene_pairs", None)
    gene_indices = out["gene_indices"]

    if cell_gene_pairs is not None and len(cell_gene_pairs) > 0:
        cell_gene_recon_loss = compute_cell_gene_recon_loss(
            out["mu_cells"],
            out["mu_genes"],
            cell_gene_pairs,
            temperature=config.cell_gene_temperature,
            num_neg=config.cell_gene_num_neg,
            device=out["mu_cells"].device,
        )
    else:
        cell_gene_recon_loss = torch.tensor(0.0, device=out["mu_cells"].device)

    H_prior = hg_data.get("H_prior", None)
    if H_prior is not None:
        gate_smooth_loss = compute_gene_gate_smoothness_loss(
            out["gate_g"], H_prior, gene_indices
        )
    else:
        gate_smooth_loss = torch.tensor(0.0, device=out["gate_g"].device)

    phase_sep_loss = compute_centroid_phase_separation_loss(out["mu_genes"], out["gate_g"])

    gate_mean = out["gate_g"].mean()
    gate_balance_loss = (gate_mean - 0.5) ** 2

    gate_entropy_loss = compute_gate_entropy_loss(out["gate_g"])
    
    gate_variance_loss = compute_gate_variance_loss(out["gate_g"])

    cell_structure_loss = torch.tensor(0.0, device=out["mu_cells"].device)
    if out.get("x_cells") is not None:
        cell_structure_loss = compute_cell_structure_loss(
            out["mu_cells"], 
            out["x_cells"],
            temperature=getattr(config, 'cell_structure_temp', 0.2)
        )

    dec_loss = torch.tensor(0.0, device=out["mu_cells"].device)
    if config.use_dec and config.num_clusters is not None and out.get("q") is not None:
        if epoch >= config.dec_start_epoch:
            p = compute_target_distribution(out["q"].detach())
            dec_loss = compute_dec_loss(out["q"], p)

    total_loss = (
        config.vae_recon_weight * vae_recon_loss
        + kl_term
        + config.cell_gene_recon_weight * cell_gene_recon_loss
        + config.phase_sep_weight * phase_sep_loss
        + config.gate_balance_weight * gate_balance_loss
        + config.gate_entropy_weight * gate_entropy_loss
        + config.gene_gate_smoothness_weight * gate_smooth_loss
        + config.dec_weight * dec_loss
        + config.cell_structure_weight * cell_structure_loss
        + config.gate_variance_weight * gate_variance_loss
    )

    diagnostics = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "vae_recon_loss": float(vae_recon_loss.detach().cpu().item()),
        "raw_kl": float(raw_kl.detach().cpu().item()),
        "kl_term": float(kl_term.detach().cpu().item()),
        "mu_abs_mean": float(out["mu"].abs().mean().detach().cpu().item()),
        "mu_norm_mean": float(out["mu"].norm(dim=1).mean().detach().cpu().item()),
        "logvar_mean": float(out["logvar"].mean().detach().cpu().item()),
        "logvar_max": float(out["logvar"].max().detach().cpu().item()),
        "h_norm_mean": float(out["h"].norm(dim=1).mean().detach().cpu().item()),
        "gate_mean": float(gate_mean.detach().cpu().item()),
        "gate_std": float(out["gate_g"].std().detach().cpu().item()),
        "cell_gene_recon_loss": float(cell_gene_recon_loss.detach().cpu().item()),
        "phase_sep_loss": float(phase_sep_loss.detach().cpu().item()),
        "gate_balance_loss": float(gate_balance_loss.detach().cpu().item()),
        "gate_entropy_loss": float(gate_entropy_loss.detach().cpu().item()),
        "gate_smooth_loss": float(gate_smooth_loss.detach().cpu().item()),
        "gate_variance_loss": float(gate_variance_loss.detach().cpu().item()),
        "cell_structure_loss": float(cell_structure_loss.detach().cpu().item()),
        "dec_loss": float(dec_loss.detach().cpu().item()),
    }

    return total_loss, diagnostics


def compute_target_distribution(q: torch.Tensor) -> torch.Tensor:
    p = q ** 2 / (q.sum(dim=0) + 1e-8)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    return p

__all__ = [name for name in globals() if not name.startswith("__")]
