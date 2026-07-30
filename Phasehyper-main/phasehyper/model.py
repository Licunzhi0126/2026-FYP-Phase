"""Shared HyperPhase model, training criterion, and optimizer builders.

Both ``run_simulation.py`` and ``run_phase.py`` use this module as the single
source of truth for the model architecture and its training objective.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phasehyper.models.phase_model import RAEDecoder, RAEEncoder
from phasehyper.models.two_level_hgnn import HypergraphChannel, scipy_to_torch_sparse


class _CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, h_causal: torch.Tensor, h_func: torch.Tensor
    ) -> torch.Tensor:
        n_nodes, dim = h_causal.shape
        heads, head_dim = self.num_heads, self.head_dim
        query = self.W_q(h_causal).view(
            n_nodes, heads, head_dim
        ).permute(1, 0, 2)
        key = self.W_k(h_func).view(
            n_nodes, heads, head_dim
        ).permute(1, 0, 2)
        value = self.W_v(h_func).view(
            n_nodes, heads, head_dim
        ).permute(1, 0, 2)
        key_t = key.transpose(-2, -1)

        if n_nodes <= 2048:
            attention = self.attn_drop(
                F.softmax(
                    query @ key_t / math.sqrt(head_dim),
                    dim=-1,
                )
            )
            cross = attention @ value
        else:
            chunks = []
            for start in range(0, n_nodes, 1024):
                logits = (
                    query[:, start:start + 1024] @ key_t
                    / math.sqrt(head_dim)
                )
                attention = self.attn_drop(F.softmax(logits, dim=-1))
                chunks.append(attention @ value)
            cross = torch.cat(chunks, dim=1)

        cross = cross.permute(1, 0, 2).contiguous().view(n_nodes, dim)
        return self.norm(h_causal + self.W_o(cross))


class _GatedDirectedAPPNP(nn.Module):
    def __init__(
        self,
        h_tail,
        h_head,
        weights,
        etype,
        n_types: int,
        in_dim: int,
        alpha: float = 0.15,
        K: int = 10,
        type_init: float = 0.9,
        content_gate: bool = True,
        gate_hidden: int = 16,
    ):
        super().__init__()
        ht = h_tail.tocsr().astype(np.float64)
        hh = h_head.tocsr().astype(np.float64)
        if ht.shape != hh.shape:
            raise ValueError(
                "directed H_tail and H_head must have the same shape, "
                f"got {ht.shape} and {hh.shape}"
            )

        n_edges = ht.shape[1]
        edge_weights = np.asarray(weights, dtype=float).ravel()
        if edge_weights.size != n_edges:
            raise ValueError(
                f"directed weights contain {edge_weights.size} values "
                f"for {n_edges} edges"
            )
        edge_types = np.asarray(etype, dtype=np.int64).ravel()
        if edge_types.size != n_edges:
            raise ValueError(
                f"directed etype contains {edge_types.size} values "
                f"for {n_edges} edges"
            )
        resolved_n_types = int(n_types)
        if resolved_n_types <= 0:
            resolved_n_types = 1
        if edge_types.size and (
            edge_types.min() < 0 or edge_types.max() >= resolved_n_types
        ):
            raise ValueError("directed etype contains an out-of-range type id")

        tail_degree = np.asarray(ht.sum(axis=0)).ravel()
        head_degree = np.asarray(
            hh.multiply(np.abs(edge_weights)[None, :]).sum(axis=1)
        ).ravel()
        tail_inv = np.zeros_like(tail_degree, dtype=float)
        head_inv = np.zeros_like(head_degree, dtype=float)
        np.divide(1.0, tail_degree, out=tail_inv, where=tail_degree > 0)
        np.divide(1.0, head_degree, out=head_inv, where=head_degree > 0)

        self.register_buffer(
            "escale",
            torch.tensor(tail_inv * edge_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "dhi", torch.tensor(head_inv, dtype=torch.float32)
        )
        self.register_buffer(
            "Htt_sp",
            scipy_to_torch_sparse(ht).transpose(0, 1).coalesce(),
        )
        self.register_buffer("Hh_sp", scipy_to_torch_sparse(hh))
        self.register_buffer(
            "etype", torch.tensor(edge_types, dtype=torch.long)
        )
        self.K = int(K)

        prior = np.clip(
            np.full(resolved_n_types, float(type_init)),
            0.02,
            0.98,
        )
        self.register_buffer(
            "type_prior", torch.tensor(prior, dtype=torch.float32)
        )
        self.type_logit = nn.Parameter(
            torch.tensor(np.log(prior / (1.0 - prior)), dtype=torch.float32)
        )
        self.alpha_logit = nn.Parameter(
            torch.full(
                (self.K,),
                float(np.log(alpha / (1.0 - alpha))),
                dtype=torch.float32,
            )
        )
        self.content_gate = bool(content_gate)
        if self.content_gate:
            self.cg = nn.Sequential(
                nn.Linear(in_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )

    def _propagate(self, x: torch.Tensor) -> torch.Tensor:
        message = torch.sparse.mm(self.Htt_sp, x)
        type_gate = torch.sigmoid(self.type_logit)[self.etype]
        message = message * (self.escale * type_gate)[:, None]
        if self.content_gate:
            content_gate = torch.sigmoid(self.cg(message)).squeeze(-1)
            message = message * content_gate[:, None]
        return torch.sparse.mm(self.Hh_sp, message) * self.dhi[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        propagated = x
        alpha = torch.sigmoid(self.alpha_logit)
        for step in range(self.K):
            propagated = (
                (1.0 - alpha[step]) * self._propagate(propagated)
                + alpha[step] * x
            )
        return propagated

    def gate_reg(self) -> torch.Tensor:
        return (
            torch.sigmoid(self.type_logit) - self.type_prior
        ).pow(2).mean()


class HyperPhaseModel(nn.Module):
    """Shared dual-channel HyperPhase architecture."""

    def __init__(
        self,
        causal_ch: nn.Module,
        func_ch: nn.Module,
        n_cells: int,
        n_genes: int,
        dc: int,
        pca_init: np.ndarray | None = None,
        hidden: int = 256,
        latent: int = 128,
        use_asym: bool = True,
    ):
        super().__init__()
        self.n_cells = int(n_cells)
        self.n_genes = int(n_genes)
        self.dc = int(dc)

        self.cell_proj = nn.Linear(self.n_genes, self.dc, bias=False)
        if pca_init is not None:
            pca_array = np.asarray(pca_init, dtype=np.float32)
            expected_shape = (self.dc, self.n_genes)
            if pca_array.shape != expected_shape:
                raise ValueError(
                    f"pca_init has shape {pca_array.shape}, "
                    f"expected {expected_shape}"
                )
            with torch.no_grad():
                self.cell_proj.weight.copy_(torch.from_numpy(pca_array))

        self.causal_ch = causal_ch
        self.func_ch = func_ch
        self.fusion = _CrossAttentionFusion(self.dc)
        self.encoder = RAEEncoder(self.dc, hidden, latent)
        self.dec_a = RAEDecoder(latent, hidden, self.dc)
        self.dec_b = RAEDecoder(latent, hidden, self.dc)
        self.penc = RAEEncoder(self.dc, hidden, latent)
        self.proj_c = nn.Sequential(
            nn.Linear(self.dc, self.dc),
            nn.GELU(),
            nn.Linear(self.dc, self.dc),
        )
        self.proj_f = nn.Sequential(
            nn.Linear(self.dc, self.dc),
            nn.GELU(),
            nn.Linear(self.dc, self.dc),
        )
        self.nce_tau = 0.2
        self.use_asym = bool(use_asym)

        if self.use_asym:
            self.asym_dir = nn.Parameter(torch.randn(latent) * 0.01)
            self.asym_scale = nn.Parameter(torch.tensor(0.1))
            self.bias_head = nn.Sequential(
                nn.Linear(self.dc, 64),
                nn.GELU(),
                nn.Linear(64, latent),
            )
            nn.init.zeros_(self.bias_head[-1].weight)
            nn.init.zeros_(self.bias_head[-1].bias)

    def info_nce(self) -> torch.Tensor:
        """Symmetric full-node InfoNCE without node sampling."""
        n_nodes = self.zc.shape[0]
        target = torch.arange(n_nodes, device=self.zc.device)
        if n_nodes <= 4096:
            logits = self.zc @ self.zf.T / self.nce_tau
            return 0.5 * (
                F.cross_entropy(logits, target)
                + F.cross_entropy(logits.T, target)
            )

        total_cf = torch.zeros((), device=self.zc.device)
        total_fc = torch.zeros((), device=self.zc.device)
        for start in range(0, n_nodes, 1024):
            stop = min(start + 1024, n_nodes)
            total_cf = total_cf + F.cross_entropy(
                self.zc[start:stop] @ self.zf.T / self.nce_tau,
                target[start:stop],
                reduction="sum",
            )
            total_fc = total_fc + F.cross_entropy(
                self.zf[start:stop] @ self.zc.T / self.nce_tau,
                target[start:stop],
                reduction="sum",
            )
        return 0.5 * (total_cf + total_fc) / n_nodes

    def forward(
        self,
        m_graph: torch.Tensor,
        gf: torch.Tensor,
        ch_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_features = self.cell_proj(m_graph)
        node_features = torch.cat([cell_features, gf], dim=0)

        out_causal = self.causal_ch(node_features)
        out_func = self.func_ch(node_features)
        gene_slice = slice(
            self.n_cells, self.n_cells + self.n_genes
        )
        self.last_causal_genes = out_causal[gene_slice].detach()
        self.last_func_genes = out_func[gene_slice].detach()

        self.zc = F.normalize(self.proj_c(out_causal), dim=1)
        self.zf = F.normalize(self.proj_f(out_func), dim=1)

        causal_cells = out_causal[:self.n_cells]
        fused = self.fusion(
            causal_cells, out_func[:self.n_cells]
        )
        self.last_fused = fused.detach()
        latent = self.encoder(fused)

        if self.use_asym:
            base = F.normalize(self.asym_dir, dim=0)[None, :]
            bias = self.asym_scale * (
                base + self.bias_head(causal_cells)
            )
            self.last_bias = bias.detach()
            phase_a_raw = self.dec_a(latent + bias)
            phase_b_raw = self.dec_b(latent - bias)
        else:
            phase_a_raw = self.dec_a(latent)
            phase_b_raw = self.dec_b(latent)

        correction = 0.5 * (
            ch_target - phase_a_raw - phase_b_raw
        )
        phase_a = phase_a_raw + correction
        phase_b = phase_b_raw + correction
        return ch_target, latent, phase_a, phase_b


class SetCriterion(nn.Module):
    """The shared six-term objective used by both HyperPhase pipelines."""

    def __init__(
        self,
        w_comp: float = 8.0,
        w_ortho: float = 4.0,
        w_nce: float = 1.0,
        w_gate: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.w_comp = float(w_comp)
        self.w_ortho = float(w_ortho)
        self.w_nce = float(w_nce)
        self.w_gate = float(w_gate)
        self.eps = float(eps)

    def forward(
        self,
        *,
        model: HyperPhaseModel,
        model_output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        gene_projection: torch.Tensor,
        compartment_indicator: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, latent, phase_a, phase_b = model_output
        phase_a_latent = model.penc(phase_a)
        phase_b_latent = model.penc(phase_b)

        cyc_comp = F.mse_loss(
            0.5 * (phase_a_latent + phase_b_latent),
            latent.detach(),
        )

        n_batch = phase_a_latent.shape[0]
        phase_a_normalized = (
            phase_a_latent - phase_a_latent.mean(dim=0)
        ) / (phase_a_latent.std(dim=0) + self.eps)
        phase_b_normalized = (
            phase_b_latent - phase_b_latent.mean(dim=0)
        ) / (phase_b_latent.std(dim=0) + self.eps)
        correlation = (
            phase_a_normalized.T @ phase_b_normalized / n_batch
        )
        off_diagonal = ~torch.eye(
            correlation.shape[0],
            dtype=torch.bool,
            device=correlation.device,
        )
        barlow = (
            correlation.diagonal().pow(2).mean()
            + 0.005 * correlation[off_diagonal].pow(2).mean()
        )

        has_compartment = (
            compartment_indicator is not None
            and bool(torch.count_nonzero(compartment_indicator).item())
        )
        if has_compartment:
            indicator = compartment_indicator.to(
                device=phase_a.device,
                dtype=phase_a.dtype,
            )
            projection = gene_projection.to(
                device=phase_a.device,
                dtype=phase_a.dtype,
            )
            phase_difference = (phase_a - phase_b) @ projection
            compartment = 1.0 - F.cosine_similarity(
                phase_difference,
                indicator.unsqueeze(0),
                dim=1,
            ).pow(2).mean()
        else:
            compartment = torch.zeros(
                (), device=phase_a.device, dtype=phase_a.dtype
            )

        phase_cosine_per_cell = (
            F.normalize(phase_a, dim=1)
            * F.normalize(phase_b, dim=1)
        ).sum(dim=1)
        orthogonality = phase_cosine_per_cell.pow(2).mean()
        info_nce = model.info_nce()
        gate_regularization = model.causal_ch.gate_reg()

        total = (
            cyc_comp
            + 0.5 * barlow
            + self.w_comp * compartment
            + self.w_ortho * orthogonality
            + self.w_nce * info_nce
            + self.w_gate * gate_regularization
        )
        terms = {
            "total": total,
            "cyc_comp": cyc_comp,
            "barlow": barlow,
            "compartment": compartment,
            "orthogonality": orthogonality,
            "info_nce": info_nce,
            "gate_regularization": gate_regularization,
            "phase_cosine": phase_cosine_per_cell.mean(),
        }
        return total, terms


def _require_keys(
    data: Mapping[str, Any],
    keys: tuple[str, ...],
    label: str,
) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(
            f"{label} data is missing required keys: {missing}"
        )


def build_model(
    *,
    directed_data: Mapping[str, Any],
    undirected_data: Mapping[str, Any],
    n_cells: int,
    n_genes: int,
    dc: int,
    pca_init: np.ndarray | None,
    hidden: int = 256,
    latent: int | None = None,
    use_asym: bool = True,
    device: torch.device | str = "cpu",
) -> HyperPhaseModel:
    """Build the shared model from either pipeline's graph dictionaries."""
    _require_keys(
        directed_data,
        ("H_tail", "H_head", "W", "etype", "n_types"),
        "directed",
    )
    _require_keys(undirected_data, ("H", "W"), "undirected")
    if dc < 1:
        raise ValueError(f"dc must be positive, got {dc}")
    resolved_latent = dc if latent is None else int(latent)

    causal_channel = _GatedDirectedAPPNP(
        directed_data["H_tail"],
        directed_data["H_head"],
        directed_data["W"],
        directed_data["etype"],
        int(directed_data["n_types"]),
        int(dc),
        alpha=0.15,
        K=10,
        type_init=0.9,
        content_gate=True,
        gate_hidden=16,
    )
    functional_channel = HypergraphChannel(
        undirected_data["H"],
        undirected_data["W"],
        in_dim=int(dc),
        hidden_dim=int(dc),
        out_dim=int(dc),
        num_layers=2,
        dropout=0.2,
    )
    model = HyperPhaseModel(
        causal_channel,
        functional_channel,
        n_cells=int(n_cells),
        n_genes=int(n_genes),
        dc=int(dc),
        pca_init=pca_init,
        hidden=int(hidden),
        latent=resolved_latent,
        use_asym=use_asym,
    )
    return model.to(device)


def build_criterion(
    *,
    w_comp: float = 8.0,
    w_ortho: float = 4.0,
    w_nce: float = 1.0,
    w_gate: float = 0.05,
) -> SetCriterion:
    return SetCriterion(
        w_comp=w_comp,
        w_ortho=w_ortho,
        w_nce=w_nce,
        w_gate=w_gate,
    )


def build_optimizer(
    model: HyperPhaseModel,
    *,
    gate_lr: float = 1e-2,
    graph_lr: float = 1e-3,
    rae_lr: float = 1e-3,
    asym_lr: float = 5e-3,
    weight_decay: float = 5e-3,
) -> torch.optim.AdamW:
    """Build AdamW with exhaustive, non-overlapping parameter groups."""
    rae_parameters = (
        list(model.encoder.parameters())
        + list(model.dec_a.parameters())
        + list(model.dec_b.parameters())
        + list(model.penc.parameters())
    )
    gate_parameters = [
        model.causal_ch.type_logit,
        model.causal_ch.alpha_logit,
    ]
    asymmetric_parameters = (
        [model.asym_dir, model.asym_scale]
        if model.use_asym
        else []
    )

    reserved_ids = {
        id(parameter)
        for parameter in (
            rae_parameters
            + gate_parameters
            + asymmetric_parameters
        )
    }
    graph_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in reserved_ids
    ]

    groups: list[dict[str, Any]] = [
        {
            "name": "gate",
            "params": gate_parameters,
            "lr": float(gate_lr),
            "weight_decay": 0.0,
        },
        {
            "name": "graph",
            "params": graph_parameters,
            "lr": float(graph_lr),
        },
        {
            "name": "rae",
            "params": rae_parameters,
            "lr": float(rae_lr),
        },
    ]
    if asymmetric_parameters:
        groups.append({
            "name": "asymmetric",
            "params": asymmetric_parameters,
            "lr": float(asym_lr),
        })

    grouped_parameters = [
        parameter
        for group in groups
        for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer parameter groups contain duplicates")

    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    actual_ids = set(grouped_ids)
    if actual_ids != expected_ids:
        missing_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in actual_ids
        ]
        extra_count = len(actual_ids - expected_ids)
        raise RuntimeError(
            "optimizer parameter groups are not exhaustive: "
            f"missing={missing_names}, extra_count={extra_count}"
        )

    return torch.optim.AdamW(
        groups, weight_decay=float(weight_decay)
    )


__all__ = [
    "HyperPhaseModel",
    "SetCriterion",
    "build_model",
    "build_criterion",
    "build_optimizer",
]
