"""统一分相 loss，按 head 分派。

additive（聚类）: 硬 recon(MSE(X_A+X_B, expr)) + UW(reg=z², diff=cos²(X_A,X_B)+能量均衡)
gated（分相）  : 纯 recon(MSE(门控混合, log_expr))

diff 前 DIFF_WARMUP_EPOCHS 轮不启用，之后线性引入；cos² 同时惩罚相同与反向，推向正交。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phasehyper.schemas import PhaseTrainingConfig


class UncertaintyWeights(nn.Module):
    """Uncertainty weighting (Kendall 2018)."""

    def __init__(self, n_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: list) -> torch.Tensor:
        total = torch.zeros((), device=self.log_vars.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def get_weights(self) -> list:
        return [float(torch.exp(-lv).item()) for lv in self.log_vars]


DIFF_WARMUP_EPOCHS = 20


def compute_total_loss(
    out: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    epoch: int,
    config: PhaseTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """统一 loss 入口：按 out["head"] 分派到 additive / gated。

    两条分支都返回 (loss, diag)，diag 至少含 total_loss / recon_loss /
    var_explained / extra（用于统一日志），让两个训练循环长得一模一样。
    """
    head = out.get("head", "additive")
    if head == "gated":
        return _gated_loss(out, aux)
    return _additive_loss(out, aux, epoch, config)


def _gated_loss(
    out: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """门控（分相/比例）头（无 Wiener）：纯 recon(log)。

    两相 logA/logB = 纯 decoder（同 additive）；gate = clip(0.5 + scale · S·residual) 不进 loss
    （per-gene |P| 与 scale 无关）。loss 只有 recon，最贴近 additive 的结构。
    """
    target = aux["recon_target"]
    recon_loss = F.mse_loss(out["recon"], target)
    total_loss = recon_loss

    with torch.no_grad():
        ss_res = (target - out["recon"]).pow(2).sum()
        ss_tot = target.pow(2).sum().clamp(min=1e-8)
        var_explained = 1.0 - ss_res / ss_tot

    diagnostics = {
        "total_loss": float(total_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "var_explained": float(var_explained.item()),
        "extra": "",
    }
    return total_loss, diagnostics


def _additive_loss(
    out: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    epoch: int,
    config: PhaseTrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = out["z_a"].device
    expr = out["expr"]
    x_a = out["phase_a_expr"]
    x_b = out["phase_b_expr"]
    x_recon = out["expr_recon"]

    # ① 加性重建（硬约束，不进 UW）
    recon_loss = F.mse_loss(x_recon, expr)

    # ② RAE L2 正则（共享 encoder，只有一个 z）
    reg_term = out["z"].pow(2).mean()

    # ③ 正交化：cos² 惩罚相同(+1)和反向(-1)，推向正交(0)
    cos_ab = F.cosine_similarity(x_a, x_b, dim=1).mean()
    ortho_loss = cos_ab.pow(2)
    # 能量均衡
    energy_a = x_a.pow(2).sum(1)
    energy_b = x_b.pow(2).sum(1)
    energy_ratio = energy_a / (energy_a + energy_b + 1e-8)
    energy_balance = (energy_ratio - 0.5).pow(2).mean()
    diff_loss = ortho_loss + energy_balance

    # diff warmup：前 N 轮只做重建，之后线性引入差异化
    diff_warmup = min(1.0, max(0.0, (epoch - DIFF_WARMUP_EPOCHS) / DIFF_WARMUP_EPOCHS))

    # ---- 总 loss = 硬 recon + UW(reg, diff) ----
    aux_losses = [reg_term, diff_loss * diff_warmup]
    aux_names = ["reg", "diff"]

    uw = aux.get("uncertainty_weights")
    if uw is not None:
        aux_total = uw(aux_losses)
        auto_weights = uw.get_weights()
    else:
        weights = [0.01, 1.0]
        aux_total = sum(w * l for w, l in zip(weights, aux_losses))
        auto_weights = weights

    total_loss = recon_loss + aux_total

    # 诊断
    with torch.no_grad():
        ss_res = (expr - x_recon).pow(2).sum()
        ss_tot = expr.pow(2).sum().clamp(min=1e-8)
        var_explained = 1.0 - ss_res / ss_tot

    diagnostics = {
        "total_loss": float(total_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "reg_term": float(reg_term.item()),
        "diff_loss": float(diff_loss.item()),
        "cos_ab": float(cos_ab.item()),
        "ortho_loss": float(ortho_loss.item()),
        "energy_balance": float(energy_balance.item()),
        "diff_warmup": diff_warmup,
        "var_explained": float(var_explained.item()),
        "z_norm": float(out["z"].norm(dim=1).mean().item()),
        "extra": f"cos_ab={cos_ab.item():.3f}",
    }
    for i, name in enumerate(aux_names):
        diagnostics[f"uw_{name}"] = auto_weights[i]

    return total_loss, diagnostics
