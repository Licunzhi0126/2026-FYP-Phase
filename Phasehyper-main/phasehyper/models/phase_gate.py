"""SlotJumpPhaseGate：过完备 SAE 特征空间 + JumpReLU + Slot 竞争 → 两相 latent 分解。

忠实 discussion_summary.md §3.3：吃细胞 embedding H (N×d)，输出两个 phase latent
H_A, H_B (N×d)，供各自 Decoder 解回表达（X_A = Decoder_A(H_A)、X_B = Decoder_B(H_B)）。

  z       = W_enc(H)                       (N, r)   过完备特征空间 r = d × expansion (r >> d)
  act_A   = JumpReLU(z, theta_A)                    Phase A 稀疏激活（每特征维一个可学习阈值）
  act_B   = JumpReLU(z, theta_B)                    Phase B 稀疏激活
  compete = softmax([act_A+assign, act_B-assign]/τ) Slot 竞争 + 逐特征相归属偏置 assign（对称破除）
  feat_A  = z * compete[..., 0]            (N, r)   Phase A 竞争后特征
  feat_B  = z * compete[..., 1]            (N, r)   Phase B 竞争后特征
  H_A     = W_dec(feat_A)                  (N, d)
  H_B     = W_dec(feat_B)                  (N, d)

因 softmax 两路相加=1 ⇒ feat_A + feat_B == z（逐元素）。但 W_dec 含 bias，故
H_A + H_B 与 H 不天然相等——由 §5 的 L_add = MSE(H_A + H_B, H) 软约束逼近
（让 W_dec∘W_enc 成为 H 的忠实自编码，并把两相绑成 H 的可加拆分）。

设计来源：过完备 SAE（Anthropic/OpenAI/DeepMind 2024）、JumpReLU 阈值
（DeepMind 2024, arXiv:2407.14435）、Slot 竞争绑定（Gated Slot Attention NeurIPS 2024 +
DINOSAUR ICLR 2024）。

输入: H (N_cells × d)  ← 细胞 embedding（取 VAE 的 mu_cells）
输出:
  H_A, H_B    (N×d)   两相 latent，喂 Decoder_A / Decoder_B
  feat_A, feat_B (N×r) 竞争特征（诊断/可选）
  compete_B   (N×r)   Phase B 竞争占比，供 balance 正则（mean → 0.5 防一相主导）
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SlotJumpPhaseGate(nn.Module):
    """两相 latent 分解（SAE + JumpReLU + Slot 竞争）。

    Args:
        latent_dim    : 细胞 embedding 维度 d（H 的列数）
        expansion     : 过完备倍率，r = latent_dim * expansion（默认 4）
        jump_bandwidth: JumpReLU sigmoid 近似的平滑带宽（越小越接近硬阈值）
    """

    def __init__(
        self,
        latent_dim: int,
        expansion: int = 4,
        jump_bandwidth: float = 0.1,
        compete_temp: float = 1.0,
        assign_init_std: float = 1.0,
    ):
        super().__init__()
        r = latent_dim * expansion
        self.r = r
        self.jump_bandwidth = jump_bandwidth
        self.compete_temp = compete_temp

        # ── 过完备 SAE：W_enc 升维到 r >> d，W_dec 解回 d ─────────────────────
        self.W_enc = nn.Linear(latent_dim, r, bias=True)
        self.W_dec = nn.Linear(r, latent_dim, bias=True)

        # ── JumpReLU 阈值：两相各自的逐特征激活门槛 ─────────────────────────
        self.theta_A = nn.Parameter(torch.zeros(r))
        self.theta_B = nn.Parameter(torch.zeros(r))

        # ── 逐特征相归属偏置（对称破除）─────────────────────────────────────
        # 实现 summary「每个特征维度天然归属一个 phase」：assign[k]>0→特征 k 偏 A，<0→偏 B。
        # 仅靠 softmax([act_A,act_B]) 在 θ 同初始化 + bandwidth 饱和时恒为 0.5（两相塌成同一解），
        # 这个可学习偏置在初始化即打破对称、再由 recon/HSIC 端到端塑形成真正的两相分配。
        # assign_init_std=0 即退回纯 summary 版（无对称破除）。
        self.assign_init_std = assign_init_std
        self.assign = nn.Parameter(torch.zeros(r))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec.weight, a=math.sqrt(5))
        nn.init.normal_(self.theta_A, mean=0.0, std=0.1)
        nn.init.normal_(self.theta_B, mean=0.0, std=0.1)
        if self.assign_init_std > 0:
            nn.init.normal_(self.assign, mean=0.0, std=self.assign_init_std)

    def _jump_relu(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """JumpReLU 的 sigmoid 可微近似：forward≈z·1[z>θ]，全程可微（带宽 jump_bandwidth）。"""
        soft_mask = torch.sigmoid((z - theta) / self.jump_bandwidth)
        return z * soft_mask

    def forward(self, H: torch.Tensor):
        # ── 1. 过完备投影 ─────────────────────────────────────────────────
        z = self.W_enc(H)                                  # (N, r)

        # ── 2. JumpReLU：两相对每个特征维各自的阈值激活 ───────────────────
        act_A = self._jump_relu(z, self.theta_A)           # (N, r)
        act_B = self._jump_relu(z, self.theta_B)           # (N, r)

        # ── 3. Slot 竞争：每个特征维在两相间竞争激活权（含相归属偏置，softmax 两路和=1）──
        logit_A = (act_A + self.assign) / self.compete_temp
        logit_B = (act_B - self.assign) / self.compete_temp
        compete = torch.softmax(
            torch.stack([logit_A, logit_B], dim=-1), dim=-1
        )                                                  # (N, r, 2)
        feat_A = z * compete[..., 0]                       # (N, r)
        feat_B = z * compete[..., 1]                       # (N, r)

        # ── 4. 解回 latent：两相 embedding ────────────────────────────────
        H_A = self.W_dec(feat_A)                           # (N, d)
        H_B = self.W_dec(feat_B)                           # (N, d)

        compete_B = compete[..., 1]                        # (N, r) Phase B 占比 → balance
        return H_A, H_B, feat_A, feat_B, compete_B
