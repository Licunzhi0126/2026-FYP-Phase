from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from ..config import DatasetBundle, PriorBundle


@dataclass(frozen=True)
class PriorConfig:
    name: str
    d_prior: int = 16
    genomic_window_bp: int = 200000
    include_window_groups: bool = True
    top_k: int = 5
    max_features: Optional[int] = 800
    denoise_candidate_top_k: int = 5
    denoise_node_feature_dim: int = 64
    denoise_hidden_dim: int = 64
    denoise_epochs: int = 20
    denoise_lr: float = 1e-3
    denoise_top_percent: float = 0.7
    allow_position_file_fallback: bool = False
    device: str = "cpu"
    seed: int = 42
    extra: Optional[Dict[str, Any]] = None


class PriorBuilder(Protocol):
    name: str
    label: str
    description: str

    def build(self, *, base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
        ...

