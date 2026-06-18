from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..config import DATA_ROOT, DatasetBundle, PriorBundle
from .base import PriorBuilder, PriorConfig


_PRIOR_REGISTRY: Dict[str, PriorBuilder] = {}


def register_prior(builder: PriorBuilder) -> PriorBuilder:
    name = str(builder.name).strip()
    if not name:
        raise ValueError("Prior builder name cannot be empty")
    if name in _PRIOR_REGISTRY:
        raise ValueError(f"Duplicate prior builder: {name}")
    _PRIOR_REGISTRY[name] = builder
    return builder


def get_prior_builder(name: str) -> PriorBuilder:
    try:
        return _PRIOR_REGISTRY[str(name)]
    except KeyError as exc:
        available = ", ".join(list_prior_builders())
        raise ValueError(f"Unknown prior builder: {name}. Available: {available}") from exc


def list_prior_builders() -> List[str]:
    return sorted(_PRIOR_REGISTRY)


def prior_builder_labels() -> Dict[str, str]:
    return {name: get_prior_builder(name).label for name in list_prior_builders()}


def build_prior(base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
    runtime_base_dir = DATA_ROOT if Path(base_dir) == Path(".") else Path(base_dir)
    return get_prior_builder(config.name).build(
        base_dir=runtime_base_dir,
        dataset=dataset,
        config=config,
    )
