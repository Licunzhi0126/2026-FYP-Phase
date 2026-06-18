from pathlib import Path

from ..config import DatasetBundle, PriorBundle
from .base import PriorConfig
from .registry import register_prior


class DatasetPriorBuilder:
    name = "dataset"
    label = "Dataset prior"
    description = "Dataset-provided KEGG, genomic-window, and PPI prior"

    def build(self, *, base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
        from ..priors import _build_dataset_prior_bundle

        return _build_dataset_prior_bundle(
            base_dir,
            dataset,
            d_prior=config.d_prior,
            allow_position_file_fallback=config.allow_position_file_fallback,
            genomic_window_bp=config.genomic_window_bp,
            include_window_groups=config.include_window_groups,
        )


register_prior(DatasetPriorBuilder())

