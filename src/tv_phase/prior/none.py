from pathlib import Path

import pandas as pd

from ..config import DatasetBundle, PriorBundle
from .base import PriorConfig
from .registry import register_prior


class NonePriorBuilder:
    name = "none"
    label = "No prior"
    description = "Empty prior control"

    def build(self, *, base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
        return PriorBundle(
            kegg_groups={},
            poswin_groups={},
            ppi_groups={},
            gene_prior_matrix=None,
            data_groups={},
            data_group_weights={},
            edge_table=pd.DataFrame(),
            metadata={
                "prior_name": self.name,
                "prior_label": self.label,
                "construction_method": "empty prior control",
                "external_prior": False,
                "labels_used": False,
                "n_edges": 0,
            },
        )


register_prior(NonePriorBuilder())

