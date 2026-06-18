from pathlib import Path

from ..config import DatasetBundle, PriorBundle
from .base import PriorConfig
from .registry import register_prior


class PGluePriorBuilder:
    name = "p_glue"
    label = "P-GLUE"
    description = "GLUE-inspired data-driven signed feature graph"

    def build(self, *, base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
        from ..priors import _build_correlation_candidate_table, _edge_table_to_data_groups

        table, nodes = _build_correlation_candidate_table(
            dataset,
            top_k=config.top_k,
            max_features=config.max_features,
            evidence_prefix="glue",
        )
        groups, weights = _edge_table_to_data_groups(table, self.name)
        return PriorBundle(
            kegg_groups={},
            poswin_groups={},
            ppi_groups={},
            gene_prior_matrix=None,
            data_groups=groups,
            data_group_weights=weights,
            edge_table=table.reset_index(drop=True),
            metadata={
                "prior_name": self.name,
                "prior_label": self.label,
                "construction_method": self.description,
                "top_k": int(config.top_k),
                "max_features": config.max_features,
                "external_prior": False,
                "labels_used": False,
                "selected_feature_count": len(nodes),
                "n_edges": len(groups),
                "density": float(len(groups) / max(1, len(dataset.common_genes) ** 2)),
                "seed": int(config.seed),
            },
        )


register_prior(PGluePriorBuilder())
