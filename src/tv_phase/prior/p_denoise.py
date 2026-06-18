from pathlib import Path

import numpy as np

from ..config import DatasetBundle, PriorBundle
from .base import PriorConfig
from .registry import register_prior


class PDenoisePriorBuilder:
    name = "p_denoise"
    label = "P-Denoise"
    description = "Data-driven candidate graph filtered by an edge-confidence model"

    def build(self, *, base_dir: Path, dataset: DatasetBundle, config: PriorConfig) -> PriorBundle:
        from ..priors import (
            _build_correlation_candidate_table,
            _edge_table_to_data_groups,
            _feature_node_matrix,
            _train_edge_confidence,
        )

        table, nodes = _build_correlation_candidate_table(
            dataset,
            top_k=config.denoise_candidate_top_k,
            max_features=config.max_features,
            evidence_prefix="denoise_candidate",
        )
        node_features = _feature_node_matrix(dataset, nodes, config.denoise_node_feature_dim, config.seed)
        table = _train_edge_confidence(
            nodes,
            table,
            node_features,
            hidden_dim=config.denoise_hidden_dim,
            epochs=config.denoise_epochs,
            lr=config.denoise_lr,
            device=config.device,
            seed=config.seed,
        )
        top_percent = float(np.clip(config.denoise_top_percent, 0.0, 1.0))
        if not table.empty and top_percent < 1.0:
            cutoff = table["confidence"].quantile(1.0 - top_percent)
            table = table[table["confidence"] >= cutoff].copy()
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
                "candidate_top_k": int(config.denoise_candidate_top_k),
                "max_features": config.max_features,
                "node_feature_dim": int(config.denoise_node_feature_dim),
                "hidden_dim": int(config.denoise_hidden_dim),
                "epochs": int(config.denoise_epochs),
                "top_percent": top_percent,
                "external_prior": False,
                "labels_used": False,
                "selected_feature_count": len(nodes),
                "n_edges": len(groups),
                "density": float(len(groups) / max(1, len(dataset.common_genes) ** 2)),
                "seed": int(config.seed),
            },
        )


register_prior(PDenoisePriorBuilder())
