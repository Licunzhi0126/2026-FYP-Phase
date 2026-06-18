"""Shared prior helpers re-exported from the legacy module during migration."""

from ..priors import (
    _build_correlation_candidate_table,
    _edge_table_to_data_groups,
    _feature_node_matrix,
    _train_edge_confidence,
    build_gene_prior_features,
)

__all__ = [
    "build_gene_prior_features",
    "_build_correlation_candidate_table",
    "_edge_table_to_data_groups",
    "_feature_node_matrix",
    "_train_edge_confidence",
]
