import unittest

import numpy as np

from phasehyper.evaluation.clustering import (
    _ari,
    _emb,
    evaluate_clustering,
    evaluate_clustering_stability,
    prepare_embedding,
)


class ClusteringEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.labels = np.repeat([0, 1], 8)
        self.embedding = np.vstack(
            [rng.normal(-3, 0.1, (8, 4)), rng.normal(3, 0.1, (8, 4))]
        )

    def test_prepare_embedding_validates_and_cleans(self):
        with self.assertRaises(ValueError):
            prepare_embedding(np.arange(5))
        dirty = self.embedding.copy()
        dirty[0, 0] = np.nan
        dirty[1, 1] = np.inf
        cleaned = prepare_embedding(dirty)
        self.assertTrue(np.isfinite(cleaned).all())

    def test_pca_dimension_and_constant_matrix(self):
        reduced = prepare_embedding(self.embedding, use_pca=True, pca_dim=100)
        self.assertEqual(reduced.shape, (16, 4))
        constant = prepare_embedding(
            np.ones((5, 3)), use_pca=True, pca_dim=30
        )
        self.assertEqual(constant.shape, (5, 3))
        self.assertTrue(np.allclose(constant, 0))

    def test_kmeans_metrics_and_label_validation(self):
        result = evaluate_clustering(
            self.embedding,
            self.labels,
            expected_clusters=2,
            seed=0,
        )
        self.assertAlmostEqual(result["ari"], 1.0)
        self.assertGreater(result["asw"], 0.9)
        with self.assertRaises(ValueError):
            evaluate_clustering(
                self.embedding, self.labels[:-1], expected_clusters=2
            )
        with self.assertRaises(ValueError):
            evaluate_clustering(
                self.embedding, self.labels, expected_clusters=17
            )

    def test_single_label_asw_is_nan(self):
        result = evaluate_clustering(
            self.embedding,
            np.zeros(16, dtype=int),
            expected_clusters=2,
        )
        self.assertTrue(np.isnan(result["asw"]))

    def test_stability_and_legacy_wrappers_are_reproducible(self):
        first = evaluate_clustering_stability(
            self.embedding, self.labels, expected_clusters=2
        )
        second = evaluate_clustering_stability(
            self.embedding, self.labels, expected_clusters=2
        )
        self.assertEqual(first["ari_runs"], second["ari_runs"])
        self.assertAlmostEqual(
            _ari(_emb(self.embedding), 2, self.labels), first["ari_mean"]
        )

    def test_graph_cluster_methods(self):
        for method in ("leiden", "louvain"):
            result = evaluate_clustering(
                self.embedding,
                self.labels,
                method=method,
                expected_clusters=2,
            )
            self.assertIn("ari", result)
            self.assertGreaterEqual(result["pred_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
