import unittest

import numpy as np

from phasehyper.evaluation.simulation import (
    evaluate_simulation_clustering,
    evaluate_simulation_expression,
    evaluate_simulation_grn,
)


class SimulationEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.maternal = rng.uniform(0.2, 1.5, (14, 6))
        self.paternal = rng.uniform(0.2, 1.5, (14, 6))
        self.combined = self.maternal + self.paternal

    def test_expression_result_and_swapped_prediction(self):
        result = evaluate_simulation_expression(
            phase_a_pred=self.paternal,
            phase_b_pred=self.maternal,
            maternal_true=self.maternal,
            paternal_true=self.paternal,
            combined=self.combined,
            seed=0,
            pre_sync_phase_a=self.paternal,
            pre_sync_phase_b=self.maternal,
        )
        self.assertEqual(result["orientation"]["n_swapped"], 1)
        self.assertTrue(
            np.allclose(result["phase_a_oriented"], self.maternal)
        )
        self.assertEqual(
            [row["name"] for row in result["headline_rows"]],
            [
                "phasehyper",
                "RandomSplit",
                "NMF2Factor",
                "[trivial] combined/2",
                "[floor] perfect@rank-dc",
            ],
        )
        self.assertIsNotNone(result["pre_sync_orientation_audit"])

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            evaluate_simulation_expression(
                phase_a_pred=self.maternal[:-1],
                phase_b_pred=self.paternal,
                maternal_true=self.maternal,
                paternal_true=self.paternal,
                combined=self.combined,
            )

    def test_clustering_auxiliary_result(self):
        rng = np.random.default_rng(32)
        labels = np.repeat([0, 1], 7)
        embedding = np.vstack(
            [rng.normal(-2, 0.1, (7, 4)), rng.normal(2, 0.1, (7, 4))]
        )
        result = evaluate_simulation_clustering(
            raw_rna=embedding,
            cell_embedding=embedding,
            phase_a_embedding=embedding,
            phase_b_embedding=embedding,
            labels=labels,
            n_clusters=2,
        )
        self.assertEqual(set(result), {"raw", "cell_h", "phase_a", "phase_b"})
        self.assertTrue(all(value == 1.0 for value in result.values()))

    def test_grn_inherits_expression_swap(self):
        rng = np.random.default_rng(33)
        true_a = rng.normal(size=(12, 10))
        true_b = rng.normal(size=(12, 10))
        result = evaluate_simulation_grn(
            grn_a_pred=true_b,
            grn_b_pred=true_a,
            grn_a_true=true_a,
            grn_b_true=true_b,
            combined_grn=true_a + true_b,
            inherited_swap=True,
            seed=0,
        )
        self.assertTrue(np.array_equal(result["phase_a_oriented"], true_a))
        self.assertEqual(result["saber_rows"][0]["orient_level"], "raw")
        self.assertEqual(
            [row["name"] for row in result["differential_rows"]],
            ["phasehyper", "RandomSplit", "NMF2Factor", "[floor] no phasing"],
        )


if __name__ == "__main__":
    unittest.main()
