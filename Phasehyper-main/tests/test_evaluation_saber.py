import unittest

import numpy as np

from phasehyper.evaluation import saber


class SaberEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.a = rng.uniform(0.2, 2.0, (12, 6))
        self.b = rng.uniform(0.2, 2.0, (12, 6))
        self.combined = self.a + self.b

    def test_perfect_prediction_and_global_swap(self):
        perfect = saber.phase_metrics(self.a, self.b, self.a, self.b)
        swapped = saber.phase_metrics(self.b, self.a, self.a, self.b)
        self.assertAlmostEqual(perfect["mse"], 0.0)
        self.assertAlmostEqual(perfect["pearson"], 1.0)
        self.assertAlmostEqual(swapped["mse"], perfect["mse"])
        _, _, info = saber.orient(self.b, self.a, self.a, self.b, "global")
        self.assertEqual(info["n_swapped"], 1)

    def test_orientation_levels_and_audit(self):
        for level in ("raw", "global", "per_gene"):
            out_a, out_b, info = saber.orient(
                self.a, self.b, self.a, self.b, level
            )
            self.assertEqual(out_a.shape, self.a.shape)
            self.assertEqual(out_b.shape, self.b.shape)
            self.assertEqual(info["level"], level)
        rows = saber.orientation_audit(
            self.b, self.a, self.a, self.b, "test"
        )
        self.assertEqual([row["level"] for row in rows], [
            "raw", "global", "per_gene"
        ])

    def test_baselines_are_complete_and_reproducible(self):
        first = saber.baseline_random_split(self.combined, seed=5)
        second = saber.baseline_random_split(self.combined, seed=5)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.allclose(first[0] + first[1], self.combined))
        half = saber.baseline_mean_fraction_shrinkage(self.combined)
        self.assertTrue(np.allclose(half[0], half[1]))
        rows = saber.run_baselines(
            self.combined, self.a, self.b, seed=5
        )
        self.assertEqual(
            [row["name"] for row in rows],
            ["RandomSplit", "MeanFractionShrinkage", "NMF2Factor"],
        )

    def test_signed_grn_and_differential_fields(self):
        rng = np.random.default_rng(12)
        true_a = rng.normal(size=(10, 8))
        true_b = rng.normal(size=(10, 8))
        row = saber.phase_metrics(
            true_a, true_b, true_a, true_b, signed=True
        )
        self.assertAlmostEqual(row["mse"], 0.0)
        differential = saber.differential_metrics(
            true_a, true_b, true_a, true_b, "perfect"
        )
        for key in (
            "pcc", "pcc_cell", "spearman", "d_ratio", "nmse",
            "skill", "alpha", "skill_cal", "auroc",
        ):
            self.assertIn(key, differential)


if __name__ == "__main__":
    unittest.main()
