import json
import unittest

import numpy as np

from phasehyper.evaluation.phase import (
    evaluate_phase_model,
    evaluate_phase_quality,
)


class PhaseEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.labels = np.repeat([0, 1], 6)
        self.raw = np.vstack(
            [rng.uniform(0.2, 1.0, (6, 5)), rng.uniform(2.0, 3.0, (6, 5))]
        )
        self.a = self.raw * 0.4
        self.b = self.raw * 0.6
        self.cell_h = self.raw[:, :3]

    def test_quality_metrics(self):
        quality = evaluate_phase_quality(self.raw, self.a, self.b)
        self.assertAlmostEqual(quality["reconstruction_relative_error"], 0.0)
        self.assertGreaterEqual(quality["phase_energy_ratio"], 0.0)
        self.assertLessEqual(quality["phase_energy_ratio"], 1.0)
        equal = evaluate_phase_quality(
            self.raw, self.raw / 2, self.raw / 2
        )
        self.assertAlmostEqual(equal["phase_imbalance"], 0.0)

    def test_complete_result_has_all_clustering_metrics_and_is_json_ready(self):
        result = evaluate_phase_model(
            raw_rna=self.raw,
            cell_embedding=self.cell_h,
            phase_a=self.a,
            phase_b=self.b,
            labels=self.labels,
            n_clusters=2,
            seed=0,
        )
        for name in ("Raw_RNA", "cell_h", "Phase_A", "Phase_B"):
            self.assertEqual(
                set(result[name]),
                {"NMI", "FMI", "ARI", "ASW", "PredClusters"},
            )
            self.assertEqual(result[name]["PredClusters"], 2)
        json.dumps(result)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_phase_quality(self.raw, self.a[:-1], self.b)


if __name__ == "__main__":
    unittest.main()
