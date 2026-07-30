import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from phasehyper.evaluation.metrics_io import (
    SABER_CSV_FIELDS,
    save_grn_evaluation,
    save_metrics_json,
    save_saber_evaluation,
)


class EvaluationMetricsIOTests(unittest.TestCase):
    def test_json_handles_numpy_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            save_metrics_json(path, {"value": np.float32(1.5)})
            self.assertEqual(json.loads(path.read_text())["value"], 1.5)

    def test_saber_and_grn_paths_and_columns(self):
        headline = [{"name": "phasehyper", "pcc_global": 0.5}]
        protocol = [{"name": "phasehyper", "mse": 0.25}]
        differential = [{"name": "phasehyper", "skill": 0.1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expression = save_saber_evaluation(
                output_dir=root / "expression",
                headline_rows=headline,
                saber_rows=protocol,
                orientation_rows=[{"level": "global", "bits": 1}],
                metadata={"seed": 0},
            )
            grn = save_grn_evaluation(
                output_dir=root / "grn",
                headline_rows=headline,
                saber_rows=protocol,
                differential_rows=differential,
                metadata={"seed": 0},
            )
            self.assertTrue(expression["orientation"].exists())
            self.assertTrue(grn["differential"].exists())
            columns = pd.read_csv(expression["metrics"]).columns.tolist()
            self.assertEqual(columns, SABER_CSV_FIELDS + ["seed"])


if __name__ == "__main__":
    unittest.main()
