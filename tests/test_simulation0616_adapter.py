import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tv_phase.simulation_adapter import adapt_simulation0616_case_to_tv_phase


class Simulation0616AdapterTests(unittest.TestCase):
    def test_small_case_is_adapted_without_legacy_position_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "case" / "output"
            input_dir = raw / "input"
            truth_dir = raw / "ground_truth"
            input_dir.mkdir(parents=True)
            truth_dir.mkdir(parents=True)
            expression = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["c1", "c2"], columns=["g1", "g2"])
            expression.index.name = "cell_id"
            expression.to_csv(input_dir / "mixed_expression.csv")
            (expression * 0.6).to_csv(truth_dir / "paternal_expression.csv")
            (expression * 0.4).to_csv(truth_dir / "maternal_expression.csv")
            pd.DataFrame(0.6, index=expression.index, columns=expression.columns).to_csv(truth_dir / "mixing_proportions.csv")
            pd.DataFrame({"cell_id": ["c1", "c2"], "cell_type": [1, 2]}).to_csv(input_dir / "cell_info.csv", index=False)
            pd.DataFrame(
                {
                    "gene_id": ["g1", "g2"],
                    "chromosome": [1, 1],
                    "start_pos": [1, 100],
                    "end_pos": [50, 150],
                    "pathway": ["p1", "p1"],
                }
            ).to_csv(input_dir / "gene_info.csv", index=False)
            pd.DataFrame({"gene1": ["g1"], "gene2": ["g2"], "weight": [-0.7]}).to_csv(
                input_dir / "synthetic_expression_ppi.csv", index=False
            )
            output = root / "adapted"
            paths = adapt_simulation0616_case_to_tv_phase("expr_position", raw.parent, output)
            for path in paths.values():
                self.assertTrue(path.exists(), path)
            self.assertFalse((output / "gene_positions_pea.txt").exists())
            self.assertEqual((output / "cell_stage.csv").read_text(encoding="utf-8"), "1,2")


if __name__ == "__main__":
    unittest.main()
