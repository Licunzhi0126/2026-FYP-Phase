import tempfile
import unittest
from pathlib import Path

from tv_phase.output import make_run_output_layout


class OutputLayoutTests(unittest.TestCase):
    def test_only_five_top_level_directories_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            layout = make_run_output_layout(root)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["config", "figures", "logs", "plot_data", "tables"],
            )
            self.assertTrue(layout.figures_chromosomes.is_dir())


if __name__ == "__main__":
    unittest.main()

