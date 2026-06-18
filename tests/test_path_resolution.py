import tempfile
import unittest
from pathlib import Path

from tv_phase.config import resolve_dataset_root, resolve_project_paths


class PathResolutionTests(unittest.TestCase):
    def test_explicit_roots_override_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            data = Path(tmp) / "datasets"
            output = Path(tmp) / "results"
            resolved = resolve_project_paths(project, data, output)
            self.assertEqual(resolved, (project.resolve(), data.resolve(), output.resolve()))
            self.assertEqual(
                resolve_dataset_root("simulation0616_expr_position", data),
                data.resolve() / "simulation_0616_tv_phase" / "expr_position",
            )


if __name__ == "__main__":
    unittest.main()

