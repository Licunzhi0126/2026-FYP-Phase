import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from phasehyper.visualization.figure3_simulation import build_figure3_data
from phasehyper.visualization.simulation_diagnostics import load_simulation_bundle
from phasehyper.visualization.simulation_imbalance import signed_imbalance
from phasehyper.visualization.simulation_pipeline import (
    VisualizationError,
    run_simulation_visualization,
)


class SimulationVisualizationTests(unittest.TestCase):
    def _write_fixture(self, root: Path):
        sim_dir = root / "simulation_data"
        result_dir = root / "result_simulation"
        (sim_dir / "input").mkdir(parents=True)
        (sim_dir / "groundtruth").mkdir(parents=True)
        (result_dir / "expression").mkdir(parents=True)
        (result_dir / "grn").mkdir(parents=True)

        rng = np.random.default_rng(42)
        cells = [f"cell_{index:02d}" for index in range(8)]
        genes = [f"gene_{index:02d}" for index in range(6)]
        maternal = rng.uniform(0.2, 1.5, (len(cells), len(genes)))
        paternal = rng.uniform(0.2, 1.5, (len(cells), len(genes)))
        combined = maternal + paternal
        axes = {"index": cells, "columns": genes}
        pd.DataFrame(combined, **axes).to_csv(
            sim_dir / "input" / "combined_true_expression.csv",
            index_label="cell_id",
        )
        pd.DataFrame(maternal, **axes).to_csv(
            sim_dir / "groundtruth" / "maternal_true_expression.csv",
            index_label="cell_id",
        )
        pd.DataFrame(paternal, **axes).to_csv(
            sim_dir / "groundtruth" / "paternal_true_expression.csv",
            index_label="cell_id",
        )
        # Deliberately swapped: the bundle must make one global correction.
        pd.DataFrame(paternal, **axes).to_csv(
            result_dir / "expression" / "phase_A.csv",
            index_label="cell_id",
        )
        pd.DataFrame(maternal, **axes).to_csv(
            result_dir / "expression" / "phase_B.csv",
            index_label="cell_id",
        )
        pd.DataFrame(
            {
                "gene_id": genes,
                "chromosome": ["chr1"] * 3 + ["chr2"] * 3,
                "start": [30, 10, 20, 5, 25, 15],
                "end": [35, 15, 25, 10, 30, 20],
            }
        ).to_csv(sim_dir / "input" / "gene_info.csv", index=False)
        pd.DataFrame(
            {
                "cell_id": cells,
                "cell_type": ["type_1"] * 4 + ["type_2"] * 4,
            }
        ).to_csv(sim_dir / "input" / "cell_metadata.csv", index=False)

        expression_rows = []
        for method, scale in (
            ("phasehyper", 0.7),
            ("RandomSplit", 0.2),
            ("NMF2Factor", 0.1),
            ("[trivial] combined/2", 0.0),
        ):
            expression_rows.append(
                {
                    "method": method,
                    "pcc_global": scale,
                    "pcc_cell": scale,
                    "pcc_A": scale,
                    "pcc_B": scale,
                    "imb": scale,
                    "imb_gene": scale,
                    "seed": 3,
                }
            )
        pd.DataFrame(expression_rows).to_csv(
            result_dir / "expression" / "metrics.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "name": method,
                    "pcc": scale,
                    "pcc_cell": scale,
                    "spearman": scale,
                    "auroc": 0.5 + max(scale, 0) / 2,
                    "skill_cal": scale,
                }
                for method, scale in (
                    ("phasehyper", 0.2),
                    ("RandomSplit", 0.05),
                    ("NMF2Factor", 0.08),
                )
            ]
        ).to_csv(result_dir / "grn" / "differential.csv", index=False)

        edge_index = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
        combined_edges = rng.normal(size=(len(cells), len(edge_index)))
        true_a = combined_edges / 2 + rng.normal(0, 0.1, combined_edges.shape)
        true_b = combined_edges - true_a
        np.savez_compressed(
            result_dir / "grn" / "edges.npz",
            edge_index=edge_index,
            combined=combined_edges,
            pred_A=true_a,
            pred_B=true_b,
            true_A=true_a,
            true_B=true_b,
            genes=np.array(genes, dtype=object),
            cell_type=np.array(["type_1"] * 4 + ["type_2"] * 4, dtype=object),
        )
        return sim_dir, result_dir, maternal, paternal, combined

    def test_bundle_alignment_and_global_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            sim_dir, result_dir, maternal, _, _ = self._write_fixture(Path(directory))
            bundle = load_simulation_bundle(sim_dir, result_dir)
            self.assertEqual(bundle.phase_mapping["n_swapped"], 1)
            self.assertTrue(np.allclose(bundle.pred_maternal, maternal))
            self.assertEqual(
                bundle.genes_for_chromosome("chr1"),
                ["gene_01", "gene_02", "gene_00"],
            )

    def test_signed_imbalance_definition(self):
        first = np.array([[3.0, -2.0]])
        second = np.array([[1.0, -6.0]])
        actual = signed_imbalance(first, second)
        expected = np.array([[0.5, 0.5]])
        self.assertTrue(np.allclose(actual, expected))

    def test_figure3c_uses_combined_half_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            sim_dir, result_dir, maternal, paternal, combined = self._write_fixture(
                Path(directory)
            )
            data = build_figure3_data(load_simulation_bundle(sim_dir, result_dir))
            expected = 0.5 * (
                ((combined / 2 - maternal) ** 2).mean(axis=0)
                + ((combined / 2 - paternal) ** 2).mean(axis=0)
            )
            self.assertTrue(
                np.allclose(data.gene_level["combined_half_mse"], expected)
            )

    def test_complete_render_stays_inside_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            sim_dir, result_dir, _, _, _ = self._write_fixture(Path(directory))
            outputs = run_simulation_visualization(
                sim_dir, result_dir, dpi=30, genes_to_plot=["gene_00"]
            )
            output_root = result_dir / "visualization"
            pngs = list(result_dir.rglob("*.png"))
            self.assertTrue(pngs)
            self.assertTrue(
                all(output_root.resolve() in path.resolve().parents for path in pngs)
            )
            expected = [
                output_root / "summary" / "expression_metrics.png",
                output_root
                / "chromosomes"
                / "chr1"
                / "chr1_phase_expression_heatmap.png",
                output_root / "genome" / "all_chromosomes_imbalance_heatmap.png",
                output_root / "genes" / "gene_00_detail.png",
                output_root / "figure3" / "fig3C_paired_gene_mse.png",
            ]
            self.assertTrue(all(path.exists() for path in expected))
            self.assertGreaterEqual(len(outputs), len(pngs))

    def test_one_figure_failure_does_not_cancel_others(self):
        with tempfile.TemporaryDirectory() as directory:
            sim_dir, result_dir, _, _, _ = self._write_fixture(Path(directory))
            with mock.patch(
                "phasehyper.visualization.simulation_pipeline.plot_expression_metrics",
                side_effect=RuntimeError("synthetic failure"),
            ):
                with self.assertRaises(VisualizationError) as raised:
                    run_simulation_visualization(
                        sim_dir,
                        result_dir,
                        dpi=30,
                        make_chromosome=False,
                        make_imbalance=False,
                        make_figure3=False,
                    )
            self.assertIn("summary/expression_metrics", str(raised.exception))
            self.assertTrue(
                (result_dir / "visualization" / "summary" / "grn_metrics.png").exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "visualization"
                    / "summary"
                    / "grn_decomposition.png"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
