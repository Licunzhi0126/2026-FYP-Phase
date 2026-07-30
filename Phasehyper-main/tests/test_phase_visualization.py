import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from phasehyper.visualization.phase import run_phase_visualization
from phasehyper.visualization.phase.allocation import compute_phase_allocation
from phasehyper.visualization.phase.loader import load_phase_visualization_bundle


class PhaseVisualizationTests(unittest.TestCase):
    def _write_result(self, root: Path):
        rng = np.random.default_rng(123)
        n_cells, n_genes = 24, 15
        cells = [f"cell_{i:02d}" for i in range(n_cells)]
        genes = [f"gene_{i:02d}" for i in range(n_genes)]
        labels = np.repeat([0, 1, 2], 8)
        label_names = [f"group_{x}" for x in labels]
        raw = rng.gamma(2, 1, (n_cells, n_genes))
        raw[:8, :5] += 2
        raw[8:16, 5:10] += 2
        phase_a = raw * rng.uniform(0.25, 0.55, (n_cells, n_genes))
        phase_b = raw - phase_a
        cell_h = np.column_stack([
            labels + rng.normal(0, 0.1, n_cells),
            rng.normal(size=(n_cells, 4)),
        ])
        axes = {"index": pd.Index(cells, name="cell_id"), "columns": genes}
        pd.DataFrame(phase_a, **axes).to_csv(root / "phase_A.csv")
        pd.DataFrame(phase_b, **axes).to_csv(root / "phase_B.csv")
        pd.DataFrame(
            cell_h, index=axes["index"], columns=[f"h_{i}" for i in range(cell_h.shape[1])]
        ).to_csv(root / "cell_h.csv")
        metrics = {
            name: {"NMI": 0.7, "FMI": 0.65, "ARI": 0.6, "ASW": 0.4, "PredClusters": 3}
            for name in ("Raw_RNA", "cell_h", "Phase_A", "Phase_B")
        }
        metrics.update(best_epoch=2, final_loss=0.4)
        (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (root / "config.json").write_text(json.dumps({"dataset": "synthetic"}), encoding="utf-8")
        pd.DataFrame({
            "cell_id": cells, "label_id": labels, "label_name": label_names
        }).to_csv(root / "cell_metadata.csv", index=False)
        pd.DataFrame({
            "channel": "directed",
            "edge_type": ["pathway_module", "grn_activate", "proximity_200kb"],
            "gate": [0.88, 0.72, 0.94],
        }).to_csv(root / "edge_gates.csv", index=False)
        summary_rows = []
        for channel in ("directed", "undirected"):
            for edge_type in ("pathway_module", "grn_activate", "proximity_200kb"):
                summary_rows.append({
                    "channel": channel, "edge_type": edge_type, "n_edges": 12,
                    "node_coverage": 0.7, "candidate_count": 14,
                    "dropped_duplicate_count": 1, "dropped_invalid_count": 1,
                })
        pd.DataFrame(summary_rows).to_csv(root / "edge_summary.csv", index=False)
        pd.DataFrame({
            "epoch": [1, 2, 3, 4],
            "loss": [1.0, 0.4, 0.5, 0.45],
            "cyc_comp": [1.0, 0.8, 0.7, 0.65],
            "barlow": [0.8, 0.6, 0.5, 0.45],
            "compartment": [0.9, 0.7, 0.6, 0.55],
            "orthogonality": [0.7, 0.5, 0.4, 0.35],
            "info_nce": [1.2, 1.0, 0.9, 0.85],
            "gate_regularization": [0.1, 0.09, 0.08, 0.07],
            "phase_cosine": [0.5, 0.4, 0.3, 0.25],
            "asym_scale": [0.1, 0.15, 0.2, 0.22],
        }).to_csv(root / "training_history.csv", index=False)
        pd.DataFrame({
            "gene_id": genes,
            "chromosome": [f"chr{1 + i // 5}" for i in range(n_genes)],
            "TSS": [1000 + (i % 5) * 100_000 for i in range(n_genes)],
            "local_gene_density": [4] * n_genes,
            "is_TF": [int(i % 4 == 0) for i in range(n_genes)],
        }).to_csv(root / "gene_annotation.csv", index=False)
        modules = [
            {"module": f"module_{i // 5}", "gene": gene}
            for i, gene in enumerate(genes)
        ]
        pd.DataFrame(modules).to_csv(root / "pathway_membership.csv", index=False)
        pd.DataFrame(modules).to_csv(root / "ppi_membership.csv", index=False)
        edge_rows = []
        for channel in ("directed", "undirected"):
            for edge_type in ("pathway_module", "grn_activate", "proximity_200kb"):
                for i, gene in enumerate(genes):
                    edge_rows.append({
                        "gene": gene, "channel": channel, "edge_type": edge_type,
                        "edge_id": f"{channel}:{edge_type}:{i // 3}", "edge_weight": 1 + i / 20,
                    })
        pd.DataFrame(edge_rows).to_csv(root / "hyperedge_membership.csv", index=False)
        return raw, labels, label_names, cells, genes

    def test_allocation_is_bounded_and_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, labels, names, cells, genes = self._write_result(root)
            bundle = load_phase_visualization_bundle(
                root, raw_rna=raw, labels=labels, label_names=names,
                cell_ids=cells, genes=genes,
            )
            allocation = compute_phase_allocation(bundle)
            self.assertEqual(allocation["cell_id"].tolist(), cells)
            self.assertTrue(allocation["allocation_score"].between(-1, 1).all())
            self.assertTrue((allocation[["energy_A", "energy_B"]] >= 0).all().all())

    def test_complete_pipeline_writes_png_and_source_data_without_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, labels, names, cells, genes = self._write_result(root)
            result = run_phase_visualization(
                root, raw_rna=raw, labels=labels, label_names=names,
                cell_ids=cells, genes=genes, dpi=45, top_genes=10,
            )
            output = root / "visualization"
            self.assertIn(result["status"], {"success", "partial"})
            self.assertTrue((output / "overview" / "01_four_representation_pca.png").exists())
            self.assertTrue((output / "phase" / "07_gene_resolution_atlas.png").exists())
            self.assertTrue((output / "source_data" / "gene_resolution_metrics.csv").exists())
            self.assertFalse(any(output.rglob("*.svg")))
            self.assertFalse((output / "visualization_report.json").exists())

    def test_one_figure_failure_does_not_cancel_later_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, labels, names, cells, genes = self._write_result(root)
            with mock.patch(
                "phasehyper.visualization.phase.pipeline._task_metrics",
                side_effect=RuntimeError("synthetic metric plot failure"),
            ):
                result = run_phase_visualization(
                    root, raw_rna=raw, labels=labels, label_names=names,
                    cell_ids=cells, genes=genes, dpi=35, top_genes=8,
                )
            self.assertEqual(
                result["failed"]["02_representation_metrics"],
                "RuntimeError: synthetic metric plot failure",
            )
            self.assertTrue(
                (root / "visualization" / "phase" / "03_phase_allocation_violin.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
