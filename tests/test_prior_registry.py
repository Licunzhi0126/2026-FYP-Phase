import unittest

import numpy as np
import pandas as pd

from tv_phase.config import DatasetBundle
from tv_phase.prior import PriorConfig, build_prior, list_prior_builders


class PriorRegistryTests(unittest.TestCase):
    def _dataset(self):
        expression = pd.DataFrame(
            np.random.default_rng(42).normal(size=(12, 4)),
            index=[f"cell_{i}" for i in range(12)],
            columns=[f"gene_{i}" for i in range(4)],
        )
        return DatasetBundle(
            dataset_type="PEA_STA",
            view1_name="test",
            view1_dfs=[],
            expression_df=expression,
            common_cells=expression.index.tolist(),
            common_genes=expression.columns.tolist(),
            labels=np.zeros(12, dtype=int),
            label_names=["type"] * 12,
            label_map={0: "type"},
        )

    def test_registered_builders(self):
        self.assertEqual(list_prior_builders(), ["dataset", "none", "p_denoise", "p_glue"])

    def test_none_builder_returns_empty_bundle(self):
        dataset = DatasetBundle(
            dataset_type="PEA_STA",
            view1_name="test",
            expression_df=pd.DataFrame([[1.0]], index=["cell"], columns=["gene"]),
            common_cells=["cell"],
            common_genes=["gene"],
            labels=np.asarray([0]),
            label_names=["type"],
            label_map={0: "type"},
        )
        bundle = build_prior(".", dataset, PriorConfig(name="none"))
        self.assertEqual(bundle.kegg_groups, {})
        self.assertEqual(bundle.poswin_groups, {})
        self.assertEqual(bundle.metadata["prior_name"], "none")

    def test_data_driven_builders_execute_independently(self):
        dataset = self._dataset()
        glue = build_prior(".", dataset, PriorConfig(name="p_glue", top_k=1, max_features=4))
        denoise = build_prior(
            ".",
            dataset,
            PriorConfig(
                name="p_denoise",
                max_features=4,
                denoise_candidate_top_k=1,
                denoise_epochs=1,
                denoise_top_percent=1.0,
            ),
        )
        self.assertEqual(glue.metadata["prior_name"], "p_glue")
        self.assertEqual(denoise.metadata["prior_name"], "p_denoise")
        self.assertGreater(len(glue.data_groups), 0)
        self.assertGreater(len(denoise.data_groups), 0)


if __name__ == "__main__":
    unittest.main()
