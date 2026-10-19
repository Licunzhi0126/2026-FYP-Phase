# V7 imprinting-fixed organized dataset

This package separates fit-side inputs from answer-side simulator truth.

## Directory layout

```text
V7_imprinting_fixed_organized/
├── input/
│   ├── combined_counts.csv
│   ├── combined_true_expression.csv
│   ├── cell_metadata.csv
│   ├── gene_info.csv
│   ├── chromosome_info.csv
│   ├── chromatin_regions.csv
│   ├── shared_TF_info.csv
│   └── shared_priors/
│       ├── regulatory_elements.csv
│       └── TF_motif_matrix.csv
├── ground_truth/
│   ├── allele_expression/
│   ├── imprinting/
│   ├── metadata/
│   ├── chromatin/
│   ├── grn/
│   │   └── per_cell/{paternal,maternal,combined}/
│   ├── simulator_latent/
│   └── derived_oracle_priors/
└── MANIFEST.csv
```

## Important leakage notes

- `input/gene_info.csv` is sanitized: imprinting status/strength and paternal/maternal imprinting gates were removed. The full original is under `ground_truth/imprinting/gene_info_full.csv`.
- `input/chromatin_regions.csv` is sanitized: `allelic_state` was removed. The full table is under `ground_truth/chromatin/`.
- `input/cell_metadata.csv` is sanitized: simulator `cell_type` was removed.
- `combined_true_expression.csv` is retained in `input/` only for compatibility with the older noiseless simulation layout. It is total expression and does not identify paternal vs maternal alleles, but it is simulator latent truth. For a hardened/no-oracle benchmark, use `combined_counts.csv` (or a separately prepared observed noise layer) instead.
- Original pathway/PPI files are **not** treated as strict inputs because their generator reads `combined_true_expression.csv`, `gene_module_membership.csv`, and `GRN_edge_table.csv`; they are therefore stored under `ground_truth/derived_oracle_priors/`.
- `combined_GRN_gene2gene.csv` and `per_cell/combined/` are also kept on the ground-truth/privileged side because they are derived from simulator GRN truth.

## Intended use

- Training / fitting: read only from `input/` unless you are deliberately running a privileged ablation.
- Evaluation / diagnostics: read from `ground_truth/` only after fitting is complete.
- See `MANIFEST.csv` for the classification of every copied file and the reason.
