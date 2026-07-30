# Figure 2 visualization workflow

This directory contains the reusable loaders, metrics, plotting code, and
dataset-specific benchmark programs.  The one-command entry point is kept one
level above this directory:

```text
scripts/
├── run_figure2_visualization.py
└── visualization/
    ├── figure2_io.py
    ├── figure2_metrics.py
    ├── figure2_style.py
    ├── figure2_expression_benchmark.py
    └── figure2_grn_benchmark.py
```

## Approved inputs

Real expression data are read from:

```text
data/answerdata/GSE45719/
data/answerdata/GSE80810/
```

For GSE45719, `RPKM` is the observed total and the `C57_hits` / `CAST_hits`
columns define the two reference-strain channels.  Entries below the configured
allelic-read threshold are retained for total-expression visualization but
excluded from truth-based scoring.

For GSE80810, the RPRT workbook is the observed total, the allelic-ratio
workbook supplies the paternal fraction, and the count workbook supplies the
read-support mask.  The old `.xls` format requires `xlrd`.

Thresholded per-cell GRNs are read from:

```text
data/per_cell_threshold_0.1/combined/
data/per_cell_threshold_0.1/maternal/
data/per_cell_threshold_0.1/paternal/
```

The three thresholded channels are loaded independently.  In particular,
`combined` is **not** assumed to equal the binary union of `maternal` and
`paternal`.  The data summary records their empirical mismatch count and match
rate.  A parental union/shared label is derived only inside the support-recovery
panel, where a deterministic 20% cell subset is held out for display.

## Outputs

The main runner writes two independent output trees:

```text
output/
├── answerdata/
│   ├── answerdata_figure2.png
│   ├── answerdata_figure2.pdf
│   ├── answerdata_metrics.csv
│   ├── answerdata_data_summary.csv
│   └── answerdata_primary_arrays.npz
└── simulationdata/
    ├── simulationdata_figure2.png
    ├── simulationdata_figure2.pdf
    ├── simulationdata_metrics.csv
    ├── simulationdata_data_summary.csv
    └── simulationdata_primary_arrays.npz
```

The built-in NMF, state, rank, random, and equal splits are observation-only
preview controls.  They test the data-to-figure pipeline but are not PhaseHyper
results.  If no trained-model predictions are supplied, `NMF2 preview` is the
default method displayed in panels A, D, and E.

## Environment and one-command run

From Anaconda Prompt:

```bat
conda activate phase
conda install -c conda-forge xlrd
cd /d "C:\Users\27139\OneDrive\Desktop\Huge Workplace\2027 FYP Phase"
python scripts\run_figure2_visualization.py
```

From PowerShell after Conda has initialized the shell:

```powershell
conda activate phase
conda install -c conda-forge xlrd
Set-Location "C:\Users\27139\OneDrive\Desktop\Huge Workplace\2027 FYP Phase"
python scripts\run_figure2_visualization.py
```

Run only one side when needed:

```powershell
python scripts\run_figure2_visualization.py --skip-simulationdata
python scripts\run_figure2_visualization.py --skip-answerdata
```

## Connecting trained-model predictions

Expression predictions may be CSV, NPY, or NPZ cell-by-gene matrices.  Both
channels are required for each connected dataset:

```powershell
python scripts\run_figure2_visualization.py `
  --gse45719-pred-a "path\to\gse45719_channel_a.csv" `
  --gse45719-pred-b "path\to\gse45719_channel_b.csv" `
  --gse80810-pred-a "path\to\gse80810_channel_a.csv" `
  --gse80810-pred-b "path\to\gse80810_channel_b.csv"
```

GRN predictions use one NPZ containing `pred_A` and `pred_B`.  Arrays may
already be cell-by-selected-edge, may be full cell-by-gene-by-gene matrices, or
may include a matching `edge_index` array:

```powershell
python scripts\run_figure2_visualization.py `
  --grn-pred-npz "path\to\grn_predictions.npz"
```

The evaluator permits one global A/B channel swap per context because the two
latent output channels are exchangeable.  It does not perform per-cell or
per-gene truth-guided swapping.
