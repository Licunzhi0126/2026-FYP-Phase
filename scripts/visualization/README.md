# HyperPhase Figure 2 workflow

The one-command entry point remains outside this directory:

```text
scripts/
├── run_figure2_visualization.py
└── visualization/
    ├── hyperphase_adapter.py
    ├── figure2_io.py
    ├── figure2_metrics.py
    ├── figure2_style.py
    ├── figure2_expression_benchmark.py
    └── figure2_grn_benchmark.py
```

No source file under `Phasehyper-main/phasehyper/` is changed.  The adapter
imports and trains the existing:

- `phasehyper.model.HyperPhaseModel`
- `phasehyper.model.SetCriterion`
- `phasehyper.model.build_optimizer`

The three controls are called directly from
`Phasehyper-main/phasehyper/evaluation/saber.py`:

- `NMF2Factor`
- `RandomSplit`
- `MeanFractionShrinkage`

Older preview helpers remain available for compatibility but are not displayed
in the formal HyperPhase figures.

## Strict truth boundary

HyperPhase receives only:

- observed total expression for GSE45719 and GSE80810;
- independently thresholded combined GRNs for the simulation benchmark.

The model-axis gene selection is based on observed total expression only.
Allelic read-support masks, C57/CAST values, maternal/paternal values, global
orientation and every truth-based metric are applied only after prediction.

For the GRN data, `combined` is not assumed to equal the binary parental union.
The empirical mismatch count and match rate are written to metadata and the data
summary.  Parental union/shared labels are derived only for post-hoc scoring.

## Server paths

Default data root:

```text
/home/jovyan/public/datasets/PHASE
```

Expected inputs:

```text
/home/jovyan/public/datasets/PHASE/GSE45719
/home/jovyan/public/datasets/PHASE/GSE80810
/home/jovyan/public/datasets/PHASE/per_cell_threshold_0.1
```

The default output is resolved from the project location:

```text
/home/jovyan/work/2026 phase/output
```

Set `PHASE_DATA_ROOT` or pass explicit command-line paths to override the data
root.

## Conda environment

Create the scientific environment:

```bash
conda create -n phase -c conda-forge \
  python=3.10 numpy pandas scipy scikit-learn matplotlib \
  xlrd openpyxl -y
conda activate phase
```

Install PyTorch.  For a CPU server:

```bash
conda install -c pytorch pytorch cpuonly -y
```

For an NVIDIA server, use the CUDA version supported by the server driver, for
example:

```bash
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
```

Verify:

```bash
python -c "import torch, numpy, pandas, scipy, sklearn, matplotlib, xlrd; print(torch.__version__, torch.cuda.is_available())"
```

## One-command run

```bash
conda activate phase
cd "/home/jovyan/work/2026 phase"
python scripts/run_figure2_visualization.py --device auto
```

Default training settings match the supplied HyperPhase report:

```text
expression epochs: 250
GRN epochs:         80
expression genes:  120
GRN edges:          300
```

Useful controls:

```bash
# Replace an incompatible/stale cached model output.
python scripts/run_figure2_visualization.py --device auto --force-model

# Reuse already generated HyperPhase matrices without fitting again.
python scripts/run_figure2_visualization.py --skip-model-fit

# Run only one benchmark.
python scripts/run_figure2_visualization.py --skip-simulationdata
python scripts/run_figure2_visualization.py --skip-answerdata
```

## Outputs

```text
output/
├── answerdata/
│   ├── answerdata_figure2_hyperphase.png
│   ├── answerdata_figure2_hyperphase.pdf
│   ├── answerdata_metrics.csv
│   ├── answerdata_data_summary.csv
│   ├── answerdata_hyperphase_arrays.npz
│   └── hyperphase_outputs/
│       ├── GSE45719/
│       │   ├── phase_A.csv
│       │   ├── phase_B.csv
│       │   ├── training_history.csv
│       │   └── metadata.json
│       └── GSE80810/
│           └── ...
└── simulationdata/
    ├── simulationdata_figure2_hyperphase.png
    ├── simulationdata_figure2_hyperphase.pdf
    ├── simulationdata_metrics.csv
    ├── simulationdata_data_summary.csv
    ├── simulationdata_hyperphase_arrays.npz
    └── hyperphase_outputs/
        ├── hyperphase_grn_predictions.npz
        ├── training_history.csv
        └── metadata.json
```

The evaluator permits one global channel swap per context because the two
latent phases are exchangeable.  It never performs per-cell or per-gene
truth-guided swaps.
