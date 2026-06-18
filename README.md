# TV-PHASE Project

Projectized TV-PHASE v11 code. The package reads datasets from `data/` by default and writes outputs under `output/` unless an output directory is supplied.

## Run a smoke experiment

```powershell
conda run -n phase python -m pip install -e . --no-deps
conda run -n phase python -m tv_phase.experiment --dataset_types sim_gene100_alpha_1_beta_1 --cluster_methods kmeans --train-epochs 1 --output-dir output/smoke_experiment
```

## Generate simulation data

`tv_phase.simulation` reproduces the data-generation logic from `simulation_v4.ipynb`,
then adapts the raw outputs into the file layout expected by TV-PHASE.

Use a new output directory first so existing datasets are not overwritten:

```powershell
conda run -n phase python -m tv_phase.simulation --alpha 1 --beta 1 --raw-output-dir data/simulation_raw/alpha_1_beta_1_trial --tv-phase-output-dir data/sim_gene100_generated/alpha_1_beta_1 --overwrite
```

To train on that generated directory without editing the static dataset registry, register
it in-process and call the main pipeline:

```powershell
conda run -n phase python -c "from pathlib import Path; from tv_phase.config import DATASET_CONFIG, DATA_ROOT, PhaseTrainingConfig; from tv_phase.pipeline import run_hgnn_vae_phase_end2end; DATASET_CONFIG['sim_gene100_generated_alpha_1_beta_1']={'name':'sim_gene100_generated_alpha_1_beta_1','description':'Generated simulation data','root':DATA_ROOT/'sim_gene100_generated'/'alpha_1_beta_1','files':{'expression':'expression_data.csv','view':[],'stage':'cell_stage.csv','kegg_prior':'kegg_prior.txt','poswin_prior':'poswin_prior.txt','ppi_prior':'ppi_prior.csv'},'has_ppi':False,'have_answer':True}; run_hgnn_vae_phase_end2end(PhaseTrainingConfig(data_name='sim_gene100_generated_alpha_1_beta_1', train_epochs=1, output_dir=Path('output/generated_smoke'), device='cpu', feature_dim=64, hidden_dim=64, latent_dim=16), version_name='TV-PHASE_v11')"
```

When you intentionally want to replace a registered dataset such as
`data/sim_gene100/alpha_1_beta_1`, pass that path as `--tv-phase-output-dir` together with
`--overwrite`.

## Run the full comparison CLI

```powershell
conda run -n phase python -m tv_phase.experiment
```

## Prepare simulation_0616

```powershell
conda run -n phase python scripts/prepare_simulation0616.py `
  --input-root data/simulation_0616 `
  --output-root data/simulation_0616_tv_phase `
  --overwrite
```

The adapter registers four independent datasets: `simulation0616_expr_position`,
`simulation0616_expr_position_kegg`, `simulation0616_ratio_position`, and
`simulation0616_ratio_position_kegg`.

## Run the prior ablation

```powershell
conda run -n phase python -m tv_phase.experiment `
  --dataset-types PEA_STA `
  --prior-builders dataset none p_glue p_denoise `
  --cluster-methods kmeans leiden louvain `
  --data-root data `
  --output-root output/prior_ablation_v1 `
  --device cuda
```

Each run writes only `figures/`, `plot_data/`, `tables/`, `logs/`, and `config/`
at its top level. Pass `--legacy-output` only when the previous deep output tree is required.

Runtime roots may also be supplied through `TV_PHASE_PROJECT_ROOT`,
`TV_PHASE_DATA_ROOT`, and `TV_PHASE_OUTPUT_ROOT`.

## Main API

```python
from tv_phase import PhaseTrainingConfig, run_hgnn_vae_phase_end2end

config = PhaseTrainingConfig(data_name="sc_GEM", train_epochs=1, device="cpu")
result = run_hgnn_vae_phase_end2end(config, version_name="TV-PHASE_v11")
```
