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

## Main API

```python
from tv_phase import PhaseTrainingConfig, run_hgnn_vae_phase_end2end

config = PhaseTrainingConfig(data_name="sc_GEM", train_epochs=1, device="cpu")
result = run_hgnn_vae_phase_end2end(config, version_name="TV-PHASE_v11")
```
