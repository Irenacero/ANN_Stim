# Personalized brain models and virtual perturbation

Code and compact analysis outputs accompanying the manuscript on personalized
whole-brain models of resting-state fMRI and in-silico perturbation.

The repository contains:

- the cleaned resting-state analysis pipeline used to train personalized neural
  network models and estimate time-resolved effective connectivity (EC) and
  bifocal effective connectivity (BEC);
- the task-fMRI validation analyses;
- the scripts used to generate the manuscript figure panels;
- compact downstream results and figure caches so that the figures can be
  reproduced without distributing the very large intermediate EC/BEC tensors.

The repository supports two modes of use:

1. **Quick figure reproduction** using the compact outputs included here.
2. **Full analysis reconstruction** starting from the original HCP data.

The final manuscript figures were assembled manually from exported panels
(e.g. in Inkscape), so the scripts reproduce the scientific panels rather than
the exact final multi-panel page composition.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── manuscript_preprint.pdf
└── codes/
    ├── HCP/
    │   ├── data/
    │   │   ├── Receptor_maps/
    │   │   ├── Task/
    │   │   └── rsn_names.mat
    │   ├── notebooks_original/
    │   ├── results/
    │   ├── scripts/
    │   ├── src/
    │   └── tests/
    └── figures/
        ├── _atlases/
        └── *.py
```

`codes/HCP/notebooks_original/` contains the original analysis notebooks for
provenance.

`codes/HCP/scripts/` contains the cleaned, portable pipeline used for
reproducibility and validation.

`codes/figures/outputs/` is generated locally and is intentionally not tracked
by Git.

---

# 1. Resting-state analysis pipeline

The cleaned resting-state pipeline is in `codes/HCP/scripts/`.

| Script | Main role | Main output |
|---|---|---|
| `01_preprocess_rest.py` | Preprocess the four HCP resting-state runs and construct ANN inputs/targets | processed signals, inputs and targets |
| `02_train_models.py` | Train one personalized feedforward ANN per participant | `ANN_model/id_<subject>_MLP.pt` |
| `03_connectivity.py` | Apply focal and bifocal perturbations to the trained models | time-resolved EC and BEC tensors |
| `04_background_dependence.py` | Relate ongoing state to perturbation response | long-format EC/BEC state-dependence dataframes |
| `05_background_dependence_summary.py` | Summary analyses of background/state dependence | subject/node summary statistics and diagnostics |
| `06_spatial_hierarchy_network_receptors_dataframe.py` | Add network, hierarchy, gradient, connectivity-strength and receptor information | final cortical spatial dataframe |

The full dependency chain is:

```text
raw HCP resting-state fMRI
        ↓
01_preprocess_rest.py
        ↓
processed resting-state activity
        ↓
02_train_models.py
        ↓
personalized ANN models
        ↓
03_connectivity.py
        ↓
time-resolved EC / BEC
        ↓
04_background_dependence.py
        ↓
state-dependence dataframe
        ↓
05_background_dependence_summary.py
        ↓
06_spatial_hierarchy_network_receptors_dataframe.py
        ↓
spatial / hierarchy / receptor dataframe
        ↓
figure-specific compact summaries
        ↓
manuscript figure scripts
```

## Resting-state input data

To reconstruct the resting-state analysis from scratch, the original HCP
resting-state data are required. They are not redistributed here.

`01_preprocess_rest.py` expects the external data root through:

```bash
export HCP_REST_ROOT=/path/to/HCP/rest/data
```

The analysis uses the four resting-state acquisitions:

```text
REST1_LR
REST1_RL
REST2_LR
REST2_RL
```

The analysis uses 450 regions: 400 Schaefer cortical parcels and 50 Tian
subcortical regions.

The cleaned code preserves the numerical parameters of the original analysis.
In particular, ANN training uses a 0.90 training proportion, as in the original
code.

---

# 2. Large intermediate outputs and distributed data

The complete time-resolved EC/BEC tensors and some downstream tables are very
large and are therefore intentionally **not distributed in the Git repository**.

For a full 100-participant reconstruction, the following are generated locally:

```text
codes/HCP/results/ECts/
codes/HCP/results/BECts/

codes/HCP/results/dataframes/
    HCP_4_df_background_dependence_ECts.csv
    HCP_4_df_background_dependence_BECts.csv
```

The full step-4 EC and BEC dataframes are each approximately 4 GB.

Instead, the repository contains downstream tables and compact caches sufficient
to reproduce the reported figures and group-level results without storing the
full intermediate tensors.

### Included example processed data

The following example participant data are included because they are required
by example and null-model figure analyses:

```text
codes/HCP/results/processed/
    id_100206_signals.npy
    id_100206_inputs.npy
    id_100206_targets.npy
```

The complete processed dataset for all participants is not distributed.

### Included ANN model

```text
codes/HCP/results/ANN_model/id_108222_MLP.pt
```

This model is required by the Figure 2 example free-running simulation.

The complete collection of 100 trained ANN models is not distributed.

### Compact example EC cache

```text
codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz
```

This provides the example EC information required by figure scripts without
including the approximately 0.8 GB full time-resolved EC tensor for the example
participant.

---

# 3. Distributed downstream analysis products

The final EC spatial dataframe is distributed in both CSV and pickle format:

```text
codes/HCP/results/dataframes/
    HCP_5_df_spatial_network_receptors_ECts_cortical400.csv
    HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl
```

It contains 40,000 rows:

```text
100 participants × 400 cortical regions
```

and includes network identity, perturbation-response measures, state-dependence
statistics, hierarchy, trophic coherence, principal gradient, EC in/out strength
and receptor information.

The pickle file is a figure-ready serialization generated directly from the CSV.

The full BEC step-6 dataframe is not distributed because the manuscript BEC
figures use compact BEC caches instead.

---

# 4. Figure-specific outputs generated from the analysis pipeline

Two important figure caches were explicitly regenerated from the full EC
step-4 dataframe and are included in the repository:

```text
codes/HCP/results/
    Figure4_PanelB_node_corr_HCP_spearman.csv
    Figure4_PanelCD_meanvar_HCP.csv
```

`Figure4_PanelB_node_corr_HCP_spearman.csv` contains the per-subject,
per-region Spearman correlations between baseline energy and perturbation effect.

`Figure4_PanelCD_meanvar_HCP.csv` contains the mean and sample variance of
global effect size and global evoked energy for:

```text
lowest 5% baseline-energy states
highest 5% baseline-energy states
200 random 5%-of-trials samples
```

These compact tables allow the state-dependence figure panels to be reproduced
without distributing the approximately 4 GB step-4 EC dataframe.

---

# 5. Precomputed compact figure caches

Some manuscript panels require analyses whose complete intermediate data are
either very large, belong to a separate analysis branch, or whose original
cache-generation step is not part of the lightweight repository.

For those panels, compact precomputed caches are included.

Examples include:

```text
Figure1E_panel_bins.npz
Figure2_fit_group.npz
Figure5_BEC_matrices.npz
BEC_focal_cache.npz
Figure5_PanelA_regime_cv_HCP.csv
FigureS_linear_null_VAR_cohort.csv
supp_S1_specificity.npz
supp_S2_pertstrength.npz
test_sparsity_A.npz
test_sparsity_B.npz
```

The receptor spatial-null analysis also uses:

```text
Figure5_variability_receptor_correlations.csv
```

stored under `codes/HCP/results/`.

The heavy provenance analysis for this file requires additional per-subject
bifocal data and large spatial-null computations, so those intermediates are not
distributed.

---

# 6. Task-fMRI validation

The task-fMRI analyses are in `codes/HCP/src/`.

The personalized models are trained using resting-state fMRI only. Task data are
used as independent validation: the analysis asks whether EC maps obtained from
rest-trained models reproduce task-evoked activation patterns and whether they
do so better than resting-state functional connectivity.

## Shared task-analysis modules

| File | Role |
|---|---|
| `task_io.py` | HCP task loaders, condition definitions and canonical contrasts |
| `task_state.py` | Converts block onsets to model-input windows; hemodynamic lag; baseline energy |
| `task_gating.py` | Loads a trained model and computes perturbation responses for task states |
| `task_ec_validation.py` | Resting EC / FC extraction |
| `task_network_behavior.py` | Yeo-7 network and hierarchy lookup |
| `spin_test.py` | Spatial spin null |
| `NPI.py` | Neural perturbational inference model utilities |
| `preprocessing_hcp.py` | HCP preprocessing utilities |

## Runnable task analyses

| Script | Output |
|---|---|
| `condition_decoding.py` | `decoding_MOTOR.npz`, `decoding_WM.npz` |
| `task_ec_figuredata.py` | `task_ec_scalars.csv`, `group_mean_EC_cortical.npy` |
| `task_ec_multi.py` | `task_ec_multitask.csv` |
| `task_ec_profiles.py` | `task_ec_profiles.npz` |
| `cognitive_state_gating.py` | `cognitive_state_S2.npz` |
| `methods_panel_data.py` | `methods_panel_data.npz` |

Compact outputs from this branch are included so the task figures can be
reproduced without requiring the original HCP task BOLD data.

## Re-running the task analysis from scratch

The task BOLD data are not distributed.

Set:

```bash
export HCP_TASK_ROOT=/path/to/HCP/task/data
```

The expected inputs are parcellated Schaefer-400 + Tian-S3 time series.

```markdown
The HCP task block-timing EV files used in the analysis are not redistributed
in this repository. They should be obtained directly from the Human Connectome
Project and placed under:

```text
codes/HCP/data/Task/HCP_TASKS_EVs/

and the paper cohort is listed in:

```text
codes/HCP/data/Task/language_subjects_paper100.txt
```

Re-running the EC-based task validation from scratch also requires the complete
set of trained ANN models and processed resting-state windows generated by
steps 01-02.

---

# 7. Manuscript figure scripts

Run figure scripts from:

```bash
cd codes/figures
```

Outputs are written to:

```text
codes/figures/outputs/
```

This directory is generated locally and is ignored by Git.

## Main figures

### Manuscript Figure 1

```text
Figure1_PanelA.py
Figure1_PanelA_observational.py
Figure1_PanelA_seedEC.py
Figure1_PanelB.py
Figure1_PanelC.py
Figure1_PanelD.py
Figure_manifold_landscape.py
```

### Manuscript Figure 2

```text
Figure2_fit_group.py
Figure2_fit_composite.py
```

### Manuscript Figure 3 — task validation

```text
Figure3_methods_taskmaps.py
render_motor_brains.py
Figure3.py
```

`render_motor_brains.py` must be run before `Figure3.py`.

### Manuscript Figure 4 — cortical hierarchy

```text
Figure3_PanelA_row.py
Figure3_PanelA_HCP.py
```

### Manuscript Figure 5 — state-dependent gating

```text
Figure4_row1.py
Figure4_PanelA_HCP.py
Figure4_gating_curve.py
Figure4_subject_hist.py
Figure4_PanelB_HCP.py
Figure4_PanelCD_meanvar.py
Figure4_PanelD_brains.py
Figure4_manifold_zooms.py
```

### Manuscript Figure 6 — bifocal stimulation

```text
Figure5_PanelBEC_matrices.py
Figure5_BEC_brain_rowA.py
Figure5_BEC_focalvsbifocal_effect_A.py
Figure5_BEC_focalvsbifocal_A.py
Figure5_BEC_scatter_GESvsCV.py
Figure5_BEC_netpair_panels.py
```

## Supplementary figures

```text
S1   Figure_task_ec_profiles.py
S2   Figure_S2_cognitive_state.py
S3   Figure_SF_fc_decoding.py
S4/5 supp_S1_S2.py
S6   Figure3_PanelC_receptor_bars_persubject.py
     Figure3_PanelC_receptor_bars.py
S7   Figure4_PanelB_HCP.py
     Figure4_PanelD_brains.py
S8   FigureS_linear_null_HCP.py
S9   test_sparsity_controls.py
S10  Figure5_BEC_suppl_var.py
```

Some filenames reflect earlier internal figure numbering and therefore do not
match the final manuscript figure numbers. The mapping above should be used as
the reference.

---

# 8. Quick figure reproduction

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Surface plots use `neuromaps`. On first use, the fsLR-32k surface files may be
downloaded automatically, so network access is required once.

A minimal example is:

```bash
cd codes/figures

python Figure1_PanelA.py
python Figure1_PanelB.py
python Figure1_PanelC.py
python Figure1_PanelD.py

python Figure2_fit_group.py
python Figure2_fit_composite.py

python Figure3_methods_taskmaps.py
python render_motor_brains.py
python Figure3.py
```

The remaining figure scripts can be run independently using the compact results
stored under `codes/HCP/results/`.

---

# 9. Requirements

The repository was validated in an environment containing:

```text
numpy==2.2.6
scipy==1.15.3
pandas==2.3.3
matplotlib==3.10.9
seaborn==0.13.2
scikit-learn==1.7.2
torch==2.13.0
h5py==3.16.0
BrainSpace==0.2.1
nibabel==5.4.2
neuromaps==0.0.7
surfplot==0.2.0
```

These versions are recorded in `requirements.txt`.

---

# 10. Reproducibility and validation

The cleaned analysis pipeline and lightweight figure package were checked
against the original analysis outputs.

## Time-resolved EC

For example participant `id_100206`, the complete EC tensor
(500 × 450 × 450; 101,250,000 values) was regenerated and compared with the
original:

```text
mean absolute difference     4.46e-07
median absolute difference   3.58e-07
99th percentile difference   1.91e-06
maximum difference           1.10e-05
relative L2 error            5.39e-04
Pearson r                    0.999999847
```

All values were within `1e-4` of the original.

## Time-resolved BEC

The corresponding BEC tensor was also compared:

```text
mean absolute difference     5.73e-07
median absolute difference   4.60e-07
99th percentile difference   2.19e-06
maximum difference           9.65e-06
relative L2 error            3.02e-05
Pearson r                    0.999999982
```

All values were within `1e-5` of the original.

## Step-4 EC/BEC state-dependence analysis

A complete subject was regenerated for both EC and BEC.

Direct quantities matched at numerical precision. ANN-forward-derived quantities
had relative L2 errors below approximately `7e-7`, with correlations effectively
equal to 1.

## Step-6 spatial dataframe

The final EC dataframe contains 40,000 × 52 entries.

The distributed dataframe was compared against the original figure-ready
reference:

```text
shape                  identical
subjects               identical
columns                identical
categorical fields     exact
numeric correlations   r = 1
maximum difference     5.68e-14
```

The principal-gradient calculation itself can show very small run-to-run
differences because the BrainSpace diffusion-map eigendecomposition is not
explicitly seeded and eigenvectors have numerical/sign ambiguity. The original
and cleaned implementations use the same calculation.

## Figure-specific state-dependence caches

`Figure4_PanelB_node_corr_HCP_spearman.csv` was regenerated from the full
step-4 EC dataframe.

Both stored correlation columns matched the original cache exactly:

```text
mean absolute difference = 0
maximum difference       = 0
```

`Figure4_PanelCD_meanvar_HCP.csv` was likewise regenerated from the full step-4
EC dataframe.

All six numerical columns:

```text
mean_low
mean_high
var_low
var_high
mean_rand
var_rand
```

matched the original cache exactly.

## Figure reproduction

The lightweight repository was tested after removing the multi-GB EC/BEC
intermediate files and previous generated figure outputs.

All 31 tested figure-generation scripts completed successfully.

The newly rendered panels were also compared against the original supplied
figure PNGs. Differences in exact PNG pixels and canvas dimensions are expected
across Matplotlib/font/rendering environments. Visual inspection of the
lowest-similarity panels confirmed equivalent scientific content, including
bar heights/signs, significance markers, distributions, matrices, brain maps and
state-dependence relationships.

### Figure 2 stochastic free-running trace

`Figure2_fit_group.py` re-simulates the example participant's long recursive
free-running trajectory.

Resetting the NumPy seed produces an exactly reproducible trajectory within the
same environment. However, small numerical differences between PyTorch/CPU
environments accumulate over thousands of autoregressive prediction steps, so
the exact pointwise trajectory can differ from the originally cached
realization.

In the validated environment:

```text
cached FC vs empirical r       0.8615
new FC vs empirical r          0.8205

cached dFC KS                  0.0256
new dFC KS                     0.0231
```

The quantitative Figure 2 cohort panels use the supplied original compact
cache; only the displayed free-running example trace is freshly simulated.

---

# 11. Tests

`codes/HCP/tests/` contains lightweight validation/smoke-test utilities.

The repository was additionally tested by removing all non-distributed heavy
intermediates and regenerating the manuscript panels from the files that remain
in the public package.

This confirms that quick figure reproduction does not depend on the omitted
multi-GB EC/BEC tensors or step-4 dataframes.

---
