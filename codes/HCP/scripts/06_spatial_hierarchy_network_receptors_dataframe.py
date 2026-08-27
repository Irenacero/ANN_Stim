#!/usr/bin/env python
# coding: utf-8

# # 5.1 — Spatial, hierarchy, network, gradient, and receptor dataframe
#
# This version always runs the analysis on the cortical block only.
#
# Default assumption for the current HCP-style dataset:
#
# - full atlas: 50 subcortical + 400 cortical ROIs
# - cortical ROIs are therefore original ROIs 50..449, remapped to cortical labels 0..399
#
# Important behavior for other datasets:
#
# - if the dataset is already cortical-only, the notebook keeps all available ROIs;
# - if the dataset has fewer cortical ROIs, for example Schaefer-100, set `CORTICAL_ROI_NUM = 100`;
# - if the dataset has cortex + subcortex, set `CORTICAL_ROI_NUM` to the cortical count and set `SUBCORTICAL_POSITION` to `"first"` or `"last"`;
# - receptor maps and RSN labels must match the cortical parcellation size.
#
# To use a different dataset, usually you only need to change variables in **Analysis parameters**:
#
# - `DATASET_NAME`
# - `CONNECTIVITY_KIND`
# - `CORTICAL_ROI_NUM`
# - `SUBCORTICAL_POSITION`
# - optional path overrides
#

# In[1]:


from pathlib import Path
import argparse
import sys
import os
import gc
import glob
import warnings

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Path and environment configuration
# -----------------------------------------------------------------------------
# Expected local/repository structure, matching the step-4 notebook style:
#
# BrainStim_ANN_fMRI_HCP/
# ├── src/
# ├── notebooks/
# └── results/
#     ├── processed/
#     ├── ANN_model/
#     ├── ECts/
#     ├── BECts/
#     ├── HCP_4_df_background_dependence_ECts.pkl
#     └── HCP_4_df_background_dependence_BECts.pkl
#
# Receptor maps are expected in a local data folder above/next to the repo.
#
# ../data/
# └── Receptor maps/
#     ├── 5HT1a_cumi_hc8_beliveau.csv
#     ├── 5HT1a_way_hc36_savli.csv
#     └── ...

script_dir = Path(__file__).resolve().parent
hcp_dir = script_dir.parent
repo_dir = hcp_dir.parent.parent

results_dir = hcp_dir / "results"
preproc_dir = results_dir / "processed"
save_dir = results_dir / "dataframes"

# Receptor maps: preserve the original candidate locations, resolved
# relative to codes/HCP rather than to the current working directory.
data_dir_candidates = [
    hcp_dir.parent / "data",
    hcp_dir / "data",
]
data_dir = next(
    (path for path in data_dir_candidates if path.exists()),
    data_dir_candidates[0],
)
receptor_dir = data_dir / "Receptor_maps"
dataframe_dir = results_dir / "dataframes"

save_dir.mkdir(parents=True, exist_ok=True)
dataframe_dir.mkdir(parents=True, exist_ok=True)

if str(hcp_dir) not in sys.path:
    sys.path.insert(0, str(hcp_dir))

print("Repository directory :", repo_dir)
print("Results directory    :", results_dir)
print("Processed data dir   :", preproc_dir)
print("Data directory       :", data_dir)
print("Receptor-map dir     :", receptor_dir)
print("Dataframe directory  :", dataframe_dir)
print("Output directory     :", save_dir)

# In[2]:


# =============================================================================
# Analysis parameters — edit only this cell for a different dataset
# =============================================================================

# Dataset prefix used in the step-4 dataframe filenames.
# Expected default filenames:
#   results/dataframes/{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.pkl
#   results/dataframes/{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.csv
DATASET_NAME = "HCP"

# Change this to switch between the two effective-connectivity families.
# Valid values: "ECts" or "BECts".
argument_parser = argparse.ArgumentParser(
    description=(
        "Build the spatial, hierarchy, network, gradient and receptor "
        "dataframe for ECts or BECts."
    )
)
argument_parser.add_argument(
    "--connectivity-kind",
    choices=["ECts", "BECts"],
    default="BECts",
    help=(
        "Connectivity family to process. The original notebook default "
        "was BECts; run once with ECts and once with BECts."
    ),
)
arguments = argument_parser.parse_args()
CONNECTIVITY_KIND = arguments.connectivity_kind

# Cortical-only analysis settings.
# CORTICAL_ROI_NUM is the number of cortical ROIs in the parcellation.
#
# Current HCP-style case:
#   450 total ROIs = 50 subcortical + 400 cortical, subcortex first
#   -> CORTICAL_ROI_NUM = 400, SUBCORTICAL_POSITION = "first"
#
# Dataset with no subcortical ROIs:
#   400 cortical-only ROIs -> keeps all 400
#   100 cortical-only ROIs -> set CORTICAL_ROI_NUM = 100 and keeps all 100
#
# SUBCORTICAL_POSITION is ignored whenever the loaded data has <= CORTICAL_ROI_NUM ROIs.
CORTICAL_ONLY = True
CORTICAL_ROI_NUM = 400
SUBCORTICAL_POSITION = "first"  # use "first" if subcortical ROIs come before cortex

# Optional overrides for another dataset.
# Leave as None for the standard project folder structure.
BACKGROUND_DF_PATH = None       # e.g. r"/path/to/my_step4_dataframe.pkl" or .csv
CONNECTIVITY_DIR_OVERRIDE = None # e.g. r"/path/to/BECts"
SIGNALS_DIR_OVERRIDE = None      # e.g. r"/path/to/processed_signals"
RECEPTOR_DIR_OVERRIDE = None     # e.g. r"/path/to/Receptor_maps"

# Other options.
HIERARCHY_USE_ABSOLUTE_CONNECTIVITY = True
KEEP_RAW_RECEPTOR_TRACER_MAPS = False
KEEP_UNGROUPED_RECEPTOR_MAPS = True
COMPUTE_PRINCIPAL_GRADIENT = True
PRINCIPAL_GRADIENT_TEST_DUR = 3000
PRINCIPAL_GRADIENT_N_COMPONENTS = 10

# Backward-compatible aliases used by the receptor/annotation cells.
# In this notebook the final dataframe is cortical-only, so final ROI count = receptor ROI count.
RECEPTOR_ROI_NUM = CORTICAL_ROI_NUM
TOTAL_ROI_NUM = CORTICAL_ROI_NUM
SUBCORTICAL_RECEPTOR_POSITION = SUBCORTICAL_POSITION
CORTICAL_ROI_POSITION = "last" if SUBCORTICAL_POSITION == "first" else "first"

CONNECTIVITY_CONFIG = {
    "ECts": {
        "dir_name": "ECts",
        "file_stem": "ECt",
        "label": "EC_t",
        "strength_prefix": "ec",
    },
    "BECts": {
        "dir_name": "BECts",
        "file_stem": "BECt",
        "label": "BEC_t",
        "strength_prefix": "bec",
    },
}

if CONNECTIVITY_KIND not in CONNECTIVITY_CONFIG:
    raise ValueError(f"CONNECTIVITY_KIND must be one of {list(CONNECTIVITY_CONFIG)}, got {CONNECTIVITY_KIND!r}.")

CONNECTIVITY_DIR_NAME = CONNECTIVITY_CONFIG[CONNECTIVITY_KIND]["dir_name"]
CONNECTIVITY_FILE_STEM = CONNECTIVITY_CONFIG[CONNECTIVITY_KIND]["file_stem"]
CONNECTIVITY_LABEL = CONNECTIVITY_CONFIG[CONNECTIVITY_KIND]["label"]
CONNECTIVITY_STRENGTH_PREFIX = CONNECTIVITY_CONFIG[CONNECTIVITY_KIND]["strength_prefix"]
CONNECTIVITY_SUFFIX = f"_{CONNECTIVITY_FILE_STEM}.npy"

# Path overrides.
if CONNECTIVITY_DIR_OVERRIDE is not None:
    connectivity_dir = Path(CONNECTIVITY_DIR_OVERRIDE)
else:
    connectivity_dir_candidates = [
        results_dir / CONNECTIVITY_DIR_NAME,
        preproc_dir / f"{CONNECTIVITY_DIR_NAME}_MLP",
        data_dir / "preprocessed_subjects" / f"{CONNECTIVITY_DIR_NAME}_MLP",
    ]
    connectivity_dir = next((p for p in connectivity_dir_candidates if p.exists()), connectivity_dir_candidates[0])

signals_dir = Path(SIGNALS_DIR_OVERRIDE) if SIGNALS_DIR_OVERRIDE is not None else preproc_dir
receptor_dir = Path(RECEPTOR_DIR_OVERRIDE) if RECEPTOR_DIR_OVERRIDE is not None else receptor_dir

# Step-4 background-dependence dataframe candidates.
if BACKGROUND_DF_PATH is not None:
    background_candidates = [Path(BACKGROUND_DF_PATH)]
else:
    background_candidates = [
        save_dir / f"{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.pkl",
        save_dir / f"{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.csv",
        results_dir / f"{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.pkl",
        results_dir / f"{DATASET_NAME}_4_df_background_dependence_{CONNECTIVITY_KIND}.csv",
    ]

# Final cortical-only outputs.
output_csv = save_dir / f"{DATASET_NAME}_5_df_spatial_network_receptors_{CONNECTIVITY_KIND}_cortical{CORTICAL_ROI_NUM}.csv"
output_pkl = save_dir / f"{DATASET_NAME}_5_df_spatial_network_receptors_{CONNECTIVITY_KIND}_cortical{CORTICAL_ROI_NUM}.pkl"


def get_cortical_indices(n_rois, cortical_roi_num=CORTICAL_ROI_NUM, subcortical_position=SUBCORTICAL_POSITION):
    """
    Return the column/ROI indices used for the cortical-only analysis.

    Examples
    --------
    n_rois = 450, cortical_roi_num = 400, subcortical_position = "first"
        -> keeps 50..449

    n_rois = 450, cortical_roi_num = 400, subcortical_position = "last"
        -> keeps 0..399

    n_rois = 400, cortical_roi_num = 400
        -> dataset is already cortical-only, keeps all 0..399

    n_rois = 100, cortical_roi_num = 100
        -> smaller cortical-only dataset, keeps all 0..99

    n_rois <= cortical_roi_num
        -> no subcortex is assumed; keeps all available ROIs
    """
    n_rois = int(n_rois)
    n_cortex = min(int(cortical_roi_num), n_rois)

    if n_rois <= cortical_roi_num:
        return np.arange(n_rois, dtype=int)

    if subcortical_position == "last":
        return np.arange(n_cortex, dtype=int)

    if subcortical_position == "first":
        return np.arange(n_rois - n_cortex, n_rois, dtype=int)

    raise ValueError("SUBCORTICAL_POSITION must be either 'last' or 'first'.")


print("Analysis configuration:")
print("  Dataset name                    :", DATASET_NAME)
print("  Connectivity kind               :", CONNECTIVITY_KIND)
print("  Connectivity directory          :", connectivity_dir)
print("  Connectivity file suffix        :", CONNECTIVITY_SUFFIX)
print("  Signals directory               :", signals_dir)
print("  Receptor-map directory          :", receptor_dir)
print("  Cortical-only                   :", CORTICAL_ONLY)
print("  Cortical ROI number             :", CORTICAL_ROI_NUM)
print("  Subcortical position            :", SUBCORTICAL_POSITION)
print("  No-subcortex behavior           : if loaded N <= CORTICAL_ROI_NUM, all ROIs are kept")
print("  Step-4 dataframe candidates     :")
for p in background_candidates:
    print("   -", p)
print("  Output CSV                      :", output_csv)
print("  Output PKL                      :", output_pkl)


# ## Load the step-4 dataframe
#
# The step-4 dataframe is expected to contain one row per `(sub_id, roi, time)` with baseline/evoked energy and perturbation-effect columns.
#

# In[3]:


# =============================================================================
# Load step-4 background-dependence dataframe
# =============================================================================
def load_step4_dataframe(paths):
    """Load the first existing .pkl or .csv step-4 dataframe from a list of candidates."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        if path.suffix.lower() in [".pkl", ".pickle"]:
            print("Loading step-4 dataframe from:", path)
            return pd.read_pickle(path)

        if path.suffix.lower() == ".csv":
            print("Loading step-4 dataframe from:", path)
            return pd.read_csv(path)

        raise ValueError(f"Unsupported step-4 dataframe format: {path}")

    raise FileNotFoundError(
        "Could not find the step-4 dataframe. Tried:\n" +
        "\n".join(f"  {Path(p)}" for p in paths)
    )


HCP_4_df_background_dependence = load_step4_dataframe(background_candidates)

# Harmonize subject-column naming.
if "sid" in HCP_4_df_background_dependence.columns and "sub_id" not in HCP_4_df_background_dependence.columns:
    HCP_4_df_background_dependence = HCP_4_df_background_dependence.rename(columns={"sid": "sub_id"})

HCP_4_df_background_dependence["sub_id"] = HCP_4_df_background_dependence["sub_id"].astype(str)
HCP_4_df_background_dependence["roi"] = pd.to_numeric(HCP_4_df_background_dependence["roi"], errors="coerce")
HCP_4_df_background_dependence = HCP_4_df_background_dependence.dropna(subset=["roi"]).copy()
HCP_4_df_background_dependence["roi"] = HCP_4_df_background_dependence["roi"].astype(int)

print("Loaded step-4 dataframe.")
print("Shape:", HCP_4_df_background_dependence.shape)
print("ROI range:", (HCP_4_df_background_dependence["roi"].min(), HCP_4_df_background_dependence["roi"].max()))
print("Number of ROIs:", HCP_4_df_background_dependence["roi"].nunique())
print("Columns:")
print(list(HCP_4_df_background_dependence.columns))
HCP_4_df_background_dependence.head()


# In[4]:


# =============================================================================
# Prepare cortical-only ROI indexing for the final dataframe
# =============================================================================
def prepare_background_df_cortical(df):
    """
    Keep only cortical ROIs and remap them to 0..N_cortex-1.

    This works for:
    - 450 ROI input with subcortex first -> keeps 50..449 and remaps to 0..399
    - 450 ROI input with subcortex last -> keeps 0..399 and remaps to 0..399
    - already-cortical 400 ROI input -> keeps all 0..399
    - smaller cortical-only datasets -> set CORTICAL_ROI_NUM to that size and keeps all ROIs

    If the loaded dataframe has <= CORTICAL_ROI_NUM ROIs, no subcortex is assumed.
    """
    df = df.copy()
    df["roi"] = pd.to_numeric(df["roi"], errors="coerce")
    df = df.dropna(subset=["roi"]).copy()
    df["roi"] = df["roi"].astype(int)

    # Convert common 1-based labels to 0-based labels.
    unique_rois = np.sort(df["roi"].unique())
    if unique_rois.min() == 1 and 0 not in unique_rois:
        df["roi"] = df["roi"] - 1
        unique_rois = np.sort(df["roi"].unique())

    # If labels are contiguous 0..N-1, use the atlas index directly.
    # Otherwise use sorted ROI positions, which is safer for a different dataset.
    if np.array_equal(unique_rois, np.arange(unique_rois.max() + 1)):
        source_rois = get_cortical_indices(unique_rois.max() + 1)
    else:
        source_positions = get_cortical_indices(len(unique_rois))
        source_rois = unique_rois[source_positions]

    source_rois = np.asarray(source_rois, dtype=int)
    roi_map = {old_roi: new_roi for new_roi, old_roi in enumerate(source_rois)}

    df = df.loc[df["roi"].isin(roi_map)].copy()
    df["roi_original"] = df["roi"].astype(int)
    df["roi"] = df["roi_original"].map(roi_map).astype(int)

    return df


background_df = prepare_background_df_cortical(HCP_4_df_background_dependence)

print("Prepared cortical-only background dataframe for step 5.")
print("Shape:", background_df.shape)
print("ROI range after remapping:", (background_df["roi"].min(), background_df["roi"].max()))
print("Original ROI range retained:", (background_df["roi_original"].min(), background_df["roi_original"].max()))
print("Number of cortical ROIs:", background_df["roi"].nunique())
background_df.head()


# ## Spatial measures from the step-4 dataframe
#
# The following cell implements the spatial metrics:
#
# - mean local evoked energy, effect size, and effect direction
# - mean global evoked energy, effect size, and effect direction
# - squared Pearson correlations for the explanatory-power terms
#

# In[5]:


# =============================================================================
# Spatial-measure utilities
# =============================================================================
SPATIAL_REQUIRED_COLUMNS = [
    "sub_id",
    "roi",
    "time",
    "global_baseline_energy",
    "global_evoked_energy",
    "global_effect_size",
    "global_effect_direction",
    "local_baseline_energy",
    "local_evoked_energy",
    "local_effect_size",
    "local_effect_direction",
]

missing_cols = [c for c in SPATIAL_REQUIRED_COLUMNS if c not in background_df.columns]
if missing_cols:
    raise ValueError(f"The step-4 dataframe is missing required columns: {missing_cols}")


def corr_squared(x, y, min_n=3):
    """Squared Pearson correlation with guards against NaNs and constant vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < min_n:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    r = np.corrcoef(x, y)[0, 1]
    return float(r ** 2)


def compute_spatial_r2_for_group(g):
    """Compute the 12 explanatory-power terms for one `(sub_id, roi)`."""
    et_local = g["local_baseline_energy"].to_numpy(dtype=float)
    et_global = g["global_baseline_energy"].to_numpy(dtype=float)

    return pd.Series({
        # Explanatory power of evoked energy
        "r2_local_energy_to_local_evoked_energy": corr_squared(et_local, g["local_evoked_energy"]),
        "r2_local_energy_to_global_evoked_energy": corr_squared(et_local, g["global_evoked_energy"]),
        "r2_global_energy_to_global_evoked_energy": corr_squared(et_global, g["global_evoked_energy"]),
        "r2_global_energy_to_local_evoked_energy": corr_squared(et_global, g["local_evoked_energy"]),

        # Explanatory power of effect size
        "r2_local_energy_to_local_effect_size": corr_squared(et_local, g["local_effect_size"]),
        "r2_local_energy_to_global_effect_size": corr_squared(et_local, g["global_effect_size"]),
        "r2_global_energy_to_global_effect_size": corr_squared(et_global, g["global_effect_size"]),
        "r2_global_energy_to_local_effect_size": corr_squared(et_global, g["local_effect_size"]),

        # Explanatory power of effect direction
        "r2_local_energy_to_local_effect_direction": corr_squared(et_local, g["local_effect_direction"]),
        "r2_local_energy_to_global_effect_direction": corr_squared(et_local, g["global_effect_direction"]),
        "r2_global_energy_to_global_effect_direction": corr_squared(et_global, g["global_effect_direction"]),
        "r2_global_energy_to_local_effect_direction": corr_squared(et_global, g["local_effect_direction"]),
    })


def compute_spatial_measures(background_df):
    """Build the ROI-level spatial-measure dataframe from the step-4 dataframe."""
    group_cols = ["sub_id", "roi"]

    mean_df = (
        background_df
        .groupby(group_cols, sort=False)
        .agg(
            roi_original=("roi_original", "first"),
            n_timepoints=("time", "nunique"),
            mean_local_evoked_energy=("local_evoked_energy", "mean"),
            mean_local_effect_size=("local_effect_size", "mean"),
            mean_local_effect_direction=("local_effect_direction", "mean"),
            mean_global_evoked_energy=("global_evoked_energy", "mean"),
            mean_global_effect_size=("global_effect_size", "mean"),
            mean_global_effect_direction=("global_effect_direction", "mean"),
        )
        .reset_index()
    )

    r2_df = (
        background_df
        .groupby(group_cols, sort=False)
        .apply(compute_spatial_r2_for_group)
        .reset_index()
    )

    spatial_df = mean_df.merge(r2_df, on=group_cols, how="left")
    return spatial_df


# In[6]:


# =============================================================================
# Compute spatial measures
# =============================================================================
spatial_df = compute_spatial_measures(background_df)

print("Spatial dataframe built successfully.")
print("Shape:", spatial_df.shape)
print("ROI range:", (spatial_df["roi"].min(), spatial_df["roi"].max()))
print("Columns:")
print(list(spatial_df.columns))
spatial_df.head()


# ## Load connectivity tensors
#
# The selected tensor is controlled by `CONNECTIVITY_KIND`:
#
# - `ECts`: `EC_t[t, j, i]`
# - `BECts`: `BEC_t[t, j, i]`
#
# In both cases, `t` is time, `j` is the stimulated/source target region, and `i` is the affected/target region.
#

# In[7]:


# =============================================================================
# Locate selected connectivity tensors
# =============================================================================
connectivity_files = sorted(connectivity_dir.glob(f"*{CONNECTIVITY_SUFFIX}"))
subject_ids_connectivity = {p.name.split(CONNECTIVITY_SUFFIX)[0] for p in connectivity_files}
subject_ids_df = set(spatial_df["sub_id"].astype(str).unique())
subject_ids = sorted(subject_ids_df.intersection(subject_ids_connectivity))

print("Number of subjects in step-4 dataframe:", len(subject_ids_df))
print(f"Number of {CONNECTIVITY_LABEL} files found       :", len(subject_ids_connectivity))
print("Number of matched subjects          :", len(subject_ids))

missing_connectivity = sorted(subject_ids_df - subject_ids_connectivity)
if missing_connectivity:
    print(f"Subjects present in step 4 but missing {CONNECTIVITY_LABEL} files:")
    print(missing_connectivity[:20], "..." if len(missing_connectivity) > 20 else "")

if len(subject_ids) == 0:
    raise RuntimeError(
        "No subjects have both a step-4 dataframe entry and a connectivity tensor file. "
        f"Check connectivity_dir: {connectivity_dir} and suffix: {CONNECTIVITY_SUFFIX}"
    )


# ## Hierarchy and trophic coherence from the selected connectivity tensor
#
# The hierarchy is computed from the subject-specific time-averaged effective connectivity matrix.
# For trophic measures, the default uses absolute weights because trophic-coherence calculations assume a non-negative directed graph.
#

# In[8]:


# =============================================================================
# Hierarchy and trophic-coherence utilities
# =============================================================================
def prepare_connectivity_for_hierarchy(C, use_absolute=True, remove_diagonal=True):
    """Prepare a connectivity matrix for trophic hierarchy/coherence."""
    A = np.asarray(C, dtype=float).copy()
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if use_absolute:
        A = np.abs(A)

    if remove_diagonal:
        np.fill_diagonal(A, 0.0)

    return A


def compute_hierarchy_and_trophic_coherence(C, use_absolute=True, remove_diagonal=True):
    """
    Compute hierarchical levels and trophic coherence from a directed connectivity matrix.

    This follows the logic used in the reference notebook:
    d      = column sums
    delta  = row sums
    u      = d + delta
    v      = d - delta
    Lambda = diag(u) - A - A.T
    Lambda * gamma = v
    """
    A = prepare_connectivity_for_hierarchy(
        C,
        use_absolute=use_absolute,
        remove_diagonal=remove_diagonal,
    )

    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"EC must be square, got {A.shape}")

    total_weight = np.sum(A)
    if total_weight <= 0:
        return np.full(N, np.nan), np.nan

    d = A.sum(axis=0)
    delta = A.sum(axis=1)
    u = d + delta
    v = d - delta
    Lambda = np.diag(u) - A - A.T

    # Anchor the first region to zero to remove the translational degree of freedom.
    Lambda_anchor = Lambda.copy()
    v_anchor = v.copy()
    Lambda_anchor[0, :] = 0.0
    Lambda_anchor[0, 0] = 1.0
    v_anchor[0] = 0.0

    try:
        gamma = np.linalg.solve(Lambda_anchor, v_anchor)
    except np.linalg.LinAlgError:
        gamma = np.linalg.pinv(Lambda_anchor) @ v_anchor

    gamma = gamma - np.nanmin(gamma)

    source_level = gamma[:, None]
    target_level = gamma[None, :]
    H = (source_level - target_level - 1.0) ** 2

    F0 = np.sum(A * H) / total_weight
    trophic_coherence = 1.0 - F0

    return gamma, float(trophic_coherence)


# In[9]:


# =============================================================================
# Compute hierarchy and trophic coherence from cortical connectivity tensors
# =============================================================================
def extract_cortical_connectivity_t(C_t):
    """
    Keep only the cortical connectivity block.

    Examples:
    - 450 ROIs with subcortex first and CORTICAL_ROI_NUM = 400:
        C_t[:, 50:450, 50:450]
    - already cortical-only, e.g. 400 or 100 ROIs:
        keeps the whole tensor
    """
    C_t = np.asarray(C_t, dtype=float)

    if C_t.ndim != 3 or C_t.shape[1] != C_t.shape[2]:
        raise ValueError(f"Connectivity tensor must have shape (T, N, N), got {C_t.shape}")

    cortical_idx = get_cortical_indices(C_t.shape[-1])
    C_t_cortical = C_t[:, cortical_idx, :][:, :, cortical_idx]

    return C_t_cortical, cortical_idx.astype(int)


def compute_connectivity_metrics_for_subject(sid, C_t):
    """Compute ROI-level network metrics from one subject's cortical connectivity tensor."""
    C_t_use, roi_original = extract_cortical_connectivity_t(C_t)

    N = C_t_use.shape[-1]
    roi = np.arange(N, dtype=int)  # remapped cortical labels: 0..N-1

    meanC_signed = np.nanmean(C_t_use, axis=0)

    hierarchy, trophic_coherence = compute_hierarchy_and_trophic_coherence(
        meanC_signed,
        use_absolute=HIERARCHY_USE_ABSOLUTE_CONNECTIVITY,
        remove_diagonal=True,
    )

    abs_meanC = np.abs(meanC_signed.copy())
    np.fill_diagonal(abs_meanC, 0.0)
    out_strength = abs_meanC.sum(axis=1)
    in_strength = abs_meanC.sum(axis=0)

    df_sub = pd.DataFrame({
        "sub_id": str(sid),
        "roi": roi,
        "roi_original_from_connectivity": roi_original,
        "connectivity_kind": CONNECTIVITY_KIND,
        "hierarchy": hierarchy,
        "trophic_coherence": trophic_coherence,
        f"mean_{CONNECTIVITY_STRENGTH_PREFIX}_abs_out_strength": out_strength,
        f"mean_{CONNECTIVITY_STRENGTH_PREFIX}_abs_in_strength": in_strength,
    })

    return df_sub


network_dfs = []
failed_network_subjects = []

for sid in subject_ids:
    print(f"Processing cortical {CONNECTIVITY_LABEL} metrics for subject {sid}")
    connectivity_path = connectivity_dir / f"{sid}{CONNECTIVITY_SUFFIX}"

    try:
        C_t = np.load(connectivity_path)
        df_sub = compute_connectivity_metrics_for_subject(sid, C_t)
        network_dfs.append(df_sub)

        del C_t, df_sub
        gc.collect()

    except Exception as exc:
        failed_network_subjects.append((sid, str(exc)))
        print(f"  Failed: {exc}")

if len(network_dfs) == 0:
    raise RuntimeError("No cortical connectivity network metrics could be created.")

network_df = pd.concat(network_dfs, axis=0, ignore_index=True)

print("Cortical network dataframe built successfully.")
print("Connectivity kind:", CONNECTIVITY_KIND)
print("Shape:", network_df.shape)
print("ROI range:", (network_df["roi"].min(), network_df["roi"].max()))
print("Original connectivity ROI range:", (network_df["roi_original_from_connectivity"].min(), network_df["roi_original_from_connectivity"].max()))
print("Failed subjects:", len(failed_network_subjects))
if failed_network_subjects:
    print(pd.DataFrame(failed_network_subjects, columns=["sub_id", "error"]))

network_df.head()


# ## Principal functional-connectivity gradient

# In[10]:


# =============================================================================
# Principal gradient from cortical empirical static FC
# =============================================================================
if COMPUTE_PRINCIPAL_GRADIENT:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from brainspace.gradient import GradientMaps
    except Exception as exc:
        raise ImportError(
            "Principal-gradient computation requires scikit-learn and brainspace. "
            "Install brainspace or set COMPUTE_PRINCIPAL_GRADIENT = False."
        ) from exc


def extract_cortical_signal_block(Z):
    """
    Keep only the cortical signal block.

    Examples:
    - 450 ROIs with subcortex first and CORTICAL_ROI_NUM = 400:
        Z[:, 50:450]
    - already cortical-only, e.g. 400 or 100 ROIs:
        keeps the whole signal matrix
    """
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError(f"Signal array must have shape (T, N), got {Z.shape}")

    cortical_idx = get_cortical_indices(Z.shape[1])
    Z_cortical = Z[:, cortical_idx]
    roi = np.arange(len(cortical_idx), dtype=int)

    return Z_cortical, roi


def compute_principal_gradient_for_subject(sid):
    """Compute the first diffusion-map gradient from one subject's cortical empirical FC."""
    sig_path = signals_dir / f"{sid}_signals.npy"

    if not sig_path.exists():
        raise FileNotFoundError(f"Missing signal file: {sig_path}")

    Z = np.load(sig_path)
    Z = np.asarray(Z, dtype=float)

    if PRINCIPAL_GRADIENT_TEST_DUR is not None:
        Z = Z[-int(PRINCIPAL_GRADIENT_TEST_DUR):, :]

    Z_use, roi = extract_cortical_signal_block(Z)

    if Z_use.shape[1] < 3:
        raise ValueError(f"Need at least 3 ROIs to compute a gradient, got {Z_use.shape[1]}")

    FC = np.corrcoef(Z_use.T)
    FC = np.nan_to_num(FC, nan=0.0, posinf=0.0, neginf=0.0)

    A_affinity = cosine_similarity(FC)
    A_affinity = np.nan_to_num(A_affinity, nan=0.0, posinf=0.0, neginf=0.0)

    gm = GradientMaps(
        n_components=PRINCIPAL_GRADIENT_N_COMPONENTS,
        approach="dm",
        kernel="normalized_angle",
    )
    gm.fit(A_affinity)

    grad = np.asarray(gm.gradients_[:, 0], dtype=float)

    return pd.DataFrame({
        "sub_id": str(sid),
        "roi": roi,
        "principal_gradient": grad,
    })


if COMPUTE_PRINCIPAL_GRADIENT:
    gradient_dfs = []
    failed_gradient_subjects = []

    for sid in subject_ids:
        print(f"Computing cortical principal gradient for subject {sid}")
        try:
            gradient_dfs.append(compute_principal_gradient_for_subject(sid))
        except Exception as exc:
            failed_gradient_subjects.append((sid, str(exc)))
            print(f"  Failed: {exc}")

    gradient_df = pd.concat(gradient_dfs, axis=0, ignore_index=True) if gradient_dfs else None
else:
    gradient_df = None
    failed_gradient_subjects = []

if gradient_df is not None:
    print("Cortical principal-gradient dataframe built successfully.")
    print("Shape:", gradient_df.shape)
    print("ROI range:", (gradient_df["roi"].min(), gradient_df["roi"].max()))
    print("ROIs per subject:")
    print(gradient_df.groupby("sub_id")["roi"].nunique().describe())
    print("Failed subjects:", len(failed_gradient_subjects))
    if failed_gradient_subjects:
        print(pd.DataFrame(failed_gradient_subjects, columns=["sub_id", "error"]))
    print(gradient_df.head())
else:
    print("Proceeding without principal-gradient columns.")
    if failed_gradient_subjects:
        print(pd.DataFrame(failed_gradient_subjects, columns=["sub_id", "error"]))


# ## Load and aggregate receptor maps from `data/Receptor maps/`
#
# Individual PET tracer maps are loaded and averaged into receptor-family columns such as `5HT1a`, `5HT2a`, `D2`, `GABA`, `mGluR5`, etc.
#
# Because this notebook is cortical-only, receptor maps are merged directly onto cortical ROI labels.
#
# Important for other datasets:
#
# - if the dataset is Schaefer-400 cortical-only, the current receptor maps should match directly;
# - if the dataset has cortex + subcortex but Schaefer-400 cortex, the subcortex has already been removed before this merge;
# - if the dataset uses a different cortical parcellation, for example Schaefer-100, you need receptor maps in that same parcellation and should set `CORTICAL_ROI_NUM` accordingly.
#

# In[11]:


# =============================================================================
# Receptor-map loading
# =============================================================================
RECEPTOR_TRACER_MAP = {
    "5HT1a": [
        "5HT1a_cumi_hc8_beliveau",
        "5HT1a_way_hc36_savli",
    ],
    "5HT1b": [
        "5HT1b_az_hc36_beliveau",
        "5HT1b_p943_hc22_savli",
        "5HT1b_p943_hc65_gallezot",
    ],
    "5HT2a": [
        "5HT2a_alt_hc19_savli",
        "5HT2a_cimbi_hc29_beliveau",
        "5HT2a_mdl_hc3_talbot",
    ],
    "5HTT": [
        "5HTT_dasb_hc100_beliveau",
        "5HTT_dasb_hc30_savli",
    ],
    "CB1": [
        "CB1_FMPEPd2_hc22_laurikainen",
        "CB1_omar_hc77_normandin",
    ],
    "D1": [
        "D1_SCH23390_hc13_kaller",
    ],
    "D2": [
        "D2_fallypride_hc49_jaworska",
        "D2_flb457_hc37_smith",
        "D2_flb457_hc55_sandiego",
        "D2_raclopride_hc7_alakurtti",
    ],
    "DAT": [
        "DAT_fepe2i_hc6_sasaki",
        "DAT_fpcit_hc174_dukart_spect",
    ],
    "GABA": [
        "GABAa-bz_flumazenil_hc16_norgaard",
        "GABAa_flumazenil_hc6_dukart",
    ],
    "MU": [
        "MU_carfentanil_hc204_kantonen",
        "MU_carfentanil_hc39_turtonen",
    ],
    "NAT": [
        "NAT_MRB_hc10_hesse",
        "NAT_MRB_hc77_ding",
    ],
    "Vacht": [
        "VAChT_feobv_hc18_aghourian_sum",
        "VAChT_feobv_hc4_tuominen",
        "VAChT_feobv_hc5_bedard_sum",
        # VAChT_feobv_hc3_spreng is intentionally excluded, as in Make_dataframe.ipynb.
    ],
    "mGluR5": [
        "mGluR5_abp_hc22_rosaneto",
        "mGluR5_abp_hc28_dubois",
        "mGluR5_abp_hc73_smart",
    ],

}


def clean_receptor_name(path):
    """Convert a receptor-map filename to a clean tracer name."""
    name = Path(path).name
    for suffix in ["_scale400.csv", ".csv", ".npy", ".txt", ".tsv", ".xlsx", ".xls"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_receptor_vector(path, n_rois=400):
    """Load one receptor-density vector from csv/npy/txt/xlsx-like files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)

    elif suffix == ".csv":
        # Standard comma-separated file.
        # Avoid sep=None because it triggers a ParserWarning.
        df = pd.read_csv(path, header=None)
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    elif suffix == ".tsv":
        # Tab-separated file.
        df = pd.read_csv(path, header=None, sep="\t")
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    elif suffix == ".txt":
        # Generic text file.
        # engine="python" avoids the ParserWarning when using sep=None.
        # If your .txt files are whitespace-separated, you can replace
        # sep=None, engine="python" with sep=r"\s+" for faster reading.
        df = pd.read_csv(path, header=None, sep=None, engine="python")
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, header=None)
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    else:
        raise ValueError(f"Unsupported receptor file format: {path}")

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        finite = arr[np.isfinite(arr)]
        if finite.size >= n_rois:
            return finite[:n_rois]

    if arr.ndim == 2:
        # Prefer a row or column with exactly n_rois numeric entries.
        for r in range(arr.shape[0]):
            vals = arr[r, :]
            vals = vals[np.isfinite(vals)]
            if vals.size == n_rois:
                return vals

        for c in range(arr.shape[1]):
            vals = arr[:, c]
            vals = vals[np.isfinite(vals)]
            if vals.size == n_rois:
                return vals

        finite = arr[np.isfinite(arr)]
        if finite.size >= n_rois:
            return finite[:n_rois]

    raise ValueError(f"Could not extract a length-{n_rois} vector from {path}; shape={arr.shape}")
def load_raw_receptor_maps(receptor_dir, n_receptor_rois=400):
    """Load all receptor-map files from `data/Receptor_maps/`."""
    receptor_dir = Path(receptor_dir)
    if not receptor_dir.exists():
        print(f"⚠️ Receptor-map folder does not exist: {receptor_dir}")
        return None

    patterns = ["*.csv", "*.npy", "*.txt", "*.tsv", "*.xlsx", "*.xls"]
    files = []
    for pattern in patterns:
        files.extend(sorted(receptor_dir.glob(pattern)))

    if len(files) == 0:
        print(f"⚠️ No receptor-map files found in {receptor_dir}")
        return None

    print(f"Found {len(files)} receptor-map files in {receptor_dir}")

    receptor_arrays = {}
    failed = []

    for f in files:
        name = clean_receptor_name(f)
        try:
            receptor_arrays[name] = load_receptor_vector(f, n_rois=n_receptor_rois)
        except Exception as exc:
            failed.append((f, str(exc)))
            print(f"  ⚠️ Failed to load {f.name}: {exc}")

    if len(receptor_arrays) == 0:
        print(f"⚠️ Receptor files were found, but no valid length-{n_receptor_rois} vectors could be loaded.")
        return None

    if failed:
        print("Some receptor files failed:")
        for f, msg in failed:
            print(f"  {f}: {msg}")

    raw_df = pd.DataFrame(receptor_arrays)
    raw_df.insert(0, "roi", np.arange(n_receptor_rois, dtype=int))
    return raw_df


def aggregate_receptor_maps(raw_receptors_df):
    """Average tracer maps into receptor-family maps."""
    if raw_receptors_df is None:
        return None

    raw = raw_receptors_df.copy()
    available = set(c for c in raw.columns if c != "roi")

    aggregated = pd.DataFrame({"roi": raw["roi"].astype(int)})
    used = set()

    for receptor_name, tracer_names in RECEPTOR_TRACER_MAP.items():
        present = [name for name in tracer_names if name in available]
        missing = [name for name in tracer_names if name not in available]

        if len(present) == 0:
            continue

        if missing:
            print(f"  ⚠️ {receptor_name}: using {len(present)}/{len(tracer_names)} tracer maps; missing {missing}")

        aggregated[f"receptor_{receptor_name}"] = raw[present].mean(axis=1)
        used.update(present)

    if KEEP_UNGROUPED_RECEPTOR_MAPS:
        leftovers = sorted(available - used)
        for col in leftovers:
            aggregated[f"receptor_raw_{col}"] = raw[col]

    if KEEP_RAW_RECEPTOR_TRACER_MAPS:
        for col in sorted(available):
            aggregated[f"receptor_tracer_{col}"] = raw[col]

    return aggregated


def map_cortical_receptors_to_final_rois(receptor_df, total_roi_num=450, receptor_roi_num=400, subcortical_position="last"):
    """Legacy helper. In this cortical-only notebook, receptors usually map directly to roi=0..N-1."""
    if receptor_df is None:
        return None

    receptor_df = receptor_df.copy()
    receptor_df["roi"] = receptor_df["roi"].astype(int)

    if total_roi_num == receptor_roi_num:
        return receptor_df

    if subcortical_position == "last":
        # User note: ROIs 400..449 are subcortical and receive NaN receptor values.
        receptor_df["roi"] = receptor_df["roi"]
    elif subcortical_position == "first":
        # Alternative atlas convention: first 50 are subcortical, cortical maps start at ROI 50.
        receptor_df["roi"] = receptor_df["roi"] + (total_roi_num - receptor_roi_num)
    else:
        raise ValueError("subcortical_position must be 'last' or 'first'")

    return receptor_df


# In[12]:


# =============================================================================
# Load receptor maps from data/Receptor maps and aggregate them
# =============================================================================
# Receptor maps are expected to match the cortical parcellation.
# For HCP/Schaefer-400 they map directly to roi = 0..399.
# For another cortical parcellation, set CORTICAL_ROI_NUM to the receptor-map length.
raw_receptor_df = load_raw_receptor_maps(
    receptor_dir,
    n_receptor_rois=CORTICAL_ROI_NUM,
)

receptor_df = aggregate_receptor_maps(raw_receptor_df)

if receptor_df is not None:
    receptor_df = receptor_df.copy()
    receptor_df["roi"] = receptor_df["roi"].astype(int)
    receptor_df = receptor_df.loc[receptor_df["roi"] < CORTICAL_ROI_NUM].copy()

    print("Cortical receptor dataframe loaded and aggregated.")
    print("Shape:", receptor_df.shape)
    print("ROI range:", (receptor_df["roi"].min(), receptor_df["roi"].max()))
    print("Columns:")
    print(list(receptor_df.columns))
    print(receptor_df.head())
else:
    print("Proceeding without receptor-map columns.")


# ## ROI/RSN annotations
#
# If a previous ROI-level dataframe exists, this cell can merge non-measure annotations such as `rsn_name`, `rsn_id`, or structural-connectivity metadata.
#

# In[13]:


# =============================================================================
# ROI/RSN annotations from rsn_names.mat
# =============================================================================
# rsn_names.mat contains Schaefer cortical parcels.
# For HCP/Schaefer-400 it maps directly to roi = 0..399.
# For another cortical parcellation, use an RSN file with the same number of cortical ROIs.
from scipy.io import loadmat


def _as_clean_string(x):
    """Convert MATLAB-loaded string-like objects to a normal Python string."""
    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return _as_clean_string(x.item())
        return "".join(str(v) for v in x.ravel())

    return str(x)


def parse_rsn_name(name):
    """Parse Schaefer names such as '7Networks_LH_Vis_1'."""
    name = _as_clean_string(name)
    parts = name.split("_")

    hemi = np.nan
    network = np.nan
    parcel_index = np.nan

    if len(parts) >= 3:
        hemi = parts[1]
        network = parts[2]

    if len(parts) >= 4:
        try:
            parcel_index = int(parts[-1])
        except Exception:
            parcel_index = np.nan

    return hemi, network, parcel_index


def find_rsn_names_mat():
    """Find rsn_names.mat in common local locations."""
    candidates = [
        script_dir / "rsn_names.mat",
        hcp_dir / "rsn_names.mat",
        repo_dir / "rsn_names.mat",
        data_dir / "rsn_names.mat",
        data_dir / "RSN" / "rsn_names.mat",
        data_dir / "Receptor_maps" / "rsn_names.mat",
        results_dir / "rsn_names.mat",
        dataframe_dir / "rsn_names.mat",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_rsn_annotations_from_mat(path=None):
    """Load Schaefer cortical RSN names from rsn_names.mat."""
    if path is None:
        path = find_rsn_names_mat()

    if path is None or not Path(path).exists():
        print("No rsn_names.mat file found. RSN annotation columns will not be added.")
        return None

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    if "rsn_names" not in mat:
        raise KeyError(f"{path} does not contain a variable named 'rsn_names'.")

    names = np.asarray(mat["rsn_names"], dtype=object).ravel()
    names = [_as_clean_string(x) for x in names]

    n_labels = len(names)
    n_use = min(n_labels, CORTICAL_ROI_NUM)

    if n_labels != CORTICAL_ROI_NUM:
        print(f"⚠️ rsn_names.mat has {n_labels} labels, expected {CORTICAL_ROI_NUM}. Using first {n_use}.")

    names = names[:n_use]
    roi = np.arange(n_use, dtype=int)

    parsed = [parse_rsn_name(name) for name in names]
    hemi, network, parcel_index = zip(*parsed)

    rsn_df = pd.DataFrame({
        "roi": roi,
        "rsn_name": names,
        "rsn_hemi": hemi,
        "rsn_network": network,
    })

    rsn_df["rsn_id"] = pd.Categorical(rsn_df["rsn_network"]).codes.astype(float)
    rsn_df.loc[rsn_df["rsn_network"].isna(), "rsn_id"] = np.nan

    print("Loaded cortical RSN annotations from:", path)
    print("RSN dataframe shape:", rsn_df.shape)
    print("RSN ROI range:", (rsn_df["roi"].min(), rsn_df["roi"].max()))
    print("RSN networks:", sorted(rsn_df["rsn_network"].dropna().unique()))

    return rsn_df


annotation_df = load_rsn_annotations_from_mat()

if annotation_df is not None:
    print(annotation_df.head())


# ## Merge and save the final dataframe
#

# In[14]:


# =============================================================================
# Merge spatial, network, receptor, gradient, and annotation dataframes
# =============================================================================
final_df = spatial_df.merge(
    network_df,
    on=["sub_id", "roi"],
    how="left",
    validate="one_to_one",
)

# Keep one original ROI column where possible.
if "roi_original_from_connectivity" in final_df.columns:
    final_df["roi_original"] = final_df["roi_original"].fillna(final_df["roi_original_from_connectivity"])
    final_df = final_df.drop(columns=["roi_original_from_connectivity"])

if gradient_df is not None:
    grad = gradient_df.copy()
    grad["sub_id"] = grad["sub_id"].astype(str)
    final_df = final_df.merge(
        grad[["sub_id", "roi", "principal_gradient"]].drop_duplicates(subset=["sub_id", "roi"]),
        on=["sub_id", "roi"],
        how="left",
    )

if annotation_df is not None:
    ann = annotation_df.copy()
    if "sub_id" in ann.columns:
        ann["sub_id"] = ann["sub_id"].astype(str)
        merge_keys = ["sub_id", "roi"]
    else:
        merge_keys = ["roi"]

    ann_cols = merge_keys + [c for c in ann.columns if c not in merge_keys and c not in final_df.columns]
    ann = ann[ann_cols].drop_duplicates(subset=merge_keys)
    final_df = final_df.merge(ann, on=merge_keys, how="left")

if receptor_df is not None:
    receptors = receptor_df.copy()
    receptor_cols = [c for c in receptors.columns if c != "roi"]
    receptors = receptors[["roi"] + receptor_cols].drop_duplicates(subset=["roi"])
    final_df = final_df.merge(receptors, on="roi", how="left")
else:
    receptor_cols = []

# Column ordering
annotation_cols = [
    "rsn_name",
    "rsn_id",
    "rsn_hemi",
    "rsn_network",
]

spatial_cols = [
    "n_timepoints",
    "mean_local_evoked_energy",
    "mean_local_effect_size",
    "mean_local_effect_direction",
    "mean_global_evoked_energy",
    "mean_global_effect_size",
    "mean_global_effect_direction",
    "r2_local_energy_to_local_evoked_energy",
    "r2_local_energy_to_global_evoked_energy",
    "r2_global_energy_to_global_evoked_energy",
    "r2_global_energy_to_local_evoked_energy",
    "r2_local_energy_to_local_effect_size",
    "r2_local_energy_to_global_effect_size",
    "r2_global_energy_to_global_effect_size",
    "r2_global_energy_to_local_effect_size",
    "r2_local_energy_to_local_effect_direction",
    "r2_local_energy_to_global_effect_direction",
    "r2_global_energy_to_global_effect_direction",
    "r2_global_energy_to_local_effect_direction",
]

network_cols = [
    "connectivity_kind",
    "hierarchy",
    "trophic_coherence",
    "principal_gradient",
    f"mean_{CONNECTIVITY_STRENGTH_PREFIX}_abs_out_strength",
    f"mean_{CONNECTIVITY_STRENGTH_PREFIX}_abs_in_strength",
]

receptor_cols = [c for c in final_df.columns if c.startswith("receptor_")]
first_cols = ["sub_id", "roi", "roi_original"]
ordered_cols = [c for c in first_cols + annotation_cols + spatial_cols + network_cols + receptor_cols if c in final_df.columns]
remaining_cols = [c for c in final_df.columns if c not in ordered_cols]
final_df = final_df[ordered_cols + remaining_cols]

print("Final dataframe built successfully.")
print("Connectivity kind:", CONNECTIVITY_KIND)
print("Shape:", final_df.shape)
print("Subjects:", final_df["sub_id"].nunique())
print("ROIs:", final_df["roi"].nunique())
print("ROI range:", (final_df["roi"].min(), final_df["roi"].max()))
print("Number of receptor-map columns:", len(receptor_cols))
print("Columns:")
print(list(final_df.columns))
final_df.head()


# In[15]:


# =============================================================================
# Quick cortical-only checks
# =============================================================================
print("Rows:", len(final_df))
print("Subjects:", final_df["sub_id"].nunique())
print("ROIs:", final_df["roi"].nunique())
print("ROI range:", (final_df["roi"].min(), final_df["roi"].max()))

expected_rois = min(CORTICAL_ROI_NUM, final_df["roi"].nunique())
if final_df["roi"].max() >= CORTICAL_ROI_NUM:
    print("⚠️ ROI labels exceed the cortical ROI range. Check ROI filtering.")
else:
    print(f"Cortical-only ROI check passed: all ROI labels are within 0..{CORTICAL_ROI_NUM - 1} or the available cortical range.")

qc_cols = [
    "mean_global_effect_size",
    "r2_global_energy_to_global_effect_size",
    "hierarchy",
    "trophic_coherence",
    "principal_gradient",
    "rsn_network",
]
qc_cols = [c for c in qc_cols if c in final_df.columns]
print("\nMissing values per selected columns:")
print(final_df[qc_cols].isna().sum())

if "rsn_network" in final_df.columns:
    print("\nRSN annotation QC:")
    print(final_df["rsn_network"].value_counts(dropna=False))

receptor_cols = [c for c in final_df.columns if c.startswith("receptor_")]
if receptor_cols:
    print("\nReceptor-map QC:")
    print("  Mean receptor NaNs:", float(final_df[receptor_cols].isna().mean().mean()))

print("\nSummary:")
summary_cols = [
    c for c in [
        "mean_global_effect_size",
        "r2_global_energy_to_global_effect_size",
        "hierarchy",
        "trophic_coherence",
        "principal_gradient",
    ] + receptor_cols[:5]
    if c in final_df.columns
]
print(final_df[summary_cols].describe().transpose())


# In[16]:


# =============================================================================
# Save final dataframe
# =============================================================================
final_df.to_csv(output_csv, index=False)
final_df.to_pickle(output_pkl)

print("Saved:")
print("  ", output_csv)
print("  ", output_pkl)
