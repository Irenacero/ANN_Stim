#!/usr/bin/env python
# coding: utf-8

# # 1 - HCP resting-state fMRI preprocessing
#
# This script preprocesses HCP resting-state fMRI data.
#
# Repository structure:
#
# repository/
# └── codes/
#     └── HCP/
#         ├── data/
#         │   └── Task/
#         │       └── language_subjects_paper100.txt
#         ├── results/
#         │   └── processed/
#         ├── scripts/
#         │   └── 01_preprocess_rest.py
#         └── src/
#
# The raw HCP resting-state data are stored outside the repository.
# Their location must be provided through the HCP_REST_ROOT environment
# variable.
#
# For example:
#
# export HCP_REST_ROOT="/path/to/Schafer400_Tian50"
#
# For each selected participant, the script:
#
# 1. Loads the four HCP resting-state runs
# 2. Fixes matrix orientation if needed
# 3. Removes the first time points of each run
# 4. Applies temporal band-pass filtering
# 5. Concatenates the runs
# 6. Builds supervised learning samples using a sliding window
# 7. Saves the outputs as .npy files
#
# Saved outputs
# -------------
#
# For each participant, three arrays are saved:
#
# 1. signals
#    Shape: (T, N)
#
# 2. inputs
#    Shape: (T - S, N * S)
#
# 3. targets
#    Shape: (T - S, N)


from pathlib import Path
import sys
import gc
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


# Occasionally needed to avoid local MKL/OpenMP conflicts when importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# =============================================================================
# Repository and external-data paths
# =============================================================================

script_dir = Path(__file__).resolve().parent
hcp_dir = script_dir.parent
repo_dir = hcp_dir.parent.parent

src_dir = hcp_dir / "src"
save_dir = hcp_dir / "results" / "processed"
subject_list_file = (
    hcp_dir
    / "data"
    / "Task"
    / "language_subjects_paper100.txt"
)

data_root = os.environ.get("HCP_REST_ROOT")

if not data_root:
    raise RuntimeError(
        "HCP_REST_ROOT is not defined. Set it to the directory containing "
        "the four parcellated HCP resting-state .mat files."
    )

data_dir = Path(data_root).expanduser().resolve()

if not data_dir.is_dir():
    raise FileNotFoundError(
        "HCP_REST_ROOT does not point to an existing directory: "
        f"{data_dir}"
    )

if not subject_list_file.is_file():
    raise FileNotFoundError(
        f"Participant list not found: {subject_list_file}"
    )

# codes/HCP must be importable so that src can be loaded as a package.
if str(hcp_dir) not in sys.path:
    sys.path.insert(0, str(hcp_dir))

print("Repository directory :", repo_dir)
print("HCP code directory   :", hcp_dir)
print("Source directory     :", src_dir)
print("Data directory       :", data_dir)
print("Participant list     :", subject_list_file)
print("Save directory       :", save_dir)

from src.preprocessing_hcp import bandpass_filter_timeseries
from src.NPI import multi2one


# =============================================================================
# Processing parameters
# =============================================================================

n_nodes = 450
remove_points = 30
using_steps = 3
dtype = np.float32


# =============================================================================
# Locate HCP fMRI run files
# =============================================================================

run_files = {
    "REST1_LR": (
        data_dir
        / "Schaefer2018_400Parcels_7Networks_order_"
          "Tian_Subcortex_S3_REST1_LR.mat"
    ),
    "REST1_RL": (
        data_dir
        / "Schaefer2018_400Parcels_7Networks_order_"
          "Tian_Subcortex_S3_REST1_RL.mat"
    ),
    "REST2_LR": (
        data_dir
        / "Schaefer2018_400Parcels_7Networks_order_"
          "Tian_Subcortex_S3_REST2_LR.mat"
    ),
    "REST2_RL": (
        data_dir
        / "Schaefer2018_400Parcels_7Networks_order_"
          "Tian_Subcortex_S3_REST2_RL.mat"
    ),
}

run_order = [
    "REST1_LR",
    "REST1_RL",
    "REST2_LR",
    "REST2_RL",
]

missing_run_files = [
    str(run_files[run_key])
    for run_key in run_order
    if not run_files[run_key].is_file()
]

if missing_run_files:
    raise FileNotFoundError(
        "The following resting-state files were not found:\n"
        + "\n".join(missing_run_files)
    )

print(f"Found {len(run_files)} fMRI runs:")
for run_key in run_order:
    print("  -", run_key)


# =============================================================================
# Select the 100 participants used in the paper
# =============================================================================

def list_subjects(h5path, run_key):
    """
    Return the participant identifiers present in one resting-state run.
    """
    with h5py.File(h5path, "r") as h5_file:
        return list(h5_file["HCP"][run_key].keys())


subject_sets = [
    set(list_subjects(run_files[run_key], run_key))
    for run_key in run_order
]

common_h5_keys = set.intersection(*subject_sets)

# HDF5 participant keys may contain a prefix such as "id_".
# Map the numeric participant ID to the complete HDF5 key.
h5_key_by_numeric_id = {
    key.split("_")[-1]: key
    for key in common_h5_keys
}

with subject_list_file.open("r", encoding="utf-8") as file:
    paper_subject_ids = [
        line.strip()
        for line in file
        if line.strip()
    ]

if len(paper_subject_ids) != 100:
    raise ValueError(
        "Expected 100 participant IDs in the paper participant list, "
        f"but found {len(paper_subject_ids)}."
    )

if len(set(paper_subject_ids)) != len(paper_subject_ids):
    raise ValueError(
        "The paper participant list contains duplicate IDs."
    )

missing_subjects = [
    subject_id
    for subject_id in paper_subject_ids
    if subject_id not in h5_key_by_numeric_id
]

if missing_subjects:
    raise ValueError(
        "The following paper participants are absent from at least one "
        "resting-state run:\n"
        + "\n".join(missing_subjects)
    )

# Preserve the order in language_subjects_paper100.txt.
subject_ids = [
    h5_key_by_numeric_id[subject_id]
    for subject_id in paper_subject_ids
]

print(f"Subjects present in all runs: {len(common_h5_keys)}")
print(f"Paper participants selected: {len(subject_ids)}")


# =============================================================================
# Main preprocessing loop
# =============================================================================

save_dir.mkdir(parents=True, exist_ok=True)

for sid in subject_ids:
    print(f"\nProcessing subject {sid}")
    subj_runs = []

    for run_key in run_order:
        with h5py.File(run_files[run_key], "r") as h5_file:
            ts = h5_file["HCP"][run_key][sid]["ts"][()]

        # Ensure shape is (time, regions).
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T

        # Remove initial TRs and keep the selected nodes.
        ts = ts[remove_points:, :n_nodes].astype(
            dtype,
            copy=False,
        )

        # Apply temporal filtering.
        ts_filt = bandpass_filter_timeseries(ts).astype(
            dtype,
            copy=False,
        )
        subj_runs.append(ts_filt)

        del ts, ts_filt
        gc.collect()

    # Concatenate all runs for the current participant.
    signals = np.concatenate(
        subj_runs,
        axis=0,
    ).astype(dtype, copy=False)

    # Build supervised samples.
    inputs, targets = multi2one(
        signals,
        steps=using_steps,
    )

    # Save outputs.
    np.save(
        save_dir / f"{sid}_signals.npy",
        signals,
    )
    np.save(
        save_dir / f"{sid}_inputs.npy",
        inputs,
    )
    np.save(
        save_dir / f"{sid}_targets.npy",
        targets,
    )

    del subj_runs, signals, inputs, targets
    gc.collect()

print("\nProcessing complete.")
