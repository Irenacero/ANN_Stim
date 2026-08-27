#!/usr/bin/env python
# coding: utf-8

"""
Build the HCP background-dependence dataframe.

This script assumes that the following steps have already been completed:

1. Resting-state preprocessing
2. Subject-specific ANN model fitting
3. Time-resolved effective-connectivity extraction

Expected inputs
---------------
For each participant:

    results/processed/<subject>_inputs.npy
    results/ANN_model/<subject>_MLP.pt

and one selected connectivity tensor:

    results/ECts/<subject>_ECt.npy
    results/BECts/<subject>_BECt.npy

Saved outputs
-------------
Depending on ``--connectivity-kind``:

    results/HCP_4_df_background_dependence_ECts.csv
    results/HCP_4_df_background_dependence_ECts.pkl
    results/HCP_4_df_background_dependence_BECts.csv
    results/HCP_4_df_background_dependence_BECts.pkl

The same dataframe-building calculations are used for EC(t) and BEC(t),
matching the way the original notebook was executed for both tensor types.
"""

from pathlib import Path
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.serialization


# Occasionally needed to avoid local MKL/OpenMP conflicts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# =============================================================================
# Repository paths
# =============================================================================

script_dir = Path(__file__).resolve().parent
hcp_dir = script_dir.parent
repo_dir = hcp_dir.parent.parent

results_dir = hcp_dir / "results"
preproc_dir = results_dir / "processed"
models_dir = results_dir / "ANN_model"
save_dir = results_dir / "dataframes"
save_dir.mkdir(parents=True, exist_ok=True)

if str(hcp_dir) not in sys.path:
    sys.path.insert(0, str(hcp_dir))

from src import NPI


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Analysis parameters
# =============================================================================

method = "MLP"
ROI_num = 450
using_steps = 3

CONNECTIVITY_CONFIG = {
    "ECts": {
        "directory": results_dir / "ECts",
        "suffix": "_ECt.npy",
        "label": "EC_t",
    },
    "BECts": {
        "directory": results_dir / "BECts",
        "suffix": "_BECt.npy",
        "label": "BEC_t",
    },
}


# Allowlist model classes for recent PyTorch versions.
torch.serialization.add_safe_globals(
    [NPI.ANN_MLP, NPI.ANN_CNN, NPI.ANN_RNN, NPI.ANN_VAR]
)


def load_model(model_path, inputs=None, targets=None):
    """Load either a full serialized model or a state-dict checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if hasattr(ckpt, "eval"):
        model = ckpt.to(device)
        model.eval()
        return model

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        method_ckpt = ckpt.get("method", method)
        roi_ckpt = ckpt.get(
            "ROI_num",
            targets.shape[-1] if targets is not None else ROI_num,
        )
        steps_ckpt = ckpt.get("using_steps", using_steps)

        model = NPI.build_model(
            method_ckpt,
            roi_ckpt,
            steps_ckpt,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    raise ValueError("Unrecognized model format.")


def recover_current_state(inputs, n_regions):
    """
    Recover X_t from the last block of each flattened ANN input window.

    Expected input shape after preprocessing:
        (n_time, using_steps * n_regions)

    The final n_regions entries correspond to the current state at time t.
    """
    inputs = np.asarray(inputs)

    if inputs.ndim != 2:
        raise ValueError(
            f"Expected inputs to be 2D, got shape {inputs.shape}"
        )

    if inputs.shape[1] % n_regions != 0:
        raise ValueError(
            f"Input width {inputs.shape[1]} is not divisible by "
            f"n_regions={n_regions}"
        )

    return inputs[:, -n_regions:]


def build_background_dependence_df_for_subject(
    sid,
    model,
    inputs,
    connectivity_t,
):
    """
    Build one long-format dataframe for a single participant.

    Parameters
    ----------
    sid : str
        Participant identifier.
    model : torch.nn.Module
        Trained participant-specific ANN model.
    inputs : ndarray, shape (T, S*N)
        Preprocessed ANN input windows.
    connectivity_t : ndarray, shape (T, N, N)
        Selected time-resolved EC(t) or BEC(t) tensor.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with one row per
        (participant, ROI, time).
    """
    T_eff, N, N_check = connectivity_t.shape

    if N != N_check:
        raise ValueError(
            "EC_t must be square in its last two dimensions, "
            f"got {connectivity_t.shape}"
        )

    # Recover baseline state X_t.
    X_t = recover_current_state(
        inputs[:T_eff],
        N,
    )

    # Predict unperturbed next state X_{t+1}.
    with torch.no_grad():
        X_tp1 = model(
            torch.tensor(
                inputs[:T_eff],
                dtype=torch.float32,
                device=device,
            )
        )
        X_tp1 = X_tp1.detach().cpu().numpy()

    # Build evoked next state X^(j)_{t+1}.
    X_evoked = X_tp1[:, None, :] + connectivity_t

    # Baseline energies.
    global_baseline_energy = np.sum(X_t ** 2, axis=1)
    local_baseline_energy = X_t ** 2

    # Global measures.
    global_evoked_energy = np.sum(X_evoked ** 2, axis=2)
    global_effect_size = np.sum(connectivity_t ** 2, axis=2)
    global_effect_direction = np.sum(connectivity_t, axis=2)

    # Local measures: stimulate j and measure j.
    roi_idx = np.arange(N, dtype=int)
    local_evoked_state = X_evoked[:, roi_idx, roi_idx]
    local_effect_state = connectivity_t[:, roi_idx, roi_idx]

    local_evoked_energy = local_evoked_state ** 2
    local_effect_size = local_effect_state ** 2
    local_effect_direction = local_effect_state

    # Long-format indexing.
    time_idx = np.arange(T_eff, dtype=int)

    sub_id_col = np.repeat(sid, T_eff * N)
    roi_col = np.tile(roi_idx, T_eff)
    time_col = np.repeat(time_idx, N)

    global_baseline_energy_col = np.repeat(
        global_baseline_energy,
        N,
    )
    global_evoked_energy_col = global_evoked_energy.reshape(-1)
    global_effect_size_col = global_effect_size.reshape(-1)
    global_effect_direction_col = global_effect_direction.reshape(-1)

    local_baseline_energy_col = local_baseline_energy.reshape(-1)
    local_evoked_energy_col = local_evoked_energy.reshape(-1)
    local_effect_size_col = local_effect_size.reshape(-1)
    local_effect_direction_col = local_effect_direction.reshape(-1)

    return pd.DataFrame(
        {
            "sub_id": sub_id_col,
            "roi": roi_col,
            "time": time_col,
            "global_baseline_energy": global_baseline_energy_col,
            "global_evoked_energy": global_evoked_energy_col,
            "global_effect_size": global_effect_size_col,
            "global_effect_direction": global_effect_direction_col,
            "local_baseline_energy": local_baseline_energy_col,
            "local_evoked_energy": local_evoked_energy_col,
            "local_effect_size": local_effect_size_col,
            "local_effect_direction": local_effect_direction_col,
        }
    )


def locate_complete_subjects(connectivity_dir, connectivity_suffix):
    """Locate participants with inputs, models and the selected tensor."""
    input_files = sorted(preproc_dir.glob("*_inputs.npy"))
    model_files = sorted(models_dir.glob(f"*_{method}.pt"))
    connectivity_files = sorted(
        connectivity_dir.glob(f"*{connectivity_suffix}")
    )

    subject_ids_inputs = {
        path.name.split("_inputs.npy")[0]
        for path in input_files
    }
    subject_ids_models = {
        path.name.split(f"_{method}.pt")[0]
        for path in model_files
    }
    subject_ids_connectivity = {
        path.name.split(connectivity_suffix)[0]
        for path in connectivity_files
    }

    return sorted(
        subject_ids_inputs
        & subject_ids_models
        & subject_ids_connectivity
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the HCP background-dependence dataframe from "
            "EC(t) or BEC(t)."
        )
    )
    parser.add_argument(
        "--connectivity-kind",
        choices=sorted(CONNECTIVITY_CONFIG),
        default="ECts",
        help=(
            "Tensor family to process. The original notebook was "
            "executed separately for ECts and BECts."
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    connectivity_kind = args.connectivity_kind
    config = CONNECTIVITY_CONFIG[connectivity_kind]

    connectivity_dir = config["directory"]
    connectivity_suffix = config["suffix"]
    connectivity_label = config["label"]

    output_csv = (
        save_dir
        / f"HCP_4_df_background_dependence_{connectivity_kind}.csv"
    )
    output_pkl = (
        save_dir
        / f"HCP_4_df_background_dependence_{connectivity_kind}.pkl"
    )

    print("Repository directory :", repo_dir)
    print("Results directory    :", results_dir)
    print("Processed data dir   :", preproc_dir)
    print("Models directory     :", models_dir)
    print(f"{connectivity_label} directory       :", connectivity_dir)
    print("Output directory     :", save_dir)
    print("Device               :", device)

    print("\nAnalysis configuration:")
    print("  Method               :", method)
    print("  Number of regions    :", ROI_num)
    print("  Window length        :", using_steps)
    print("  Connectivity kind    :", connectivity_kind)
    print("  Tensor suffix        :", connectivity_suffix)
    print("  Output CSV           :", output_csv)
    print("  Output PKL           :", output_pkl)

    subject_ids = locate_complete_subjects(
        connectivity_dir,
        connectivity_suffix,
    )

    print(
        f"\nSubjects with complete inputs/models/{connectivity_label}:",
        len(subject_ids),
    )

    if subject_ids:
        print("First subjects:", subject_ids[:10])
    else:
        raise RuntimeError(
            "No complete participants found. Check processed data, "
            "trained models and selected connectivity outputs."
        )

    all_subject_dfs = []
    failed_subjects = []

    for sid in subject_ids:
        print(f"Processing subject {sid}")

        input_path = preproc_dir / f"{sid}_inputs.npy"
        model_path = models_dir / f"{sid}_{method}.pt"
        connectivity_path = (
            connectivity_dir / f"{sid}{connectivity_suffix}"
        )

        try:
            inputs = np.load(input_path)
            connectivity_t = np.load(connectivity_path)
            model = load_model(
                model_path,
                inputs=inputs,
            )

            df_sub = build_background_dependence_df_for_subject(
                sid=sid,
                model=model,
                inputs=inputs,
                connectivity_t=connectivity_t,
            )

            all_subject_dfs.append(df_sub)

            del model, inputs, connectivity_t, df_sub
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            failed_subjects.append((sid, str(exc)))
            print(f"  Failed: {exc}")

    if not all_subject_dfs:
        raise RuntimeError(
            "No participant dataframe could be created."
        )

    background_dependence_df = pd.concat(
        all_subject_dfs,
        axis=0,
        ignore_index=True,
    )

    print("\nFinal dataframe built successfully.")
    print("Shape:", background_dependence_df.shape)
    print("Columns:")
    print(list(background_dependence_df.columns))

    if failed_subjects:
        print("\nParticipants that failed during processing:")
        for sid, message in failed_subjects:
            print(f"  {sid}: {message}")

    background_dependence_df.to_csv(
        output_csv,
        index=False,
    )
    background_dependence_df.to_pickle(output_pkl)

    print("\nSaved:")
    print(" ", output_csv)
    print(" ", output_pkl)


if __name__ == "__main__":
    main()
