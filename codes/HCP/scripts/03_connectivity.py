#!/usr/bin/env python
# coding: utf-8

"""
Extract time-resolved effective connectivity (EC(t)) and bifocal effective
connectivity (BEC(t)) from the fitted participant-specific ANN models.

Expected inputs
---------------
For each participant:

    results/processed/<subject>_inputs.npy
    results/processed/<subject>_targets.npy
    results/ANN_model/<subject>_MLP.pt

Saved outputs
-------------

    results/ECts/<subject>_ECt.npy
    results/BECts/<subject>_BECt.npy

The numerical parameters and calls to NPI.model_ECt / NPI.model_BECt are kept
identical to the original analysis notebook.
"""

from pathlib import Path
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
ects_dir = results_dir / "ECts"
bects_dir = results_dir / "BECts"

ects_dir.mkdir(parents=True, exist_ok=True)
bects_dir.mkdir(parents=True, exist_ok=True)

if str(hcp_dir) not in sys.path:
    sys.path.insert(0, str(hcp_dir))

from src import NPI


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Original extraction parameters
# =============================================================================

method = "MLP"
ROI_num = 450
using_steps = 3
pert_strength = 0.1
max_timepoints = 500
compute_ECt = True
compute_BECt = True
bec_metric = "l2"
skip_existing = True


# =============================================================================
# Utilities
# =============================================================================

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
        roi_num_ckpt = ckpt.get(
            "ROI_num",
            targets.shape[-1] if targets is not None else ROI_num,
        )

        if inputs is not None and inputs.ndim == 2:
            using_steps_ckpt = ckpt.get(
                "using_steps",
                inputs.shape[1] // roi_num_ckpt,
            )
        else:
            using_steps_ckpt = ckpt.get("using_steps", using_steps)

        model = NPI.build_model(
            method_ckpt,
            roi_num_ckpt,
            using_steps_ckpt,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    raise ValueError(f"Unrecognized model file format: {model_path}")


def maybe_trim_timepoints(X, Y, max_tp=None):
    """Restrict the number of windows used for EC/BEC extraction."""
    if max_tp is None:
        return X, Y

    n_use = min(len(X), len(Y), max_tp)
    return X[:n_use], Y[:n_use]


def find_complete_subjects():
    """Return participants with inputs, targets, and a fitted model."""
    input_files = sorted(preproc_dir.glob("*_inputs.npy"))
    target_files = sorted(preproc_dir.glob("*_targets.npy"))
    model_files = sorted(models_dir.glob(f"*_{method}.pt"))

    subject_ids_inputs = {
        path.name.split("_inputs.npy")[0]
        for path in input_files
    }
    subject_ids_targets = {
        path.name.split("_targets.npy")[0]
        for path in target_files
    }
    subject_ids_models = {
        path.name.split(f"_{method}.pt")[0]
        for path in model_files
    }

    return sorted(
        subject_ids_inputs
        & subject_ids_targets
        & subject_ids_models
    )


def extract_ec(subjects):
    """Compute and save EC(t) for all valid participants."""
    summary_rows = []

    if not compute_ECt:
        print("compute_ECt = False -> skipping EC(t) extraction.")
        return pd.DataFrame(summary_rows)

    for sid in subjects:
        print(f"\n================ {sid} | EC(t) ================")

        inp_path = preproc_dir / f"{sid}_inputs.npy"
        tgt_path = preproc_dir / f"{sid}_targets.npy"
        mdl_path = models_dir / f"{sid}_{method}.pt"
        ect_path = ects_dir / f"{sid}_ECt.npy"

        if not inp_path.exists() or not tgt_path.exists() or not mdl_path.exists():
            print(f"Missing inputs, targets, or model for {sid}. Skipping.")
            continue

        if skip_existing and ect_path.exists():
            print("EC(t) file already exists. Skipping.")
            continue

        X = np.load(inp_path)
        Y = np.load(tgt_path)
        X_use, Y_use = maybe_trim_timepoints(X, Y, max_timepoints)

        print("Input windows shape   :", X.shape)
        print("Target windows shape  :", Y.shape)
        print("Using windows         :", X_use.shape[0])

        model = load_model(mdl_path, X, Y)
        print("Model loaded.")

        row = {
            "subject_id": sid,
            "n_windows": X_use.shape[0],
            "ECt_saved": False,
            "status": "ok",
        }

        try:
            with torch.no_grad():
                EC_t = NPI.model_ECt(
                    model,
                    input_X=X_use,
                    target_Y=Y_use,
                    pert_strength=pert_strength,
                )

            np.save(ect_path, EC_t)
            print("EC(t) computed        :", EC_t.shape)
            print("Saved EC(t)           :", ect_path)
            row["ECt_saved"] = True
            del EC_t

        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}"
            print(f"EC(t) extraction failed for {sid}: {exc}")

        summary_rows.append(row)

        del X, Y, X_use, Y_use, model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(summary_rows)


def extract_bec(subjects):
    """Compute and save BEC(t) for all valid participants."""
    summary_rows = []

    if not compute_BECt:
        print("compute_BECt = False -> skipping BEC(t) extraction.")
        return pd.DataFrame(summary_rows)

    for sid in subjects:
        print(f"\n================ {sid} | BEC(t) ================")

        inp_path = preproc_dir / f"{sid}_inputs.npy"
        tgt_path = preproc_dir / f"{sid}_targets.npy"
        mdl_path = models_dir / f"{sid}_{method}.pt"
        bect_path = bects_dir / f"{sid}_BECt.npy"

        if not inp_path.exists() or not tgt_path.exists() or not mdl_path.exists():
            print(f"Missing inputs, targets, or model for {sid}. Skipping.")
            summary_rows.append(
                {
                    "subject_id": sid,
                    "n_windows": np.nan,
                    "BECt_saved": False,
                    "status": "missing_inputs_or_model",
                }
            )
            continue

        if skip_existing and bect_path.exists():
            print("BEC(t) file already exists. Skipping.")
            summary_rows.append(
                {
                    "subject_id": sid,
                    "n_windows": np.nan,
                    "BECt_saved": True,
                    "status": "skipped_existing",
                }
            )
            continue

        X = np.load(inp_path)
        Y = np.load(tgt_path)
        X_use, Y_use = maybe_trim_timepoints(X, Y, max_timepoints)

        print("Input windows shape   :", X.shape)
        print("Target windows shape  :", Y.shape)
        print("Using windows         :", X_use.shape[0])

        model = load_model(mdl_path, X, Y)
        print("Model loaded.")

        row = {
            "subject_id": sid,
            "n_windows": X_use.shape[0],
            "BECt_saved": False,
            "status": "ok",
        }

        try:
            with torch.no_grad():
                BEC_t = NPI.model_BECt(
                    model,
                    input_X=X_use,
                    target_Y=Y_use,
                    pert_strength=pert_strength,
                    metric=bec_metric,
                )

            np.save(bect_path, BEC_t)
            print("BEC(t) computed       :", BEC_t.shape)
            print("Saved BEC(t)          :", bect_path)
            row["BECt_saved"] = True
            del BEC_t

        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}"
            print(f"BEC(t) extraction failed for {sid}: {exc}")

        summary_rows.append(row)

        del X, Y, X_use, Y_use, model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(summary_rows)


def print_summary(label, summary_df, output_pattern, output_dir):
    """Print a compact extraction summary."""
    if summary_df.empty:
        print(f"No {label} extraction summary available.")
        return

    print(f"\n{label} status summary:")
    print(summary_df["status"].value_counts(dropna=False).to_string())
    print(
        f"\n{label} files found:",
        len(list(output_dir.glob(output_pattern))),
    )


# =============================================================================
# Main execution
# =============================================================================


def main():
    print("Repository directory :", repo_dir)
    print("Results directory    :", results_dir)
    print("Processed data dir   :", preproc_dir)
    print("Models directory     :", models_dir)
    print("EC_t directory       :", ects_dir)
    print("BEC_t directory      :", bects_dir)
    print("PyTorch version      :", torch.__version__)
    print("CUDA available       :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Running on           :", torch.cuda.get_device_name(0))
    else:
        print("Running on           : CPU")

    print("\nShared extraction configuration:")
    print("  Method               :", method)
    print("  Number of regions    :", ROI_num)
    print("  Window length        :", using_steps)
    print("  Perturbation strength:", pert_strength)
    print("  Max time points      :", max_timepoints)
    print("  Compute EC(t)        :", compute_ECt)
    print("  Compute BEC(t)       :", compute_BECt)
    print("  BEC metric           :", bec_metric)
    print("  Skip existing files  :", skip_existing)

    subjects = find_complete_subjects()
    print(
        f"\nFound {len(subjects)} subjects with inputs, targets, "
        f"and {method} models."
    )

    if subjects:
        print("First subjects        :", subjects[:10])
    else:
        print(
            "No valid subjects found. Check results/processed and "
            "results/ANN_model."
        )
        return

    ec_summary_df = extract_ec(subjects)
    print_summary("EC(t)", ec_summary_df, "*_ECt.npy", ects_dir)

    bec_summary_df = extract_bec(subjects)
    print_summary("BEC(t)", bec_summary_df, "*_BECt.npy", bects_dir)


if __name__ == "__main__":
    main()
