#!/usr/bin/env python
# coding: utf-8

"""
Smoke-test the cleaned HCP pipeline using one participant and existing outputs.

This test does not overwrite analysis files. It checks:

1. Processed arrays and their expected shapes.
2. Loading the saved participant-specific ANN model.
3. A forward prediction from the model.
4. Existing EC(t) and BEC(t) tensor shapes and finite values.
5. Step-4 dataframe construction on a small number of time points.
6. Step-5 summary calculations on those small dataframes.
7. Prerequisites needed by step 6.
8. Optionally, reproduction of step-1 preprocessing for one participant from
   the four raw resting-state HDF5 files.

Place this file at:

    codes/HCP/tests/smoke_test_existing_outputs.py

Example:

    python codes/HCP/tests/smoke_test_existing_outputs.py \
        --subject 100206 \
        --timepoints 5

Optional step-1 comparison:

    python codes/HCP/tests/smoke_test_existing_outputs.py \
        --subject 100206 \
        --timepoints 5 \
        --rest-root /path/to/Schaefer400_Tian50 \
        --compare-preprocessing
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch


EXPECTED_ROIS = 450
EXPECTED_STEPS = 3
EXPECTED_INPUT_WIDTH = EXPECTED_ROIS * EXPECTED_STEPS

EXPECTED_STEP4_COLUMNS = {
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
}


class SmokeTestFailure(RuntimeError):
    """Raised when an expected pipeline property is not satisfied."""


def infer_hcp_dir(explicit_hcp_dir: str | None) -> Path:
    """Locate codes/HCP from an explicit argument, script location, or CWD."""
    candidates: list[Path] = []

    if explicit_hcp_dir:
        candidates.append(Path(explicit_hcp_dir).expanduser())

    script_path = Path(__file__).resolve()
    candidates.extend(
        [
            script_path.parent.parent,
            Path.cwd() / "codes" / "HCP",
            Path.cwd(),
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (
            (candidate / "scripts").is_dir()
            and (candidate / "src").is_dir()
            and (candidate / "results").is_dir()
        ):
            return candidate

    raise SmokeTestFailure(
        "Could not locate codes/HCP. Pass --hcp-dir /path/to/repository/codes/HCP."
    )


def import_script(path: Path, module_name: str) -> ModuleType:
    """Import a numbered analysis script without executing its main block."""
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise SmokeTestFailure(f"Could not import script: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    return subject if subject.startswith("id_") else f"id_{subject}"


def subjects_from_suffix(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()

    return {
        path.name[: -len(suffix)]
        for path in directory.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    }


def choose_subject(hcp_dir: Path, requested_subject: str | None) -> str:
    results_dir = hcp_dir / "results"

    subject_sets = {
        "inputs": subjects_from_suffix(
            results_dir / "processed",
            "_inputs.npy",
        ),
        "targets": subjects_from_suffix(
            results_dir / "processed",
            "_targets.npy",
        ),
        "models": subjects_from_suffix(
            results_dir / "ANN_model",
            "_MLP.pt",
        ),
    }

    common = set.intersection(*subject_sets.values())

    if requested_subject:
        subject = normalize_subject(requested_subject)

        if subject not in common:
            missing = [
                label
                for label, subjects in subject_sets.items()
                if subject not in subjects
            ]
            raise SmokeTestFailure(
                f"{subject} is missing required files for: {', '.join(missing)}"
            )

        return subject

    if not common:
        counts = ", ".join(
            f"{label}={len(subjects)}"
            for label, subjects in subject_sets.items()
        )
        raise SmokeTestFailure(
            "No participant has inputs, targets, and a trained model. "
            f"Available counts: {counts}"
        )

    return sorted(common)[0]


def require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise SmokeTestFailure(
            f"{name} contains {count} non-finite values."
        )


def validate_processed_arrays(
    hcp_dir: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    processed_dir = hcp_dir / "results" / "processed"

    inputs_path = processed_dir / f"{subject}_inputs.npy"
    targets_path = processed_dir / f"{subject}_targets.npy"
    signals_path = processed_dir / f"{subject}_signals.npy"

    inputs = np.load(inputs_path, mmap_mode="r")
    targets = np.load(targets_path, mmap_mode="r")
    signals = (
        np.load(signals_path, mmap_mode="r")
        if signals_path.is_file()
        else None
    )

    if inputs.ndim != 2 or inputs.shape[1] != EXPECTED_INPUT_WIDTH:
        raise SmokeTestFailure(
            f"Unexpected inputs shape: {inputs.shape}; expected (*, "
            f"{EXPECTED_INPUT_WIDTH})."
        )

    if targets.ndim != 2 or targets.shape[1] != EXPECTED_ROIS:
        raise SmokeTestFailure(
            f"Unexpected targets shape: {targets.shape}; expected (*, "
            f"{EXPECTED_ROIS})."
        )

    if inputs.shape[0] != targets.shape[0]:
        raise SmokeTestFailure(
            "Inputs and targets have different numbers of samples: "
            f"{inputs.shape[0]} and {targets.shape[0]}."
        )

    require_finite("inputs sample", np.asarray(inputs[:5]))
    require_finite("targets sample", np.asarray(targets[:5]))

    if signals is not None:
        if signals.ndim != 2 or signals.shape[1] != EXPECTED_ROIS:
            raise SmokeTestFailure(
                f"Unexpected signals shape: {signals.shape}; expected (*, "
                f"{EXPECTED_ROIS})."
            )

        if signals.shape[0] - EXPECTED_STEPS != inputs.shape[0]:
            raise SmokeTestFailure(
                "signals, inputs, and window length are inconsistent: "
                f"{signals.shape[0]} - {EXPECTED_STEPS} != {inputs.shape[0]}."
            )

        require_finite("signals sample", np.asarray(signals[:5]))

    print("[OK] Processed arrays")
    print("     inputs :", inputs.shape, inputs.dtype)
    print("     targets:", targets.shape, targets.dtype)
    print(
        "     signals:",
        "MISSING" if signals is None else f"{signals.shape} {signals.dtype}",
    )

    return inputs, targets, signals


def validate_model(
    hcp_dir: Path,
    subject: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    connectivity_module: ModuleType,
) -> torch.nn.Module:
    model_path = (
        hcp_dir
        / "results"
        / "ANN_model"
        / f"{subject}_MLP.pt"
    )

    model = connectivity_module.load_model(
        model_path,
        inputs=inputs,
        targets=targets,
    )

    device = connectivity_module.device
    sample = torch.tensor(
        np.asarray(inputs[:2]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        prediction = model(sample).detach().cpu().numpy()

    if prediction.shape != (2, EXPECTED_ROIS):
        raise SmokeTestFailure(
            f"Unexpected model prediction shape: {prediction.shape}; "
            f"expected (2, {EXPECTED_ROIS})."
        )

    require_finite("model prediction", prediction)

    print("[OK] Saved model loads and predicts")
    print("     prediction:", prediction.shape, prediction.dtype)
    print("     device    :", device)

    return model


def validate_connectivity_tensor(
    path: Path,
    label: str,
    expected_timepoints: int,
    smoke_timepoints: int,
) -> np.ndarray:
    if not path.is_file():
        raise SmokeTestFailure(f"Missing {label} tensor: {path}")

    tensor = np.load(path, mmap_mode="r")

    if tensor.ndim != 3:
        raise SmokeTestFailure(
            f"{label} must be 3D, got {tensor.shape}."
        )

    if tensor.shape[1:] != (EXPECTED_ROIS, EXPECTED_ROIS):
        raise SmokeTestFailure(
            f"Unexpected {label} shape: {tensor.shape}; expected "
            f"(T, {EXPECTED_ROIS}, {EXPECTED_ROIS})."
        )

    if tensor.shape[0] != expected_timepoints:
        raise SmokeTestFailure(
            f"Unexpected {label} time dimension: {tensor.shape[0]}; "
            f"expected {expected_timepoints} from step 3."
        )

    n_use = min(smoke_timepoints, tensor.shape[0])
    sample = np.asarray(tensor[:n_use])
    require_finite(f"{label} sample", sample)

    print(f"[OK] Existing {label} tensor")
    print("     path :", path)
    print("     shape:", tensor.shape, tensor.dtype)

    return sample


def validate_step4_and_step5(
    subject: str,
    model: torch.nn.Module,
    inputs: np.ndarray,
    connectivity_sample: np.ndarray,
    connectivity_kind: str,
    background_module: ModuleType,
    summary_module: ModuleType,
) -> None:
    df = background_module.build_background_dependence_df_for_subject(
        sid=subject,
        model=model,
        inputs=inputs,
        connectivity_t=connectivity_sample,
    )

    expected_rows = connectivity_sample.shape[0] * EXPECTED_ROIS

    if len(df) != expected_rows:
        raise SmokeTestFailure(
            f"{connectivity_kind} step-4 dataframe has {len(df)} rows; "
            f"expected {expected_rows}."
        )

    missing_columns = EXPECTED_STEP4_COLUMNS - set(df.columns)

    if missing_columns:
        raise SmokeTestFailure(
            f"{connectivity_kind} step-4 dataframe is missing columns: "
            f"{sorted(missing_columns)}"
        )

    numeric = df.select_dtypes(include=[np.number]).to_numpy()
    require_finite(
        f"{connectivity_kind} step-4 dataframe",
        numeric,
    )

    subject_global = summary_module.compute_subject_global_table(df)

    if len(subject_global) != connectivity_sample.shape[0]:
        raise SmokeTestFailure(
            f"{connectivity_kind} subject summary has "
            f"{len(subject_global)} rows; expected "
            f"{connectivity_sample.shape[0]}."
        )

    corr_global, _ = summary_module.compute_global_effect_correlations(
        subject_global
    )
    corr_poststim, _ = summary_module.compute_poststim_correlations(
        subject_global
    )
    node_corr = summary_module.compute_node_correlations(df)

    if len(corr_global) != 1 or len(corr_poststim) != 1:
        raise SmokeTestFailure(
            f"{connectivity_kind} participant-level summaries should "
            "contain exactly one participant."
        )

    if len(node_corr) != EXPECTED_ROIS:
        raise SmokeTestFailure(
            f"{connectivity_kind} node summary has {len(node_corr)} rows; "
            f"expected {EXPECTED_ROIS}."
        )

    print(
        f"[OK] Steps 4 and 5 on {connectivity_kind} "
        f"({connectivity_sample.shape[0]} time points)"
    )
    print("     step-4 dataframe:", df.shape)
    print("     node summary    :", node_corr.shape)


def find_rsn_names(hcp_dir: Path) -> Path | None:
    repo_dir = hcp_dir.parent.parent
    data_dir_candidates = [
        hcp_dir.parent / "data",
        hcp_dir / "data",
    ]
    data_dir = next(
        (path for path in data_dir_candidates if path.exists()),
        data_dir_candidates[0],
    )

    candidates = [
        hcp_dir / "scripts" / "rsn_names.mat",
        hcp_dir / "rsn_names.mat",
        repo_dir / "rsn_names.mat",
        data_dir / "rsn_names.mat",
        data_dir / "RSN" / "rsn_names.mat",
        data_dir / "Receptor_maps" / "rsn_names.mat",
        hcp_dir / "results" / "rsn_names.mat",
        hcp_dir / "results" / "dataframes" / "rsn_names.mat",
    ]

    return next((path for path in candidates if path.is_file()), None)


def validate_step6_prerequisites(
    hcp_dir: Path,
    subject: str,
) -> None:
    failures: list[str] = []

    try:
        import brainspace  # noqa: F401

        brainspace_status = "installed"
    except Exception as exc:
        brainspace_status = f"missing ({type(exc).__name__}: {exc})"
        failures.append("brainspace")

    data_dir_candidates = [
        hcp_dir.parent / "data",
        hcp_dir / "data",
    ]
    data_dir = next(
        (path for path in data_dir_candidates if path.exists()),
        data_dir_candidates[0],
    )
    receptor_dir = data_dir / "Receptor_maps"

    receptor_patterns = [
        "*.csv",
        "*.npy",
        "*.txt",
        "*.tsv",
        "*.xlsx",
        "*.xls",
    ]
    receptor_files: list[Path] = []

    if receptor_dir.is_dir():
        for pattern in receptor_patterns:
            receptor_files.extend(receptor_dir.glob(pattern))

    if not receptor_files:
        failures.append("receptor maps")

    rsn_names = find_rsn_names(hcp_dir)

    if rsn_names is None:
        failures.append("rsn_names.mat")

    signals_path = (
        hcp_dir
        / "results"
        / "processed"
        / f"{subject}_signals.npy"
    )

    if not signals_path.is_file():
        failures.append(f"{subject}_signals.npy")

    print("[INFO] Step-6 prerequisites")
    print("       brainspace    :", brainspace_status)
    print("       receptor dir  :", receptor_dir)
    print("       receptor files:", len(receptor_files))
    print("       rsn_names.mat :", rsn_names or "MISSING")
    print("       signals       :", signals_path if signals_path.is_file() else "MISSING")

    if failures:
        print(
            "[WARN] Step 6 cannot be fully executed yet; missing: "
            + ", ".join(failures)
        )
    else:
        print("[OK] Step-6 prerequisites are present")


def compare_preprocessing(
    hcp_dir: Path,
    subject: str,
    rest_root: Path,
    stored_inputs: np.ndarray,
    stored_targets: np.ndarray,
    stored_signals: np.ndarray | None,
) -> None:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise SmokeTestFailure(
            "h5py is required for --compare-preprocessing."
        ) from exc

    if stored_signals is None:
        raise SmokeTestFailure(
            "The stored signals file is required to compare step 1."
        )

    if str(hcp_dir) not in sys.path:
        sys.path.insert(0, str(hcp_dir))

    from src.NPI import multi2one
    from src.preprocessing_hcp import bandpass_filter_timeseries

    run_files = {
        "REST1_LR": (
            rest_root
            / "Schaefer2018_400Parcels_7Networks_order_"
              "Tian_Subcortex_S3_REST1_LR.mat"
        ),
        "REST1_RL": (
            rest_root
            / "Schaefer2018_400Parcels_7Networks_order_"
              "Tian_Subcortex_S3_REST1_RL.mat"
        ),
        "REST2_LR": (
            rest_root
            / "Schaefer2018_400Parcels_7Networks_order_"
              "Tian_Subcortex_S3_REST2_LR.mat"
        ),
        "REST2_RL": (
            rest_root
            / "Schaefer2018_400Parcels_7Networks_order_"
              "Tian_Subcortex_S3_REST2_RL.mat"
        ),
    }

    missing = [
        path
        for path in run_files.values()
        if not path.is_file()
    ]

    if missing:
        raise SmokeTestFailure(
            "Missing raw resting-state files:\n"
            + "\n".join(str(path) for path in missing)
        )

    numeric_subject = subject.split("_")[-1]
    subject_runs: list[np.ndarray] = []

    for run_key, path in run_files.items():
        with h5py.File(path, "r") as h5_file:
            run_group = h5_file["HCP"][run_key]
            matching_keys = [
                key
                for key in run_group.keys()
                if key.split("_")[-1] == numeric_subject
            ]

            if len(matching_keys) != 1:
                raise SmokeTestFailure(
                    f"Could not uniquely resolve {subject} in {path}; "
                    f"matches={matching_keys}"
                )

            ts = run_group[matching_keys[0]]["ts"][()]

        if ts.shape[0] < ts.shape[1]:
            ts = ts.T

        ts = ts[30:, :EXPECTED_ROIS].astype(
            np.float32,
            copy=False,
        )
        ts_filtered = bandpass_filter_timeseries(ts).astype(
            np.float32,
            copy=False,
        )
        subject_runs.append(ts_filtered)

    signals = np.concatenate(subject_runs, axis=0).astype(
        np.float32,
        copy=False,
    )
    inputs, targets = multi2one(
        signals,
        steps=EXPECTED_STEPS,
    )

    comparisons = {
        "signals": (
            signals,
            np.asarray(stored_signals),
        ),
        "inputs": (
            inputs,
            np.asarray(stored_inputs),
        ),
        "targets": (
            targets,
            np.asarray(stored_targets),
        ),
    }

    for label, (new, old) in comparisons.items():
        if new.shape != old.shape:
            raise SmokeTestFailure(
                f"Step-1 {label} shape mismatch: regenerated={new.shape}, "
                f"stored={old.shape}."
            )

        max_abs_diff = float(
            np.max(
                np.abs(
                    new.astype(np.float64)
                    - old.astype(np.float64)
                )
            )
        )
        matches = np.allclose(
            new,
            old,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

        print(
            f"[{'OK' if matches else 'FAIL'}] Step-1 {label} comparison: "
            f"max |difference| = {max_abs_diff:.3e}"
        )

        if not matches:
            raise SmokeTestFailure(
                f"Regenerated step-1 {label} does not match the stored output."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test one participant through the cleaned HCP pipeline "
            "using existing processed/model/EC/BEC outputs."
        )
    )
    parser.add_argument(
        "--hcp-dir",
        default=None,
        help="Path to repository/codes/HCP. Usually inferred automatically.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Participant to test, for example 100206 or id_100206. "
            "Defaults to the first participant with inputs, targets, and model."
        ),
    )
    parser.add_argument(
        "--timepoints",
        type=int,
        default=5,
        help=(
            "Number of existing EC/BEC time points used for the small "
            "step-4/5 test. Must be at least 3."
        ),
    )
    parser.add_argument(
        "--skip-ec",
        action="store_true",
        help="Skip testing the existing EC(t) output.",
    )
    parser.add_argument(
        "--skip-bec",
        action="store_true",
        help="Skip testing the existing BEC(t) output.",
    )
    parser.add_argument(
        "--rest-root",
        default=None,
        help=(
            "Directory containing the four raw resting-state HDF5 .mat files."
        ),
    )
    parser.add_argument(
        "--compare-preprocessing",
        action="store_true",
        help=(
            "Regenerate step-1 arrays for the selected participant and "
            "compare them with the stored arrays. Requires --rest-root and h5py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.timepoints < 3:
        raise SmokeTestFailure("--timepoints must be at least 3.")

    if args.compare_preprocessing and not args.rest_root:
        raise SmokeTestFailure(
            "--compare-preprocessing requires --rest-root."
        )

    hcp_dir = infer_hcp_dir(args.hcp_dir)

    if str(hcp_dir) not in sys.path:
        sys.path.insert(0, str(hcp_dir))

    scripts_dir = hcp_dir / "scripts"
    subject = choose_subject(hcp_dir, args.subject)

    print("HCP directory :", hcp_dir)
    print("Participant   :", subject)
    print("Smoke points  :", args.timepoints)
    print()

    connectivity_module = import_script(
        scripts_dir / "03_connectivity.py",
        "hcp_step03_connectivity",
    )
    background_module = import_script(
        scripts_dir / "04_background_dependence.py",
        "hcp_step04_background",
    )
    summary_module = import_script(
        scripts_dir / "05_background_dependence_summary.py",
        "hcp_step05_summary",
    )

    inputs, targets, signals = validate_processed_arrays(
        hcp_dir,
        subject,
    )
    model = validate_model(
        hcp_dir,
        subject,
        inputs,
        targets,
        connectivity_module,
    )

    expected_timepoints = min(
        len(inputs),
        len(targets),
        connectivity_module.max_timepoints,
    )

    connectivity_tests = []

    if not args.skip_ec:
        connectivity_tests.append(
            (
                "ECts",
                hcp_dir
                / "results"
                / "ECts"
                / f"{subject}_ECt.npy",
            )
        )

    if not args.skip_bec:
        connectivity_tests.append(
            (
                "BECts",
                hcp_dir
                / "results"
                / "BECts"
                / f"{subject}_BECt.npy",
            )
        )

    for connectivity_kind, path in connectivity_tests:
        sample = validate_connectivity_tensor(
            path=path,
            label=connectivity_kind,
            expected_timepoints=expected_timepoints,
            smoke_timepoints=args.timepoints,
        )
        validate_step4_and_step5(
            subject=subject,
            model=model,
            inputs=inputs,
            connectivity_sample=sample,
            connectivity_kind=connectivity_kind,
            background_module=background_module,
            summary_module=summary_module,
        )

    validate_step6_prerequisites(
        hcp_dir,
        subject,
    )

    if args.compare_preprocessing:
        compare_preprocessing(
            hcp_dir=hcp_dir,
            subject=subject,
            rest_root=Path(args.rest_root).expanduser().resolve(),
            stored_inputs=inputs,
            stored_targets=targets,
            stored_signals=signals,
        )

    print()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except SmokeTestFailure as exc:
        print()
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
