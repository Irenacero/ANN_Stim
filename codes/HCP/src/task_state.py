"""
task_state.py — Window extractor for the task-state gating experiment.

Goal: take a participant's HCP LANGUAGE task BOLD and its block-onset timing,
and produce S-step input windows (the format the trained twin consumes) whose
*current state* x(t) sits at the "before", "during", or "after" epoch of each
task block. Feeding these windows to the twin (NPI.model_ECt / collect_state_
effect_pairs) then yields the evoked response conditioned on a task-driven state,
the task-side counterpart of the resting-state gating analysis.

Conventions (must match the trained twins):
- BOLD is (T, N) = timepoints x regions after transposing the (N, T) .mat array.
- Preprocessing matches the rest pipeline: linear detrend -> 2nd-order Butterworth
  band-pass 0.008-0.08 Hz -> per-region z-score (preprocessing_hcp).
- Window length S = 3, regions N = 450, TR = 0.72 s.
- A window ending at current-time t is time_series[t-S+1 : t+1] flattened to (S*N,),
  i.e. multi2one()'s row i corresponds to current-time t = i + S - 1. The last
  frame of the window is x(t); baseline energy is E(t) = sum_i x_i(t)^2.

Hemodynamic lag: a block onset at volume v_onset only shows up in the BOLD state
~4-6 s later, so the task-reflecting ("during") state is anchored at v_onset + lag,
while the state AT v_onset is effectively the pre-task ("before") baseline.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import scipy.io as sio

from preprocessing_hcp import bandpass_filter_timeseries, TR

# Geometry of the trained twins
S_WINDOW = 3
N_ROI = 450
DEFAULT_LAG_S = 5.0   # hemodynamic delay used to anchor the "during" state

# Default location of the task data (relative to codes/HCP)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "Task")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_language_bold(subject: str, data_root: str = DATA_ROOT, preprocess: bool = True) -> np.ndarray:
    """
    Load one participant's LANGUAGE LR parcellated BOLD.

    Returns (T, N) = (316, 450). If preprocess=True, applies the same
    detrend + band-pass + z-score used to train the twins.
    """
    path = os.path.join(data_root, "Language", f"{subject}_LANGUAGE_LR_schaefer_400_Tian_S3.mat")
    ts = sio.loadmat(path)["schaefer_tian_ts"]  # (N, T)
    ts = np.asarray(ts, dtype=float).T           # (T, N)
    if preprocess:
        ts = bandpass_filter_timeseries(ts)      # (T, N), z-scored per region
    return ts


def load_block_onsets(
    subject: str,
    conditions: tuple[str, ...] = ("story", "math"),
    data_root: str = DATA_ROOT,
) -> list[dict]:
    """
    Read the LANGUAGE_LR block-onset EVs (FSL 3-column: onset_s, dur_s, weight).

    Pools the requested conditions (default story + math = every task block) and
    returns a chronologically sorted list of dicts with keys
    {'condition', 'onset_s', 'dur_s'}.
    """
    ev_dir = os.path.join(
        data_root, "HCP_TASKS_EVs", subject,
        "MNINonLinear", "Results", "tfMRI_LANGUAGE_LR", "EVs",
    )
    blocks = []
    for cond in conditions:
        f = os.path.join(ev_dir, f"{cond}.txt")
        if not os.path.isfile(f):
            continue
        rows = np.loadtxt(f, ndmin=2)            # (n_blocks, 3)
        for onset_s, dur_s, _w in rows:
            blocks.append({"condition": cond, "onset_s": float(onset_s), "dur_s": float(dur_s)})
    blocks.sort(key=lambda b: b["onset_s"])
    return blocks


def list_subjects(data_root: str = DATA_ROOT) -> list[str]:
    """Subject ids that have both LANGUAGE BOLD and LANGUAGE_LR EVs."""
    out = []
    for p in sorted(glob.glob(os.path.join(data_root, "Language", "*_LANGUAGE_LR_*.mat"))):
        subj = os.path.basename(p).split("_")[0]
        ev = os.path.join(data_root, "HCP_TASKS_EVs", subj,
                          "MNINonLinear", "Results", "tfMRI_LANGUAGE_LR", "EVs")
        if os.path.isdir(ev):
            out.append(subj)
    return out


# -----------------------------------------------------------------------------
# Anchor times (which volumes to read out per epoch)
# -----------------------------------------------------------------------------
def block_epoch_anchors(
    blocks: list[dict],
    T: int,
    tr: float = TR,
    lag_s: float = DEFAULT_LAG_S,
    pre_s: float = 0.0,
    post_s: float = 0.0,
    s_window: int = S_WINDOW,
) -> dict[str, list[dict]]:
    """
    Convert block onsets into integer current-time volumes for three epochs.

    For each block (onset v_onset, end v_end), with lag L = round(lag_s/tr):
      - 'before' : v_onset - round(pre_s/tr)      (pre-task baseline; lag NOT applied,
                   because at onset the BOLD has not yet responded)
      - 'during' : v_onset + L                     (task-reflecting state)
      - 'after'  : v_end   + L + round(post_s/tr)  (post-block state)

    Anchors whose window would fall outside [s_window-1, T-1] are dropped, so the
    three epochs may have slightly different counts; each returned record carries
    its source condition for later grouping.

    Returns {'before': [...], 'during': [...], 'after': [...]} where each item is
    {'vol': int, 'condition': str, 'onset_s': float}.
    """
    L = int(round(lag_s / tr))
    pre = int(round(pre_s / tr))
    post = int(round(post_s / tr))
    lo, hi = s_window - 1, T - 1

    epochs: dict[str, list[dict]] = {"before": [], "during": [], "after": []}
    for b in blocks:
        v_onset = int(round(b["onset_s"] / tr))
        v_end = int(round((b["onset_s"] + b["dur_s"]) / tr))
        cand = {
            "before": v_onset - pre,
            "during": v_onset + L,
            "after": v_end + L + post,
        }
        for epoch, v in cand.items():
            if lo <= v <= hi:
                epochs[epoch].append({"vol": v, "condition": b["condition"], "onset_s": b["onset_s"]})
    return epochs


def block_window_anchors(
    blocks: list[dict],
    T: int,
    tr: float = TR,
    lag_s: float = DEFAULT_LAG_S,
    s_window: int = S_WINDOW,
) -> dict[str, list[dict]]:
    """
    Alternative to block_epoch_anchors that samples *every* within-block volume
    as a 'during' anchor (range [v_onset+L, v_end+L]) and every volume strictly
    outside any lagged block as 'rest'. Useful for a denser during-vs-rest contrast
    rather than a single state per block.
    """
    L = int(round(lag_s / tr))
    lo, hi = s_window - 1, T - 1
    in_block = np.zeros(T, dtype=bool)
    during = []
    for b in blocks:
        v0 = int(round(b["onset_s"] / tr)) + L
        v1 = int(round((b["onset_s"] + b["dur_s"]) / tr)) + L
        for v in range(max(v0, lo), min(v1, hi) + 1):
            in_block[v] = True
            during.append({"vol": v, "condition": b["condition"], "onset_s": b["onset_s"]})
    rest = [{"vol": v, "condition": "rest", "onset_s": np.nan}
            for v in range(lo, hi + 1) if not in_block[v]]
    return {"during": during, "rest": rest}


# -----------------------------------------------------------------------------
# Window construction (the actual extractor)
# -----------------------------------------------------------------------------
def windows_at(ts: np.ndarray, anchor_vols, s_window: int = S_WINDOW) -> np.ndarray:
    """
    Build twin-ready input windows whose *current state* sits at each anchor volume.

    Args:
        ts:          (T, N) preprocessed BOLD.
        anchor_vols: iterable of integer current-time volumes t (each window is
                     ts[t-S+1 : t+1]). Must satisfy S-1 <= t <= T-1.
        s_window:    S.

    Returns:
        input_X: (len(anchor_vols), S*N) float array, last N columns = x(t).
                 Drop-in for NPI.model_ECt / collect_state_effect_pairs.
    """
    T, N = ts.shape
    anchor_vols = np.asarray(list(anchor_vols), dtype=int)
    bad = (anchor_vols < s_window - 1) | (anchor_vols > T - 1)
    if bad.any():
        raise ValueError(f"{bad.sum()} anchors outside valid range [{s_window-1}, {T-1}]")
    X = np.empty((anchor_vols.size, s_window * N), dtype=float)
    for k, t in enumerate(anchor_vols):
        X[k] = ts[t - s_window + 1 : t + 1].reshape(-1)
    return X


def baseline_energy(input_X: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """E(t) = sum_i x_i(t)^2 for each window (the last frame). Returns (M,)."""
    M = input_X.shape[0]
    last_frame = input_X.reshape(M, -1, n_roi)[:, -1, :]   # (M, N)
    return np.sum(last_frame ** 2, axis=1)


def extract_subject(
    subject: str,
    data_root: str = DATA_ROOT,
    lag_s: float = DEFAULT_LAG_S,
    conditions: tuple[str, ...] = ("story", "math"),
    s_window: int = S_WINDOW,
) -> dict:
    """
    End-to-end convenience: load + preprocess one subject, build before/during/after
    windows, and return everything needed to feed the twin and group the results.

    Returns dict with, per epoch in {'before','during','after'}:
        input_X[epoch]   : (M_e, S*N) windows
        energy[epoch]    : (M_e,) baseline energy per window
        condition[epoch] : (M_e,) source condition (story/math) per window
    plus 'meta' (subject, T, N, lag_s, anchor volumes).
    """
    ts = load_language_bold(subject, data_root, preprocess=True)
    T, N = ts.shape
    blocks = load_block_onsets(subject, conditions, data_root)
    anchors = block_epoch_anchors(blocks, T, lag_s=lag_s, s_window=s_window)

    out = {"input_X": {}, "energy": {}, "condition": {},
           "meta": {"subject": subject, "T": T, "N": N, "lag_s": lag_s,
                    "n_blocks": len(blocks), "anchors": anchors}}
    for epoch, items in anchors.items():
        vols = [it["vol"] for it in items]
        X = windows_at(ts, vols, s_window)
        out["input_X"][epoch] = X
        out["energy"][epoch] = baseline_energy(X, N)
        out["condition"][epoch] = np.array([it["condition"] for it in items])
    return out


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    subj = "100206"
    res = extract_subject(subj)
    T, N = res["meta"]["T"], res["meta"]["N"]
    print(f"subject {subj}: BOLD {T}x{N}, {res['meta']['n_blocks']} task blocks, "
          f"lag={res['meta']['lag_s']}s")
    for epoch in ("before", "during", "after"):
        X = res["input_X"][epoch]
        E = res["energy"][epoch]
        conds = res["condition"][epoch]
        ns = (conds == "story").sum()
        nm = (conds == "math").sum()
        print(f"  {epoch:6s}: {X.shape[0]:2d} windows  (story {ns}, math {nm})  "
              f"E(t) mean={E.mean():8.2f}  median={np.median(E):8.2f}")
    # Sanity: the during-task state should differ from the before state.
    eb, ed = res["energy"]["before"], res["energy"]["during"]
    print(f"\n  before vs during baseline energy: "
          f"mean {eb.mean():.1f} -> {ed.mean():.1f} "
          f"(Δ {ed.mean()-eb.mean():+.1f})")
