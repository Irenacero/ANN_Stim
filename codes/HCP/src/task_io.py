"""
task_io.py — Generalized loaders + canonical activation contrasts for all 7 HCP
tasks (parcellated to Schaefer-400 + Tian-S3). Task BOLD lives on the external
drive; rest data / twins / processed inputs stay local.

Each task defines a standard activation contrast a = mean(POS states) - mean(NEG
states) of the lagged, z-scored task BOLD (cortical 400). NEG can be an explicit
condition set or "REST" (volumes outside any modelled block = fixation), which is
only meaningful for tasks with genuine rest blocks (e.g. MOTOR).
"""

from __future__ import annotations
import os
import numpy as np
import scipy.io as sio

from preprocessing_hcp import bandpass_filter_timeseries, TR
from task_state import block_window_anchors, windows_at, baseline_energy, N_ROI, S_WINDOW

CORTICAL = np.arange(50, 450)

# Root of the parcellated task fMRI (see README, "What you need to supply").
# Set the HCP_TASK_ROOT environment variable, or edit the fallback below.
EXT_ROOT = os.environ.get(
    "HCP_TASK_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Task"),
)
DEFAULT_LAG_S = 5.0

TASKS = {
    "LANGUAGE":   dict(folder="Language",       suffix="LANGUAGE_LR",
                       pos=["story"],                                   neg=["math"]),
    "MOTOR":      dict(folder="Motor_LR",       suffix="MOTOR_LR",
                       pos=["lf", "rf", "lh", "rh", "t"],               neg="REST"),
    "MOTOR_HF":   dict(folder="Motor_LR",       suffix="MOTOR_LR", ev_task="MOTOR",
                       pos=["lh", "rh"],                                neg=["lf", "rf"]),
    "WM":         dict(folder="WM_LR",          suffix="WM_LR",
                       pos=["2bk_body", "2bk_faces", "2bk_places", "2bk_tools"],
                       neg=["0bk_body", "0bk_faces", "0bk_places", "0bk_tools"]),
    "GAMBLING":   dict(folder="Gambling_LR",    suffix="GAMBLING_LR",
                       pos=["win"],                                     neg=["loss"]),
    "EMOTION":    dict(folder="Emotion_LR",     suffix="EMOTION_LR",
                       pos=["fear"],                                    neg=["neut"]),
    "RELATIONAL": dict(folder="Relational_LR",  suffix="RELATIONAL_LR",
                       pos=["relation"],                                neg=["match"]),
    "SOCIAL":     dict(folder="Social_LR",      suffix="SOCIAL_LR",
                       pos=["mental"],                                  neg=["rnd"]),
}


def mat_path(task, subject, root=EXT_ROOT):
    cfg = TASKS[task]
    return os.path.join(root, cfg["folder"], f"{subject}_{cfg['suffix']}_schaefer_400_Tian_S3.mat")


def has_subject(task, subject, root=EXT_ROOT):
    return os.path.isfile(mat_path(task, subject, root))


def load_bold(task, subject, root=EXT_ROOT, preprocess=True):
    ts = sio.loadmat(mat_path(task, subject, root))["schaefer_tian_ts"]   # (450, T)
    ts = np.asarray(ts, dtype=float).T                                   # (T, 450)
    return bandpass_filter_timeseries(ts) if preprocess else ts


def load_onsets(task, subject, conditions, root=EXT_ROOT):
    """Block list [{condition,onset_s,dur_s}] for the given conditions, sorted."""
    cfg = TASKS[task]
    ev_task = cfg.get("ev_task", task)
    ev_dir = os.path.join(root, "HCP_TASKS_EVs", subject, "MNINonLinear",
                          "Results", f"tfMRI_{ev_task}_LR", "EVs")
    blocks = []
    for c in conditions:
        f = os.path.join(ev_dir, f"{c}.txt")
        if not os.path.isfile(f):
            continue
        for onset_s, dur_s, _w in np.loadtxt(f, ndmin=2):
            blocks.append({"condition": c, "onset_s": float(onset_s), "dur_s": float(dur_s)})
    blocks.sort(key=lambda b: b["onset_s"])
    return blocks


def task_activation(task, subject, lag_s=DEFAULT_LAG_S, root=EXT_ROOT):
    """Canonical contrast a = mean(POS states) - mean(NEG states), cortical (400,)."""
    cfg = TASKS[task]
    ts = load_bold(task, subject, root, preprocess=True)
    T = ts.shape[0]
    neg_is_rest = cfg["neg"] == "REST"
    conds = list(cfg["pos"]) + ([] if neg_is_rest else list(cfg["neg"]))
    blocks = load_onsets(task, subject, conds, root)
    anch = block_window_anchors(blocks, T, lag_s=lag_s)
    pos_vols = [it["vol"] for it in anch["during"] if it["condition"] in cfg["pos"]]
    if neg_is_rest:
        neg_vols = [it["vol"] for it in anch["rest"]]
    else:
        neg_vols = [it["vol"] for it in anch["during"] if it["condition"] in cfg["neg"]]
    a = ts[pos_vols].mean(0) - ts[neg_vols].mean(0)
    return a[CORTICAL]
