"""
task_network_behavior.py — Connect task-state stimulability to (i) the network
hierarchy and (ii) behavioural performance, keeping the analysis tied to the
paper's themes.

For each participant we feed the rest-trained twin its LANGUAGE task states
(within-block "during" volumes, labelled story / math), compute the global
effect size (GES) at every cortical target, and summarise:

  per target  -> mean GES across during-states  (and math-only / story-only)
  per subject -> mean GES, math-state GES, story-state GES, gating slope, mean E

We then ask:
  (1) Network: does task-state responsiveness follow the unimodal->transmodal
      hierarchy (as it does at rest, Fig 3A), and which Yeo-7 networks respond
      most when stimulated from a task state?
  (2) Behaviour: across subjects, does task-state GES predict LANGUAGE
      performance (MATH accuracy / adaptive difficulty, story accuracy, RT)?

Outputs (codes/HCP/results/):
  task_state_per_target_GES.npz       (100 x 400) during / math / story GES
  task_state_subject_behavior.csv     per-subject GES summaries + behaviour
"""

from __future__ import annotations

import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HCP_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _HCP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import task_state as tsm
from task_gating import load_twin, ges_for_windows, CORTICAL, RESULTS_DIR

DF_DIR = os.path.join(RESULTS_DIR, "dataframes")
EV = "MNINonLinear/Results/tfMRI_LANGUAGE_LR/EVs"


# -----------------------------------------------------------------------------
# Per-subject GES on task states (per target, split by condition)
# -----------------------------------------------------------------------------
def subject_task_ges(subject: str, model, lag_s: float = tsm.DEFAULT_LAG_S):
    """
    Returns:
        per_target: dict with 'during','math','story' -> (400,) mean GES per cortical target
        scalars:    dict of per-subject summaries (gating rho, mean E, mean GES, ...)
    """
    ts = tsm.load_language_bold(subject, preprocess=True)
    T, N = ts.shape
    blocks = tsm.load_block_onsets(subject)
    anchors = tsm.block_window_anchors(blocks, T, lag_s=lag_s)

    vols = np.array([it["vol"] for it in anchors["during"]])
    cond = np.array([it["condition"] for it in anchors["during"]])
    order = np.argsort(vols)
    vols, cond = vols[order], cond[order]

    X = tsm.windows_at(ts, vols)               # (M, S*N)
    E = tsm.baseline_energy(X, N)              # (M,)
    GES = ges_for_windows(model, X)            # (M, 400)
    ges_w = GES.mean(axis=1)                   # (M,) mean over cortical targets

    is_math = cond == "math"
    is_story = cond == "story"
    per_target = {
        "during": GES.mean(axis=0),                                  # (400,)
        "math": GES[is_math].mean(axis=0) if is_math.any() else np.full(GES.shape[1], np.nan),
        "story": GES[is_story].mean(axis=0) if is_story.any() else np.full(GES.shape[1], np.nan),
    }
    rho, _ = spearmanr(E, ges_w)
    scalars = {
        "sub": subject,
        "gating_rho": float(rho),
        "E_task_mean": float(E.mean()),
        "GES_task_mean": float(ges_w.mean()),
        "GES_math_mean": float(ges_w[is_math].mean()) if is_math.any() else np.nan,
        "GES_story_mean": float(ges_w[is_story].mean()) if is_story.any() else np.nan,
        "E_math_mean": float(E[is_math].mean()) if is_math.any() else np.nan,
        "E_story_mean": float(E[is_story].mean()) if is_story.any() else np.nan,
        "n_math": int(is_math.sum()),
        "n_story": int(is_story.sum()),
    }
    return per_target, scalars


# -----------------------------------------------------------------------------
# Helpers: behaviour + network labels
# -----------------------------------------------------------------------------
def load_behavior(subjects):
    rows = []
    for s in subjects:
        p = os.path.join(tsm.DATA_ROOT, "HCP_TASKS_EVs", s, EV, "LANGUAGE_Stats.csv")
        d = pd.read_csv(p)
        row = {"sub": s}
        for _, r in d.iterrows():
            row[f"{r.ConditionName}_{r.Measure}"] = r.Value
        rows.append(row)
    return pd.DataFrame(rows)


def load_network_table():
    """roi (50..449) -> rsn_network, principal_gradient, hierarchy (one row per ROI)."""
    f = os.path.join(DF_DIR, "HCP_5_df_spatial_network_receptors_ECts_cortical400.csv")
    df = pd.read_csv(f, usecols=["roi", "rsn_network", "principal_gradient", "hierarchy"])
    df = df.drop_duplicates("roi").sort_values("roi").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def main(subjects):
    n = len(subjects)
    dur = np.empty((n, len(CORTICAL)))
    math = np.empty((n, len(CORTICAL)))
    story = np.empty((n, len(CORTICAL)))
    scal = []
    print(f"Computing per-target task-state GES for {n} subjects...")
    for i, s in enumerate(subjects, 1):
        model = load_twin(s)
        pt, sc = subject_task_ges(s, model)
        dur[i - 1], math[i - 1], story[i - 1] = pt["during"], pt["math"], pt["story"]
        scal.append(sc)
        del model
        if i % 20 == 0 or i == n:
            print(f"  {i}/{n}", flush=True)

    np.savez(os.path.join(RESULTS_DIR, "task_state_per_target_GES.npz"),
             during=dur, math=math, story=story,
             subjects=np.array(subjects), rois=CORTICAL)

    beh = load_behavior(subjects)
    df = pd.DataFrame(scal).merge(beh, on="sub")
    df.to_csv(os.path.join(RESULTS_DIR, "task_state_subject_behavior.csv"), index=False)

    # ---------------- (1) Network / hierarchy ----------------
    net = load_network_table()
    order7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default", "Limbic"]
    grp_gradient = net["principal_gradient"].values
    grp_net = net["rsn_network"].values

    target_mean = np.nanmean(dur, axis=0)            # (400,) group-mean task-state GES per target
    print("\n=== (1) Network hierarchy of task-state responsiveness ===")
    print("  network-mean GES across during-task states (unimodal -> transmodal):")
    net_means = {}
    for name in order7:
        m = grp_net == name
        if m.any():
            net_means[name] = float(np.nanmean(target_mean[m]))
            print(f"    {name:12s} {net_means[name]:.4g}   (n={m.sum()})")
    # hierarchy gradient correlation (per target) and per-subject
    rho_grad, p_grad = spearmanr(grp_gradient, target_mean, nan_policy="omit")
    print(f"  GES vs principal gradient (group): Spearman rho={rho_grad:+.3f} (p={p_grad:.1e})")
    persub = []
    for i in range(n):
        r, _ = spearmanr(grp_gradient, dur[i], nan_policy="omit")
        persub.append(r)
    persub = np.array(persub)
    print(f"  per-subject GES-gradient rho: median {np.median(persub):+.3f}, "
          f"positive in {(persub>0).mean()*100:.0f}% (Wilcoxon p={wilcoxon(persub).pvalue:.1e})")

    # ---------------- (2) Behaviour ----------------
    print("\n=== (2) Task-state GES as a predictor of LANGUAGE performance ===")
    predictors = ["GES_task_mean", "GES_math_mean", "GES_story_mean", "gating_rho", "E_task_mean"]
    targets = ["MATH_ACC", "MATH_AVG_DIFFICULTY_LEVEL", "MATH_MEDIAN_RT",
               "STORY_ACC", "STORY_AVG_DIFFICULTY_LEVEL", "STORY_MEDIAN_RT"]
    print(f"  Spearman rho (n={n}); * p<0.05 uncorrected")
    print("  " + "predictor".ljust(16) + "".join(t[:11].rjust(13) for t in targets))
    for pcol in predictors:
        cells = []
        for tcol in targets:
            r, p = spearmanr(df[pcol], df[tcol], nan_policy="omit")
            cells.append(f"{r:+.2f}{'*' if p < 0.05 else ' '}".rjust(13))
        print("  " + pcol.ljust(16) + "".join(cells))
    print("\n  (math-state GES paired with MATH metrics, story-state GES with STORY metrics is the")
    print("   condition-matched test; off-diagonal blocks are controls.)")
    return df


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(tsm.DATA_ROOT, "language_subjects_paper100.txt"))]
    if len(sys.argv) > 1:
        subs = subs[: int(sys.argv[1])]
    main(subs)
