"""
task_ec_profiles.py — Per-region EC-vs-task-activation correlation profiles.

For every task and every cortical seed j, m_EC[j] = corr(EC[j,:], a) where EC is
the rest twin's effective connectivity (computed once per subject) and a is the
task activation contrast. Saves the group-mean profile per task (400,) for the
"which region's stimulation best reproduces the task" plot, plus the per-subject
scalar T1 / EC-vs-FC / Alexander-Bloch spin for MOTOR_HF (and any task requested).

Output: results/task_ec_profiles.npz  (m_EC group mean + sem per task, a mean,
best-seed name) and prints a scalar summary table.
"""

from __future__ import annotations
import os, sys
_SRC = os.path.dirname(os.path.abspath(__file__)); _HCP = os.path.dirname(_SRC)
for _p in (_SRC, _HCP):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import task_io
from task_io import CORTICAL
from task_gating import load_twin, RESULTS_DIR
from task_ec_validation import rest_static_EC, rest_FC, row_match, SPIN_IDX, N_SPIN
from spin_test import load_centroids

TASKS = ["MOTOR", "MOTOR_HF", "WM", "GAMBLING", "RELATIONAL", "SOCIAL", "EMOTION", "LANGUAGE"]
_, _, ROI_NAMES = load_centroids()      # (400,) Schaefer names, CORTICAL order


def short_name(full):
    # '7Networks_LH_SomMot_8' -> 'LH SomMot 8'
    p = full.split("_")
    return " ".join(p[1:]) if len(p) > 1 else full


def main(subjects, tasks=TASKS, do_spin_for=("MOTOR_HF",)):
    mEC = {t: [] for t in tasks}        # per-subject m_EC profiles
    A = {t: [] for t in tasks}
    scal = {t: [] for t in tasks}       # (t1_ec, t1_fc, p_spin) for do_spin_for tasks
    print(f"EC-vs-task profiles: {len(subjects)} subjects x {len(tasks)} tasks...")
    for n, s in enumerate(subjects, 1):
        model = load_twin(s)
        EC = rest_static_EC(s, model)
        FC = rest_FC(s)
        del model
        for t in tasks:
            if not task_io.has_subject(t, s):
                continue
            a = task_io.task_activation(t, s)
            m = row_match(EC, a)
            mEC[t].append(m); A[t].append(a)
            if t in do_spin_for:
                t1 = spearmanr(a, m).statistic
                t1f = spearmanr(a, row_match(FC, a)).statistic
                null = np.array([spearmanr(a[SPIN_IDX[:, k]], row_match(EC, a[SPIN_IDX[:, k]])).statistic
                                 for k in range(N_SPIN)])
                p = (np.sum(np.abs(null) >= abs(t1)) + 1) / (N_SPIN + 1)
                scal[t].append((t1, t1f, p))
        if n % 10 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)}", flush=True)

    out = {"rois": CORTICAL, "roi_names": ROI_NAMES, "tasks": np.array(tasks)}
    print("\nGroup-mean profiles (peak region = best stimulation target):")
    for t in tasks:
        if not mEC[t]:
            continue
        M = np.array(mEC[t])                  # (n_sub, 400)
        mu = M.mean(0); sem = M.std(0) / np.sqrt(len(M))
        out[f"mEC_{t}"] = mu
        out[f"mECsem_{t}"] = sem
        out[f"a_{t}"] = np.array(A[t]).mean(0)
        peak = int(np.argmax(mu))
        print(f"  {t:11s} n={len(M):3d}  peak: {short_name(ROI_NAMES[peak]):18s} "
              f"(m_EC={mu[peak]:.3f})")
    np.savez(os.path.join(RESULTS_DIR, "task_ec_profiles.npz"), **out)
    print(f"\nwrote {os.path.join(RESULTS_DIR, 'task_ec_profiles.npz')}")

    for t in do_spin_for:
        if not scal[t]:
            continue
        arr = np.array(scal[t])
        t1, t1f, p = arr[:, 0], arr[:, 1], arr[:, 2]
        print(f"\n=== {t} part (c) ===")
        print(f"  T1(EC) median {np.median(t1):+.3f}, {100*(t1>0).mean():.0f}% positive, "
              f"spin<.05 in {100*(p<0.05).mean():.0f}%")
        print(f"  EC>FC in {100*(t1>t1f).mean():.0f}% (median EC {np.median(t1):+.3f} vs FC {np.median(t1f):+.3f}, "
              f"Wilcoxon p={wilcoxon(t1,t1f).pvalue:.1e})")


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(_HCP, "data", "Task", "language_subjects_paper100.txt"))]
    args = sys.argv[1:]
    if args and args[0].isdigit():
        subs = subs[: int(args[0])]; args = args[1:]
    tasks = [a.upper() for a in args] if args else TASKS
    main(subs, tasks)
