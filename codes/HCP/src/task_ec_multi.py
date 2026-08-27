"""
task_ec_multi.py — Part (c) across all HCP tasks: does the rest twin's effective
connectivity reproduce each task's activation, and beat FC?

Rest EC and FC are computed ONCE per subject (independent of task) and reused for
every task's activation map a. Per (task, subject):
  m_EC[j]=corr(EC[j,:],a); T1=Spearman(a,m_EC); C2 vs FC; C1 Alexander-Bloch spin.

Outputs results/task_ec_multitask.csv (one row per task x subject) + a per-task
summary table.
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
from task_io import TASKS, CORTICAL
from task_gating import load_twin, RESULTS_DIR
from task_ec_validation import rest_static_EC, rest_FC, row_match, SPIN_IDX, N_SPIN
from task_network_behavior import load_network_table

TASK_ORDER = ["MOTOR", "WM", "GAMBLING", "RELATIONAL", "SOCIAL", "EMOTION", "LANGUAGE"]


def main(subjects, tasks=TASK_ORDER):
    net_arr = load_network_table()["rsn_network"].values
    rows = []
    print(f"Multi-task part (c): {len(subjects)} subjects x {len(tasks)} tasks "
          f"(EC/FC once per subject, {N_SPIN} spins)...")
    for n, s in enumerate(subjects, 1):
        model = load_twin(s)
        EC = rest_static_EC(s, model)        # (400,400) rest, computed once
        FC = rest_FC(s)
        del model
        for task in tasks:
            if not task_io.has_subject(task, s):
                continue
            a = task_io.task_activation(task, s)
            m_EC = row_match(EC, a)
            m_FC = row_match(FC, a)
            t1 = spearmanr(a, m_EC).statistic
            t1f = spearmanr(a, m_FC).statistic
            null = np.array([spearmanr(a[SPIN_IDX[:, k]], row_match(EC, a[SPIN_IDX[:, k]])).statistic
                             for k in range(N_SPIN)])
            p_spin = (np.sum(np.abs(null) >= abs(t1)) + 1) / (N_SPIN + 1)
            best_pos = int(np.argmax(m_EC))
            rows.append({"task": task, "sub": s, "t1_ec": float(t1), "t1_fc": float(t1f),
                         "p_spin": float(p_spin), "best_seed_net": net_arr[best_pos]})
        if n % 10 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)} subjects", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "task_ec_multitask.csv"), index=False)

    print("\n=== Part (c) per task: EC reproduces activation & beats FC ===")
    hdr = f"  {'task':11s} {'n':>3s} {'T1(EC)':>8s} {'%pos':>5s} {'spin<.05':>8s} "\
          f"{'FC':>7s} {'EC>FC%':>7s} {'topNet(%)':>14s}"
    print(hdr)
    for task in tasks:
        d = df[df.task == task]
        if len(d) == 0:
            continue
        t1 = d.t1_ec.values; tf = d.t1_fc.values
        top = d.best_seed_net.value_counts()
        topnet = f"{top.index[0]}({100*top.iloc[0]/len(d):.0f})"
        print(f"  {task:11s} {len(d):>3d} {np.median(t1):>+8.3f} {100*(t1>0).mean():>4.0f}% "
              f"{100*(d.p_spin<0.05).mean():>7.0f}% {np.median(tf):>+7.3f} "
              f"{100*(t1>tf).mean():>6.0f}% {topnet:>14s}")
    return df


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "Task", "language_subjects_paper100.txt"))]
    tasks = TASK_ORDER
    args = [a for a in sys.argv[1:]]
    if args and args[0].isdigit():
        subs = subs[: int(args[0])]; args = args[1:]
    if args:
        tasks = [a.upper() for a in args]
    main(subs, tasks)
