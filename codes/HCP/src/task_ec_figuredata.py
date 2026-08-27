"""
task_ec_figuredata.py — data prep for Figure 3.

Computes, reusing rest EC/FC once per subject:
  - per (task, subject) scalars t1_ec, t1_fc  (Spearman a vs m_EC / m_FC) -> CSV (Panel B)
  - group-mean rest EC matrix (400x400) -> npy (Panel A evoked map from peak seed)
Uses the canonical task set with MOTOR = hand-foot (MOTOR_HF).
"""
from __future__ import annotations
import os, sys
_SRC = os.path.dirname(os.path.abspath(__file__)); _HCP = os.path.dirname(_SRC)
for _p in (_SRC, _HCP):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import task_io
from task_gating import load_twin, RESULTS_DIR
from task_ec_validation import rest_static_EC, rest_FC, row_match

TASKS = ["MOTOR_HF", "WM", "RELATIONAL", "LANGUAGE", "EMOTION", "SOCIAL", "GAMBLING"]


def main(subjects):
    rows = []
    EC_sum = np.zeros((400, 400)); n_ec = 0
    print(f"Figure-3 data: {len(subjects)} subjects x {len(TASKS)} tasks...")
    for n, s in enumerate(subjects, 1):
        model = load_twin(s)
        EC = rest_static_EC(s, model); FC = rest_FC(s); del model
        EC_sum += EC; n_ec += 1
        for t in TASKS:
            if not task_io.has_subject(t, s):
                continue
            a = task_io.task_activation(t, s)
            rows.append({"task": t, "sub": s,
                         "t1_ec": float(spearmanr(a, row_match(EC, a)).statistic),
                         "t1_fc": float(spearmanr(a, row_match(FC, a)).statistic)})
        if n % 20 == 0 or n == len(subjects):
            print(f"  {n}/{len(subjects)}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "task_ec_scalars.csv"), index=False)
    np.save(os.path.join(RESULTS_DIR, "group_mean_EC_cortical.npy"), EC_sum / n_ec)
    print(f"wrote task_ec_scalars.csv ({len(rows)} rows) and group_mean_EC_cortical.npy")


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(_HCP, "data", "Task", "language_subjects_paper100.txt"))]
    if len(sys.argv) > 1:
        subs = subs[: int(sys.argv[1])]
    main(subs)
