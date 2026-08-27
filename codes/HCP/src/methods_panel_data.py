"""
methods_panel_data.py — real data for the Figure-3 methods schematic
(Figure3_methods_taskmaps.py). Computes, from the actual HCP MOTOR task:

  * carpet : one subject's z-scored cortical BOLD (400 ROIs x T), with ROIs
             ordered by Yeo-7 network, plus that subject's block design
             (onset/duration in volume units + condition per block).
  * stage3 : group-mean <x>_blk (L hand), <x>_fix, and a_c = blk - fix  (400,)
  * stage4 : group A (400 x 5) raw per-condition activation a_c, and the
             demeaned A~ = A - mean_c(A); condition order Lh,Rh,Lf,Rf,T.

Saves codes/HCP/results/methods_panel_data.npz.  Run with the mounted task drive.
"""
from __future__ import annotations
import os, sys
_SRC = os.path.dirname(os.path.abspath(__file__)); _HCP = os.path.dirname(_SRC)
for _p in (_SRC, _HCP):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import task_io
from task_io import CORTICAL
from task_state import block_window_anchors
from task_network_behavior import load_network_table
from preprocessing_hcp import TR

RESULTS = os.path.join(_HCP, "results")
CONDS = ["lh", "rh", "lf", "rf", "t"]
YEO7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default", "Limbic"]
LAG = 5.0


def cond_block_fix(subj):
    """Per-condition block-mean and shared fixation-mean, cortical (400,)."""
    ts = task_io.load_bold("MOTOR", subj)                      # (T, 450), bandpassed
    T = ts.shape[0]
    an = block_window_anchors(task_io.load_onsets("MOTOR", subj, CONDS), T, lag_s=LAG)
    fixv = [it["vol"] for it in an["rest"]]
    fixm = ts[fixv].mean(0)[CORTICAL]
    blk = {}
    for c in CONDS:
        v = [it["vol"] for it in an["during"] if it["condition"] == c]
        blk[c] = ts[v].mean(0)[CORTICAL] if v else None
    return blk, fixm


def main(subjects, n_group=20):
    subjects = [s for s in subjects if task_io.has_subject("MOTOR", s)]
    net = load_network_table()["rsn_network"].to_numpy()
    order = np.concatenate([np.where(net == n)[0] for n in YEO7])
    net_ord = net[order]

    # ---- carpet: first usable subject ----
    csub = subjects[0]
    ts = task_io.load_bold("MOTOR", csub)[:, CORTICAL]         # (T, 400)
    carpet = ((ts - ts.mean(0)) / (ts.std(0) + 1e-9)).T[order]  # (400, T) z-scored, net-ordered
    blocks = task_io.load_onsets("MOTOR", csub, CONDS)
    Tc = ts.shape[0]
    blk_on = np.array([b["onset_s"] / TR / Tc for b in blocks])
    blk_w = np.array([b["dur_s"] / TR / Tc for b in blocks])
    blk_c = np.array([CONDS.index(b["condition"]) for b in blocks])
    print(f"carpet subject {csub}: T={Tc}, {len(blocks)} blocks")

    # ---- group a_c (raw) over n_group subjects ----
    gs = subjects[:n_group]
    Blk = {c: [] for c in CONDS}; Fix = []
    for i, s in enumerate(gs, 1):
        blk, fixm = cond_block_fix(s)
        if any(blk[c] is None for c in CONDS):
            continue
        Fix.append(fixm)
        for c in CONDS:
            Blk[c].append(blk[c])
        if i % 5 == 0 or i == len(gs):
            print(f"  group {i}/{len(gs)}", flush=True)
    fixm = np.mean(Fix, 0)                                     # (400,)
    blk_lh = np.mean(Blk["lh"], 0)                             # (400,)
    A = np.stack([np.mean(Blk[c], 0) - fixm for c in CONDS], axis=1)   # (400, 5) raw a_c
    Ad = A - A.mean(1, keepdims=True)                          # demeaned across conditions

    np.savez(os.path.join(RESULTS, "methods_panel_data.npz"),
             carpet=carpet, net_ord=net_ord, yeo7=np.array(YEO7),
             blk_on=blk_on, blk_w=blk_w, blk_c=blk_c, conds=np.array(CONDS),
             stage3_blk=blk_lh, stage3_fix=fixm, stage3_ac=blk_lh - fixm,
             A=A, Ad=Ad, n_group=len(Fix), carpet_subj=str(csub))
    print(f"\nwrote methods_panel_data.npz (group n={len(Fix)})")


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(_HCP, "data", "Task",
                                                  "language_subjects_paper100.txt"))]
    if len(sys.argv) > 1:
        subs = subs[: int(sys.argv[1])]
    main(subs)
