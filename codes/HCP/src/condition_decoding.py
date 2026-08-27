"""
condition_decoding.py — within-task condition decoding (EC vs FC) for the
multi-condition tasks (MOTOR: 5 body parts; WM: 4 visual categories).

Per condition, activation = mean(condition states) - mean across the task's
conditions (demeaned). Group-fixed seeds (one per condition, from group EC and,
separately, group FC). Each participant is decoded by assigning each condition to
the seed whose evoked / FC map best matches that condition's activation; 5- or
4-way accuracy vs chance, EC and FC, with a label-permutation null on EC.

Saves results/decoding_<TASK>.npz: activations, M (EC confusion), ec_acc, fc_acc,
ec_seeds, conds, ec_perm_p.
"""
from __future__ import annotations
import os, sys
_SRC = os.path.dirname(os.path.abspath(__file__)); _HCP = os.path.dirname(_SRC)
for _p in (_SRC, _HCP):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import task_io
from task_io import CORTICAL
from task_state import block_window_anchors
from task_gating import load_twin, RESULTS_DIR, PERT, device
from spin_test import load_centroids

_, _, NAMES = load_centroids()
RNG = np.random.default_rng(0)

TASKS = {
    "MOTOR": {"L hand": ["lh"], "R hand": ["rh"], "L foot": ["lf"], "R foot": ["rf"], "tongue": ["t"]},
    "WM": {"body": ["2bk_body", "0bk_body"], "faces": ["2bk_faces", "0bk_faces"],
           "places": ["2bk_places", "0bk_places"], "tools": ["2bk_tools", "0bk_tools"]},
}


def activations(task, conds, s):
    ts = task_io.load_bold(task, s); T = ts.shape[0]
    allc = [c for g in conds.values() for c in g]
    an = block_window_anchors(task_io.load_onsets(task, s, allc), T, lag_s=5.0)
    A = np.full((len(conds), 400), np.nan)
    for i, k in enumerate(conds):
        v = [it["vol"] for it in an["during"] if it["condition"] in conds[k]]
        if v:
            A[i] = ts[v].mean(0)[CORTICAL]
    return A - np.nanmean(A, 0)


def ec_rows(model, s, seeds):
    inp = np.load(os.path.join(RESULTS_DIR, "processed", f"id_{s}_inputs.npy"), mmap_mode="r")
    idx = RNG.choice(inp.shape[0], 500, replace=False)
    X = torch.tensor(np.asarray(inp[idx], dtype=np.float32), device=device)
    with torch.no_grad():
        un = model(X).cpu().numpy()
    o = np.empty((len(seeds), 400))
    for k, p in enumerate(seeds):
        dd = torch.zeros(1350, device=device); dd[2 * 450 + CORTICAL[p]] = PERT
        with torch.no_grad():
            o[k] = (model(X + dd).cpu().numpy() - un).mean(0)[CORTICAL]
    return o


def fc_sub(s):
    t = np.load(os.path.join(RESULTS_DIR, "processed", f"id_{s}_targets.npy"), mmap_mode="r")
    return np.corrcoef(np.asarray(t[:, CORTICAL]).T)


def run(task):
    conds = TASKS[task]; CL = list(conds); n = len(CL)
    subs = [l.strip() for l in open(os.path.join(_HCP, "data", "Task", "language_subjects_paper100.txt"))]
    subs = [s for s in subs if task_io.has_subject(task, s)]
    ECg = np.load(os.path.join(RESULTS_DIR, "group_mean_EC_cortical.npy"))
    FCg = np.load(os.path.join(RESULTS_DIR, "group_FC_cortical.npy"))

    Asub = {s: activations(task, conds, s) for s in subs}
    Ag = np.nanmean(np.stack([Asub[s] for s in subs]), 0)
    es = [int(np.argmax([np.corrcoef(ECg[j], Ag[i])[0, 1] for j in range(400)])) for i in range(n)]
    fs = [int(np.argmax([np.corrcoef(FCg[j], Ag[i])[0, 1] for j in range(400)])) for i in range(n)]
    print(f"{task} EC seeds: " + ", ".join(f"{CL[i]}={' '.join(NAMES[es[i]].split('_')[1:])}" for i in range(n)))

    Ms, Mfs, ea, fa = [], [], [], []
    for k, s in enumerate(subs, 1):
        m = load_twin(s); er = ec_rows(m, s, es); del m
        fr = fc_sub(s)[fs]; A = Asub[s]
        Me = np.array([[np.corrcoef(A[i], er[j])[0, 1] for j in range(n)] for i in range(n)])
        Mf = np.array([[np.corrcoef(A[i], fr[j])[0, 1] for j in range(n)] for i in range(n)])
        Ms.append(Me); Mfs.append(Mf)
        ea.append(np.mean([np.argmax(Me[i]) == i for i in range(n)]))
        fa.append(np.mean([np.argmax(Mf[i]) == i for i in range(n)]))
        if k % 25 == 0 or k == len(subs):
            print(f"  {k}/{len(subs)}", flush=True)
    Ms = np.array(Ms); Mfs = np.array(Mfs); ea = np.array(ea); fa = np.array(fa)

    null = []
    for _ in range(5000):
        pm = RNG.permutation(n)
        null.append(np.mean([[np.argmax(M[i][pm]) == i for i in range(n)] for M in Ms]))
    p = (np.sum(np.array(null) >= ea.mean()) + 1) / 5001

    np.savez(os.path.join(RESULTS_DIR, f"decoding_{task}.npz"),
             activations=Ag, M=Ms.mean(0), M_fc=Mfs.mean(0), ec_acc=ea, fc_acc=fa,
             ec_seeds=np.array(es), fc_seeds=np.array(fs),
             ec_seed_names=np.array([" ".join(NAMES[p_].split("_")[1:]) for p_ in es]),
             conds=np.array(CL), ec_perm_p=p)
    print(f"  {task}: EC {ea.mean()*100:.0f}% vs FC {fa.mean()*100:.0f}% (chance {100/n:.0f}%), "
          f"EC>FC {100*(ea>fa).mean():.0f}% subj, perm p={p:.1e}\n")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["MOTOR", "WM"]):
        run(t.upper())
