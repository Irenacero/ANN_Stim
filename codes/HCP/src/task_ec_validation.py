"""
task_ec_validation.py — Part (c): do rest-derived seed-stimulation maps reproduce
the task's activation pattern, and does the causal (EC) map beat plain FC?

Per subject:
  a   (400,)      LANGUAGE task activation contrast = mean(story states) - mean(math
                  states) of the lagged, z-scored task BOLD (cortical).
  EC  (400,400)   rest static effective connectivity from the twin: row j = mean over
                  rest states of (perturbed - unperturbed), seed j -> cortical downstream.
  FC  (400,400)   rest cortical functional connectivity (from the twin's training targets).
  m_EC[j] = corr(EC[j,:], a)   how task-like is the map evoked by stimulating j
  m_FC[j] = corr(FC[j,:], a)   the FC-only comparison

Tests:
  T1  (main)        Spearman(a, m_EC) > 0  -> the regions the task recruits are the
                    regions whose stimulation evokes the task pattern (story areas ->
                    story-like map, math areas -> math-like map).
  C1  (specificity) Alexander-Bloch spin null on a (spatial-autocorrelation-preserving)
                    + the Yeo-7 network of the best-matching seed.
  C2  (EC vs FC)    paired Spearman(a,m_EC) vs Spearman(a,m_FC) across subjects. If EC
                    wins, the stimulation framing earns main-text placement.
"""

from __future__ import annotations
import os, sys
_SRC = os.path.dirname(os.path.abspath(__file__)); _HCP = os.path.dirname(_SRC)
for _p in (_SRC, _HCP):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

import task_state as tsm
from task_gating import load_twin, CORTICAL, RESULTS_DIR, PERT, device
from task_network_behavior import load_network_table
from spin_test import load_centroids, gen_spin_indices

PROC = os.path.join(RESULTS_DIR, "processed")
N_REST = 500          # rest windows sampled to estimate the static EC
N_SPIN = 1000         # Alexander-Bloch spatial spins for the C1 null
RNG = np.random.default_rng(0)

# Precompute spin reassignment indices once (shared across subjects)
_COORDS, _HEMI, _ = load_centroids()
SPIN_IDX = gen_spin_indices(_COORDS, _HEMI, n_rotate=N_SPIN, seed=0)   # (400, N_SPIN)


def task_activation(subject, lag_s=tsm.DEFAULT_LAG_S):
    """Story - Math contrast of lagged z-scored task BOLD, cortical (400,)."""
    ts = tsm.load_language_bold(subject, preprocess=True)        # (T, 450)
    T = ts.shape[0]
    blocks = tsm.load_block_onsets(subject)
    anchors = tsm.block_window_anchors(blocks, T, lag_s=lag_s)
    story = [it["vol"] for it in anchors["during"] if it["condition"] == "story"]
    math = [it["vol"] for it in anchors["during"] if it["condition"] == "math"]
    a = ts[story].mean(0) - ts[math].mean(0)                     # (450,)
    return a[CORTICAL]                                           # (400,)


def rest_static_EC(subject, model, n_rest=N_REST):
    """Mean perturbation response over sampled rest windows: (400 seeds, 400 cortical)."""
    inp = np.load(os.path.join(PROC, f"id_{subject}_inputs.npy"), mmap_mode="r")  # (T,1350)
    idx = RNG.choice(inp.shape[0], size=min(n_rest, inp.shape[0]), replace=False)
    X = torch.tensor(np.asarray(inp[idx], dtype=np.float32), device=device)        # (R,1350)
    N = tsm.N_ROI; S = X.shape[1] // N
    with torch.no_grad():
        unpert = model(X).cpu().numpy()                          # (R, 450)
    EC = np.empty((len(CORTICAL), len(CORTICAL)))
    for k, j in enumerate(CORTICAL):
        d = torch.zeros(S * N, device=device); d[(S - 1) * N + j] = PERT
        with torch.no_grad():
            pert = model(X + d).cpu().numpy()                    # (R, 450)
        EC[k] = (pert - unpert).mean(0)[CORTICAL]                # cortical downstream
    return EC


def rest_FC(subject):
    tgt = np.load(os.path.join(PROC, f"id_{subject}_targets.npy"), mmap_mode="r")  # (T,450)
    return np.corrcoef(np.asarray(tgt[:, CORTICAL]).T)           # (400,400)


def row_match(M, a):
    """corr(M[j,:], a) for every row j -> (n_rows,)."""
    Mc = M - M.mean(1, keepdims=True)
    ac = a - a.mean()
    num = Mc @ ac
    den = np.sqrt((Mc**2).sum(1) * (ac**2).sum()) + 1e-12
    return num / den


def analyze(subject):
    model = load_twin(subject)
    a = task_activation(subject)
    EC = rest_static_EC(subject, model)
    FC = rest_FC(subject)
    del model

    m_EC = row_match(EC, a)                  # (400,)
    m_FC = row_match(FC, a)
    t1_ec = spearmanr(a, m_EC).statistic
    t1_fc = spearmanr(a, m_FC).statistic

    # C1 Alexander-Bloch spin null: spin the task map, recompute its EC alignment
    null = np.empty(N_SPIN)
    for s in range(N_SPIN):
        ap = a[SPIN_IDX[:, s]]
        null[s] = spearmanr(ap, row_match(EC, ap)).statistic
    p_spin = (np.sum(np.abs(null) >= abs(t1_ec)) + 1) / (N_SPIN + 1)

    best_pos = int(np.argmax(m_EC))
    return {"sub": subject, "t1_ec": float(t1_ec), "t1_fc": float(t1_fc),
            "p_spin": float(p_spin), "best_pos": best_pos,
            "best_seed_roi": int(CORTICAL[best_pos]),
            "a_at_best": float(a[best_pos])}


def main(subjects):
    net_arr = load_network_table()["rsn_network"].values   # (400,) aligned to CORTICAL order
    rows = []
    print(f"Part (c): EC-vs-task validation on {len(subjects)} subjects "
          f"({N_REST} rest states for EC, {N_SPIN} Alexander-Bloch spins)...")
    for i, s in enumerate(subjects, 1):
        r = analyze(s)
        r["best_seed_net"] = net_arr[r["best_pos"]]
        rows.append(r)
        if i % 5 == 0 or i == len(subjects):
            print(f"  {i}/{len(subjects)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "task_ec_validation.csv"), index=False)

    t1 = df.t1_ec.values
    print("\n=== T1: task-activated regions evoke task-like maps ===")
    print(f"  Spearman(a, m_EC): median {np.median(t1):+.3f}, "
          f"positive in {(t1>0).mean()*100:.0f}% of subjects, "
          f"group Wilcoxon p={wilcoxon(t1).pvalue:.1e}")
    print(f"  Alexander-Bloch spin null: median p_spin={np.median(df.p_spin):.3f}, "
          f"sig (<0.05) in {(df.p_spin<0.05).mean()*100:.0f}% of subjects")

    print("\n=== C1: where does the best-matching seed land? ===")
    print(df.best_seed_net.value_counts().to_string())

    print("\n=== C2: does EC beat FC? ===")
    tfc = df.t1_fc.values
    w = wilcoxon(t1, tfc)
    print(f"  Spearman(a,m_EC) median {np.median(t1):+.3f}  vs  "
          f"Spearman(a,m_FC) median {np.median(tfc):+.3f}")
    print(f"  EC>FC in {(t1>tfc).mean()*100:.0f}% of subjects, paired Wilcoxon p={w.pvalue:.1e}")
    return df


if __name__ == "__main__":
    subs = [l.strip() for l in open(os.path.join(tsm.DATA_ROOT, "language_subjects_paper100.txt"))]
    if len(sys.argv) > 1:
        subs = subs[: int(sys.argv[1])]
    main(subs)
