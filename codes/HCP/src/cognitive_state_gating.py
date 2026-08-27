"""
cognitive_state_gating.py — data for the upgraded Supplementary Figure S2,
"Cognitive state modulates the gating axis."

Three things, all per participant, saved to results/cognitive_state_S2.npz:

(A) Model-free: does the empirical baseline energy E(t)=sum_i x_i(t)^2 differ
    across cognitive states? Within-subject Delta E expressed in units of that
    subject's temporal SD of E. Four contrasts:
      math_vs_story     (LANGUAGE, demand; both mid-run task -> no edge confound)
      WM_task_vs_fix    (WM, engagement vs INTERIOR fixation -> edge-controlled)
      motor_vs_fix      (MOTOR, vs interior fixation -> null after control)
      WM_2bk_vs_0bk     (WM, parametric load -> null)

(B) Model-based: feeding task states to the rest-trained twin, does the gating
    law hold (per-subject Spearman rho(E, GES) < 0)? Tasks: LANGUAGE, WM.

(C) Model-based: does the unimodal->transmodal hierarchy of responsiveness
    persist on task states (per-subject Spearman rho(network-mean GES, rank))?

GES convention matches the paper / task_gating.py: GES = sum_i (perturbed -
unperturbed)^2 over 450 regions, targets = 400 cortical parcels, amplitude 0.1.
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
from task_state import block_window_anchors, windows_at, baseline_energy
from task_gating import load_twin, ges_for_windows, CORTICAL, RESULTS_DIR
from preprocessing_hcp import TR

YEO7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default", "Limbic"]
NETCSV = os.path.join(_HCP, "results", "dataframes",
                      "HCP_5_df_spatial_network_receptors_ECts_cortical400.csv")
L = int(round(5.0 / TR))
SUBS = [l.strip() for l in open(os.path.join(_HCP, "data", "Task",
        "language_subjects_paper100.txt")) if l.strip()]

WM_HI = ["2bk_body", "2bk_faces", "2bk_places", "2bk_tools"]
WM_LO = ["0bk_body", "0bk_faces", "0bk_places", "0bk_tools"]
MOTOR = ["lf", "rf", "lh", "rh", "t"]


def _net_masks():
    net = pd.read_csv(NETCSV, usecols=["roi", "rsn_network"]).drop_duplicates("roi")
    net = net.sort_values("roi")["rsn_network"].to_numpy()
    return [net == n for n in YEO7]


def _vols(anch, conds):
    return [it["vol"] for it in anch["during"] if it["condition"] in conds]


def _interior_rest(anch, blocks):
    first = min(int(round(b["onset_s"]/TR)) + L for b in blocks)
    last = max(int(round((b["onset_s"]+b["dur_s"])/TR)) + L for b in blocks)
    return [it["vol"] for it in anch["rest"] if first < it["vol"] < last]


def _dE_sd(ts, vols_hi, vols_lo, ref_vols):
    """within-subject (mean E_hi - mean E_lo) / SD(E over ref states)."""
    if not vols_hi or not vols_lo:
        return np.nan
    Ehi = baseline_energy(windows_at(ts, vols_hi)).mean()
    Elo = baseline_energy(windows_at(ts, vols_lo)).mean()
    sd = baseline_energy(windows_at(ts, ref_vols)).std()
    return (Ehi - Elo) / sd


# ---------------- (A) model-free energy contrasts ----------------
def energy_contrasts():
    out = {"math_vs_story": [], "WM_task_vs_fix": [], "motor_vs_fix": [], "WM_2bk_vs_0bk": []}
    for s in SUBS:
        # LANGUAGE: math vs story (during-only)
        if task_io.has_subject("LANGUAGE", s):
            ts = task_io.load_bold("LANGUAGE", s); T = ts.shape[0]
            an = block_window_anchors(task_io.load_onsets("LANGUAGE", s, ["story", "math"]), T, lag_s=5.0)
            ref = _vols(an, ["story", "math"]) + [it["vol"] for it in an["rest"]]
            out["math_vs_story"].append(_dE_sd(ts, _vols(an, ["math"]), _vols(an, ["story"]), ref))
        # WM: task vs interior fixation; 2bk vs 0bk
        if task_io.has_subject("WM", s):
            ts = task_io.load_bold("WM", s); T = ts.shape[0]
            blk = task_io.load_onsets("WM", s, WM_HI + WM_LO)
            an = block_window_anchors(blk, T, lag_s=5.0)
            ref = _vols(an, WM_HI + WM_LO) + [it["vol"] for it in an["rest"]]
            ri = _interior_rest(an, blk)
            out["WM_task_vs_fix"].append(_dE_sd(ts, _vols(an, WM_HI + WM_LO), ri, ref))
            out["WM_2bk_vs_0bk"].append(_dE_sd(ts, _vols(an, WM_HI), _vols(an, WM_LO), ref))
        # MOTOR: movement vs interior fixation
        if task_io.has_subject("MOTOR", s):
            ts = task_io.load_bold("MOTOR", s); T = ts.shape[0]
            blk = task_io.load_onsets("MOTOR", s, MOTOR)
            an = block_window_anchors(blk, T, lag_s=5.0)
            ref = _vols(an, MOTOR) + [it["vol"] for it in an["rest"]]
            ri = _interior_rest(an, blk)
            out["motor_vs_fix"].append(_dE_sd(ts, _vols(an, MOTOR), ri, ref))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


# ---------------- (B,C) twin gating + hierarchy on task states ----------------
def twin_gating(task, conds, masks):
    rho_gate, rho_hier = [], []
    n = 0
    for s in SUBS:
        if not task_io.has_subject(task, s):
            continue
        ts = task_io.load_bold(task, s); T = ts.shape[0]
        an = block_window_anchors(task_io.load_onsets(task, s, conds), T, lag_s=5.0)
        dur = _vols(an, conds)
        rest = [it["vol"] for it in an["rest"]]
        vols = np.array(sorted(dur + rest))
        is_dur = np.isin(vols, dur)
        X = windows_at(ts, vols)
        E = baseline_energy(X)
        m = load_twin(s)
        GES = ges_for_windows(m, X)              # (M, 400)
        del m
        ges_w = GES.mean(axis=1)                 # (M,)
        rho_gate.append(spearmanr(E, ges_w).statistic)
        # hierarchy from during-state per-target GES
        per_target = GES[is_dur].mean(axis=0)    # (400,)
        net_means = [np.nanmean(per_target[mk]) for mk in masks]
        rho_hier.append(spearmanr(range(7), net_means).statistic)
        n += 1
        if n % 20 == 0 or n == len(SUBS):
            print(f"  {task} {n} subjects", flush=True)
    return np.array(rho_gate), np.array(rho_hier)


def main():
    masks = _net_masks()
    print("A) model-free energy contrasts ...")
    en = energy_contrasts()
    print("B/C) twin gating + hierarchy on task states ...")
    lang_gate, lang_hier = twin_gating("LANGUAGE", ["story", "math"], masks)
    wm_gate, wm_hier = twin_gating("WM", WM_HI + WM_LO, masks)

    out = os.path.join(RESULTS_DIR, "cognitive_state_S2.npz")
    np.savez(out,
             dE_math_vs_story=en["math_vs_story"], dE_WM_task_vs_fix=en["WM_task_vs_fix"],
             dE_motor_vs_fix=en["motor_vs_fix"], dE_WM_2bk_vs_0bk=en["WM_2bk_vs_0bk"],
             gate_language=lang_gate, gate_wm=wm_gate,
             hier_language=lang_hier, hier_wm=wm_hier)
    print(f"\nwrote {out}")
    from scipy.stats import wilcoxon
    for k, v in en.items():
        v = v[~np.isnan(v)]
        print(f"  {k:16s}: median {np.median(v):+.3f} SD, <0 in {100*(v<0).mean():.0f}%, p={wilcoxon(v).pvalue:.1e}")
    for nm, g, h in [("LANGUAGE", lang_gate, lang_hier), ("WM", wm_gate, wm_hier)]:
        print(f"  {nm}: gating rho median {np.median(g):+.3f} (<0 {100*(g<0).mean():.0f}%); "
              f"hierarchy rho median {np.median(h):+.3f} (>0 {100*(h>0).mean():.0f}%)")


if __name__ == "__main__":
    main()
