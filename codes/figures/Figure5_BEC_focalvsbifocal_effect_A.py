"""
Focal vs bifocal EFFECT SIZE for one target and its best effect-increasing
partner. Companion to Figure5_BEC_focalvsbifocal_A (which shows the CV /
variability); here we show the magnitude -- the mean global effect size
(Sigma = BEC^2 = GES, averaged over trials) -- demonstrating that co-stimulation
INCREASES the effect rather than just stabilizing it.

    focal A      = <Sigma>[A, A]   single-site stimulation of A
    focal B      = <Sigma>[B, B]   single-site stimulation of partner B
    bifocal A+B  = <Sigma>[A, B]   co-stimulation

A = roi 92 (SomMot_LH_12); B = its group best effect-increase partner (argmax of
the subject-averaged eff_incr row, cortical) = roi 96 (SomMot_LH_16). Boxplots
across the 100 HCP subjects with paired Wilcoxon tests -- same style/size as
Figure5_BEC_focalvsbifocal_A and Figure 4 C/D.

Output: codes/figures/outputs/Figure5_BEC_focalvsbifocal_effect_A.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
from Figure5_PanelA_HCP import _stars, _bracket
from Figure5_BEC_focalvsbifocal_A import (FOCAL_A_COLOR, FOCAL_B_COLOR,
                                          BIFOCAL_COLOR, roi_name, CACHE_DIR,
                                          NPZ, OUT_DIR, SUB, A_ROI)
setup()


def main():
    # Group best effect-increase partner B for A: most-positive eff_incr in A's
    # row (cortical, B != A) -> roi 96 = SomMot_LH_16.
    eff = np.load(NPZ)["eff_incr"].astype(float)[A_ROI].copy()
    eff[A_ROI] = -np.inf
    eff[:SUB] = -np.inf                          # cortical partners only
    B_ROI = int(np.argmax(eff))

    # Per-subject mean global effect size (GES) for the three conditions.
    rows = {"A": [], "B": [], "AB": []}
    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.glob("id_*.npz")):
            d = np.load(f)
            m = d["msq_all"]                      # <Sigma> over trials, (450,450)
            rows["A"].append(m[A_ROI, A_ROI])
            rows["B"].append(m[B_ROI, B_ROI])
            rows["AB"].append(m[A_ROI, B_ROI])
    else:  # consolidated cache (diag + row A_ROI) shipped with the bundle
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        d = np.load(root / "codes/HCP/results/BEC_focal_cache.npz")
        assert int(d["a_roi"]) == A_ROI
        rows["A"] = list(d["rowA_msq"][:, A_ROI])
        rows["B"] = list(d["diag_msq"][:, B_ROI])
        rows["AB"] = list(d["rowA_msq"][:, B_ROI])
    raw = {k: np.array(v) for k, v in rows.items()}
    labels = ["Focal A", "Focal B", "Bifocal\nA + B"]
    colors = [FOCAL_A_COLOR, FOCAL_B_COLOR, BIFOCAL_COLOR]

    # Fold the order of magnitude (~1e-4) into the y-label.
    exp = int(np.floor(np.log10(np.median(np.concatenate(list(raw.values()))))))
    data = [raw[k] * 10.0 ** (-exp) for k in ("A", "B", "AB")]

    fig, ax = plt.subplots(figsize=figsize_mm(58, 54), constrained_layout=True)
    pos = np.arange(1, 4)
    bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False,
                    patch_artist=True, zorder=2)
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, edgecolor="#3a3a3a", linewidth=1.0, alpha=0.9)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set(color="#3a3a3a", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#1a1a1a", linewidth=1.6)

    pairs = [(0, 2), (1, 2), (0, 1)]
    pv = {p: wilcoxon(data[p[0]], data[p[1]]).pvalue for p in pairs}
    caps = bp["caps"]
    top = max(caps[2 * i + 1].get_ydata()[0] for i in range(3))
    lo = min(caps[2 * i].get_ydata()[0] for i in range(3))
    span = top - lo; step = span * 0.09
    for lvl, p in enumerate(pairs):
        _bracket(ax, pos[p[0]], pos[p[1]], top + span * 0.04 + lvl * step,
                 step * 0.3, _stars(pv[p]))
    ax.set_ylim(lo - span * 0.06, top + span * 0.36)

    ax.set_xticks(pos); ax.set_xticklabels(labels)
    ax.set_ylabel(rf"GES (per subject)   ($\times 10^{{{exp}}}$)")
    ax.tick_params(width=0.8, length=3, direction="out")

    print(f"A=roi {A_ROI} ({roi_name(A_ROI)}), B=roi {B_ROI} ({roi_name(B_ROI)})")
    for nm in ("A", "B", "AB"):
        print(f"  median GES {nm} = {np.median(raw[nm]):.4g}")
    for p in pairs:
        print(f"  {labels[p[0]].split(chr(10))[0]} vs "
              f"{labels[p[1]].split(chr(10))[0]}: p={pv[p]:.2e}")
    save_panel(fig, OUT_DIR / "Figure5_BEC_focalvsbifocal_effect_A")


if __name__ == "__main__":
    main()
