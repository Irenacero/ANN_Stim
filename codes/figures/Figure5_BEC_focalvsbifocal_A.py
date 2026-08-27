"""
Focal vs bifocal CV(GES) for the most reproducible pair and its single sites.

Per subject, the relative trial-to-trial variability CV(GES) = std/mean of the
response (all trials, no timing) for:
    focal A      = CV[A, A]   (single-site stimulation of A)
    focal B      = CV[B, B]   (single-site stimulation of the partner B)
    bifocal A+B  = CV[A, B]   (co-stimulation)
A,B are the most-reproducible cortical pair: the pair whose co-stimulation most
lowers CV(GES) relative to its better single site (most-negative impr_naive in the
subject-averaged matrix). GES = BEC^2, so CV = ssq_all/msq_all per (subject, pair).
Boxplots across the 100 HCP subjects with paired Wilcoxon tests.

Output: codes/figures/outputs/Figure5_BEC_focalvsbifocal_cv_A.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.colors as mcolors
from style import setup, save_panel, figsize_mm
from Figure5_PanelA_HCP import _stars, _bracket
setup()


def _tint(c, f):
    """Blend color `c` toward white by fraction `f` (0 = c, 1 = white)."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# Focal = tab:blue (open-loop / focal scheme), two gradations for the two single
# targets; bifocal = tab:green.
TAB_BLUE = "#1f77b4"
TAB_GREEN = "#2ca02c"
FOCAL_A_COLOR = TAB_BLUE              # vivid blue
FOCAL_B_COLOR = _tint(TAB_BLUE, 0.5)  # pale blue
BIFOCAL_COLOR = TAB_GREEN            # green

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "codes/HCP/results/BECts_reduced"
NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50


def roi_name(roi):
    nm = LABEL_TXT.read_text().splitlines()[2 * roi].strip()
    if nm.startswith("7Networks_"):
        p = nm.split("_"); return f"{p[2]}_{p[1]}_{'_'.join(p[3:])}"
    return nm


# Example pair (same as the brain-row map and the win-win scatter mark):
# A = SomMot_LH_12, B = Cont_RH_PFCl_6, which is A's best CV partner
# (rank 1/400 by impr_naive, -5.9%). Module-level so the effect-size companion
# panel (Figure5_BEC_focalvsbifocal_effect_A) can import the shared A_ROI.
A_ROI, B_ROI = 92, 395


def main():

    # Per-subject CV(GES) for the three conditions (CV = ssq_all / msq_all).
    rows = {"A": [], "B": [], "AB": []}
    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.glob("id_*.npz")):
            d = np.load(f)
            cv = d["ssq_all"].astype(float) / d["msq_all"].astype(float)
            rows["A"].append(cv[A_ROI, A_ROI])
            rows["B"].append(cv[B_ROI, B_ROI])
            rows["AB"].append(cv[A_ROI, B_ROI])
    else:  # consolidated cache (diag + row A_ROI) shipped with the bundle
        d = np.load(ROOT / "codes/HCP/results/BEC_focal_cache.npz")
        assert int(d["a_roi"]) == A_ROI
        rows["A"] = list(d["rowA_ssq"][:, A_ROI] / d["rowA_msq"][:, A_ROI])
        rows["B"] = list(d["diag_ssq"][:, B_ROI] / d["diag_msq"][:, B_ROI])
        rows["AB"] = list(d["rowA_ssq"][:, B_ROI] / d["rowA_msq"][:, B_ROI])
    data = [np.array(rows["A"]), np.array(rows["B"]), np.array(rows["AB"])]
    labels = ["Focal A", "Focal B", "Bifocal\nA + B"]
    colors = [FOCAL_A_COLOR, FOCAL_B_COLOR, BIFOCAL_COLOR]

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
    ax.set_ylabel(r"CV(GES) (per subject)")
    ax.tick_params(width=0.8, length=3, direction="out")

    print(f"A=roi {A_ROI} ({roi_name(A_ROI)}), B=roi {B_ROI} ({roi_name(B_ROI)})")
    for nm, p in zip(("A", "B", "AB"), ("A", "B", "AB")):
        print(f"  median CV {nm} = {np.median(rows[p]):.4g}")
    for p in pairs:
        print(f"  {labels[p[0]].split(chr(10))[0]} vs {labels[p[1]].split(chr(10))[0]}: p={pv[p]:.2e}")
    save_panel(fig, OUT_DIR / "Figure5_BEC_focalvsbifocal_cv_A")


if __name__ == "__main__":
    main()
