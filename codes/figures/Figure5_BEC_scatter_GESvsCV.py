"""
Figure 5 -- Effect-size gain vs variability change for every cortical region pair.

Each point is one cortical pair (i, j), i < j (79,800 pairs). Axes are the
group-level relative changes of bifocal co-stimulation vs the single-site
("focal") baselines, from Figure5_BEC_matrices.npz:

    x = impr_naive  : % change in CV(GES) of bifocal(i,j) vs the BEST (lowest-CV)
                      focal of i, j      ( < 0  -> more reproducible )
    y = eff_incr    : % change in mean GES of bifocal(i,j) vs the MAX focal of
                      i, j               ( > 0  -> larger effect )

The top-left quadrant (x < 0, y > 0) is the "win-win" region (more reproducible
AND larger effect). Points are colored by local density (brightPMY). The two
example pairs from the focal-vs-bifocal panels are marked:
    blue  star : SomMot_LH_12 + Cont_RH_PFCl_6  (variability-optimal partner)
    green star : SomMot_LH_12 + SomMot_LH_16     (effect-optimal partner)

Input
    codes/HCP/results/Figure5_BEC_matrices.npz
Output
    codes/figures/outputs/Figure5_BEC_scatter_GESvsCV.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, SEQ_CMAP, GUIDE, INK
setup()

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50                       # rois 0..49 subcortical, 50..449 cortical

TAB_BLUE = "#1f77b4"
TAB_GREEN = "#2ca02c"
# (roi_i, roi_j, color, label, label-offset-in-points) example pairs to mark.
MARKS = [(92, 395, TAB_BLUE,  "SomMot_LH_12 +\nCont_RH_PFCl_6", (9, 7)),
         (92,  96, TAB_GREEN, "SomMot_LH_12 +\nSomMot_LH_16", (9, -16))]


def main():
    g = np.load(NPZ)
    impr = g["impr_naive"].astype(float)     # x: %Δ CV vs best focal
    eff = g["eff_incr"].astype(float)        # y: %Δ GES vs max focal

    ii, jj = np.triu_indices(450, k=1)
    m = (ii >= SUB) & (jj >= SUB)
    ii, jj = ii[m], jj[m]
    x, y = impr[ii, jj], eff[ii, jj]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    # Local density for coloring (log counts on a 2-D grid), densest plotted last.
    nb = 300
    H, xe, ye = np.histogram2d(x, y, bins=nb)
    ix = np.clip(np.searchsorted(xe, x) - 1, 0, nb - 1)
    iy = np.clip(np.searchsorted(ye, y) - 1, 0, nb - 1)
    dens = np.log10(H[ix, iy] + 1.0)
    order = np.argsort(dens)

    # Wider than tall so the plotting area reads ~square and aligns, height-matched,
    # with the network-pair matrices in the same row.
    fig, ax = plt.subplots(figsize=figsize_mm(95, 60), constrained_layout=True)

    # Robust limits for the bulk, then widened to include the marked examples.
    xlo, xhi = np.nanpercentile(x, 0.2), np.nanpercentile(x, 99.8)
    ylo, yhi = np.nanpercentile(y, 0.2), np.nanpercentile(y, 99.8)
    mx = [impr[i, j] for i, j, *_ in MARKS]
    my = [eff[i, j] for i, j, *_ in MARKS]
    ax.axvline(0, color=GUIDE, ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.axhline(0, color=GUIDE, ls=(0, (3, 2)), lw=0.8, zorder=1)

    sc = ax.scatter(x[order], y[order], c=dens[order], s=2.2, cmap=SEQ_CMAP,
                    norm=Normalize(0, np.nanpercentile(dens, 99.5)),
                    linewidths=0, rasterized=True, zorder=2)

    # Example pairs.
    for i, j, col, lab, off in MARKS:
        ax.scatter(impr[i, j], eff[i, j], s=46, marker="*", facecolor=col,
                   edgecolor="#1a1a1a", linewidths=0.6, zorder=5)
        ax.annotate(lab, (impr[i, j], eff[i, j]),
                    textcoords="offset points", xytext=off, fontsize=6,
                    color="#1a1a1a", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=col, lw=0.6, alpha=0.85))

    ax.set_xlim(min(xlo, min(mx)) - 1.5, xhi + 1.5)
    ax.set_ylim(min(ylo, min(my)) - 2, max(yhi, max(my)) + 4)
    ax.set_xlabel(r"$\Delta$CV vs best focal (%)   ($<0$ = more reproducible)")
    ax.set_ylabel(r"$\Delta$GES vs strongest focal (%)   ($>0$ = larger effect)")
    ax.tick_params(width=0.8, length=3, direction="out")

    # "win-win" label in the top-left quadrant.
    ax.text(0.02, 0.97, "win-win\n(↓variability, ↑effect)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=INK, style="italic")

    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"pairs per bin ($\log_{10}$)", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.8, length=2.5)
    cb.outline.set_linewidth(0.6)

    n = len(x)
    ww = ((x < 0) & (y > 0)).mean() * 100
    print(f"[Figure5_BEC_scatter_GESvsCV] {n:,} cortical pairs  "
          f"win-win quadrant (x<0,y>0) = {ww:.1f}%  "
          f"| y>0: {(y>0).mean()*100:.1f}%  x<0: {(x<0).mean()*100:.1f}%")
    save_panel(fig, OUT_DIR / "Figure5_BEC_scatter_GESvsCV")


if __name__ == "__main__":
    main()
