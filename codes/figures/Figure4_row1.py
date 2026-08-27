"""
Figure 4, top row -- the four aligned columns in a single figure, ready to drop
into the assembly (no manual alignment needed). Reuses the axes-level plot
functions of the standalone panels so there is one source of truth:

    A | (1) baseline energy vs GES scatter      [Figure4_PanelA_HCP.plot_effect]
      | (2) pooled gating curve                 [Figure4_gating_curve.plot_gating]
      | (3) per-subject Spearman-r histogram    [Figure4_subject_hist.plot_hist]
    B | (4) baseline energy vs evoked scatter   [Figure4_PanelA_HCP.plot_evoked]

Columns (1) and (4) are the same (subject, region) example and carry the title;
columns (2) and (3) are cohort-level. The four plot boxes are placed at fixed,
equal positions so they are exactly the same size and perfectly aligned.

Input
    codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv
    codes/HCP/results/Figure1E_panel_bins.npz
    codes/HCP/results/Figure4_PanelB_node_corr_HCP_spearman.csv

Output
    codes/figures/outputs/Figure4_row1.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
from Figure4_PanelA_HCP import (load_hcp_subject_roi, plot_effect, plot_evoked,
                                HCP_SUBJECT, HCP_REGION, HCP_REGION_LABEL,
                                OUT_DIR)
from Figure4_gating_curve import plot_gating
from Figure4_subject_hist import plot_hist
setup()

# Four equal axes boxes (figure fractions): [left, bottom, width, height].
# Fixed positions -> identical size and aligned tops/bottoms regardless of the
# differing y-labels. Wider gaps keep each y-label out of its neighbor; the box
# height comes from BOX_H x FIG_H_MM (taller than wide for breathing room).
FIG_W_MM, FIG_H_MM = 174, 60
BOX_W, BOX_H = 0.17, 0.56
BOTTOM = 0.26
LEFTS = [0.070, 0.307, 0.543, 0.780]


def main():
    df = load_hcp_subject_roi(HCP_SUBJECT, HCP_REGION)
    baseline = df["global_baseline_energy"].to_numpy()
    effect = df["global_effect_size"].to_numpy()
    evoked = df["global_evoked_energy"].to_numpy()

    fig = plt.figure(figsize=figsize_mm(FIG_W_MM, FIG_H_MM))
    ax1, ax2, ax3, ax4 = (fig.add_axes([x, BOTTOM, BOX_W, BOX_H])
                          for x in LEFTS)

    re, pe = plot_effect(ax1, baseline, effect)
    plot_gating(ax2)
    plot_hist(ax3)
    rv, pv = plot_evoked(ax4, baseline, evoked)

    # The two single-(subject, region) example columns carry the title.
    title = f"{HCP_SUBJECT}  ·  {HCP_REGION_LABEL}"
    ax1.set_title(title, fontsize=8)
    ax4.set_title(title, fontsize=8)

    # Panel letters above the two example columns (above the title line).
    for ax, letter in ((ax1, "A"), (ax4, "B")):
        ax.text(-0.36, 1.30, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    print(f"[Figure4_row1] {HCP_SUBJECT} {HCP_REGION_LABEL}  "
          f"r(effect)={re:+.3f} (p={pe:.1e})  r(evoked)={rv:+.3f} (p={pv:.1e})")
    save_panel(fig, OUT_DIR / "Figure4_row1")


if __name__ == "__main__":
    main()
