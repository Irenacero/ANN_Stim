"""
Figure 4 -- Per-subject gating strength.

Distribution, across the 100 HCP participants, of the subject-level correlation
between baseline energy and global effect size: for each subject we take the
mean across cortical regions of the per-region Spearman r (the Panel B values).
The distribution sits well below zero, so the gating holds in essentially every
individual, not just in the pooled data.

Input
    codes/HCP/results/Figure4_PanelB_node_corr_HCP_spearman.csv   (Panel B cache)

Output
    codes/figures/outputs/Figure4_subject_hist.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
setup()

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "codes/HCP/results/Figure4_PanelB_node_corr_HCP_spearman.csv"
OUT_DIR = ROOT / "codes/figures/outputs"
SUBCORT_CORT_BOUNDARY = 50


def plot_hist(ax):
    """Draw the per-subject Spearman-r distribution on `ax`. Returns the
    per-subject values."""
    rtab = pd.read_csv(CACHE)
    cort = rtab[rtab.roi >= SUBCORT_CORT_BOUNDARY]
    # Per-subject mean of the node-wise Spearman r (baseline vs global effect).
    per_sub = cort.groupby("sub_id")["r_global_effect"].mean().to_numpy()

    ax.hist(per_sub, bins=16, color="#b9b9b9", edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)

    ax.set_xlabel(r"Spearman $r$ (per subject)")
    ax.set_ylabel("Participants")
    ax.tick_params(width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return per_sub


def main():
    fig, ax = plt.subplots(figsize=figsize_mm(50, 48), constrained_layout=True)
    per_sub = plot_hist(ax)
    print(f"n={per_sub.size}  median r={np.median(per_sub):.3f}  "
          f"{(per_sub < 0).mean()*100:.0f}% negative")
    save_panel(fig, OUT_DIR / "Figure4_subject_hist")


if __name__ == "__main__":
    main()
