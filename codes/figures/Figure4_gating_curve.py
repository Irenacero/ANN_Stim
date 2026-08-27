"""
Figure 4 -- Pooled gating curve (the headline result).

Mean global effect size of stimulation as a function of the brain's normalized
baseline energy at the moment of stimulation, pooled across the HCP cohort and
all cortical targets: low ongoing activity -> larger response.

Reuses the Figure-1E 2-D bins (normalized log E_t x principal gradient); we
marginalize over the gradient axis to get the 1-D energy curve. The light band
is the spread of the mean effect across gradient bins (across-target spread).

Input
    codes/HCP/results/Figure1E_panel_bins.npz   (energy x gradient bin sums)

Output
    codes/figures/outputs/Figure4_gating_curve.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
setup()

# Single line/band color: matplotlib tab:blue (matches the Panel A scatter).
TAB_BLUE = "#1f77b4"

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "codes/HCP/results/Figure1E_panel_bins.npz"
OUT_DIR = ROOT / "codes/figures/outputs"


def plot_gating(ax):
    """Draw the pooled gating curve (mean GES vs normalized baseline energy,
    with the across-target spread band) on `ax`. Returns (x, col_mean)."""
    d = np.load(CACHE)
    se, cnt = d["sum_eff"], d["cnt"]            # (gradient, energy)
    x_edges = d["x_edges"]
    x = (x_edges[:-1] + x_edges[1:]) / 2        # energy-bin centers, 0..1

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_mean = se / np.where(cnt > 0, cnt, np.nan)        # per (grad, energy)
    col_mean = np.nansum(se, axis=0) / np.nansum(cnt, axis=0)  # weighted, per energy
    col_std = np.nanstd(cell_mean, axis=0)                     # across-target spread

    ax.fill_between(x, col_mean - col_std, col_mean + col_std,
                    color=TAB_BLUE, alpha=0.18, lw=0, zorder=2)
    ax.plot(x, col_mean, color=TAB_BLUE, lw=1.6, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized baseline energy")
    ax.set_ylabel(r"GES")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(width=0.8, length=3, direction="out")
    return x, col_mean


def main():
    fig, ax = plt.subplots(figsize=figsize_mm(60, 52), constrained_layout=True)
    x, col_mean = plot_gating(ax)

    lo = col_mean[x < 0.25].mean()
    hi = col_mean[x > 0.75].mean()
    print(f"low-quartile mean={lo:.3e}  high-quartile mean={hi:.3e}  "
          f"ratio={lo/hi:.2f}  (first bin/last bin={col_mean[0]/col_mean[-1]:.2f})")
    save_panel(fig, OUT_DIR / "Figure4_gating_curve")


if __name__ == "__main__":
    main()
