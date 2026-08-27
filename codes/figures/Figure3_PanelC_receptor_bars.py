"""
Figure 3 -- Receptor-correlation bar panel (group level), all 19 PET maps.

For each receptor/transporter map, the Spearman correlation of its cortical density
with two node-level stimulation properties:
    left  : mean(GES)   -- response magnitude (responsiveness)
    right : CV(GES)     -- relative trial-to-trial variability (std/mean)

One row, two columns; receptors share a single bar color and a fixed order (sorted
by the mean-GES correlation) so the two columns are directly comparable. Both
spatial-autocorrelation nulls are marked above each bar (the bar height, the
observed rho, is identical for the two nulls; only the p-value differs):
    star  (open / filled) : spin test       nominal p<0.05 / survives FDR q<0.05
    triangle (open/filled): Moran S.R.       nominal p<0.05 / survives FDR q<0.05

Reads the correlation table produced by Figure5_variability_receptors.py.

Input
    codes/HCP/results/Figure5_variability_receptor_correlations.csv
Output
    codes/figures/outputs/Figure3_PanelC_receptor_bars.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, ACCENT_COOL
setup()

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "codes/HCP/results/Figure5_variability_receptor_correlations.csv"
OUT_DIR = ROOT / "codes/figures/outputs"

BAR = ACCENT_COOL          # single color for all bars
FADE = 0.4                 # alpha for non-(nominally)-significant bars


SPIN_C = "#1a1a1a"   # spin-null marker (star)
MORAN_C = "#dd8452"  # Moran-null marker (triangle), warm accent


def _sig_marker(ax, xi, ytip, sign, p, q, marker, color, off, dx, ms):
    """One null's significance marker, horizontally offset by dx from the bar centre
    and placed just past the tip: open if nominal p<0.05, filled if FDR q<0.05."""
    if not (p < 0.05):
        return
    filled = q < 0.05
    ax.plot(xi + dx, ytip + sign * off, marker=marker, ms=ms + (1.5 if filled else 0),
            mfc=color if filled else "none", mec=color, mew=0.9, zorder=6,
            clip_on=False)


def _bars(ax, names, rho, ps, qs, pm, qm, title):
    x = np.arange(len(names))
    span = max(rho.max(), 0) - min(rho.min(), 0) + 1e-9
    off = 0.055 * span
    ax.bar(x, rho, width=0.78, color=BAR, edgecolor="#2f3b52", linewidth=0.4, zorder=3)
    for bar, p_s, p_m in zip(ax.patches, ps, pm):     # fade if neither null nominal
        bar.set_alpha(1.0 if (p_s < 0.05 or p_m < 0.05) else FADE)
    ax.axhline(0, color="#555555", lw=0.8, zorder=2)

    # Two nulls side by side above each bar tip: spin star (left), Moran triangle (right).
    for xi, r, p_s, q_s, p_m, q_m in zip(x, rho, ps, qs, pm, qm):
        sign = 1 if r >= 0 else -1
        _sig_marker(ax, xi, r, sign, p_s, q_s, "*", SPIN_C, off, -0.20, 7.0)
        _sig_marker(ax, xi, r, sign, p_m, q_m, "^", MORAN_C, off, 0.20, 5.0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.4)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title(title, pad=4, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    t = pd.read_csv(CSV)
    t = t.sort_values("rho_resp", ascending=False).reset_index(drop=True)
    names = t["name"].to_numpy()

    lim = 1.20 * np.nanmax(np.abs(np.r_[t.rho_resp, t.rho_cv]))  # headroom for markers
    fig, (axE, axC) = plt.subplots(1, 2, figsize=figsize_mm(150, 52),
                                   constrained_layout=True)
    _bars(axE, names, t.rho_resp.to_numpy(),
          t.p_resp.to_numpy(), t.q_resp.to_numpy(),
          t.p_resp_moran.to_numpy(), t.q_resp_moran.to_numpy(),
          "vs effect size  mean(GES)")
    _bars(axC, names, t.rho_cv.to_numpy(),
          t.p_cv.to_numpy(), t.q_cv.to_numpy(),
          t.p_cv_moran.to_numpy(), t.q_cv_moran.to_numpy(),
          "vs relative variability  CV(GES)")
    for ax in (axE, axC):
        ax.set_ylim(-lim, lim)

    # Legend: marker shape = null, fill = FDR.
    axC.annotate(r"$\ast$ spin   $\blacktriangle$ Moran" + "\n"
                 r"open: $p<0.05$   filled: $q_{\mathrm{FDR}}<0.05$",
                 xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=5.2, color="#333333")

    ns = lambda p, q: f"{int((p<0.05).sum())} nom, {int((q<0.05).sum())} FDR"
    print(f"mean(GES): spin {ns(t.p_resp, t.q_resp)} | Moran {ns(t.p_resp_moran, t.q_resp_moran)}")
    print(f"CV(GES)  : spin {ns(t.p_cv, t.q_cv)} | Moran {ns(t.p_cv_moran, t.q_cv_moran)}")
    save_panel(fig, OUT_DIR / "Figure3_PanelC_receptor_bars")


if __name__ == "__main__":
    main()
