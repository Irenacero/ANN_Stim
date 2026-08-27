"""
Figure 1 Panel D -- How.

Three stimulation strategies:
    1. Focal       : single-site, state-naive          -> high variability (measured).
    2. Closed-loop : single-site, state-dependent      -> predicted lower (?)
    3. Bifocal     : two-site, state-naive             -> predicted lower (?)

Bar chart of effects variability with question marks on the predicted bars.
(The brain renderings live in the subpanels for separate Inkscape composition.)

Output
    codes/figures/outputs/Figure1_PanelD.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, categorical, INK
setup()

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_COLORS = categorical(3)

# --- Toggles for improvements over the rough sketch ---
SHOW_ERROR_BAR_FOCAL = True
HATCH_PREDICTED_BARS = True
SHOW_FOCAL_REFERENCE_LINE = True


def build_panel(fig):
    """Draw Panel D (How) -- the effects-variability bar chart -- into fig."""
    bar_ax = fig.add_axes([0.12, 0.12, 0.84, 0.80])
    names = ["Focal", "Closed-loop", "Bifocal"]
    values = [1.00, 0.42, 0.48]                # illustrative
    focal_err = 0.18                            # illustrative
    bars = bar_ax.bar(names, values, color=BAR_COLORS, width=0.80,
                      edgecolor=INK, linewidth=0.8)

    if HATCH_PREDICTED_BARS:
        for b in bars[1:]:
            b.set_hatch("///")
            b.set_edgecolor(INK)
    if SHOW_ERROR_BAR_FOCAL:
        bar_ax.errorbar([0], [values[0]], yerr=[[focal_err], [focal_err]],
                        fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
    if SHOW_FOCAL_REFERENCE_LINE:
        bar_ax.axhline(values[0], color="#666", linestyle=(0, (3, 3)),
                       linewidth=0.8, alpha=0.7, zorder=0)

    # Question marks on the predicted bars
    for idx in (1, 2):
        bar_ax.text(idx, values[idx] + 0.16, "?", fontsize=16,
                    fontweight="bold", color=INK,
                    ha="center", va="center", zorder=4)

    bar_ax.set_ylabel("Effects variability")
    bar_ax.set_yticks([])
    bar_ax.tick_params(axis="x", length=0, pad=4)
    bar_ax.set_ylim(0, max(values) + 0.40)
    for s in ("top", "right", "left"):
        bar_ax.spines[s].set_visible(False)
    bar_ax.spines["bottom"].set_linewidth(0.8)


def main():
    fig = plt.figure(figsize=figsize_mm(60, 52))
    build_panel(fig)
    save_panel(fig, OUT_DIR / "Figure1_PanelD")


if __name__ == "__main__":
    main()
