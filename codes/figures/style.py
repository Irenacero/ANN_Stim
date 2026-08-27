"""
Shared publication style for all paper figures. Import in every figure script:

    from style import setup, save_panel, figsize_mm, DIV_CMAP, SEQ_CMAP
    setup()
    fig, ax = plt.subplots(figsize=figsize_mm(85, 55))
    ...
    save_panel(fig, OUT_DIR / "panel_A")     # -> SVG + PDF, transparent, editable text

Design target: a vector pipeline for high-impact neuroscience journals
(Python -> SVG/PDF panels -> Inkscape assembly -> final figure). Therefore:

  * panels are generated at final publication size -- never rescale later
  * Arial, 7-8 pt; text stays editable in SVG (``svg.fonttype = "none"``)
  * fonts and linewidths are centralized here, not repeated per script
  * canonical colormaps live here so every panel stays consistent:
        DIV_CMAP  brightRdBu  -- brain maps + effective-connectivity matrices
        SEQ_CMAP  brightPMY   -- responsiveness / global effect-size maps
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Importing custom_cmaps registers brightRdBu / brightPMY (+ _r, _lin) with
# matplotlib, so they are usable by string name (e.g. in surfplot) too.
from custom_cmaps import brightRdBu, brightPMY, brightRdBu_lin, brightPMY_lin  # noqa: F401


# ---- figure sizing (millimetres) ----------------------------------------
MM = 1.0 / 25.4                  # millimetres -> inches
SINGLE_COL_MM = 85.0             # typical single-column width
ONEHALF_COL_MM = 114.0
DOUBLE_COL_MM = 174.0            # typical double-column width


def figsize_mm(w_mm: float, h_mm: float) -> tuple[float, float]:
    """Figure size in inches from a width/height given in millimetres."""
    return (w_mm * MM, h_mm * MM)


# ---- rcParams ------------------------------------------------------------
def setup():
    """Apply the publication rcParams. Call once at the top of every script."""
    plt.rcParams.update({
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "regular",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "figure.labelsize": 8,
        # lines / axes / ticks
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # vector cleanliness + editable text
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # output: transparent by default for clean Inkscape compositing
        "figure.dpi": 150,
        "savefig.dpi": 600,          # only affects rasterized insets (brains)
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    })


# ---- canonical colormaps -------------------------------------------------
# Diverging blue <-> red for brain maps and effective-connectivity matrices.
DIV_CMAP = brightRdBu
# Sequential purple -> yellow for responsiveness / global effect-size maps.
SEQ_CMAP = brightPMY
# Registered string names, for libraries that need a name (e.g. surfplot/VTK).
DIV_CMAP_NAME = "brightRdBu"
SEQ_CMAP_NAME = "brightPMY"


# ---- export --------------------------------------------------------------
def save_panel(fig, path, *, formats=("svg", "pdf"), png_preview=True, dpi=600,
               close=True):
    """Export a modular panel for Inkscape assembly.

    Writes one file per format with transparent background and editable text
    (vector SVG by default, plus PDF). ``path`` may omit an extension.

    Parameters
    ----------
    formats      : vector formats to write (default SVG + PDF).
    png_preview  : also drop a raster PNG for quick visual checks.
    dpi          : raster resolution (PNG, and any rasterized brain insets).
    close        : close the figure afterwards (avoids state leaking).
    """
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    exts = list(formats) + (["png"] if png_preview else [])
    for ext in exts:
        fig.savefig(stem.with_suffix(f".{ext}"),
                    bbox_inches="tight", transparent=True,
                    dpi=(dpi if ext == "png" else None))
    if close:
        plt.close(fig)
    print(f"  -> {stem.name}.{{{','.join(exts)}}}")
    return stem


# ---- coherent categorical palette ---------------------------------------
# Every categorical color is sampled from the two data colormaps so the whole
# figure set stays on two hue families (brightRdBu / brightPMY) + neutrals.

INK = "#1a1a1a"        # primary dark: single lines (energy), emphasis, text
GUIDE = "#9aa0a8"      # dashed guide lines / subtle dividers
MUTE = "#f2f2f5"       # neutral light fill (schematic boxes)
BRAIN_FILL = "#e6e6f0"  # neutral brain silhouette fill
BRAIN_EDGE = "#3a3a55"  # neutral brain silhouette edge


# Categorical colors come from seaborn's "muted" qualitative palette: soft,
# colorblind-friendly, and distinct (no muddy gray midpoint like cividis, no
# brilliance like brightPMY -- which stays reserved for the responsiveness
# heatmap). One source keeps every discrete slot coherent, including the 7
# Yeo networks below.
import seaborn as _sns  # noqa: E402
_CAT_QUAL = _sns.color_palette("muted")        # 10 soft qualitative colors


def categorical(n, palette=None):
    """``n`` distinct, soft, colorblind-friendly colors for any 'few discrete
    categories' slot -- ROI labels, bar groups, grouped scatters. Drawn from
    seaborn's muted qualitative palette so every categorical slot is coherent.
    """
    pal = palette if palette is not None else _CAT_QUAL
    return list(pal[:n]) if n <= len(pal) else _sns.color_palette("muted", n)


def sequential_colors(n, cmap=SEQ_CMAP, lo=0.05, hi=0.95):
    """``n`` colors for an ordered stack / gradient, e.g. stacked BOLD traces.
    Defaults to brightPMY so a continuous gradient reads as 'the responsiveness
    family'."""
    return [cmap(x) for x in np.linspace(lo, hi, n)]


def text_on(color):
    """Return black or white, whichever is legible on ``color`` (for labels
    drawn on colored markers/bars)."""
    r, g, b = mpl.colors.to_rgb(color)
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "#ffffff"


# Distinct, soft colors for the 7 Yeo networks (Vis, SomMot, DorsAttn,
# SalVentAttn, Cont, Default, Limbic), from seaborn's muted qualitative
# palette -- distinguishable but not garish. Exposed as a list and a dict.
YEO7_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                 "Cont", "Default", "Limbic"]
YEO7_COLOR_LIST = _sns.color_palette("muted", len(YEO7_NETWORKS))
YEO7_COLORS = dict(zip(YEO7_NETWORKS, YEO7_COLOR_LIST))

# Binary warm/cool accents (e.g. state-naive vs state-dependent stimulation).
# Colorblind-friendly orange/blue pair.
ACCENT_WARM = "#dd8452"          # muted orange (state-naive)
ACCENT_COOL = "#4c72b0"          # muted blue   (state-dependent)

# Region colors for the 4 illustrative ROIs across panels A--C, as a dict
# (A/B/C/D) and a list. Derived from cividis so ROI markers and their BOLD
# traces match the rest of the categorical palette.
ROI_COLOR_LIST = categorical(4)
ROI_COLORS = dict(zip("ABCD", ROI_COLOR_LIST))


def clean_axes(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(width=0.8, direction="out", length=3)


def rounded_box(ax, x, y, w, h, fc="#e6e6f0", ec="#3a3a55", lw=1.2, rounding=0.05):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    return box


def lateral_brain_silhouette(scale=1.0, anchor=(0.0, 0.0)):
    """Return (x, y) of a stylized lateral brain silhouette.

    The shape is anchored with frontal pole on the left, occipital pole on the
    right, and temporal lobe on the bottom-front, matching a typical
    left-hemisphere lateral view.
    """
    from scipy.interpolate import splprep, splev
    anchors = np.array([
        [0.02, 0.58],   # frontal pole (prominent forward bulge)
        [0.04, 0.78],   # upper frontal rise
        [0.16, 0.93],   # frontal-top corner
        [0.40, 1.00],   # top-frontal
        [0.62, 0.99],   # top-parietal
        [0.82, 0.92],   # parieto-occipital
        [0.95, 0.74],   # occipital upper
        [1.00, 0.55],   # occipital pole
        [0.96, 0.38],   # occipital lower
        [0.86, 0.30],   # temporo-occipital
        [0.66, 0.26],   # temporal mid
        [0.42, 0.24],   # temporal anterior
        [0.22, 0.27],   # temporal pole
        [0.10, 0.36],   # inferior frontal
        [0.04, 0.48],   # back to frontal pole
    ])
    tck, _ = splprep([anchors[:, 0], anchors[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, 300)
    x, y = splev(u, tck)
    return x * scale + anchor[0], y * scale + anchor[1]


# Backwards-compatible aliases (older scripts import these names).
DIV_CMAP_LEGACY = "RdYlBu_r"
SEQ_CMAP_LEGACY = "viridis"
