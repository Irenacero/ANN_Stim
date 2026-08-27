"""
Brain map of one row of the bifocal improvement matrix.

For the most-reproducible target A (default roi 92 = SomMot_LH_12, rank 1 in the
top-20), show impr_naive[A, B] for every cortical target B on the cortical
surface: how much pairing A with B changes the response CV vs the best single
site. Same colormap as the matrix -- diverging brightRdBu centered at 0
(blue = bifocal more reproducible, red = less). A's own parcel is left blank.

Input  : codes/HCP/results/Figure5_BEC_matrices.npz
Output : codes/figures/outputs/Figure5_BEC_brain_rowA_{cv,eff}.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, DIV_CMAP, DIV_CMAP_NAME
from brain_render import (schaefer_parcels_to_vertices, render_hemi_lateral_array,
                          lh_inflated_surface, rh_inflated_surface)
from Figure1_PanelB import crop_transparent
from surfplot import Plot
setup()

A_COLOR = "#39ff14"     # lime  -> target A
B_COLOR = "#ff00ff"     # magenta -> best partner B


def render_with_outline(hemi, view, stat, color_range, outline_layers):
    """`view` ('lateral'/'medial') of hemisphere `stat` (brightRdBu); outline each
    (mask, color). Medial is needed so medial-wall targets (precuneus/PCC) show."""
    surf = lh_inflated_surface() if hemi == "lh" else rh_inflated_surface()
    side = "left" if hemi == "lh" else "right"
    kw = (dict(surf_lh=surf, surf_rh=None) if hemi == "lh"
          else dict(surf_lh=None, surf_rh=surf))
    p = Plot(**kw, views=view, size=(600, 480), zoom=1.25, layout="row",
             brightness=1.0)
    p.add_layer({side: stat}, cmap=DIV_CMAP_NAME, color_range=color_range, cbar=False)
    for mask, color in outline_layers:
        p.add_layer({side: mask}, cmap=ListedColormap([color]), as_outline=True,
                    cbar=False, color_range=(0, 1))
    return p.render().to_numpy(transparent_bg=True, scale=(2, 2))

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50
N_CORT = 400

# Which matrices to map as a row of A. best="min" picks the most-negative partner
# (CV reduction), "max" the most-positive (effect increase). a_roi is the featured
# target A (absolute BEC index): both the reproducibility (CV) and the effect panel
# feature SomMot_LH_12 (92); its best CV partner is Cont_RH_PFCl_6 (395), its best
# effect partner is within somatomotor cortex.
# views: list of (hemisphere, view) panels. A=92 is LH and its partners are RH
# lateral, so show LH lateral + RH lateral. asym=False uses a symmetric diverging
# scale centred at 0 (blue = bifocal more reproducible / larger, red = worse).
CONFIGS = {
    "cv": dict(field="impr_naive", best="min", a_roi=92, out="Figure5_BEC_brain_rowA_cv",
               views=[("lh", "lateral"), ("rh", "lateral")], asym=False,
               cbar=r"% change in CV(GES)  (<0: bifocal more reproducible)",
               title="CV change pairing {A} with each target B"),
    "eff": dict(field="eff_incr", best="max", a_roi=92, out="Figure5_BEC_brain_rowA_eff",
                views=[("lh", "lateral"), ("rh", "lateral")], asym=False,
                cbar=r"% change in mean(GES)  (>0: bifocal larger)",
                title="Effect-size change pairing {A} with each target B"),
}


def roi_name(roi):
    lines = LABEL_TXT.read_text().splitlines()
    nm = lines[2 * roi].strip()
    if nm.startswith("7Networks_"):
        p = nm.split("_")
        return f"{p[2]}_{p[1]}_{'_'.join(p[3:])}"
    return nm


def make_brain(cfg):
    A_ROI = cfg["a_roi"]
    M = np.load(NPZ)[cfg["field"]].astype(float)
    # Group best partner B for A, in the direction that means "best" for this matrix.
    arow = M[A_ROI].copy()
    arow[A_ROI] = np.nan; arow[:SUB] = np.nan
    B_ROI = int(np.nanargmin(arow) if cfg["best"] == "min" else np.nanargmax(arow))

    row = M[A_ROI, SUB:].copy()            # (400,) over cortical targets B
    row[A_ROI - SUB] = np.nan              # blank A's own parcel

    # Colour scale: variance -> asymmetric (vmin=-30, centre 0, vmax=max) so the
    # sparse blue values read well; effect -> symmetric.
    if cfg.get("asym"):
        vmax = float(np.nanmax(row))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-30.0, vmax=vmax)
    else:
        vmax = float(np.nanpercentile(np.abs(row), 98))
        norm = Normalize(-vmax, vmax)
    print(f"[{cfg['out']}] A=roi {A_ROI} ({roi_name(A_ROI)}) "
          f"B=roi {B_ROI} ({roi_name(B_ROI)})  range=[{norm.vmin:.1f},{norm.vmax:.1f}]%  "
          f"row mean={np.nanmean(row):+.2f}%")

    # Pre-map parcel values through `norm` to [0,1] so surfplot's linear
    # color_range=(0,1) reproduces the (possibly two-slope) normalization.
    row01 = norm(np.ma.masked_invalid(row)).filled(np.nan)
    vmaps = schaefer_parcels_to_vertices(row01)
    # Separate masks per region so A and B can be coloured distinctly.
    mA = np.full(N_CORT, np.nan); mA[A_ROI - SUB] = 1.0
    mB = np.full(N_CORT, np.nan); mB[B_ROI - SUB] = 1.0
    vA, vB = schaefer_parcels_to_vertices(mA), schaefer_parcels_to_vertices(mB)

    def layers(h):
        out = []
        if np.isfinite(vA[h]).any(): out.append((vA[h], A_COLOR))
        if np.isfinite(vB[h]).any(): out.append((vB[h], B_COLOR))
        return out

    panels = cfg["views"]                  # list of (hemi, view)
    arrs = [crop_transparent(render_with_outline(h, v, vmaps[h], (0.0, 1.0), layers(h)))
            for (h, v) in panels]

    ncol = len(panels)
    fig = plt.figure(figsize=figsize_mm(90, 56))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.09], hspace=0.12,
                          wspace=0.02, left=0.02, right=0.98, top=0.78, bottom=0.07)
    for k, (h, v) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(arrs[k]); ax.set_axis_off()
        ax.set_title(f"{'Left' if h == 'lh' else 'Right'} {v}", fontsize=7, pad=2)
    fig.suptitle(cfg["title"].format(A=roi_name(A_ROI)), y=0.97, fontsize=8)
    fig.text(0.5, 0.84, f"A = {roi_name(A_ROI)} (lime)    "
             f"B = {roi_name(B_ROI)} (magenta, best partner)",
             ha="center", va="center", fontsize=6.5, color="#555555")

    cax = fig.add_subplot(gs[1, :])
    sm = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(cfg["cbar"], labelpad=2)
    if cfg.get("asym"):
        cb.set_ticks([norm.vmin, 0.0, norm.vmax])
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.7, length=2.5)
    cax.set_position([0.22, cax.get_position().y0, 0.56, cax.get_position().height])
    save_panel(fig, OUT_DIR / cfg["out"])


def main():
    for cfg in CONFIGS.values():
        make_brain(cfg)


if __name__ == "__main__":
    main()
