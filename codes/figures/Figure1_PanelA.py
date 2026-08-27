"""
Figure 1 Panel A: BOLD fMRI -> ANN (unperturbed -> virtual perturbation)
-> Seed-based effective connectivity.

The panel is a wide single-row figure with four sub-cells:
    1. BOLD fMRI     : observational LH brain with 4 colored ROI markers and
                       4 corner BOLD-trace boxes (a, b, c, d).
    2. Unperturbed   : ANN schematic X(t) -> ANN -> X(t+1) with integer values.
    3. Virtual perturbation: same schematic with the b row of X(t) perturbed,
                       and the affected rows of X(t+1) highlighted using the
                       Panel B EC colormap.
    4. Seed-based EC : LH lateral brain colored by EC[t, L-TPJ, :] (cortical),
                       in the project's brain-render style.

Output
    codes/figures/outputs/Figure1_PanelA.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import (setup, save_panel, figsize_mm, categorical, text_on,
                   DIV_CMAP, DIV_CMAP_NAME, INK, MUTE)
from brain_render import (
    N_SUBCORTICAL,
    fig_size_inches,
    render_neutral_hemi_array,
    render_hemi_lateral_array,
    schaefer_parcels_to_vertices,
)
setup()

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_NPY = ROOT / "codes/HCP/results/processed/id_100206_signals.npy"
EC_NPY = ROOT / "codes/HCP/results/ECts/id_100206_ECt.npy"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Helpers ------------------------------------------------------------
def parcel_id_by_name(name: str) -> int:
    lines = LABEL_TXT.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == name:
            return int(lines[i + 1].split()[0])
    raise ValueError(f"{name!r} not in label file")


def crop_transparent(arr: np.ndarray, pad: int = 0,
                     alpha_thresh: int = 32) -> np.ndarray:
    if arr.shape[-1] != 4:
        return arr
    alpha = arr[..., 3] > alpha_thresh
    rows = np.any(alpha, axis=1); cols = np.any(alpha, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return arr[max(0, r0 - pad):min(arr.shape[0], r1 + pad),
               max(0, c0 - pad):min(arr.shape[1], c1 + pad)]


def asymmetric_diverging_cmap(neg_frac: float,
                              sample_range=(0.16, 0.90),
                              base=DIV_CMAP,
                              n: int = 512) -> LinearSegmentedColormap:
    base_cmap = mpl.colormaps[base] if isinstance(base, str) else base
    s_lo, s_hi = sample_range
    n_neg = max(2, int(round(n * neg_frac)))
    n_pos = max(2, n - n_neg)
    blue_part = base_cmap(np.linspace(s_lo, 0.5, n_neg))
    red_part = base_cmap(np.linspace(0.5, s_hi, n_pos))
    colors = np.vstack([blue_part, red_part])
    return LinearSegmentedColormap.from_list("asym_div", colors, N=n)


# ---- Observational sub-cell ---------------------------------------------
TR = 0.72
N_T_OBS = 90

# 4 ROI colors from the shared categorical (muted) palette -- these match the
# bold-trace subpanels (bold_trace_a..d).
ROI_COLORS = categorical(4)

ROIS = [
    dict(label="a", color=ROI_COLORS[0], name="7Networks_LH_Cont_PFCl_4",
         pos=(0.13, 0.66), corner="tl"),
    dict(label="b", color=ROI_COLORS[1], name="7Networks_LH_DorsAttn_Post_13",
         pos=(0.52, 0.85), corner="tr"),
    dict(label="c", color=ROI_COLORS[2], name="7Networks_LH_Default_PFC_10",
         pos=(0.30, 0.30), corner="bl"),
    dict(label="d", color=ROI_COLORS[3], name="7Networks_LH_Vis_31",
         pos=(0.88, 0.55), corner="br"),
]


def build_observational(fig, bbox):
    """BOLD-fMRI sub-cell: neutral LH brain with the 4 colored ROI markers.

    The ROI BOLD time series are emitted separately as subpanels
    (bold_trace_a..d) for independent Inkscape composition, so only the brain
    and its markers live here.
    """
    bx0, by0, bw, bh = bbox

    brain = crop_transparent(render_neutral_hemi_array("lh", zoom=1.3))
    img_h, img_w = brain.shape[:2]
    img_aspect = img_w / img_h
    fig_w, fig_h = fig_size_inches(fig)

    brain_w_frac = bw * 0.92
    brain_h_frac = brain_w_frac * fig_w / img_aspect / fig_h
    if brain_h_frac > bh * 0.92:
        brain_h_frac = bh * 0.92
        brain_w_frac = brain_h_frac * fig_h * img_aspect / fig_w
    brain_left = bx0 + (bw - brain_w_frac) / 2
    brain_bottom = by0 + (bh - brain_h_frac) / 2
    ax_brain = fig.add_axes([brain_left, brain_bottom, brain_w_frac, brain_h_frac])
    ax_brain.imshow(brain); ax_brain.set_axis_off()

    h, w = brain.shape[:2]
    radius_px = 0.075 * min(h, w)
    for roi in ROIS:
        fx, fy = roi["pos"]
        cx, cy = fx * w, (1 - fy) * h
        c = Circle((cx, cy), radius=radius_px, facecolor=roi["color"],
                   edgecolor=INK, linewidth=1.0, zorder=6)
        ax_brain.add_patch(c)
        ax_brain.text(cx, cy, roi["label"], color=text_on(roi["color"]),
                      fontsize=7, fontweight="bold", ha="center", va="center",
                      zorder=7)


# ---- Schematic sub-cells (unperturbed / perturbation) -------------------
SCHEMATIC_LABELS = ["a", "b", "c", "d"]
SCHEMATIC_FC_NORMAL = MUTE
SCHEMATIC_EC = "#3a3a55"
ANN_FC = "#c8dff5"             # brighter pale blue
ANN_EC = "#4a5d7a"
LABEL_GRAY = "#3a3a3a"


def _highlight_color(kind: str, scheme_cmap):
    """Extract pale-red / pale-blue from the EC colormap for highlight fills."""
    if kind == "red":
        return scheme_cmap(0.82)
    elif kind == "blue":
        return scheme_cmap(0.10)
    else:
        raise ValueError(kind)


def build_schematic(fig, bbox, *, x_vals, y_vals,
                    x_hi=None, y_hi=None, superscript: str = "",
                    scheme_cmap=None):
    """Draw an X(t) -> ANN -> X(t+1) schematic into `bbox` of `fig`.

    x_vals, y_vals: lists of 4 numeric values (input and output).
    x_hi, y_hi: dict {row_idx: 'red'|'blue'} for highlighted rows.
    superscript: '' for unperturbed, '(b)' (or similar) for perturbed.
    """
    x_hi = x_hi or {}
    y_hi = y_hi or {}
    bx0, by0, bw, bh = bbox

    ax = fig.add_axes([bx0, by0, bw, bh])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.axis("off")

    # Geometry: 4 attached boxes per vector, spanning most of the cell height
    # so the schematic fills its bbox like the brain panels do.
    box_w, box_h = 0.10, 0.18                     # boxes touch vertically
    y_centers = np.array([0.81, 0.63, 0.45, 0.27])  # 0.18 apart -> attached
    x_in_label = 0.14
    x_in = 0.25
    ann_x0, ann_x1 = 0.40, 0.70
    ann_y0, ann_y1 = 0.13, 0.93
    x_out = 0.82

    # --- Vector column labels (X(t), X^(b)(t)) -----
    if superscript:
        x_top_lbl = rf"$X^{{{superscript}}}(t)$"
        y_top_lbl = rf"$X^{{{superscript}}}(t+1)$"
    else:
        x_top_lbl = r"$X(t)$"
        y_top_lbl = r"$X(t+1)$"
    ax.text(x_in, 0.96, x_top_lbl, ha="center", va="bottom", fontsize=8)
    ax.text(x_out, 0.96, y_top_lbl, ha="center", va="bottom", fontsize=8)

    # --- ANN rounded rectangle -----
    ann = FancyBboxPatch(
        (ann_x0, ann_y0), ann_x1 - ann_x0, ann_y1 - ann_y0,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.4, edgecolor=ANN_EC, facecolor=ANN_FC, zorder=2,
    )
    ax.add_patch(ann)
    ax.text((ann_x0 + ann_x1) / 2, (ann_y0 + ann_y1) / 2, "ANN",
            ha="center", va="center", fontsize=10, color="#1f3d5e",
            fontweight="bold", zorder=3)

    # --- Input boxes (attached squares) + letter labels ----
    for i, (val, y) in enumerate(zip(x_vals, y_centers)):
        fc = (_highlight_color(x_hi[i], scheme_cmap) if i in x_hi
              else SCHEMATIC_FC_NORMAL)
        box = FancyBboxPatch(
            (x_in - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="square,pad=0.0",
            linewidth=1.0, edgecolor=SCHEMATIC_EC, facecolor=fc, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x_in, y, f"{val}", ha="center", va="center",
                fontsize=8, color="#101010", zorder=4)
        ax.text(x_in_label, y, SCHEMATIC_LABELS[i], ha="center", va="center",
                fontsize=7, color=LABEL_GRAY, zorder=4)

    # --- Output boxes (attached squares) ----
    for i, (val, y) in enumerate(zip(y_vals, y_centers)):
        fc = (_highlight_color(y_hi[i], scheme_cmap) if i in y_hi
              else SCHEMATIC_FC_NORMAL)
        box = FancyBboxPatch(
            (x_out - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="square,pad=0.0",
            linewidth=1.0, edgecolor=SCHEMATIC_EC, facecolor=fc, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x_out, y, f"{val}", ha="center", va="center",
                fontsize=8, color="#101010", zorder=4)

    # --- Arrows -----
    arrow_kw = dict(arrowstyle="->", color=SCHEMATIC_EC,
                    linewidth=1.4, mutation_scale=16, zorder=2)
    ax.add_patch(FancyArrowPatch(
        (x_in + box_w / 2 + 0.01, 0.53), (ann_x0 - 0.005, 0.53), **arrow_kw,
    ))
    ax.add_patch(FancyArrowPatch(
        (ann_x1 + 0.005, 0.53), (x_out - box_w / 2 - 0.01, 0.53), **arrow_kw,
    ))


# ---- Seed-based EC sub-cell ---------------------------------------------
SEED_NAME = "7Networks_LH_Default_Par_2"
T_FIXED = 200


EC_CACHE = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"


def build_seed_ec(fig, bbox):
    bx0, by0, bw, bh = bbox

    seed_idx = parcel_id_by_name(SEED_NAME) - 1
    if EC_NPY.exists():
        ec = np.load(EC_NPY, mmap_mode="r")
        row_cort = np.asarray(ec[T_FIXED, seed_idx])[N_SUBCORTICAL:]
    else:  # seed-row cache shipped with the figure bundle (same values)
        d = np.load(EC_CACHE)
        assert int(d["seed_idx"]) == seed_idx
        row_cort = d["seed_rows_tpj"][T_FIXED][N_SUBCORTICAL:]

    seed_cort = seed_idx - N_SUBCORTICAL
    mask = np.ones_like(row_cort, dtype=bool); mask[seed_cort] = False
    vlim = float(np.max(np.abs(row_cort[mask]))) * 1.15

    vmaps = schaefer_parcels_to_vertices(row_cort)
    arr = crop_transparent(render_hemi_lateral_array(
        "lh", vmaps["lh"], cmap=DIV_CMAP_NAME, color_range=(-vlim, vlim), zoom=1.3,
    ))

    img_h, img_w = arr.shape[:2]
    img_aspect = img_w / img_h
    fig_w, fig_h = fig_size_inches(fig)

    brain_w_frac = bw * 0.92
    brain_h_frac = brain_w_frac * fig_w / img_aspect / fig_h
    if brain_h_frac > bh * 0.92:
        brain_h_frac = bh * 0.92
        brain_w_frac = brain_h_frac * fig_h * img_aspect / fig_w
    bl = bx0 + (bw - brain_w_frac) / 2
    bb = by0 + (bh - brain_h_frac) / 2
    ax_b = fig.add_axes([bl, bb, brain_w_frac, brain_h_frac])
    ax_b.imshow(arr); ax_b.set_axis_off()


# ---- Orchestrator -------------------------------------------------------
TITLES = ["BOLD fMRI", "Unperturbed", "Virtual Perturbation",
          "Seed-based EC (L-TPJ)"]


def build_panel(fig):
    """Draw the four sub-cells of Panel A into the given figure (or SubFigure).
    Coordinates are relative to the passed figure (0..1)."""
    n_cells = 4
    cell_w = 1.0 / n_cells
    top_band = 0.88          # leave room for titles
    bot = 0.04
    cell_h = top_band - bot

    # Reusable highlight cmap (matches Panel B's asymmetric cmap)
    scheme_cmap = asymmetric_diverging_cmap(neg_frac=1 / 6,
                                            sample_range=(0.16, 0.90))

    # ---- Cell 1: BOLD fMRI (observational) ----
    build_observational(fig, bbox=(0.0,         bot, cell_w, cell_h))

    # ---- Cell 2: Unperturbed schematic ----
    build_schematic(fig, bbox=(cell_w,           bot, cell_w, cell_h),
                    x_vals=[1, 2, -1, 3], y_vals=[2, 1, 0, 1],
                    scheme_cmap=scheme_cmap)

    # ---- Cell 3: Virtual Perturbation (perturb b: 2 -> 3) ----
    build_schematic(fig, bbox=(2 * cell_w,       bot, cell_w, cell_h),
                    x_vals=[1, 3, -1, 3], y_vals=[3, 1, -1, 1],
                    x_hi={1: "red"},
                    y_hi={0: "red", 2: "blue"},
                    superscript="(b)",
                    scheme_cmap=scheme_cmap)

    # ---- Cell 4: Seed-based EC ----
    build_seed_ec(fig, bbox=(3 * cell_w,         bot, cell_w, cell_h))

    # ---- Titles ----
    for i, title in enumerate(TITLES):
        fig.text(i * cell_w + cell_w / 2, 0.93, title,
                 ha="center", va="bottom", fontsize=plt.rcParams["axes.titlesize"])


def main():
    # Wide single-row, double-column panel at final size.
    fig = plt.figure(figsize=figsize_mm(174, 46))
    build_panel(fig)
    save_panel(fig, OUT_DIR / "Figure1_PanelA")


if __name__ == "__main__":
    main()
