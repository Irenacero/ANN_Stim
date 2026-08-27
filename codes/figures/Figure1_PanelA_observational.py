"""
Figure 1 Panel A, leftmost subpanel -- Observational signals.

A neutral LH lateral cortical surface in the project's brain-render style,
with four colored ROI markers (a, b, c, d) positioned on the cortex.  Each
ROI connects via a dashed line to a small dashed-bordered box containing
that region's BOLD signal trace, in the same color as the marker.

Inputs
    codes/HCP/results/processed/id_100206_signals.npy   (4680, 450)

Output
    codes/figures/outputs/Figure1_PanelA_observational.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup
from brain_render import N_SUBCORTICAL, render_neutral_hemi_array
setup()

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_NPY = ROOT / "codes/HCP/results/processed/id_100206_signals.npy"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TR = 0.72
N_T = 90                # first ~65 s of signal for each trace

DOT_EDGE = "#101010"        # black rim
LABEL_COLOR = "white"       # white letter

# Four ROIs at distinct, lateral-visible locations on LH. Each is colored
# (node fill + trace) with a distinct hue.
ROIS = [
    dict(label="a", color="#ff7f0e",                  # tab orange
         name="7Networks_LH_Cont_PFCl_4",             # frontal: DLPFC
         pos=(0.13, 0.66), corner="tl"),
    dict(label="b", color="#1f77b4",                  # tab blue
         name="7Networks_LH_DorsAttn_Post_13",        # parietal: SPL
         pos=(0.52, 0.85), corner="tr"),
    dict(label="c", color="#9467bd",                  # purple
         name="7Networks_LH_Default_PFC_10",          # temporal-ish: IFG
         pos=(0.30, 0.30), corner="bl"),
    dict(label="d", color="#2ca02c",                  # green
         name="7Networks_LH_Vis_31",                   # occipital: lateral visual
         pos=(0.88, 0.55), corner="br"),
]


def parcel_id_by_name(label_txt: Path, name: str) -> int:
    lines = label_txt.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == name:
            return int(lines[i + 1].split()[0])
    raise ValueError(f"{name!r} not in {label_txt}")


def crop_transparent(arr: np.ndarray, pad: int = 0,
                     alpha_thresh: int = 16) -> np.ndarray:
    """Trim transparent / near-transparent border around a rendered brain."""
    if arr.shape[-1] != 4:
        return arr
    alpha = arr[..., 3] > alpha_thresh
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0 = max(0, r0 - pad); r1 = min(arr.shape[0], r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(arr.shape[1], c1 + pad)
    return arr[r0:r1, c0:c1]


def main():
    # Load BOLD signals for the 4 ROIs, smooth, and z-score (clean look)
    sig = np.load(SIGNAL_NPY, mmap_mode="r")[:N_T]   # (N_T, 450)
    for roi in ROIS:
        pid = parcel_id_by_name(LABEL_TXT, roi["name"])
        s = sig[:, pid - 1].astype(float)
        s = savgol_filter(s, window_length=9, polyorder=3)
        s = (s - s.mean()) / s.std()
        roi["trace"] = s

    # Neutral brain (gray cortex). Modest zoom so the frontal pole and
    # occipital pole aren't clipped by the render canvas, then tight crop.
    brain = crop_transparent(
        render_neutral_hemi_array("lh", zoom=1.3),
        pad=0, alpha_thresh=32,
    )
    img_h, img_w = brain.shape[:2]
    img_aspect = img_w / img_h          # ~1.3 (wider than tall)

    # ---- Figure layout: explicit absolute axes rectangles ----
    fig = plt.figure(figsize=(8.0, 8.0))    # square

    # Brain: as wide as the figure allows, height tied to the image aspect so
    # there's no internal whitespace in the brain axes.
    brain_w_frac = 0.84
    brain_h_frac = brain_w_frac * fig.get_size_inches()[0] / img_aspect / fig.get_size_inches()[1]
    brain_left = (1.0 - brain_w_frac) / 2
    brain_bottom = (1.0 - brain_h_frac) / 2
    ax_brain = fig.add_axes([brain_left, brain_bottom, brain_w_frac, brain_h_frac])

    # Time-series boxes: x almost twice as long as y. Pushed into the corners
    # but close to the brain.
    ts_w_frac, ts_h_frac = 0.39, 0.13   # x ~50% longer than before
    margin = 0.02
    corner_axes = {
        "tl": fig.add_axes([margin, 1 - margin - ts_h_frac, ts_w_frac, ts_h_frac]),
        "tr": fig.add_axes([1 - margin - ts_w_frac, 1 - margin - ts_h_frac, ts_w_frac, ts_h_frac]),
        "bl": fig.add_axes([margin, margin, ts_w_frac, ts_h_frac]),
        "br": fig.add_axes([1 - margin - ts_w_frac, margin, ts_w_frac, ts_h_frac]),
    }

    # --- Brain ---
    ax_brain.imshow(brain)
    ax_brain.set_axis_off()
    # Use data coords for ROI markers so the circles are truly circular,
    # not stretched into ovals by an axes box that isn't square.
    h, w = brain.shape[:2]
    radius_px = 0.075 * min(h, w)

    for roi in ROIS:
        fx, fy = roi["pos"]
        cx, cy = fx * w, (1.0 - fy) * h          # data coords (origin upper)
        c = Circle((cx, cy), radius=radius_px,
                   facecolor=roi["color"], edgecolor=DOT_EDGE, linewidth=1.6,
                   zorder=6)
        ax_brain.add_patch(c)
        ax_brain.text(cx, cy, roi["label"],
                      color=LABEL_COLOR, fontsize=26, fontweight="bold",
                      ha="center", va="center", zorder=7)

    # --- Time-series boxes ---
    t_axis = np.arange(N_T) * TR
    for roi in ROIS:
        ax = corner_axes[roi["corner"]]
        ax.plot(t_axis, roi["trace"], color=roi["color"], linewidth=2.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(t_axis[0], t_axis[-1])
        for s in ax.spines.values():
            s.set_linestyle((0, (3, 2)))
            s.set_color("#333")
            s.set_linewidth(1.0)

    # --- Dashed connecting lines: from the node border (not center) to the
    # (0, 0) corner of each TS box (intersection of its x and y axes).
    # Need fully laid-out transforms, so draw first.
    fig.canvas.draw()
    for roi in ROIS:
        ax_ts = corner_axes[roi["corner"]]
        fx, fy = roi["pos"]
        cx, cy = fx * w, (1.0 - fy) * h
        # Display-coord endpoints
        node_disp = ax_brain.transData.transform((cx, cy))
        ts_disp = ax_ts.transAxes.transform((0.0, 0.0))
        dx, dy = ts_disp[0] - node_disp[0], ts_disp[1] - node_disp[1]
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        # Node radius in display pixels
        r_disp_x = (ax_brain.transData.transform((radius_px, 0))[0]
                    - ax_brain.transData.transform((0, 0))[0])
        radius_disp = abs(r_disp_x)
        # Offset start outward to the border
        start_disp = (node_disp[0] + ux * radius_disp,
                      node_disp[1] + uy * radius_disp)
        start_data = ax_brain.transData.inverted().transform(start_disp)
        con = ConnectionPatch(
            xyA=tuple(start_data), coordsA=ax_brain.transData,
            xyB=(0.0, 0.0), coordsB=ax_ts.transAxes,
            linestyle=(0, (4, 3)), color="#555", linewidth=0.9, alpha=0.9,
        )
        fig.add_artist(con)

    out_png = OUT_DIR / "Figure1_PanelA_observational.png"
    out_pdf = OUT_DIR / "Figure1_PanelA_observational.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}\nSaved {out_pdf}")


if __name__ == "__main__":
    main()
