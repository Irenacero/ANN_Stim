"""
Figure 1 Panel C -- When.

Top:    Raster of all 400 cortical ROIs over the first 500 BOLD timepoints
        (HCP TR = 0.72 s -> ~360 s window).
Middle: Global energy E(t) = sum_i X_i^2(t) over all 450 ROIs.
Bottom: Three LH lateral seed-based EC surface plots for L-TPJ stimulation at
        a low-energy, a higher-than-median-energy, and the peak-energy state.

Inputs
    codes/HCP/results/processed/id_100206_signals.npy      (4680, 450)
    codes/HCP/results/ECts/id_100206_ECt.npy               (500, 450, 450)

Output
    codes/figures/outputs/Figure1_PanelC.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, sequential_colors, INK
from brain_render import (
    N_SUBCORTICAL,
    cortical_network_only_ordering,
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

T_LIM = 500             # timepoints to display (matches EC tensor length)
TR = 0.72               # HCP TR (seconds)
SEED_NAME = "7Networks_LH_Default_Par_2"  # left TPJ / angular gyrus (~MNI -46 -58 20)
ENERGY_COLOR = INK           # neutral dark line for the energy

# Stacked z-scored BOLD traces (replaces the carpet heatmap), colored along a
# neutral grayscale gradient (light -> dark up the stack).
N_TRACES = 40                 # every 10th cortical ROI -> 40 traces
TRACE_OFFSET = 1.5            # vertical spacing in z-score units
TRACE_ALPHA = 0.8
TRACE_LW = 0.6


def parcel_id_by_name(label_txt: Path, name: str) -> int:
    lines = label_txt.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == name:
            return int(lines[i + 1].split()[0])
    raise ValueError(f"{name!r} not in {label_txt}")


def crop_transparent(arr: np.ndarray, pad: int = 4) -> np.ndarray:
    """Trim a transparent RGBA border around a rendered brain (alpha == 0).
    Keeps `pad` pixels of margin so the brain doesn't bleed into the axes edge.
    """
    if arr.shape[-1] != 4:
        return arr
    alpha = arr[..., 3] > 0
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0 = max(0, r0 - pad)
    r1 = min(arr.shape[0], r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(arr.shape[1], c1 + pad)
    return arr[r0:r1, c0:c1]


def pick_timepoints(E: np.ndarray):
    """Return (t_low, t_mid, t_high):
      - low : argmin in the first third
      - mid : local peak in the middle third, capped at 240 s
              (per user request: the visible bump just before 240 s)
      - high: argmax across the whole window.
    """
    n = len(E)
    t_low = int(np.argmin(E[: n // 3]))
    # The peak just before 240 s (user-picked region of the trace).
    mid_lo, mid_hi = int(220 / TR), int(240 / TR)
    t_mid = mid_lo + int(np.argmax(E[mid_lo:mid_hi]))
    t_high = int(np.argmax(E))
    return t_low, t_mid, t_high


EC_CACHE = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"


def build_panel(fig):
    """Draw Panel C (When) into the given figure or SubFigure."""
    signals = np.load(SIGNAL_NPY, mmap_mode="r")[:T_LIM]   # (500, 450)
    if EC_NPY.exists():
        ec = np.load(EC_NPY, mmap_mode="r")                # (500, 450, 450)
    else:  # seed-row cache shipped with the figure bundle (same values)
        ec = None
        ec_cache = np.load(EC_CACHE)

    E = (signals.astype(float) ** 2).sum(axis=1)
    raster = np.asarray(signals[:, N_SUBCORTICAL:]).astype(float)  # (500, 400)

    t_low, t_mid, t_high = pick_timepoints(E)
    picks = [(r"Low $E(t)$",  t_low),
             (r"Mid $E(t)$",  t_mid),
             (r"High $E(t)$", t_high)]
    guide_color = "#333333"
    print("Picked timepoints:")
    for lab, t in picks:
        print(f"  {lab}  t={t:>3}  ({t*TR:6.2f} s)  E={E[t]:.3g}")

    # Seed and per-timepoint EC rows; shared symmetric vmax across the three
    seed_pid = parcel_id_by_name(LABEL_TXT, SEED_NAME)
    seed_idx = seed_pid - 1
    seed_cort_idx = seed_idx - N_SUBCORTICAL
    if ec is not None:
        ec_rows = [np.asarray(ec[t, seed_idx])[N_SUBCORTICAL:] for _, t in picks]
    else:
        assert int(ec_cache["seed_idx"]) == seed_idx
        ec_rows = [ec_cache["seed_rows_tpj"][t][N_SUBCORTICAL:] for _, t in picks]
    mask = np.ones(400, dtype=bool); mask[seed_cort_idx] = False
    vlim = max(np.max(np.abs(r[mask])) for r in ec_rows) * 1.15

    # Render LH brains (shared color range); crop the transparent border so
    # the visible brain fills its axes box.
    brain_arrs = []
    for row_cort in ec_rows:
        vmaps = schaefer_parcels_to_vertices(row_cort)
        arr = render_hemi_lateral_array("lh", vmaps["lh"],
                                        color_range=(-vlim, vlim))
        brain_arrs.append(crop_transparent(arr))

    # Raster + energy share a narrow gridspec (margins reserved for E(t) ylabel
    # on the left and the BOLD colorbar on the right).
    gs_top = gridspec.GridSpec(
        2, 1,
        height_ratios=[3.0, 1.2],
        hspace=0.42,
        top=0.96, bottom=0.40, left=0.10, right=0.95,
    )
    ax_raster = fig.add_subplot(gs_top[0])
    ax_energy = fig.add_subplot(gs_top[1])

    # Brain row uses a separate, wider gridspec -- the brains push out past the
    # raster/energy horizontal extent toward the figure edges.
    # More vertical gap between energy axis and brains (top dropped 0.32 -> 0.26).
    gs_bot = gridspec.GridSpec(
        1, 3,
        wspace=0.04,
        top=0.26, bottom=0.02, left=0.02, right=0.98,
    )
    ax_brains = [fig.add_subplot(gs_bot[0, j]) for j in range(3)]

    time_s = np.arange(T_LIM) * TR

    x_ticks = [0, 60, 120, 180, 240, 300, 360]

    # --- Stacked z-scored BOLD traces ---
    # Order the 400 cortical ROIs by the Yeo-7 hierarchy (unimodal -> transmodal,
    # Limbic last) so the stack matches Panel B; bottom = Visual, top = Limbic.
    perm, _, _ = cortical_network_only_ordering(LABEL_TXT)
    sig_z = (raster - raster.mean(axis=0)) / raster.std(axis=0)   # (500, 400)
    sig_z = sig_z[:, perm]
    indices = np.linspace(0, raster.shape[1] - 1, N_TRACES, dtype=int)
    # Grayscale gradient (medium gray -> near-black up the stack); avoid the
    # white end of Greys so every trace stays visible on the white canvas.
    trace_colors = sequential_colors(N_TRACES, cmap=plt.cm.Greys, lo=0.35, hi=0.9)
    for k, roi in enumerate(indices):
        ax_raster.plot(
            time_s, sig_z[:, roi] + k * TRACE_OFFSET,
            color=trace_colors[k], alpha=TRACE_ALPHA, linewidth=TRACE_LW,
        )
    ymin = -3.0
    ymax = (N_TRACES - 1) * TRACE_OFFSET + 3.0
    ax_raster.set_xlim(time_s[0], time_s[-1])
    ax_raster.set_ylim(ymin, ymax)
    ax_raster.set_ylabel("ROIs")
    ax_raster.set_yticks([])
    ax_raster.set_xticks(x_ticks)
    ax_raster.set_xticklabels([])
    ax_raster.tick_params(axis="x", length=3, width=0.8, direction="out")
    for s in ("top", "right"):
        ax_raster.spines[s].set_visible(False)
    ax_raster.set_title("Empirical BOLD", pad=6)

    # --- Energy line ---
    ax_energy.plot(time_s, E, color=ENERGY_COLOR, linewidth=1.0)
    ax_energy.fill_between(time_s, E, E.min(), color=ENERGY_COLOR, alpha=0.15)
    ax_energy.set_ylabel(r"$E(t)$", rotation=90, va="center", labelpad=10)
    ax_energy.set_xlabel("Time (s)", labelpad=4)
    ax_energy.set_xlim(time_s[0], time_s[-1])
    ax_energy.set_xticks(x_ticks)
    for s in ("top", "right"):
        ax_energy.spines[s].set_visible(False)
    # Dashed guide lines on each picked timepoint (inside the energy panel)
    for _, t in picks:
        ax_energy.axvline(t * TR, color=guide_color, linewidth=0.9,
                          linestyle=(0, (4, 3)), alpha=0.9, zorder=3)

    # --- Brains ---
    title_fs = plt.rcParams["axes.labelsize"]  # match the axis-label size
    for ax, arr, (lab, t) in zip(ax_brains, brain_arrs, picks):
        ax.imshow(arr)
        ax.set_axis_off()
        ax.set_title(f"{lab}\n$t={t*TR:.1f}$ s",
                     color="black", pad=10, fontsize=title_fs)

    # --- Dashed guides crossing the gaps ---
    # 1) Raster bottom edge -> energy top edge: short vertical segment per pick
    raster_ymin = ax_raster.get_ylim()[0]
    for _, t in picks:
        con = ConnectionPatch(
            xyA=(t * TR, raster_ymin), coordsA=ax_raster.transData,
            xyB=(t * TR, ax_energy.get_ylim()[1]), coordsB=ax_energy.transData,
            linestyle=(0, (4, 3)), color=guide_color, linewidth=0.9, alpha=0.9,
        )
        fig.add_artist(con)

    # 2) Energy bottom edge -> just above the brain title (slightly tilted)
    for (lab, t), ax_b in zip(picks, ax_brains):
        con = ConnectionPatch(
            xyA=(t * TR, ax_energy.get_ylim()[0]), coordsA=ax_energy.transData,
            xyB=(0.5, 1.30), coordsB=ax_b.transAxes,
            linestyle=(0, (4, 3)), color=guide_color, linewidth=0.9, alpha=0.9,
        )
        fig.add_artist(con)


def main():
    fig = plt.figure(figsize=figsize_mm(85, 110))
    build_panel(fig)
    save_panel(fig, OUT_DIR / "Figure1_PanelC")


if __name__ == "__main__":
    main()
