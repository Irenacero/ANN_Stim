"""
Figure 1 Panel B -- Where.

Top:    Time-averaged effective connectivity matrix, cortical 400 x 400, sorted
        by Yeo-7 networks. Network boundaries marked with dark-gray dividers.
Bottom: Responsiveness brain map -- for each cortical source j, the
        time-averaged sum of squared effects across all cortical targets,
        rendered on the LH inflated surface with a sequential colormap.

Inputs
    codes/HCP/results/ECts/id_100206_ECt.npy                (500, 450, 450)

Output
    codes/figures/outputs/Figure1_PanelB.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import (setup, save_panel, figsize_mm,
                   DIV_CMAP, SEQ_CMAP, SEQ_CMAP_NAME, YEO7_COLORS)
from brain_render import (
    N_SUBCORTICAL,
    cortical_network_only_ordering,
    render_hemi_lateral_array,
    schaefer_parcels_to_vertices,
)
setup()

ROOT = Path(__file__).resolve().parents[2]
EC_NPY = ROOT / "codes/HCP/results/ECts/id_100206_ECt.npy"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def asymmetric_diverging_cmap(neg_frac: float,
                              sample_range: tuple[float, float] = (0.20, 0.85),
                              base=DIV_CMAP,
                              n: int = 512,
                              red_gamma: float = 0.5,
                              blue_gamma: float = 1.0,
                              center_white: float = 0.55,
                              white_sigma_frac: float = 0.07,
                              ) -> LinearSegmentedColormap:
    """Diverging colormap with white at fractional position `neg_frac` from
    the bottom (not at 0.5), so that with a regular `Normalize(vmin, vmax)`
    the colorbar's blue:red proportions match |vmin|:vmax.

    `sample_range` controls how much of the base colormap's extremes to use
    -- (0.20, 0.85) skips the darkest 20% / 15% to give softer extremes.

    Two shaping controls tune the positive (red) half and the zero-center:

    `red_gamma` (< 1) bends the red ramp so saturated reds arrive *sooner*
    above zero -- the base map only turns red past ~0.65, so a linear ramp
    leaves a long washed-out strip just above zero; gamma=0.5 pulls the reds
    down toward the center.

    `center_white` blends the colors around the zero position toward white
    (peak weight at zero, Gaussian falloff of width `white_sigma_frac * n`),
    so the center reads as light/near-white rather than the base map's cool
    gray (#c5d5dd). The blue half and the red extremes are left untouched.
    """
    base_cmap = mpl.colormaps[base] if isinstance(base, str) else base
    s_lo, s_hi = sample_range
    n_neg = max(2, int(round(n * neg_frac)))
    n_pos = max(2, n - n_neg)
    # Blue half: gamma ramp from the center (0.5) out to s_lo. `blue_gamma` < 1
    # pulls saturated blue closer to zero (u = 0 at center, 1 at vmin).
    u_blue = np.linspace(1.0, 0.0, n_neg) ** blue_gamma
    blue_part = base_cmap(0.5 + (s_lo - 0.5) * u_blue)
    # Red half: gamma ramp from the center (0.5) out to s_hi so reds appear
    # close to zero instead of only near vmax.
    t_red = np.linspace(0.0, 1.0, n_pos) ** red_gamma
    red_part = base_cmap(0.5 + (s_hi - 0.5) * t_red)
    colors = np.vstack([blue_part, red_part])

    # Lighten the zero-center toward white with a narrow Gaussian weight.
    if center_white > 0.0:
        idx = np.arange(colors.shape[0])
        sigma = max(1.0, white_sigma_frac * n)
        w = center_white * np.exp(-0.5 * ((idx - n_neg) / sigma) ** 2)
        white = np.ones(colors.shape[1]); white[3:] = colors[:, 3:].mean()
        colors = colors * (1.0 - w[:, None]) + white[None, :] * w[:, None]
    return LinearSegmentedColormap.from_list("asym_div", colors, N=n)


def crop_transparent(arr: np.ndarray, pad: int = 4) -> np.ndarray:
    if arr.shape[-1] != 4:
        return arr
    alpha = arr[..., 3] > 0
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0 = max(0, r0 - pad); r1 = min(arr.shape[0], r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(arr.shape[1], c1 + pad)
    return arr[r0:r1, c0:c1]


def compute_ec_mean_and_responsiveness(ec_path: Path, chunk: int = 50):
    """Time-averaged EC[j, i] (cortical x cortical) and per-source
    responsiveness = mean_t sum_i EC_t[t, j, i]^2 (cortical i only)."""
    ec = np.load(ec_path, mmap_mode="r")
    T = ec.shape[0]
    N = 400
    ec_sum = np.zeros((N, N), dtype=np.float64)
    sq_sum = np.zeros(N, dtype=np.float64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        sl = np.asarray(ec[t0:t1, N_SUBCORTICAL:, N_SUBCORTICAL:],
                        dtype=np.float64)         # (c, 400, 400)
        ec_sum += sl.sum(axis=0)
        sq_sum += (sl ** 2).sum(axis=2).sum(axis=0)
    ec_mean = ec_sum / T                          # (400, 400)
    responsiveness = sq_sum / T                   # (400,)
    return ec_mean, responsiveness


EC_CACHE = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"


def build_panel(fig):
    """Draw Panel B (Where) into the given figure or SubFigure."""
    if EC_NPY.exists():
        ec_mean, responsiveness = compute_ec_mean_and_responsiveness(EC_NPY)
    else:  # reduction cache shipped with the figure bundle (same values)
        d = np.load(EC_CACHE)
        ec_mean, responsiveness = d["ec_mean"], d["responsiveness"]

    # Reorder by network only (LH + RH parcels interleaved within each network)
    perm, boundaries, net_labels = cortical_network_only_ordering(LABEL_TXT)
    ec_mean = ec_mean[np.ix_(perm, perm)]

    # ---- Figure layout ----
    gs_top = gridspec.GridSpec(
        1, 1,
        top=0.96, bottom=0.50, left=0.14, right=0.94,
    )
    ax_mat = fig.add_subplot(gs_top[0])

    # Brain row aligned to the same horizontal extent as the matrix above.
    gs_bot = gridspec.GridSpec(
        1, 1,
        top=0.43, bottom=0.04, left=0.14, right=0.94,
    )
    ax_brain = fig.add_subplot(gs_bot[0])

    # --- EC matrix ---
    # Light spatial smoothing softens the salt-and-pepper noise between
    # off-diagonal entries while keeping the network blocks identifiable.
    ec_show = gaussian_filter(ec_mean, sigma=0.8, mode="nearest")

    # Asymmetric vmin/vmax with a custom colormap that puts white at the
    # data-zero position (1/6 from the bottom). Using a regular Normalize
    # (not TwoSlopeNorm) so the colorbar visually reflects the asymmetric
    # ranges -- a short blue strip and a long red gradient. The cmap is
    # sampled away from the darkest red/blue extremes so saturation is soft.
    n = ec_mean.shape[0]
    off = ec_mean[~np.eye(n, dtype=bool)]
    pos_vmax = float(np.percentile(np.abs(off), 99)) * 1.2  # tighter headroom
    neg_vmin = -pos_vmax / 5.0                              # 1:5 blue:red
    neg_frac = abs(neg_vmin) / (pos_vmax - neg_vmin)
    asym_cmap = asymmetric_diverging_cmap(neg_frac=neg_frac,
                                          sample_range=(0.16, 0.90),
                                          red_gamma=0.62, blue_gamma=0.7,
                                          center_white=0.55)
    norm = Normalize(vmin=neg_vmin, vmax=pos_vmax)
    im = ax_mat.imshow(
        ec_show, cmap=asym_cmap, norm=norm,
        aspect="equal", origin="upper",
        interpolation="bilinear",
    )
    ax_mat.set_xlabel("ROIs")
    ax_mat.set_ylabel("ROIs")
    ax_mat.set_xticks([]); ax_mat.set_yticks([])
    ax_mat.set_title("Average effective connectivity", pad=8)

    # Dark-gray dividers between Yeo-7 networks (no LH/RH split)
    divider_kw = dict(color="#3a3a3a", linewidth=0.5, alpha=0.8)
    for b in boundaries[:-1]:
        ax_mat.axvline(b - 0.5, **divider_kw)
        ax_mat.axhline(b - 0.5, **divider_kw)

    # Colored Yeo-7 network bands along the top and left edges
    from matplotlib.patches import Rectangle
    starts = [0] + list(boundaries[:-1])
    band = ec_show.shape[0] * 0.02
    for s0, e0, lab in zip(starts, boundaries, net_labels):
        c = YEO7_COLORS[lab]
        ax_mat.add_patch(Rectangle((s0 - 0.5, -0.5 - band), e0 - s0, band,
                                   facecolor=c, edgecolor="none",
                                   clip_on=False, zorder=5))
        ax_mat.add_patch(Rectangle((-0.5 - band, s0 - 0.5), band, e0 - s0,
                                   facecolor=c, edgecolor="none",
                                   clip_on=False, zorder=5))
    # All four spines in black (the global style hides top/right)
    for sp in ax_mat.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)

    # Colorbar for the matrix
    mat_pos = ax_mat.get_position()
    cax = fig.add_axes([0.955, mat_pos.y0, 0.012, mat_pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\overline{\mathrm{EC}}_{ij}$")
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.8, length=3)

    # --- Responsiveness brain ---
    rvmin = 0.0
    rvmax = float(np.percentile(responsiveness, 99))
    vmaps = schaefer_parcels_to_vertices(responsiveness)
    arr = render_hemi_lateral_array(
        "lh", vmaps["lh"], cmap=SEQ_CMAP_NAME,
        color_range=(rvmin, rvmax),
    )
    ax_brain.imshow(crop_transparent(arr))
    ax_brain.set_axis_off()
    ax_brain.set_title("Responsiveness", pad=10,
                       fontsize=plt.rcParams["axes.titlesize"])

    # Brain colorbar aligned at the same x as the matrix colorbar
    b_pos = ax_brain.get_position()
    bcax = fig.add_axes([0.955, b_pos.y0 + 0.18 * b_pos.height,
                         0.012, 0.50 * b_pos.height])
    sm = plt.cm.ScalarMappable(
        cmap=SEQ_CMAP,
        norm=plt.Normalize(vmin=rvmin, vmax=rvmax),
    )
    bcb = fig.colorbar(sm, cax=bcax)
    bcb.set_label(r"$\sum_i\,\overline{\mathrm{EC}}^{\,2}_{ji}$")
    bcb.outline.set_visible(False)
    bcb.ax.tick_params(width=0.8, length=3)


def main():
    fig = plt.figure(figsize=figsize_mm(85, 156))
    build_panel(fig)
    save_panel(fig, OUT_DIR / "Figure1_PanelB")


if __name__ == "__main__":
    main()
