"""
Figure 3 Panel A (HCP) -- Global effect size across Yeo-7 networks.

For each cortical ROI, the global effect size <Sigma^(j)>_t (already time-averaged
over the EC timepoints, summed over all 450 targets; column
'mean_global_effect_size') is averaged across subjects, giving one value per
region. Regions are grouped by Yeo-7 network and ordered along the cortical
hierarchy (unimodal -> transmodal, Limbic last). Each dot is one region, colored
by its network; a box summarizes the per-network distribution. The inset shows
the distribution of per-subject network-level Spearman rho (gradient strength).

`draw_panel` draws onto a given Axes and is reused by Figure3_PanelA_TMS.py and
SupplementaryFigure1.py (with other measures / the TMS dataframe).

Input
    codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl

Output
    codes/figures/outputs/Figure3_PanelA_HCP.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, YEO7_COLORS
from brain_render import YEO7_NETWORKS  # hierarchy order (Limbic last)
setup()

ROOT = Path(__file__).resolve().parents[2]
HCP_DF_PKL = ROOT / "codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALUE_COL = "mean_global_effect_size"
GLOBAL_EFFECT_YLABEL = r"Mean GES"

# Network colors come from the shared muted palette (style.YEO7_COLORS), so the
# dots here match the Yeo-7 bands on the EC matrix (Panel B / subpanels).
# Standard resting-state-network acronyms (SN = salience / ventral attention,
# FPN = frontoparietal control, LN = limbic).
PRETTY = {
    "Vis": "VIS", "SomMot": "SMN", "DorsAttn": "DAN",
    "SalVentAttn": "SN", "Cont": "FPN", "Default": "DMN",
    "Limbic": "LN",
}

JITTER = 0.16


def darken(color: str, f: float = 0.55) -> tuple[float, float, float]:
    """Return a darker shade of `color` for soft marker edges."""
    r, g, b = mcolors.to_rgb(color)
    return (r * f, g * f, b * f)


def per_subject_rhos(df, value_col):
    """Per-subject network-level gradient strength.

    For each subject, take the mean of `value_col` within each of the 7
    networks, then the Spearman rho between those 7 values and their
    hierarchy rank (unimodal -> transmodal, 1..7). Returns the rho array.
    """
    rhos = []
    for _, g in df.groupby("sub_id"):
        net_mean = g.groupby("rsn_network")[value_col].mean()
        y = np.array([net_mean[net] for net in YEO7_NETWORKS])
        rho, _ = spearmanr(np.arange(len(YEO7_NETWORKS)), y)
        rhos.append(rho)
    return np.asarray(rhos)


def draw_panel(ax, df, value_col, ylabel, inset_rect=None, ylim=None,
               rng=None, show_xlabel=True, sci_y=True):
    """Draw the per-region-by-network panel for `value_col` onto `ax`.

    If `inset_rect` is given, an inset histogram of the per-subject Spearman
    rho is drawn; pass None (default) for a clean panel with no inset.
    Returns a dict with the per-network medians and per-subject Spearman stats.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # One value per region: mean of `value_col` across subjects.
    per_roi = (df.groupby("roi")
                 .agg(value=(value_col, "mean"),
                      net=("rsn_network", "first"))
                 .reset_index())

    box_data = [per_roi.loc[per_roi.net == net, "value"].to_numpy()
                for net in YEO7_NETWORKS]
    positions = np.arange(1, len(YEO7_NETWORKS) + 1)

    # Light neutral boxplots underneath (median + IQR per network).
    bp = ax.boxplot(
        box_data, positions=positions, widths=0.62,
        showfliers=False, patch_artist=True, zorder=2,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor="#f2f2f2", edgecolor="#9a9a9a", linewidth=1.0)
    for elem in ("whiskers", "caps"):
        for ln in bp[elem]:
            ln.set(color="#9a9a9a", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#333333", linewidth=1.8)

    # Subtle dashed trend through the network medians to read the gradient.
    medians = [np.median(d) for d in box_data]
    ax.plot(positions, medians, color="#555555", linewidth=1.2,
            linestyle=(0, (4, 3)), alpha=0.7, zorder=3)

    # Jittered colored dots: one per region. Soft (low alpha, faint darker edge).
    for pos, net in zip(positions, YEO7_NETWORKS):
        vals = box_data[YEO7_NETWORKS.index(net)]
        x = pos + rng.uniform(-JITTER, JITTER, size=vals.size)
        ax.scatter(x, vals, s=18, color=YEO7_COLORS[net],
                   edgecolor=darken(YEO7_COLORS[net]), linewidth=0.3,
                   alpha=0.6, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[n] for n in YEO7_NETWORKS],
                       rotation=30, ha="right")
    ax.set_xlim(0.4, len(YEO7_NETWORKS) + 0.6)
    if show_xlabel:
        ax.set_xlabel("Resting-state network")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if sci_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(width=0.8, length=3, direction="out")

    # --- Per-subject gradient statistic (network level) -----------------
    rhos = per_subject_rhos(df, value_col)
    med = float(np.median(rhos))
    frac_pos = float((rhos > 0).mean())
    _, pval = wilcoxon(rhos)  # H0: median rho == 0

    # Optional inset (top-left) with the distribution of per-subject rho.
    if inset_rect is not None:
        axin = ax.inset_axes(list(inset_rect))
        axin.hist(rhos, bins=16, color="#b9b9b9", edgecolor="white", linewidth=0.4)
        axin.axvline(0.0, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
        axin.set_yticks([])
        axin.set_xlabel(r"Spearman $\rho$ (per subject)", labelpad=3)
        axin.tick_params(axis="x", width=0.7, length=2.5)
        for s in ("top", "right", "left"):
            axin.spines[s].set_visible(False)

    return {"n_regions": len(per_roi), "n_subjects": df.sub_id.nunique(),
            "medians": medians, "rho_median": med, "rhos": rhos,
            "frac_pos": frac_pos, "pval": pval}


def make_panel(df_pkl: Path, out_stem: str, inset_rect=None):
    """Build the standalone global-effect-size panel from a HCP_5/TMS_5 df.

    No inset by default; wide-and-short so two of these (HCP + TMS) tile as the
    first two columns of a Figure-3 row, with the Spearman panel as the third.
    """
    df = pd.read_pickle(df_pkl)
    fig, ax = plt.subplots(figsize=figsize_mm(70, 42))
    st = draw_panel(ax, df, VALUE_COL, GLOBAL_EFFECT_YLABEL, inset_rect,
                    ylim=(2.5e-4, 1.2e-3))
    print(f"[{out_stem}] regions={st['n_regions']}  subjects={st['n_subjects']}")
    print(f"  per-subject network-level Spearman rho: median={st['rho_median']:.3f}, "
          f"{st['frac_pos']*100:.0f}% positive, Wilcoxon p={st['pval']:.2e}")
    save_panel(fig, OUT_DIR / out_stem)


def main():
    make_panel(HCP_DF_PKL, "Figure3_PanelA_HCP")


if __name__ == "__main__":
    main()
