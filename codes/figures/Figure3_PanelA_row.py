"""
Figure 3 Panel A -- effect size vs relative variability across the cortical hierarchy.

A 2x2 block contrasting the global effect SIZE with its trial-to-trial relative
variability (coefficient of variation, CV = std/mean), both organized by Yeo-7
resting-state network:

    row 1 (boxplots, dot = region, by network, unimodal -> transmodal)
        left : mean global effect size  <Sigma^(j)>      (from the HCP_5 df)
        right: CV of the effect          CV(GES)          (from BECts_reduced)
    row 2 (per-subject Spearman rho between the 7 network means and their
           hierarchy rank; >0 = increases along the hierarchy)
        left : effect size       right: CV(GES)

The effect size climbs the hierarchy reproducibly (rho ~ +0.8), but the relative
variability is essentially FLAT (CV ~ 0.13 in every network, rho ~ 0): the absolute
variance Var(GES) scales as mean^2 and so carries no spatial information beyond the
mean, whereas the CV divides that scaling out. GES = BEC^2, so mean(GES) = msq_all
(mean of squares) and std(GES) = ssq_all; CV = ssq_all / msq_all.

Inputs
    codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl
    codes/HCP/results/BECts_reduced/{sid}.npz   (CV(GES) = diag(ssq_all)/diag(msq_all))
Output
    codes/figures/outputs/Figure3_PanelA_row.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
from Figure3_PanelA_HCP import (draw_panel, per_subject_rhos,
                                VALUE_COL, GLOBAL_EFFECT_YLABEL, ROOT)
setup()

HCP_DF_PKL = ROOT / "codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl"
BECTS = ROOT / "codes/HCP/results/BECts_reduced"
# Consolidated per-subject focal diagonals (diag msq_all / ssq_all), shipped
# with the figure bundle so this panel runs without the 927 MB BECts_reduced/.
BEC_CACHE = ROOT / "codes/HCP/results/BEC_focal_cache.npz"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50
CV_COL = "ges_cv"
CV_YLABEL = r"CV(GES) = std/mean"


def build_cv_df(ref_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(subject, cortical roi) focal CV(GES) = diag(ssq_all)/diag(msq_all), the
    trial-to-trial relative variability of the global effect size. GES = BEC^2, so
    over trials its mean is msq_all (mean of squares) and its std is ssq_all; their
    ratio is the CV (matching cv_bec in Figure5_BEC_matrices.npz). The roi->network
    map is taken from `ref_df` (constant across subjects)."""
    rois = np.arange(400)
    roi_net = ref_df.groupby("roi")["rsn_network"].first().reindex(rois).to_numpy()
    parts = []
    if BECTS.exists():
        per_subj = ((f.stem, np.diag(np.load(f)["ssq_all"].astype(float)),
                     np.diag(np.load(f)["msq_all"].astype(float)))
                    for f in sorted(BECTS.glob("id_*.npz")))
    else:
        d = np.load(BEC_CACHE)
        per_subj = zip(d["sids"], d["diag_ssq"], d["diag_msq"])
    for sid, ss_diag, ms_diag in per_subj:
        ss = ss_diag[SUB:]                                   # std of GES over trials
        ms = ms_diag[SUB:]                                   # mean of GES over trials
        cv = ss / ms                                         # CV(GES)
        parts.append(pd.DataFrame({"sub_id": str(sid), "roi": rois,
                                   "rsn_network": roi_net, CV_COL: cv}))
    return pd.concat(parts, ignore_index=True)


def _hist(ax, rhos, show_ylabel=True):
    ax.hist(rhos, bins=16, range=(-1, 1), color="#b9b9b9", edgecolor="white",
            linewidth=0.4)
    ax.axvline(0.0, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(r"Spearman $\rho$ (per subject)")
    if show_ylabel:
        ax.set_ylabel("Participants")
    ax.tick_params(width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    df = pd.read_pickle(HCP_DF_PKL)
    cv_df = build_cv_df(df)

    fig = plt.figure(figsize=figsize_mm(150, 82))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1.0], wspace=0.34,
                           hspace=0.62, left=0.10, right=0.97,
                           top=0.93, bottom=0.12)
    axES, axCV = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axESh, axCVh = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # Row 1 -- boxplots by network.
    st_es = draw_panel(axES, df, VALUE_COL, GLOBAL_EFFECT_YLABEL,
                       ylim=(2.5e-4, 1.2e-3))
    axES.set_title("Effect size", pad=4)
    # CV sits in a very narrow band (~0.13 +/- 0.005 across all regions); zoom in
    # to show the per-region spread. NB the absolute scale is tiny -- the flatness
    # is the point (cf. the per-subject gradient histogram below, rho ~ 0).
    draw_panel(axCV, cv_df, CV_COL, CV_YLABEL, ylim=(0.127, 0.142))
    axCV.set_title("Relative variability (CV)", pad=4)

    # Row 2 -- per-subject hierarchy-rho histograms.
    rho_es = per_subject_rhos(df, VALUE_COL)
    rho_cv = per_subject_rhos(cv_df, CV_COL)
    _hist(axESh, rho_es, show_ylabel=True)
    _hist(axCVh, rho_cv, show_ylabel=True)

    print(f"effect size : hierarchy rho median={np.median(rho_es):+.2f}  "
          f"{(rho_es>0).mean()*100:.0f}% positive")
    print(f"CV          : hierarchy rho median={np.median(rho_cv):+.2f}  "
          f"{(rho_cv>0).mean()*100:.0f}% positive")
    save_panel(fig, OUT_DIR / "Figure3_PanelA_row")


if __name__ == "__main__":
    main()
