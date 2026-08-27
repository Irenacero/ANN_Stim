"""
Figure 3 -- Per-subject receptor-correlation bar panel (effect size vs variability).

Per-subject version of Figure3_PanelC_receptor_bars: for each participant we
correlate (Spearman, across the 400 cortical regions) each PET receptor map with
that subject's own focal mean(GES) map and focal CV(GES) map, then plot the
MEAN correlation across the 100 participants with a 95% CI error bar.

    left  : receptor density vs mean(GES)  -- response magnitude
    right : receptor density vs CV(GES)    -- relative variability (std/mean)

Bars share one color; bars are faded if the across-subject correlation is not
significant after FDR (Wilcoxon signed-rank vs 0, BH across the 20 maps).

NOTE: this is a *reproducibility* test (is the correlation consistent across
subjects), not a spatial-autocorrelation-controlled one. It is anti-conservative
relative to the spin test in Figure3_PanelC_receptor_bars; use that panel for the
spatial-null significance.

Inputs
    codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl
    codes/HCP/results/BECts_reduced/{sid}.npz
    codes/HCP/results/Figure5_variability_receptor_correlations.csv  (pretty names)
Output
    codes/figures/outputs/Figure3_PanelC_receptor_bars_persubject.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, ACCENT_COOL
setup()

ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl"
BECTS = ROOT / "codes/HCP/results/BECts_reduced"
NAMES_CSV = ROOT / "codes/HCP/results/Figure5_variability_receptor_correlations.csv"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50
BAR = ACCENT_COOL
FADE = 0.4


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def per_subject_table(prop_mat, receptor_maps):
    """prop_mat: (n_sub, 400). receptor_maps: dict short->(400,).
    Returns DataFrame: receptor, mean_rho, ci95, p, q (Wilcoxon vs 0, FDR)."""
    M = prop_mat
    rows = []
    for short, x in receptor_maps.items():
        rhos = np.array([spearmanr(x, M[i]).correlation for i in range(M.shape[0])])
        rhos = rhos[np.isfinite(rhos)]
        m = rhos.mean()
        ci = 1.96 * rhos.std(ddof=1) / np.sqrt(rhos.size)
        p = wilcoxon(rhos).pvalue
        rows.append((short, m, ci, p))
    t = pd.DataFrame(rows, columns=["receptor", "mean_rho", "ci95", "p"])
    t["q"] = bh(t.p.to_numpy())
    return t


def _bars(ax, names, mean, ci, q, title):
    x = np.arange(len(names))
    alphas = [1.0 if qi < 0.05 else FADE for qi in q]
    ax.bar(x, mean, width=0.78, color=BAR, edgecolor="#2f3b52", linewidth=0.4,
           zorder=3)
    for bar, a in zip(ax.patches, alphas):
        bar.set_alpha(a)
    ax.errorbar(x, mean, yerr=ci, fmt="none", ecolor="#1a1a1a", elinewidth=0.7,
                capsize=1.6, capthick=0.7, zorder=5)
    ax.axhline(0, color="#555555", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.4)
    ax.set_ylabel(r"mean Spearman $\rho$ (per subject)", fontsize=7)
    ax.set_title(title, pad=4, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    df = pd.read_pickle(DF)
    # raw_VAChT is a single PET source of the normalised VAChT map ('Vacht'); drop
    # it so this panel shows the same 19 maps as the group/Moran panel.
    recs = [c for c in df.columns if c.startswith("receptor")
            and c != "receptor_raw_VAChT_feobv_hc3_spreng"]
    pretty = pd.read_csv(NAMES_CSV).set_index("receptor")["name"].to_dict()
    R = {c.replace("receptor_", ""): df.groupby("roi")[c].mean().to_numpy()
         for c in recs}

    # per-subject effect-size map (n_sub x 400), roi-ordered
    resp = (df.pivot_table(index="sub_id", columns="roi",
                           values="mean_global_effect_size").sort_index())
    sids = list(resp.index)
    resp_mat = resp.to_numpy()

    # per-subject CV(GES) map from BECts_reduced, aligned to the same subjects.
    # GES = BEC^2 -> mean over trials = msq_all, std = ssq_all; CV = ssq_all/msq_all.
    cv = {}
    if BECTS.exists():
        for f in sorted(BECTS.glob("id_*.npz")):
            d = np.load(f)
            cv[f.stem] = np.diag(d["ssq_all"])[SUB:] / np.diag(d["msq_all"])[SUB:]
    else:  # consolidated focal-diagonal cache shipped with the figure bundle
        d = np.load(ROOT / "codes/HCP/results/BEC_focal_cache.npz")
        for sid, ss, ms in zip(d["sids"], d["diag_ssq"], d["diag_msq"]):
            cv[str(sid)] = ss[SUB:] / ms[SUB:]
    cv_mat = np.vstack([cv[s] for s in sids])

    te = per_subject_table(resp_mat, R)
    tc = per_subject_table(cv_mat, R)
    order = te.sort_values("mean_rho", ascending=False).receptor.tolist()
    te = te.set_index("receptor").loc[order]
    tc = tc.set_index("receptor").loc[order]
    names = [pretty.get(r, r) for r in order]

    fig, (axE, axC) = plt.subplots(1, 2, figsize=figsize_mm(150, 52),
                                   constrained_layout=True)
    _bars(axE, names, te.mean_rho.to_numpy(), te.ci95.to_numpy(),
          te["q"].to_numpy(), "vs effect size  mean(GES)")
    _bars(axC, names, tc.mean_rho.to_numpy(), tc.ci95.to_numpy(),
          tc["q"].to_numpy(), "vs relative variability  CV(GES)")
    # Shared y-axis so the two columns are honestly comparable: per subject the
    # CV-receptor correlations are an order of magnitude smaller than the mean-GES
    # ones (the CV signal is a group-level spatial property, not a within-subject one).
    lim = 1.1 * np.nanmax(np.abs(np.r_[te.mean_rho + te.ci95, te.mean_rho - te.ci95,
                                       tc.mean_rho + tc.ci95, tc.mean_rho - tc.ci95]))
    for ax in (axE, axC):
        ax.set_ylim(-lim, lim)
    axE.annotate("faded = not FDR-significant\n(Wilcoxon across subjects)",
                 xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                 fontsize=5.4, color="#555555")

    print(f"mean(GES): {(te['q']<0.05).sum()}/{len(te)} FDR-significant  "
          f"(range mean_rho {te.mean_rho.min():.2f}..{te.mean_rho.max():.2f})")
    print(f"CV(GES)  : {(tc['q']<0.05).sum()}/{len(tc)} FDR-significant  "
          f"(range mean_rho {tc.mean_rho.min():.3f}..{tc.mean_rho.max():.3f})")
    save_panel(fig, OUT_DIR / "Figure3_PanelC_receptor_bars_persubject")


if __name__ == "__main__":
    main()
