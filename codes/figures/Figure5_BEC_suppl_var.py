"""
Supplementary bifocal figure (var-based), replacing the outdated CV version.

Cohort-level summary of bifocal stimulation over all cortical region pairs, on the
ABSOLUTE scale (var(GES)); the relative-variability (CV) results and the Yeo-7
network-pair breakdowns are in the main bifocal figure (Figure 6).

Layout (2 rows x 2 cols), one row per measure (mean effect / variability):
  A mean(GES) bifocal matrix | B % change vs stronger single
  C var(GES) bifocal matrix  | D % change vs less-variable single

The network-pair matrices (formerly C, F) now live in Figure 6F,G; the
network_fraction / cortical_network_index helpers are kept for that script
(Figure5_BEC_netpair_panels.py).

Source: subject-averaged Figure5_BEC_matrices.npz (mean_bec, var_bec, eff_incr,
impr_var). Output: paper .../PDF/SF10.pdf  and  outputs/Figure5_BEC_suppl_var.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, figsize_mm, INK
from brain_render import YEO7_NETWORKS   # same network order as Figure 3
setup()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
DF = ROOT / "codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl"
PDFDIR = ROOT / "paper/ANN_fMRI_HCP/Figures/Inkscape/PDF"
OUT = ROOT / "codes/figures/outputs"
if not PDFDIR.parent.exists():          # figure bundle: no paper/ tree
    PDFDIR = OUT
SUB = 50
BLUE = "#1f77b4"
PRETTY = {"Vis": "VIS", "SomMot": "SMN", "DorsAttn": "DAN", "SalVentAttn": "SN",
          "Cont": "FPN", "Default": "DMN", "Limbic": "LN"}
LABELS = [PRETTY[n] for n in YEO7_NETWORKS]


def cortical_network_index():
    """Yeo-7 network index per cortical parcel (0..399), in YEO7_NETWORKS order,
    taken from the same dataframe Figure 3 uses."""
    d = pd.read_pickle(DF)
    per = d.drop_duplicates("roi").sort_values("roi")["rsn_network"].to_numpy()
    return np.array([YEO7_NETWORKS.index(n) for n in per])


def network_fraction(M, net_idx, positive):
    """For each Yeo-7 network pair, percentage of region pairs with M>0 (positive=True)
    or M<0, over the upper triangle of the cortical matrix."""
    K = len(YEO7_NETWORKS)
    F = np.full((K, K), np.nan)
    iu = np.triu_indices(M.shape[0], k=1)
    ai, bi = iu
    na, nb, vals = net_idx[ai], net_idx[bi], M[ai, bi]
    for i in range(K):
        for j in range(K):
            m = (((na == i) & (nb == j)) | ((na == j) & (nb == i))) & np.isfinite(vals)
            if m.sum():
                F[i, j] = (np.mean(vals[m] > 0) if positive else np.mean(vals[m] < 0)) * 100
    return F


def _hist(ax, vals, xlabel, good_positive):
    vals = vals[np.isfinite(vals)]
    mean = vals.mean()
    frac = ((vals > 0) if good_positive else (vals < 0)).mean() * 100
    sign = ">" if good_positive else "<"
    txt = (">0 (bifocal\nlarger effect)" if good_positive else "<0 (bifocal\nless variable)")
    ax.hist(vals, bins=60, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.axvline(0.0, color="black", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.axvline(mean, color=INK, linewidth=1.4, label=f"mean = {mean:+.1f}%")
    ax.set_yticks([]); ax.set_xlabel(xlabel)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.annotate(f"{frac:.0f}% {sign} 0\n{txt}", xy=(0.03, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=6.5, color="#444444")


def _matrix(ax, M, title, cbar_label):
    lo, hi = np.nanpercentile(M, [2, 98])
    im = ax.imshow(M, cmap="magma", vmin=lo, vmax=hi, aspect="equal")
    ax.set_xlabel("ROIs"); ax.set_ylabel("ROIs"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=6.5); cb.ax.tick_params(labelsize=6)


def _netmat(ax, F, title, cbar_label):
    im = ax.imshow(F, cmap="cividis", vmin=0, vmax=np.nanmax(F), aspect="equal")
    K = len(LABELS)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(LABELS, fontsize=6, rotation=90); ax.set_yticklabels(LABELS, fontsize=6)
    thr = np.nanmax(F) * 0.55
    for i in range(K):
        for j in range(K):
            if np.isfinite(F[i, j]):
                ax.text(j, i, f"{F[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if F[i, j] < thr else "black")
    ax.set_title(title, fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=6.5); cb.ax.tick_params(labelsize=6)


def main():
    d = np.load(NPZ); n = int(d["n_subjects"])
    mean_bec = d["mean_bec"].astype(float)[SUB:, SUB:]
    var_bec = d["var_bec"].astype(float)[SUB:, SUB:]
    eff = d["eff_incr"].astype(float)[SUB:, SUB:]
    imv = d["impr_var"].astype(float)[SUB:, SUB:]
    iu = np.triu_indices_from(eff, k=1)

    # 2x2: matrices + distributions only. The Yeo-7 network-pair breakdowns moved
    # to the main bifocal figure (Figure 6F,G).
    fig, ax = plt.subplots(2, 2, figsize=figsize_mm(120, 115), constrained_layout=True)
    _matrix(ax[0, 0], mean_bec, "Bifocal mean(GES)", "mean(GES)")
    _hist(ax[0, 1], eff[iu], r"% change in mean(GES)" + "\n" + r"(bifocal $-$ stronger single)", True)
    ax[0, 1].set_title(f"All cortical pairs (n={eff[iu].size:,})", fontsize=8)
    _matrix(ax[1, 0], var_bec, "Bifocal var(GES)", "var(GES)")
    _hist(ax[1, 1], imv[iu], r"% change in var(GES)" + "\n" + r"(bifocal $-$ less-variable single)", False)

    for a, lab in zip(ax.ravel(), "ABCD"):
        a.text(-0.22, 1.05, lab, transform=a.transAxes, fontsize=11, fontweight="bold")
    fig.suptitle(f"Bifocal stimulation, cohort summary (HCP, N={n})", fontsize=9)

    fig.savefig(PDFDIR / "SF10.pdf", dpi=600)
    fig.savefig(OUT / "Figure5_BEC_suppl_var.png", dpi=200)
    plt.close(fig)
    print(f"  eff: {(eff[iu] > 0).mean()*100:.0f}% larger | var(abs): {(imv[iu] < 0).mean()*100:.0f}% less variable")
    print(f"  -> {PDFDIR/'SF10.pdf'}")


if __name__ == "__main__":
    main()
