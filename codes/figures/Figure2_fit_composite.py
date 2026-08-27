"""
Figure 2 -- composite fit figure (single A4 sheet, 2 rows x 4 columns).

Assembles the eight personalized-fit panels into one figure:

  Row 1 (FC) :   FC split matrix | emp-vs-sim FC value distribution
                 | per-participant FC correlation | per-participant FC KS distance
  Row 2 (dFC):   dFC split matrix | emp-vs-sim dFC value distribution
                 | per-participant dFC variance scatter | per-participant dFC KS distance

The first two columns describe the best-fit example participant; the last two
columns summarise the whole cohort. All numbers come from the cache written by
``Figure2_fit_group.py`` (re-run that first if the cache predates the KS arrays).

Output
    codes/figures/outputs/Figure2_fit_composite.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, sequential_colors, INK
from brain_render import cortical_network_only_ordering, N_SUBCORTICAL

ROOT = Path(__file__).resolve().parents[2]
HCP_DIR = ROOT / "codes/HCP"
OUT_DIR = ROOT / "codes/figures/outputs"
CACHE = HCP_DIR / "results/Figure2_fit_group.npz"
LABEL_TXT = HCP_DIR / "data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"

N = 450
MODEL = "#1f77b4"           # tab:blue -- simulated / model fit
EMP = "#4d4d4d"             # dark gray -- empirical

# Simulated-BOLD trace panel (matches Figure2_best_traces).
TR = 0.72
N_TRACES = 40
TRACE_OFFSET = 1.5
TRACES_SECONDS = 360.0

setup()
# A4-sheet typography: 9 pt axis labels, 10 pt panel titles.
plt.rcParams.update({
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def combine(lower, upper, diag=1.0):
    """One matrix: strict lower triangle from `lower`, strict upper from `upper`."""
    n = lower.shape[0]
    M = np.empty((n, n), dtype=float)
    M[np.tril_indices(n, -1)] = lower[np.tril_indices(n, -1)]
    M[np.triu_indices(n, 1)] = upper[np.triu_indices(n, 1)]
    np.fill_diagonal(M, diag)
    return M


def draw_split(fig, ax, mat, *, cmap, title, kind, rval, axis_label):
    """Square split matrix (lower = simulated, upper = empirical) with the
    Image-2 title convention: 'FC'/'dFC' on top, halves labelled inside, and the
    empirical-vs-simulated similarity r reported in the corner."""
    n = mat.shape[0]
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper",
                   interpolation="nearest")
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="white", lw=1.0)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(axis_label); ax.set_ylabel(axis_label)
    ax.set_title(title, pad=4)
    ax.text(0.72 * n, 0.13 * n, f"Empirical\n{kind}", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(0.28 * n, 0.87 * n, f"Simulated\n{kind}", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(0.97, 0.04, rf"$r = {rval:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(kind); cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.7, length=2.5)


def draw_traces(ax, traces, *, title):
    """Stacked z-scored simulated cortical BOLD traces (Figure2_best_traces)."""
    raster = traces[:, N_SUBCORTICAL:].astype(float)
    perm, _, _ = cortical_network_only_ordering(LABEL_TXT)
    sig_z = ((raster - raster.mean(0)) / raster.std(0))[:, perm]
    idx = np.linspace(0, raster.shape[1] - 1, N_TRACES, dtype=int)
    colors = sequential_colors(N_TRACES, cmap=plt.cm.Greys, lo=0.35, hi=0.9)
    t = np.arange(raster.shape[0]) * TR
    for k, roi in enumerate(idx):
        ax.plot(t, sig_z[:, roi] + k * TRACE_OFFSET, color=colors[k],
                lw=0.6, alpha=0.8)
    ax.set_xlim(0, TRACES_SECONDS)
    ax.set_ylim(-3, (N_TRACES - 1) * TRACE_OFFSET + 3)
    ax.set_xticks(list(range(0, int(TRACES_SECONDS) + 1, 60))); ax.set_yticks([])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ROIs"); ax.set_title(title, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.8, length=3, direction="out")


def draw_value_dist(ax, emp_vals, sim_vals, *, xlabel, ks, title):
    """Overlaid empirical (gray) vs simulated (blue) value distributions."""
    e = emp_vals[np.isfinite(emp_vals)]
    s = sim_vals[np.isfinite(sim_vals)]
    bins = np.linspace(min(e.min(), s.min()), max(e.max(), s.max()), 41)
    ax.hist(e, bins=bins, density=True, histtype="stepfilled", color=EMP,
            alpha=0.4, edgecolor=EMP, linewidth=1.0, label="Empirical")
    ax.hist(s, bins=bins, density=True, histtype="stepfilled", color=MODEL,
            alpha=0.4, edgecolor=MODEL, linewidth=1.0, label="Simulated")
    ax.annotate(rf"KS $= {ks:.2f}$", xy=(0.96, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.set_title(title, pad=4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0,
              borderpad=0.2, labelspacing=0.3)
    ax.tick_params(width=0.8, length=3, direction="out")


def draw_corr_hist(ax, fc_corr, *, title):
    r = fc_corr[np.isfinite(fc_corr)]
    med = np.median(r)
    ax.hist(r, bins=16, color=MODEL, edgecolor="white", linewidth=0.4)
    ax.axvline(med, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.annotate(rf"median $= {med:.2f}$", xy=(med, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", ha="left", va="top", fontsize=8)
    ax.set_xlabel(r"FC correlation  $r$"); ax.set_ylabel("Participants")
    ax.set_title(title, pad=4)
    ax.tick_params(width=0.8, length=3, direction="out")


def draw_dfc_scatter(ax, flu_emp, flu_sim, *, title):
    m = np.isfinite(flu_emp) & np.isfinite(flu_sim)
    x, y = flu_emp[m], flu_sim[m]
    q1, q3 = np.percentile(y, [25, 75])
    keep = y <= q3 + 1.5 * (q3 - q1)
    x, y = x[keep], y[keep]
    rho, _ = stats.spearmanr(x, y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.scatter(x, y, s=14, color=MODEL, edgecolor="#13496f",
               linewidths=0.3, alpha=0.75, zorder=3)
    ax.plot(xs, b * xs + a, color=INK, lw=1.2, zorder=4)
    ax.annotate(rf"Spearman $r = {rho:.2f}$", xy=(0.05, 0.95),
                xycoords="axes fraction", ha="left", va="top", fontsize=8)
    ax.set_xlabel("dFC variance (empirical)")
    ax.set_ylabel("dFC variance (simulated)")
    ax.set_title(title, pad=4)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.tick_params(width=0.8, length=3, direction="out")


def draw_ks_hist(ax, ks, *, xlabel, title):
    k = ks[np.isfinite(ks)]
    med = np.median(k)
    ax.hist(k, bins=16, color=MODEL, edgecolor="white", linewidth=0.4)
    ax.axvline(med, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.annotate(rf"median $= {med:.2f}$", xy=(med, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", ha="left", va="top", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("Participants"); ax.set_title(title, pad=4)
    ax.tick_params(width=0.8, length=3, direction="out")


def main():
    d = np.load(CACHE, allow_pickle=True)
    if "ks_fc" not in d.files:
        raise SystemExit("Cache lacks KS arrays -- run Figure2_fit_group.py first.")

    iuN = np.triu_indices(N, 1)
    fc_emp_v, fc_sim_v = d["best_FC_emp"][iuN], d["best_FC_sim"][iuN]
    r_fc = stats.pearsonr(fc_emp_v, fc_sim_v)[0]
    ks_fc_best = stats.ks_2samp(fc_emp_v, fc_sim_v).statistic

    nT = d["best_dfc_emp"].shape[0]
    iuT = np.triu_indices(nT, 1)
    dfc_emp_v, dfc_sim_v = d["best_dfc_emp"][iuT], d["best_dfc_sim"][iuT]
    mt = np.isfinite(dfc_emp_v) & np.isfinite(dfc_sim_v)
    r_dfc = stats.pearsonr(dfc_emp_v[mt], dfc_sim_v[mt])[0]
    ks_dfc_best = stats.ks_2samp(dfc_emp_v[np.isfinite(dfc_emp_v)],
                                 dfc_sim_v[np.isfinite(dfc_sim_v)]).statistic

    # 4 rows x 3 cols. Rows 1-2: FC matrix (centre) + dFC matrix (right), each
    # spanning two rows; the top-left 2x1 block is left empty (filled later).
    # Row 3: the three remaining FC panels. Row 4: the three remaining dFC panels.
    fig = plt.figure(figsize=figsize_mm(297, 210), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    # Top-left (gs[0, 0]) left empty -- filled in later. Simulated BOLD traces
    # sit just below it, in the second row of the first column.
    draw_traces(fig.add_subplot(gs[1, 0]), d["best_traces"], title="Simulated BOLD")

    ax_fc = fig.add_subplot(gs[0:2, 1])
    ax_dfc = fig.add_subplot(gs[0:2, 2])
    draw_split(fig, ax_fc, combine(d["best_FC_sim"], d["best_FC_emp"]),
               cmap="plasma", title="FC", kind="FC", rval=r_fc, axis_label="ROIs")
    draw_split(fig, ax_dfc, combine(d["best_dfc_sim"], d["best_dfc_emp"]),
               cmap="Spectral_r", title="dFC", kind="dFC", rval=r_dfc,
               axis_label="Time")

    # --- Row 3: FC summary panels ---------------------------------------
    draw_value_dist(fig.add_subplot(gs[2, 0]), fc_emp_v, fc_sim_v, xlabel="FC",
                    ks=ks_fc_best, title="Value distribution")
    draw_corr_hist(fig.add_subplot(gs[2, 1]), d["fc_corr"], title="FC correlation")
    draw_ks_hist(fig.add_subplot(gs[2, 2]), d["ks_fc"],
                 xlabel="KS distance (FC)", title="KS distance")

    # --- Row 4: dFC summary panels --------------------------------------
    draw_value_dist(fig.add_subplot(gs[3, 0]), dfc_emp_v, dfc_sim_v, xlabel="dFC",
                    ks=ks_dfc_best, title="Value distribution")
    draw_dfc_scatter(fig.add_subplot(gs[3, 1]), d["flu_emp"], d["flu_sim"],
                     title="dFC variance")
    draw_ks_hist(fig.add_subplot(gs[3, 2]), d["ks_dfc"],
                 xlabel="KS distance (dFC)", title="KS distance")

    print(f"best subject {str(d['best_sid'])}  r_FC={r_fc:.3f}  r_dFC={r_dfc:.3f}  "
          f"KS_FC={ks_fc_best:.3f}  KS_dFC={ks_dfc_best:.3f}")
    save_panel(fig, OUT_DIR / "Figure2_fit_composite")


if __name__ == "__main__":
    main()
