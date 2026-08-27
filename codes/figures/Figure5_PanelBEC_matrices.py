"""
Figure 5 bifocal matrices -- focal vs closed-loop-focal vs bifocal variability.

Built from the small per-subject caches written by Figure5_BEC_reduce.py (no need
to reload the 77 GB of full BEC tensors). All quantities use the SAME metric as
the bifocal EC: BEC_t[t,i,j] = whole-brain L2 response of co-stimulating i,j;
the diagonal i,i is the focal (single-region) effect. Trial-to-trial variability
= coefficient of variation CV = std_t / mean_t.

Four region x region matrices (averaged across participants):
  1. mean_bec   : mean bifocal effective connectivity  <BEC[i,j]>
  2. cv_bec      : trial-to-trial variability (CV) of BEC[i,j]
  3. impr_naive  : % change in CV of bifocal(i,j) vs the BEST focal of i,j,
                   both with no state selection (all trials).
                   = 100 * (CV_all[i,j] - min(CV_all[i,i], CV_all[j,j])) / min(...)
  4. impr_cloop  : % change in CV of bifocal(i,j) at RANDOM 5% trials vs the BEST
                   focal CLOSED-LOOP (5% lowest-energy) of i,j.
                   = 100 * (CV_rand[i,j] - min(CV_low[i,i], CV_low[j,j])) / min(...)
  (3,4: negative = bifocal more reproducible than the best single-site option.)

Per-subject CVs are averaged across subjects, then the improvement matrices are
computed on those average matrices. Displayed cortical-only (400), ordered by
Yeo-7 network.

Outputs
    codes/HCP/results/Figure5_BEC_matrices.npz
    codes/figures/outputs/Figure5_BEC_{mean,cv,impr_naive,impr_cloop}.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, SEQ_CMAP, DIV_CMAP, YEO7_COLORS
from brain_render import cortical_network_only_ordering
setup()

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "codes/HCP/results/BECts_reduced"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 450
N_SUBCORT = 50          # cortical = roi 50..449


def _cv(mean, std):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mean != 0, std / mean, np.nan)


def _two_slope_cmap(base, vmin, vcenter, vmax, n=256):
    """Bake a TwoSlopeNorm into a ListedColormap, returned with a *linear*
    Normalize. The image keeps 0 at the colormap centre, but the colorbar is
    linear in data (proportional) instead of giving each slope equal length."""
    tsn = TwoSlopeNorm(vcenter, vmin, vmax)
    return (ListedColormap(base(tsn(np.linspace(vmin, vmax, n)))),
            Normalize(vmin, vmax))


def load_average():
    """Average per-subject GLOBAL EFFECT SIZE matrices across cached subjects.

    Everything is in terms of Sigma = BEC^2 (global effect size; diagonal =
    unifocal Sigma^(j)). msq_* = mean Sigma over a regime, ssq_* = its std.
    """
    files = sorted(CACHE_DIR.glob("id_*.npz")) if CACHE_DIR.exists() else []
    if not files and OUT_NPZ.exists():
        # Figure bundle: rebuild the group averages from the saved group npz
        # (same values; the per-subject BECts_reduced caches are not shipped).
        g = np.load(OUT_NPZ)
        acc = {"mean_eff": g["mean_bec"], "cv_all": g["cv_bec"],
               "cv_low": g["cv_low"], "cv_rand": g["cv_rand"],
               "var_all": g["var_bec"], "var_low": g["var_low"],
               "var_rand": g["var_rand"]}
        n = int(g["n_subjects"])
        print(f"Loaded group cache {OUT_NPZ} (n={n})")
        return acc, n
    if not files:
        raise FileNotFoundError(f"No caches in {CACHE_DIR}; run Figure5_BEC_reduce.py")
    if "msq_all" not in np.load(files[0]).files:
        raise RuntimeError("Caches lack Sigma (msq_*) moments; rerun Figure5_BEC_reduce.py")
    acc = {k: np.zeros((N, N)) for k in ("mean_eff", "cv_all", "cv_low", "cv_rand",
                                         "var_all", "var_low", "var_rand")}
    for f in files:
        d = np.load(f)
        acc["mean_eff"] += d["msq_all"]                       # <Sigma> over all trials
        acc["cv_all"] += _cv(d["msq_all"], d["ssq_all"])      # CV of Sigma
        acc["cv_low"] += _cv(d["msq_low"], d["ssq_low"])
        acc["cv_rand"] += _cv(d["msq_rand"], d["ssq_rand"])
        acc["var_all"] += d["ssq_all"].astype(float) ** 2     # Var(GES) = std_t(Sigma)^2
        acc["var_low"] += d["ssq_low"].astype(float) ** 2
        acc["var_rand"] += d["ssq_rand"].astype(float) ** 2
    n = len(files)
    for k in acc:
        acc[k] /= n
    print(f"Averaged {n} subjects (global effect size Sigma)")
    return acc, n


def rel_change(bifocal, focal_diag, agg):
    """% change of bifocal[i,j] vs a reference focal of i,j.

    agg=np.minimum -> vs the BEST (lowest) single site  (used for CV: <0 better)
    agg=np.maximum -> vs the MAX single site            (used for effect: >0 bigger)
    """
    ref = agg(focal_diag[:, None], focal_diag[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * (bifocal - ref) / ref


def cortical_network_ordered(M):
    """Cortical 400x400 submatrix reordered by Yeo-7 network."""
    perm, boundaries, labels = cortical_network_only_ordering(LABEL_TXT)
    sub = M[N_SUBCORT:, N_SUBCORT:]
    return sub[np.ix_(perm, perm)], boundaries, labels


def _draw_matrix(M, out_stem, title, cmap, norm, cbar_label):
    Mo, boundaries, labels = cortical_network_ordered(M)
    fig, ax = plt.subplots(figsize=figsize_mm(88, 84), constrained_layout=True)
    im = ax.imshow(Mo, cmap=cmap, norm=norm, aspect="equal", origin="upper",
                   interpolation="nearest")
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="#3a3a3a", lw=0.4, alpha=0.7)
        ax.axhline(b - 0.5, color="#3a3a3a", lw=0.4, alpha=0.7)
    # Yeo-7 colour bands along the edges
    from matplotlib.patches import Rectangle
    starts = [0] + list(boundaries[:-1])
    band = Mo.shape[0] * 0.02
    for s0, e0, lab in zip(starts, boundaries, labels):
        c = YEO7_COLORS[lab]
        ax.add_patch(Rectangle((s0 - 0.5, -0.5 - band), e0 - s0, band,
                               facecolor=c, edgecolor="none", clip_on=False))
        ax.add_patch(Rectangle((-0.5 - band, s0 - 0.5), band, e0 - s0,
                               facecolor=c, edgecolor="none", clip_on=False))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Region j (Yeo-7 ordered)")
    ax.set_ylabel("Region i")
    ax.set_title(title, pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.7, length=2.5)
    save_panel(fig, OUT_DIR / out_stem)


SIG = r"\Sigma^{(jk)}"     # bifocal global effect size = BEC^2 (diagonal = unifocal Sigma^(j))


def main():
    acc, n = load_average()
    cv_all, cv_low, cv_rand = acc["cv_all"], acc["cv_low"], acc["cv_rand"]
    var_all, var_low, var_rand = acc["var_all"], acc["var_low"], acc["var_rand"]
    mean_eff = acc["mean_eff"]                                     # <Sigma> global effect size
    impr_naive = rel_change(cv_all, np.diag(cv_all), np.minimum)   # vs best focal CV(Sigma)
    impr_cloop = rel_change(cv_rand, np.diag(cv_low), np.minimum)  # vs best closed-loop CV(Sigma)
    # Variance analogues (Var(GES) = std_t(Sigma)^2): the reproducibility change
    # measured as a change in variance instead of CV.
    impr_var = rel_change(var_all, np.diag(var_all), np.minimum)   # vs best focal Var(GES)
    impr_var_cloop = rel_change(var_rand, np.diag(var_low), np.minimum)
    eff_incr = rel_change(mean_eff, np.diag(mean_eff), np.maximum) # vs MAX focal <Sigma>
    np.savez(OUT_NPZ, mean_bec=mean_eff, cv_bec=cv_all, cv_low=cv_low,
             cv_rand=cv_rand, impr_naive=impr_naive, impr_cloop=impr_cloop,
             var_bec=var_all, var_low=var_low, var_rand=var_rand,
             impr_var=impr_var, impr_var_cloop=impr_var_cloop,
             eff_incr=eff_incr, n_subjects=n)
    print(f"Saved {OUT_NPZ}")

    # 1) mean bifocal global effect size (sequential)
    sub = mean_eff[N_SUBCORT:, N_SUBCORT:]
    _draw_matrix(mean_eff, "Figure5_BEC_mean",
                 f"Mean bifocal global effect size $\\langle{SIG}\\rangle$ (N={n})",
                 SEQ_CMAP, Normalize(np.nanpercentile(sub, 1),
                                     np.nanpercentile(sub, 99)),
                 rf"$\langle{SIG}\rangle$")

    # 2) trial-to-trial variability of the bifocal global effect size (sequential)
    subcv = cv_all[N_SUBCORT:, N_SUBCORT:]
    _draw_matrix(cv_all, "Figure5_BEC_cv",
                 f"Variability of bifocal global effect size (CV of ${SIG}$, N={n})",
                 SEQ_CMAP, Normalize(np.nanpercentile(subcv, 1),
                                     np.nanpercentile(subcv, 99)),
                 rf"CV$_t$ of ${SIG}$")

    # 3,4) CV-based reproducibility-improvement matrices (legacy; kept for
    #      reference, not used in the variance version of Figure 5).
    for M, stem, ttl in [
        (impr_naive, "Figure5_BEC_impr_naive",
         "Bifocal vs best focal — reproducibility (no timing)"),
        (impr_cloop, "Figure5_BEC_impr_cloop",
         "Bifocal (random) vs best focal closed-loop (low energy)")]:
        s = M[N_SUBCORT:, N_SUBCORT:]
        vmax = np.nanpercentile(np.abs(s), 98)
        _draw_matrix(M, stem, ttl, DIV_CMAP, TwoSlopeNorm(0.0, -vmax, vmax),
                     "% change in CV  (<0: bifocal more reproducible)")

    # 3v,4v) variance-based improvement matrices (Panel C = impr_var, all trials).
    # Asymmetric scale: vmin=-30, centre 0, vmax=max -- the blue (variance-
    # reducing) values are sparse and small, so give them the full blue range.
    for M, stem, ttl in [
        (impr_var, "Figure5_BEC_impr_var",
         "Bifocal vs best focal — variance of GES (no timing)"),
        (impr_var_cloop, "Figure5_BEC_impr_var_cloop",
         "Bifocal (random) vs best focal closed-loop (low energy)")]:
        s = M[N_SUBCORT:, N_SUBCORT:]
        vmax = float(np.nanpercentile(s, 98))     # cap red at the 98th pct
        cmap2, norm2 = _two_slope_cmap(DIV_CMAP, -30.0, 0.0, vmax)
        _draw_matrix(M, stem, ttl, cmap2, norm2,
                     "% change in variance of GES  (<0: bifocal less variable)")

    # 5) effect-size increase vs MAX focal (diverging, >0 = bifocal larger)
    s = eff_incr[N_SUBCORT:, N_SUBCORT:]
    vmax = np.nanpercentile(np.abs(s), 98)
    _draw_matrix(eff_incr, "Figure5_BEC_eff_incr",
                 "Bifocal vs max focal — mean(GES)",
                 DIV_CMAP, TwoSlopeNorm(0.0, -vmax, vmax),
                 "% change in mean(GES)  (>0: bifocal larger)")


if __name__ == "__main__":
    main()
