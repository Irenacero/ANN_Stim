"""
Figure 2 -- Personalized ANN fit across all HCP participants.

Free-runs every fitted ANN (NPI.model_time_series, tlen=4600, noise=0.1, exactly
as HCP_2_Fit_data), compares simulated vs empirical FC / dFC, and produces:

  best_traces : simulated BOLD of the best-fit participant, 0-180 s, Fig-1 style.
  best_FC     : split matrix, lower = simulated FC, upper = empirical FC.
  best_dFC    : split matrix, lower = simulated dFC, upper = empirical dFC.
  FC_corr_hist: distribution across participants of the empirical-vs-simulated
                FC edge correlation.
  dFC_var_scatter : per participant, variance of empirical dFC (x) vs simulated
                dFC (y), with regression line and Spearman r.

Per-participant scalars are cached so re-plotting is instant.

Output
    codes/HCP/results/Figure2_fit_group.npz          (cache)
    codes/figures/outputs/Figure2_best_traces.{svg,pdf,png}
    codes/figures/outputs/Figure2_best_FC.{svg,pdf,png}
    codes/figures/outputs/Figure2_best_dFC.{svg,pdf,png}
    codes/figures/outputs/Figure2_FC_corr_hist.{svg,pdf,png}
    codes/figures/outputs/Figure2_dFC_var_scatter.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import os
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
sys.path.insert(0, str(HCP_DIR))           # so the model unpickles as `src.NPI`
from src import NPI                         # noqa: E402
import torch                               # noqa: E402

setup()

PROC = HCP_DIR / "results/processed"
MODELS = HCP_DIR / "results/ANN_model"
LABEL_TXT = HCP_DIR / "data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"

S, N = 3, 450
TLEN = 4600
# Noise level for the free-run simulation (HCP_2 default 0.1). Override with
# e.g. SIM_NOISE=0.05; non-default values get a filename suffix so runs are kept
# separate.
NOISE = float(os.environ.get("SIM_NOISE", "0.1"))
SUFFIX = "" if abs(NOISE - 0.1) < 1e-9 else f"_noise{int(round(NOISE * 100)):02d}"
CACHE = HCP_DIR / f"results/Figure2_fit_group{SUFFIX}.npz"
TEST_DUR_FC = 3000
TEST_DUR_DFC = 1500
TR = 0.72
SEED = 0

# Emp-vs-sim colors (shared by every fit panel): the personalized model in
# tab:blue, the empirical data in a neutral dark gray.
MODEL = "#1f77b4"           # tab:blue -- simulated / model fit
EMP = "#4d4d4d"             # dark gray -- empirical

# Best-participant traces panel (Fig-1 style).
N_TRACES = 40
TRACE_OFFSET = 1.5
TRACES_SECONDS = 360.0
T_TRACES = int(round(TRACES_SECONDS / TR))     # ~500 timepoints


def load_model(sid):
    torch.serialization.add_safe_globals(
        [NPI.ANN_MLP, NPI.ANN_CNN, NPI.ANN_RNN, NPI.ANN_VAR])
    m = torch.load(MODELS / f"{sid}_MLP.pt", map_location=NPI.device,
                   weights_only=False)
    m.eval()
    return m


def go_edge(ts):
    n = ts.shape[1]
    iu = np.triu_indices(n, k=1)
    gz = stats.zscore(ts, axis=0).astype(np.float32)
    return gz[:, iu[0]] * gz[:, iu[1]]


def dFC(ts):
    return np.corrcoef(go_edge(ts))


def fitted_subjects():
    return sorted(p.name.split("_MLP.pt")[0] for p in MODELS.glob("*_MLP.pt"))


def compute_group():
    """Per-subject FC correlation and dFC variance; keep the best subject's
    full matrices + simulated trace window."""
    sids = fitted_subjects()
    iuN = np.triu_indices(N, 1)
    fc_corr, flu_emp, flu_sim, kept = [], [], [], []
    ks_fc, ks_dfc = [], []          # per-subject KS distance, emp vs sim values
    best = {"corr": -np.inf}
    # Group-average FC accumulators (Luo et al.: r=0.97 is corr of the
    # subject-averaged model FC vs subject-averaged empirical FC).
    gfe = np.zeros((N, N)); gfs = np.zeros((N, N)); n_group = 0

    for k, sid in enumerate(sids):
        try:
            Z = np.load(PROC / f"{sid}_signals.npy")
            model = load_model(sid)
        except Exception as e:
            print(f"  skip {sid}: {e}"); continue
        np.random.seed(SEED)
        Zsim = NPI.model_time_series(model, np.zeros((S, N)), tlen=TLEN,
                                     noise_strength=NOISE)

        FC_emp = np.corrcoef(Z[-TEST_DUR_FC:].T)
        FC_sim = np.corrcoef(Zsim[-TEST_DUR_FC:].T)
        x, y = FC_emp[iuN], FC_sim[iuN]
        m = np.isfinite(x) & np.isfinite(y)
        r = stats.pearsonr(x[m], y[m])[0] if m.sum() > 2 else np.nan
        if np.isfinite(FC_emp).all() and np.isfinite(FC_sim).all():
            gfe += FC_emp; gfs += FC_sim; n_group += 1

        dfc_emp = dFC(Z[-TEST_DUR_DFC:])
        dfc_sim = dFC(Zsim[-TEST_DUR_DFC:])
        iuT = np.triu_indices(dfc_emp.shape[0], 1)
        ve = np.nanvar(dfc_emp[iuT]); vs = np.nanvar(dfc_sim[iuT])

        # KS distance between the empirical and simulated value distributions
        # (FC edges over the upper triangle; dFC off-diagonal entries).
        kfc = stats.ks_2samp(x[m], y[m]).statistic if m.sum() > 2 else np.nan
        de, ds = dfc_emp[iuT], dfc_sim[iuT]
        de, ds = de[np.isfinite(de)], ds[np.isfinite(ds)]
        kdfc = (stats.ks_2samp(de, ds).statistic
                if de.size > 2 and ds.size > 2 else np.nan)

        fc_corr.append(r); flu_emp.append(ve); flu_sim.append(vs); kept.append(sid)
        ks_fc.append(kfc); ks_dfc.append(kdfc)
        if np.isfinite(r) and r > best["corr"]:
            best = {"corr": r, "sid": sid,
                    "FC_emp": FC_emp.astype(np.float32),
                    "FC_sim": FC_sim.astype(np.float32),
                    "dfc_emp": dfc_emp.astype(np.float32),
                    "dfc_sim": dfc_sim.astype(np.float32),
                    "traces": Zsim[-T_TRACES:].astype(np.float32)}
        print(f"  [{k+1}/{len(sids)}] {sid}  r_FC={r:.3f}  "
              f"flu_emp={ve:.4g} flu_sim={vs:.4g}", flush=True)
        del Z, Zsim, model, dfc_emp, dfc_sim

    np.savez(CACHE, sids=np.array(kept), fc_corr=np.array(fc_corr),
             flu_emp=np.array(flu_emp), flu_sim=np.array(flu_sim),
             ks_fc=np.array(ks_fc), ks_dfc=np.array(ks_dfc),
             best_sid=best["sid"], best_FC_emp=best["FC_emp"],
             best_FC_sim=best["FC_sim"], best_dfc_emp=best["dfc_emp"],
             best_dfc_sim=best["dfc_sim"], best_traces=best["traces"],
             group_FC_emp=(gfe / n_group).astype(np.float32),
             group_FC_sim=(gfs / n_group).astype(np.float32), n_group=n_group)
    print(f"Cached -> {CACHE}  (best = {best['sid']}, r={best['corr']:.3f})")
    return np.load(CACHE, allow_pickle=True)


def combine(lower, upper, diag=1.0):
    n = lower.shape[0]
    M = np.empty((n, n), dtype=float)
    M[np.tril_indices(n, -1)] = lower[np.tril_indices(n, -1)]
    M[np.triu_indices(n, 1)] = upper[np.triu_indices(n, 1)]
    np.fill_diagonal(M, diag)
    return M


def plot_split(mat, out_stem, *, cmap, cbar_label, axis_label, annot=None):
    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=figsize_mm(58, 58), constrained_layout=True)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper",
                   interpolation="nearest")
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="white", lw=1.2)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(axis_label); ax.set_ylabel(axis_label)
    ax.text(0.72 * n, 0.12 * n, "Empirical", ha="center", va="center", fontsize=8)
    ax.text(0.28 * n, 0.88 * n, "Simulated", ha="center", va="center", fontsize=8)
    if annot:
        ax.text(0.5, 1.03, annot, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label); cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.7, length=2.5)
    save_panel(fig, OUT_DIR / f"{out_stem}{SUFFIX}")


def plot_best_traces(traces):
    raster = traces[:, N_SUBCORTICAL:].astype(float)
    perm, _, _ = cortical_network_only_ordering(LABEL_TXT)
    sig_z = ((raster - raster.mean(0)) / raster.std(0))[:, perm]
    idx = np.linspace(0, raster.shape[1] - 1, N_TRACES, dtype=int)
    colors = sequential_colors(N_TRACES, cmap=plt.cm.Greys, lo=0.35, hi=0.9)
    t = np.arange(raster.shape[0]) * TR

    fig, ax = plt.subplots(figsize=figsize_mm(90, 55))
    for k, roi in enumerate(idx):
        ax.plot(t, sig_z[:, roi] + k * TRACE_OFFSET, color=colors[k],
                lw=0.6, alpha=0.8)
    ax.set_xlim(0, TRACES_SECONDS)
    ax.set_ylim(-3, (N_TRACES - 1) * TRACE_OFFSET + 3)
    ax.set_xticks(list(range(0, int(TRACES_SECONDS) + 1, 60))); ax.set_yticks([])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ROIs")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.8, length=3, direction="out")
    save_panel(fig, OUT_DIR / f"Figure2_best_traces{SUFFIX}")


def plot_fc_hist(fc_corr):
    r = fc_corr[np.isfinite(fc_corr)]
    med = np.median(r)
    fig, ax = plt.subplots(figsize=figsize_mm(52, 46), constrained_layout=True)
    ax.hist(r, bins=16, color=MODEL, edgecolor="white", linewidth=0.4)
    ax.axvline(med, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.annotate(rf"median $= {med:.2f}$", xy=(med, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", ha="left", va="top", fontsize=7)
    ax.set_xlabel(r"FC correlation  $r$")
    ax.set_ylabel("Participants")
    ax.tick_params(width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(f"FC corr: n={r.size}  median={np.median(r):.3f}  "
          f"max={r.max():.3f}  min={r.min():.3f}")
    save_panel(fig, OUT_DIR / f"Figure2_FC_corr_hist{SUFFIX}")


def plot_dfc_scatter(flu_emp, flu_sim):
    m = np.isfinite(flu_emp) & np.isfinite(flu_sim)
    x, y = flu_emp[m], flu_sim[m]
    # Drop unstable free-run simulations (Tukey outliers on the simulated dFC
    # variance) -- the paper excludes participants whose simulation failed.
    q1, q3 = np.percentile(y, [25, 75])
    keep = y <= q3 + 1.5 * (q3 - q1)
    n_excl = int((~keep).sum())
    x, y = x[keep], y[keep]
    rho, p = stats.spearmanr(x, y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)

    fig, ax = plt.subplots(figsize=figsize_mm(54, 50), constrained_layout=True)
    ax.scatter(x, y, s=14, color=MODEL, edgecolor="#13496f",
               linewidths=0.3, alpha=0.75, zorder=3)
    ax.plot(xs, b * xs + a, color=INK, lw=1.2, zorder=4)
    ax.annotate(rf"Spearman $r = {rho:.2f}$", xy=(0.05, 0.95),
                xycoords="axes fraction", ha="left", va="top")
    ax.set_xlabel("dFC variance (empirical)")
    ax.set_ylabel("dFC variance (simulated)")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.tick_params(width=0.8, length=3, direction="out")
    print(f"dFC variance: Spearman r={rho:.3f}  p={p:.2e}  n={x.size} "
          f"({n_excl} unstable sims excluded)")
    save_panel(fig, OUT_DIR / f"Figure2_dFC_var_scatter{SUFFIX}")


def plot_value_dist(emp_vals, sim_vals, out_stem, *, xlabel, ks):
    """Overlaid empirical (gray) vs simulated (blue) value distributions for the
    best-fit participant, with the KS distance annotated."""
    e = emp_vals[np.isfinite(emp_vals)]
    s = sim_vals[np.isfinite(sim_vals)]
    lo = min(e.min(), s.min()); hi = max(e.max(), s.max())
    bins = np.linspace(lo, hi, 41)
    fig, ax = plt.subplots(figsize=figsize_mm(52, 46), constrained_layout=True)
    ax.hist(e, bins=bins, density=True, histtype="stepfilled", color=EMP,
            alpha=0.4, edgecolor=EMP, linewidth=1.0, label="Empirical")
    ax.hist(s, bins=bins, density=True, histtype="stepfilled", color=MODEL,
            alpha=0.4, edgecolor=MODEL, linewidth=1.0, label="Simulated")
    ax.annotate(rf"KS $= {ks:.2f}$", xy=(0.96, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=7)
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0,
              borderpad=0.2, labelspacing=0.3, fontsize=7)
    ax.tick_params(width=0.8, length=3, direction="out")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    print(f"{out_stem}: KS={ks:.3f}  (emp n={e.size}, sim n={s.size})")
    save_panel(fig, OUT_DIR / f"{out_stem}{SUFFIX}")


def plot_ks_hist(ks, out_stem, *, xlabel):
    """Per-participant distribution of the empirical-vs-simulated KS distance."""
    k = ks[np.isfinite(ks)]
    med = np.median(k)
    fig, ax = plt.subplots(figsize=figsize_mm(52, 46), constrained_layout=True)
    ax.hist(k, bins=16, color=MODEL, edgecolor="white", linewidth=0.4)
    ax.axvline(med, color="#222222", linestyle=(0, (3, 2)), linewidth=1.2)
    ax.annotate(rf"median $= {med:.2f}$", xy=(med, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", ha="left", va="top", fontsize=7)
    ax.set_xlabel(xlabel); ax.set_ylabel("Participants")
    ax.tick_params(width=0.8, length=3, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(f"{out_stem}: n={k.size}  median={med:.3f}  "
          f"min={k.min():.3f}  max={k.max():.3f}")
    save_panel(fig, OUT_DIR / f"{out_stem}{SUFFIX}")


def main():
    d = np.load(CACHE, allow_pickle=True) if CACHE.exists() else compute_group()
    if "ks_fc" not in d.files:        # legacy cache predates the KS panels
        print("cache missing KS distances -> recomputing group")
        d = compute_group()
    best_sid = str(d["best_sid"])
    print(f"best participant = {best_sid}")
    # Re-simulate the best subject for a long enough trace window (cache stores
    # only a short window); cheap (~0.3 s).
    model = load_model(best_sid)
    np.random.seed(SEED)
    Zsim = NPI.model_time_series(model, np.zeros((S, N)), tlen=TLEN,
                                 noise_strength=NOISE)
    plot_best_traces(Zsim[-T_TRACES:])
    plot_split(combine(d["best_FC_sim"], d["best_FC_emp"]), "Figure2_best_FC",
               cmap="plasma", cbar_label="FC", axis_label="ROIs")
    plot_split(combine(d["best_dfc_sim"], d["best_dfc_emp"]), "Figure2_best_dFC",
               cmap="Spectral_r", cbar_label="dFC", axis_label="Time")
    plot_fc_hist(d["fc_corr"])
    plot_dfc_scatter(d["flu_emp"], d["flu_sim"])

    # --- Row B col 2 / Row C col 2: emp-vs-sim value distributions (best
    #     participant) with KS distance; col 4: per-participant KS histograms.
    iuN = np.triu_indices(N, 1)
    fc_emp_v, fc_sim_v = d["best_FC_emp"][iuN], d["best_FC_sim"][iuN]
    ks_fc_best = stats.ks_2samp(fc_emp_v[np.isfinite(fc_emp_v)],
                                fc_sim_v[np.isfinite(fc_sim_v)]).statistic
    plot_value_dist(fc_emp_v, fc_sim_v, "Figure2_FC_value_dist",
                    xlabel="FC", ks=ks_fc_best)
    plot_ks_hist(d["ks_fc"], "Figure2_FC_ks_hist", xlabel="KS distance (FC)")

    nT = d["best_dfc_emp"].shape[0]
    iuT = np.triu_indices(nT, 1)
    dfc_emp_v, dfc_sim_v = d["best_dfc_emp"][iuT], d["best_dfc_sim"][iuT]
    ks_dfc_best = stats.ks_2samp(dfc_emp_v[np.isfinite(dfc_emp_v)],
                                 dfc_sim_v[np.isfinite(dfc_sim_v)]).statistic
    plot_value_dist(dfc_emp_v, dfc_sim_v, "Figure2_dFC_value_dist",
                    xlabel="dFC", ks=ks_dfc_best)
    plot_ks_hist(d["ks_dfc"], "Figure2_dFC_ks_hist", xlabel="KS distance (dFC)")

    # Group-level FC (Luo et al. reproduction): correlate subject-averaged
    # model FC vs subject-averaged empirical FC.
    gfe, gfs = d["group_FC_emp"], d["group_FC_sim"]
    iuN = np.triu_indices(gfe.shape[0], 1)
    rg = stats.pearsonr(gfe[iuN], gfs[iuN])[0]
    print(f"GROUP-LEVEL FC corr (n={int(d['n_group'])} subjects): r = {rg:.3f}")
    plot_split(combine(gfs, gfe), "Figure2_group_FC", cmap="plasma",
               cbar_label="FC", axis_label="ROIs", annot=rf"group  $r = {rg:.2f}$")


if __name__ == "__main__":
    main()
