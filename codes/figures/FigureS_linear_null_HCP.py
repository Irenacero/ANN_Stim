"""
Supplementary linear-model control for the state-gating (paper point #3).

Question: is the dependence of the global effect size (GES) on baseline energy
E(t) a genuine nonlinear, state-dependent property of the trained twin, or an
artifact of the perturbation/normalization pipeline that any model would show?

Control: a *linear* surrogate has a state-independent perturbation response. If
the one-step map is f(X)=AX+b, then EC = f(X+delta) - f(X) = A.delta, a fixed
vector independent of the current state. Hence GES(t)=||A.delta||^2 is constant
in time and cannot depend on E(t). We fit the optimal linear model (ordinary
least squares, the best possible VAR with window S=3) to each subject and run
the *identical* perturbation pipeline used for the MLP.

Panels
  A  Example (subject id_100206, target roi 100 = L SomMot 20, same as Fig 4A):
     GES(t) vs baseline energy E(t), MLP (negative gating, rho=-0.41) vs the
     linear model (flat line: identical response at every time point).
  B  Cohort: distribution over cortical targets of the per-target gating
     correlation rho(E, GES) for the MLP (median ~ -0.36, 99.8% negative).
     The linear model has Var_t(GES)=0 at every target, so this correlation is
     undefined: the gating is absent by construction.

Outputs
  codes/figures/outputs/FigureS_linear_null_scatter.{svg,pdf,png}
  codes/figures/outputs/FigureS_linear_null_hist.{svg,pdf,png}
  codes/HCP/results/FigureS_linear_null_VAR_cohort.csv   (per-subject VAR CV check)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, INK
setup()

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "codes/HCP/results/processed"
ECTS = ROOT / "codes/HCP/results/ECts"
MLP_CORR = ROOT / "codes/HCP/results/Figure4_PanelB_node_corr_HCP_spearman.csv"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COHORT_CSV = ROOT / "codes/HCP/results/FigureS_linear_null_VAR_cohort.csv"

N = 450
S = 3
PERT = 0.1
LAST = (S - 1) * N            # offset of the current-frame block in the window
EXAMPLE_SID = "id_100206"
EXAMPLE_ROI = 100             # L SomMot 20, the Fig 4A example
EXAMPLE_LABEL = "L SomMot 20"

MLP_BLUE = "#1f77b4"
VAR_ORANGE = "#e8853a"


def fit_var(inp: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Optimal linear one-step map A (N, S*N) by ordinary least squares, with
    intercept. tgt ~ inp @ A.T + b."""
    A1 = np.hstack([inp, np.ones((inp.shape[0], 1))])
    W, *_ = np.linalg.lstsq(A1, tgt, rcond=None)   # (S*N+1, N)
    return W[:-1, :].T                              # (N, S*N)


def var_ges(A: np.ndarray, roi: int) -> float:
    """Constant GES evoked by perturbing `roi` in the linear model."""
    ec = PERT * A[:, LAST + roi]                    # (N,) response, state-independent
    return float(np.sum(ec ** 2))


# ---------------------------------------------------------------------------
# Panel A: example scatter, MLP gating vs flat linear response
# ---------------------------------------------------------------------------
def panel_example():
    import matplotlib.pyplot as plt

    inp = np.load(PROC / f"{EXAMPLE_SID}_inputs.npy")    # (T, S*N)
    tgt = np.load(PROC / f"{EXAMPLE_SID}_targets.npy")   # (T, N)
    ect_npy = ECTS / f"{EXAMPLE_SID}_ECt.npy"
    if ect_npy.exists():
        ect = np.load(ect_npy)                           # (Te, N, N)
        ges_mlp = np.sum(ect[:, EXAMPLE_ROI, :] ** 2, axis=1)  # (Te,)
    else:  # GES series cache shipped with the figure bundle (same values)
        cache = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"
        ges_mlp = np.load(cache)["ges_roi100"]
    Te = ges_mlp.shape[0]

    Xt = inp[:Te, -N:]
    E = np.sum(Xt ** 2, axis=1)                          # (Te,)
    rho_mlp, _ = spearmanr(E, ges_mlp)

    A = fit_var(inp, tgt)
    ges_var_const = var_ges(A, EXAMPLE_ROI)
    std_var = 0.0

    # The two models have very different response magnitudes (the linear fit has
    # a large gain, the MLP is contractive), so we compare state-dependence:
    # GES normalized to each model's own temporal mean. The linear model is a
    # flat line at 1; the MLP varies with state (negative gating).
    ges_mlp_n = ges_mlp / np.mean(ges_mlp)

    fig, ax = plt.subplots(figsize=figsize_mm(58, 50), constrained_layout=True)
    ax.scatter(E, ges_mlp_n, s=9, color=MLP_BLUE, alpha=0.55,
               edgecolor=(0.12, 0.27, 0.42), linewidths=0.2, zorder=3,
               label=f"MLP  ($\\rho={rho_mlp:+.2f}$)")
    ax.axhline(1.0, color=VAR_ORANGE, lw=2.0, zorder=4,
               label="Linear (VAR): constant")
    ax.set_xscale("log")
    ax.set_xlabel("Baseline energy")
    ax.set_ylabel("GES / mean(GES)")
    ax.set_title(f"{EXAMPLE_SID}  ·  {EXAMPLE_LABEL}", fontsize=8)
    ax.grid(True, which="major", linestyle=(0, (4, 3)), linewidth=0.4,
            color="#c9c9c9", alpha=0.6, zorder=0)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    save_panel(fig, OUT_DIR / "FigureS_linear_null_scatter")
    print(f"[A] MLP rho={rho_mlp:+.3f}  VAR GES const={ges_var_const:.3e} "
          f"temporal std={std_var:.1e}")
    return rho_mlp


# ---------------------------------------------------------------------------
# Panel B: cohort MLP gating-correlation histogram
# ---------------------------------------------------------------------------
def panel_cohort_hist():
    import matplotlib.pyplot as plt

    d = pd.read_csv(MLP_CORR)
    cort = d[d.roi >= 50]
    r = cort.r_global_effect.to_numpy()
    med = np.median(r)
    frac_neg = float((r < 0).mean())

    fig, ax = plt.subplots(figsize=figsize_mm(58, 50), constrained_layout=True)
    ax.hist(r, bins=40, color=MLP_BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0.0, color=VAR_ORANGE, lw=2.0,
               label="Linear (VAR):\nno gating (Var$_t$ GES = 0)")
    ax.axvline(med, color=INK, lw=1.0, linestyle=(0, (4, 2)),
               label=f"MLP median = {med:+.2f}")
    ax.set_xlabel(r"Gating correlation  $\rho(E,\,$GES$)$  per target")
    ax.set_ylabel("Cortical targets")
    ax.set_title(f"MLP: {frac_neg*100:.1f}% of targets negative", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    save_panel(fig, OUT_DIR / "FigureS_linear_null_hist")
    print(f"[B] MLP cortical rho: median={med:+.3f}  frac_neg={frac_neg:.3f}  n={len(r)}")
    return med, frac_neg


# ---------------------------------------------------------------------------
# Cohort VAR check: confirm GES temporal invariance across all subjects
# ---------------------------------------------------------------------------
def cohort_var_check(n_targets=20, seed_list=None):
    if COHORT_CSV.exists():
        df = pd.read_csv(COHORT_CSV)
        print(f"Loaded cache {COHORT_CSV}  ({len(df)} subjects)")
        print(f"[cohort] VAR max temporal CV of GES over {len(df)} subjects: "
              f"max={df.max_temporal_cv_GES.max():.1e}  "
              f"median={df.max_temporal_cv_GES.median():.1e}")
        return df
    sids = sorted(p.name.split("_inputs.npy")[0]
                  for p in PROC.glob("*_inputs.npy"))
    rng_targets = np.arange(50, N)        # cortical targets
    rows = []
    for k, sid in enumerate(sids):
        inp = np.load(PROC / f"{sid}_inputs.npy")
        tgt = np.load(PROC / f"{sid}_targets.npy")
        A = fit_var(inp, tgt)
        # empirical temporal std of GES via the actual perturbation pipeline,
        # for a subset of cortical targets, over the first 500 windows
        Te = min(500, inp.shape[0])
        sub = rng_targets[:: max(1, len(rng_targets) // n_targets)][:n_targets]
        max_cv = 0.0
        for roi in sub:
            pert_flat = np.zeros(S * N)
            pert_flat[LAST + roi] = PERT
            ec_t = (inp[:Te] + pert_flat) @ A.T - inp[:Te] @ A.T   # (Te, N)
            ges_t = np.sum(ec_t ** 2, axis=1)                      # (Te,)
            cv = float(np.std(ges_t) / (np.mean(ges_t) + 1e-300))
            max_cv = max(max_cv, cv)
        rows.append({"sub_id": sid, "max_temporal_cv_GES": max_cv})
        if k % 20 == 0:
            print(f"  VAR cohort {k+1}/{len(sids)}  {sid}  max CV={max_cv:.1e}")
    df = pd.DataFrame(rows)
    df.to_csv(COHORT_CSV, index=False)
    print(f"[cohort] VAR max temporal CV of GES over {len(df)} subjects: "
          f"max={df.max_temporal_cv_GES.max():.1e}  "
          f"median={df.max_temporal_cv_GES.median():.1e}")
    return df


def panel_combined():
    """Combined two-panel draft (A scatter, B histogram) saved straight into the
    paper figure folder so the LaTeX compiles; refine/replace in Inkscape."""
    import matplotlib.pyplot as plt

    inp = np.load(PROC / f"{EXAMPLE_SID}_inputs.npy")
    tgt = np.load(PROC / f"{EXAMPLE_SID}_targets.npy")
    ect_npy = ECTS / f"{EXAMPLE_SID}_ECt.npy"
    if ect_npy.exists():
        ect = np.load(ect_npy)
        ges_mlp = np.sum(ect[:, EXAMPLE_ROI, :] ** 2, axis=1)
    else:  # GES series cache shipped with the figure bundle (same values)
        cache = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"
        ges_mlp = np.load(cache)["ges_roi100"]
    Te = ges_mlp.shape[0]
    E = np.sum(inp[:Te, -N:] ** 2, axis=1)
    rho_mlp, _ = spearmanr(E, ges_mlp)
    ges_mlp_n = ges_mlp / np.mean(ges_mlp)

    d = pd.read_csv(MLP_CORR)
    r = d[d.roi >= 50].r_global_effect.to_numpy()
    med = np.median(r); frac_neg = float((r < 0).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_mm(150, 58),
                                   constrained_layout=True)
    ax1.scatter(E, ges_mlp_n, s=9, color=MLP_BLUE, alpha=0.55,
                edgecolor=(0.12, 0.27, 0.42), linewidths=0.2, zorder=3,
                label=f"MLP  ($\\rho={rho_mlp:+.2f}$)")
    ax1.axhline(1.0, color=VAR_ORANGE, lw=2.0, zorder=4, label="Linear (VAR)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Baseline energy"); ax1.set_ylabel("GES / mean(GES)")
    ax1.set_title(f"{EXAMPLE_SID}  ·  {EXAMPLE_LABEL}", fontsize=8)
    ax1.grid(True, which="major", linestyle=(0, (4, 3)), linewidth=0.4,
             color="#c9c9c9", alpha=0.6, zorder=0)
    ax1.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    ax1.text(-0.18, 1.02, "A", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    ax2.hist(r, bins=40, color=MLP_BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axvline(0.0, color=VAR_ORANGE, lw=2.0, label="Linear (VAR): no gating")
    ax2.axvline(med, color=INK, lw=1.0, linestyle=(0, (4, 2)),
                label=f"MLP median = {med:+.2f}")
    ax2.set_xlabel(r"Gating correlation $\rho(E,$ GES$)$ per target")
    ax2.set_ylabel("Cortical targets")
    ax2.set_title(f"{frac_neg*100:.1f}% of targets negative", fontsize=8)
    ax2.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    ax2.text(-0.18, 1.02, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    pdf_dir = ROOT / "paper/ANN_fMRI_HCP/Figures/Inkscape/PDF"
    if not pdf_dir.parent.parent.exists():   # figure bundle: no paper/ tree
        pdf_dir = OUT_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / "SF5.pdf", dpi=600)
    fig.savefig(OUT_DIR / "FigureS_linear_null_combined.png", dpi=300)
    plt.close(fig)
    print(f"  -> {pdf_dir/'SF5.pdf'}")


def main():
    rho = panel_example()
    med, frac = panel_cohort_hist()
    df = cohort_var_check()
    print("\nSUMMARY for paper:")
    print(f"  MLP example rho = {rho:+.2f}")
    print(f"  MLP cohort: median rho = {med:+.2f}, {frac*100:.1f}% negative")
    print(f"  VAR: GES temporal CV <= {df.max_temporal_cv_GES.max():.0e} "
          f"(state-independent by construction)")


if __name__ == "__main__":
    main()
