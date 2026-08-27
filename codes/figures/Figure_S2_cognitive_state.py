"""
Supplementary Figure S2 — Cognitive state modulates the gating axis.

(A) Model-free: within-subject change in baseline energy E(t) between cognitive
    states (SD units). Engagement/demand lower the energy (math<story, WM task<
    interior fixation); brief movement and parametric WM load do not.
(B) The gating law generalizes: feeding task states to the rest-trained twin, the
    per-participant Spearman rho(E, GES) stays negative (LANGUAGE, WM).
(C) The unimodal->transmodal hierarchy of responsiveness persists on task states.

Reads codes/HCP/results/cognitive_state_S2.npz (built by
codes/HCP/src/cognitive_state_gating.py). Output: outputs/Figure_S2_cognitive_state.
"""
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from style import setup, save_panel, figsize_mm, categorical, ACCENT_COOL, clean_axes

ROOT = Path(__file__).resolve().parents[2]
D = np.load(ROOT / "codes/HCP/results/cognitive_state_S2.npz")
OUT = Path(__file__).resolve().parent / "outputs" / "Figure_S2_cognitive_state"

GREY = "0.6"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def box(ax, data, pos, color, width=0.55):
    bp = ax.boxplot([data], positions=[pos], widths=width, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.15", lw=1.3),
                    whiskerprops=dict(color="0.4", lw=0.8), capprops=dict(color="0.4", lw=0.8),
                    boxprops=dict(lw=0.8))
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.55, edgecolor="0.3")
    x = np.random.default_rng(0).normal(pos, 0.06, size=len(data))
    ax.scatter(x, data, s=3, color=color, alpha=0.5, edgecolor="none", zorder=3)


def main():
    setup()
    fig = plt.figure(figsize=figsize_mm(180, 56))
    gs = GridSpec(1, 3, width_ratios=[2.1, 1, 1], wspace=0.45,
                  left=0.07, right=0.985, bottom=0.26, top=0.88)

    # ---------------- (A) energy by cognitive state ----------------
    axA = fig.add_subplot(gs[0])
    contrasts = [
        ("dE_math_vs_story", "math vs\nstory", ACCENT_COOL),
        ("dE_WM_task_vs_fix", "WM task vs\nfixation", ACCENT_COOL),
        ("dE_motor_vs_fix", "movement vs\nfixation", GREY),
        ("dE_WM_2bk_vs_0bk", "2-back vs\n0-back", GREY),
    ]
    axA.axhline(0, color="0.3", lw=0.8, ls="--", zorder=0)
    for i, (key, lab, col) in enumerate(contrasts):
        v = D[key]; v = v[~np.isnan(v)]
        box(axA, v, i, col)
        p = wilcoxon(v).pvalue
        ax_top = max(v.max(), 0.1)
        axA.text(i, 0.62, stars(p), ha="center", va="bottom", fontsize=7.5,
                 color="0.15" if p < 0.05 else "0.5")
        axA.text(i, -0.92, f"{np.median(v):+.2f}", ha="center", va="top", fontsize=6.5, color=col)
    axA.set_xticks(range(4)); axA.set_xticklabels([c[1] for c in contrasts], fontsize=7)
    axA.set_ylabel(r"$\Delta E$  (within-participant, SD units)", fontsize=8)
    axA.set_ylim(-1.0, 0.8)
    axA.set_title("Empirical baseline energy by cognitive state", fontsize=8, loc="left")
    clean_axes(axA)

    # ---------------- (B) gating generalizes ----------------
    axB = fig.add_subplot(gs[1])
    gate = [("gate_language", "LANG"), ("gate_wm", "WM")]
    cols = categorical(2)
    axB.axhline(0, color="0.3", lw=0.8, ls="--", zorder=0)
    for i, (key, lab) in enumerate(gate):
        v = D[key]
        box(axB, v, i, cols[i])
        axB.text(i, 0.18, f"{100*(v<0).mean():.0f}%<0", ha="center", fontsize=6.2, color="0.3")
        axB.text(i, -0.92, f"{np.median(v):+.2f}", ha="center", va="top", fontsize=6.5, color=cols[i])
    axB.set_xticks(range(2)); axB.set_xticklabels([g[1] for g in gate], fontsize=7)
    axB.set_ylabel(r"$\rho\,(E,\ \mathrm{GES})$ on task states", fontsize=7.5)
    axB.set_ylim(-1.0, 0.35)
    axB.set_title("Gating persists", fontsize=8, loc="left")
    clean_axes(axB)

    # ---------------- (C) hierarchy persists ----------------
    axC = fig.add_subplot(gs[2])
    hier = [("hier_language", "LANG"), ("hier_wm", "WM")]
    axC.axhline(0, color="0.3", lw=0.8, ls="--", zorder=0)
    for i, (key, lab) in enumerate(hier):
        v = D[key]
        box(axC, v, i, cols[i])
        axC.text(i, -0.05, f"{100*(v>0).mean():.0f}%>0", ha="center", va="top", fontsize=6.2, color="0.3")
        axC.text(i, -0.92, f"{np.median(v):+.2f}", ha="center", va="top", fontsize=6.5, color=cols[i])
    axC.set_xticks(range(2)); axC.set_xticklabels([h[1] for h in hier], fontsize=7)
    axC.set_ylabel(r"$\rho\,(\mathrm{network\ resp.,\ rank})$", fontsize=7.5)
    axC.set_ylim(-1.0, 1.0)
    axC.set_title("Hierarchy persists", fontsize=8, loc="left")
    clean_axes(axC)

    for ax, lab in [(axA, "A"), (axB, "B"), (axC, "C")]:
        ax.text(-0.02, 1.12, lab, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="right")

    save_panel(fig, OUT)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
