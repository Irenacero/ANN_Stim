"""
Figure 3 — top methods panel: how task-activation maps are extracted.

A wide, short horizontal schematic (≈A4 text width) meant to sit ON TOP of the
existing Figure 3, explaining (motor task as the worked example) the pipeline used
in task_condition_data.py / motor_somatotopy.py:

  1. Task block design  — REAL carpet (400 ROIs × time) + plasma condition blocks
  2. Hemodynamic lag    — +5 s shift picks "during" (in-block) vs "fixation" volumes
  3. Block-average      — REAL columns ⟨x⟩_blk, ⟨x⟩_fix (length 400) → a_c = blk − fix
  4. Demean across cond — REAL a_c, ã_c as 400-ROI × 5-condition matrices; ã_c = a_c − ā

Real inputs are precomputed by HCP/src/methods_panel_data.py (run with the task
drive mounted) into results/methods_panel_data.npz:
  carpet = one subject's z-scored cortical BOLD, ROIs ordered by Yeo-7 network;
  stage3 = group-mean (n=20) L-hand block mean, fixation mean, and their difference;
  A / Ad = group raw per-condition a_c (400×5) and its demean across conditions.
Numbers: Schaefer-400 cortical parcels, TR = 0.72 s, lag 5 s ≈ 7 vols, C = 5.
Only the stage-2 HRF/lag cartoon is schematic.

Outputs: codes/figures/outputs/Figure3_methods_taskmaps.{svg,pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from style import setup, save_panel, figsize_mm, rounded_box, DIV_CMAP, INK, GUIDE, MUTE

ROOT = Path(__file__).resolve().parents[2]
DATA = np.load(ROOT / "codes/HCP/results/methods_panel_data.npz", allow_pickle=True)
OUT = Path(__file__).resolve().parent / "outputs" / "Figure3_methods_taskmaps"

CABBR = ["Lh", "Rh", "Lf", "Rf", "T"]
PLASMA = plt.get_cmap("plasma")
PCOL = [PLASMA(x) for x in np.linspace(0.12, 0.86, 5)]   # 5 plasma shades for conditions

# ---- stage card geometry (figure fraction) ------------------------------
X0, GAP = 0.015, 0.030
WIDTHS = [0.285, 0.180, 0.150, 0.230]                    # carpet, lag, columns, matrices
XS = [X0]
for w in WIDTHS[:-1]:
    XS.append(XS[-1] + w + GAP)
GY0, GH = 0.165, 0.555                                    # graphic axes band
TITLES = ["Task block design", "Hemodynamic lag", "Block-average contrast",
          "Demean across conditions"]
CAPS = ["real BOLD carpet (ROIs × time);  plasma = conditions",
        r"averaging window = 12 s block, shifted $+5\,$s",
        r"$a_c=\langle x\rangle_{\rm blk}-\langle x\rangle_{\rm fix}$  (length 400)",
        r"$\tilde a_c=a_c-\bar a$   removes shared component"]


def stage_axes(i):
    ax = plt.gcf().add_axes([XS[i], GY0, WIDTHS[i], GH]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def stage1():
    ax = stage_axes(0)
    C = DATA["carpet"]                                   # (400, T) z-scored, net-ordered
    ax.imshow(C, cmap=DIV_CMAP, aspect="auto", extent=(0, 1, 0.10, 0.80),
              vmin=-2, vmax=2, interpolation="nearest", rasterized=True)
    # design strip: grey fixation bar + real plasma condition blocks + labels
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.12, fc="#d9d9d9", ec="none"))
    for o, w, ci in zip(DATA["blk_on"], DATA["blk_w"], DATA["blk_c"]):
        ax.add_patch(plt.Rectangle((o, 0.86), w, 0.12, fc=PCOL[int(ci)], ec="none"))
        ax.text(o + w / 2, 1.0, CABBR[int(ci)], ha="center", va="bottom",
                fontsize=4.2, color="0.2")
    ax.text(0.012, 0.91, "fix", fontsize=4.6, color="0.4", ha="left", va="center")
    ax.text(-0.015, 0.45, "ROIs", rotation=90, ha="right", va="center", fontsize=6,
            color="0.4")
    # time axis (seconds): T volumes x TR 0.72 s
    Tdur = C.shape[1] * 0.72
    for s in (0, 60, 120, 180):
        ax.plot([s / Tdur, s / Tdur], [0.06, 0.03], color="0.4", lw=0.7)
        ax.text(s / Tdur, 0.0, str(s), ha="center", va="top", fontsize=4.8, color="0.4")
    ax.plot([0, 180 / Tdur], [0.06, 0.06], color="0.4", lw=0.7)
    ax.text(1.0, 0.0, " time (s)", ha="left", va="top", fontsize=4.8, color="0.4")


def stage2():
    ax = stage_axes(1)
    TMAX = 30.0
    X = lambda s: s / TMAX                               # seconds -> axis fraction
    on_s, off_s, lag = 6.0, 18.0, 5.0
    dw0, dw1 = on_s + lag, off_s + lag                    # 11 s, 23 s (lagged window)
    # fixation grey = volumes outside the lagged block
    ax.axvspan(0.0, X(dw0), color=MUTE, alpha=0.6)
    ax.axvspan(X(dw1), 1.0, color=MUTE, alpha=0.6)
    # the averaged "during" window (12 s, lagged)
    ax.add_patch(plt.Rectangle((X(dw0), 0.10), X(dw1) - X(dw0), 0.62, fc=PCOL[0],
                               ec="none", alpha=0.18))
    ax.text(X((dw0 + dw1) / 2), 0.665, "during (12 s)", ha="center", fontsize=5.0, color="0.2")
    # stimulus block (12 s) on top
    ax.add_patch(plt.Rectangle((X(on_s), 0.84), X(off_s) - X(on_s), 0.10, fc=PCOL[0], ec="none"))
    ax.text(X((on_s + off_s) / 2), 0.89, "block (12 s)", ha="center", va="center",
            fontsize=4.8, color="white")
    # onset / during-onset markers + the +5 s lag arrow between them
    for s in (on_s, dw0):
        ax.plot([X(s), X(s)], [0.05, 0.82], color="0.55", lw=0.6, ls=(0, (2, 2)))
    ax.annotate("", xy=(X(dw0), 0.78), xytext=(X(on_s), 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK))
    ax.text(X((on_s + dw0) / 2), 0.80, r"$+5\,$s", ha="center", va="bottom",
            fontsize=5.6, color=INK)
    # schematic HRF
    t = np.linspace(0, TMAX, 300)
    d = np.clip(t - on_s, 0, None)
    h = d ** 2 * np.exp(-d / 2.2); h = h / (h.max() + 1e-9)
    ax.plot(t / TMAX, 0.14 + 0.52 * h, color=INK, lw=1.0)
    ax.text(X(2.5), 0.40, "fix", ha="center", fontsize=5.4, color="0.45")
    # time axis (seconds)
    ax.plot([0, 1], [0.0, 0.0], color="0.4", lw=0.7)
    for s in (0, 10, 20, 30):
        ax.plot([X(s), X(s)], [0.0, -0.03], color="0.4", lw=0.7)
        ax.text(X(s), -0.06, str(s), ha="center", va="top", fontsize=4.8, color="0.4")
    ax.text(1.02, -0.06, "time (s)", ha="left", va="top", fontsize=4.8, color="0.4")


def _col(ax, v, x0, x1, y0=0.10, y1=0.90):
    vm = np.percentile(np.abs(v), 97)
    ax.imshow(v[:, None], cmap=DIV_CMAP, aspect="auto", extent=(x0, x1, y0, y1),
              vmin=-vm, vmax=vm, interpolation="nearest", rasterized=True)


def stage3():
    ax = stage_axes(2)
    _col(ax, DATA["stage3_blk"], 0.12, 0.26)
    _col(ax, DATA["stage3_fix"], 0.40, 0.54)
    _col(ax, DATA["stage3_ac"], 0.78, 0.92)
    ax.text(0.19, 0.95, r"$\langle x\rangle_{\rm blk}$", ha="center", fontsize=6)
    ax.text(0.47, 0.95, r"$\langle x\rangle_{\rm fix}$", ha="center", fontsize=6)
    ax.text(0.85, 0.95, r"$a_c$", ha="center", fontsize=6.5)
    ax.text(0.33, 0.50, "−", ha="center", va="center", fontsize=11, color="0.3")
    ax.text(0.66, 0.50, "=", ha="center", va="center", fontsize=11, color="0.3")
    ax.text(0.03, 0.50, "ROIs", rotation=90, ha="right", va="center", fontsize=6,
            color="0.4")


def _matrix(ax, M, x0, x1, y0=0.30, y1=0.95):
    """M is (nR, 5): ROIs × conditions; heatmap + black column dividers + plasma chips."""
    vm = np.percentile(np.abs(M), 97)
    ax.imshow(M, cmap=DIV_CMAP, aspect="auto", extent=(x0, x1, y0, y1), vmin=-vm, vmax=vm,
              interpolation="nearest", rasterized=True)
    w = (x1 - x0) / 5
    for k in range(1, 5):
        ax.plot([x0 + k * w, x0 + k * w], [y0, y1], color="black", lw=0.9)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=0.9))
    for k in range(5):                                   # plasma chips + labels below
        ax.add_patch(plt.Rectangle((x0 + k * w + 0.004, y0 - 0.11), w - 0.008, 0.07,
                                   fc=PCOL[k], ec="none"))
        ax.text(x0 + (k + 0.5) * w, y0 - 0.185, CABBR[k], ha="center", va="top",
                fontsize=4.8, color="0.3")


def stage4():
    ax = stage_axes(3)
    _matrix(ax, DATA["A"], 0.05, 0.40)
    _matrix(ax, DATA["Ad"], 0.60, 0.95)
    ax.text(0.225, 0.99, r"$a_c$", ha="center", fontsize=6.5)
    ax.text(0.775, 0.99, r"$\tilde a_c$", ha="center", fontsize=6.5)
    ax.annotate("", xy=(0.58, 0.62), xytext=(0.42, 0.62),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK))
    ax.text(0.50, 0.70, r"$-\,\bar a$", ha="center", fontsize=6, color=INK)
    ax.text(-0.01, 0.62, "ROIs", rotation=90, ha="right", va="center", fontsize=6,
            color="0.4")


def main():
    setup()
    plt.figure(figsize=figsize_mm(185, 60))
    fig = plt.gcf()
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    for i, x in enumerate(XS):
        w = WIDTHS[i]
        rounded_box(bg, x - 0.006, 0.075, w + 0.012, 0.745,
                    fc="#fbfbfd", ec=GUIDE, lw=0.7, rounding=0.015)
        bg.text(x + 0.002, 0.755, f"{i+1}", fontsize=8.5, fontweight="bold",
                color=INK, va="center", ha="left")
        bg.text(x + 0.026, 0.755, TITLES[i], fontsize=6.8, fontweight="bold",
                color=INK, va="center", ha="left")
        bg.text(x + w / 2, 0.035, CAPS[i], fontsize=5.4, color="0.35",
                va="center", ha="center")

    for i in range(3):
        bg.annotate("", xy=(XS[i + 1] - 0.010, 0.44), xytext=(XS[i] + WIDTHS[i] + 0.004, 0.44),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))

    bg.text(0.001, 0.99, "A", fontsize=12, fontweight="bold", va="top")
    bg.text(0.032, 0.965, "Extraction of task-activation maps",
            fontsize=8.5, fontweight="bold", va="top")
    bg.text(0.40, 0.965, "(motor task as worked example)", fontsize=6.5,
            style="italic", color="0.4", va="top")

    stage1(); stage2(); stage3(); stage4()
    save_panel(fig, OUT)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
