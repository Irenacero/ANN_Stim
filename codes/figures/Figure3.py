"""
Figure 3 (3x3 layout).

Row 1: L hand, R hand, tongue (motor activation, dorsal view, best EC seed green).
Row 2: L foot, R foot, motor 5-way somatotopy decoding matrix.
Row 3: working-memory 4-way category decoding matrix, and EC-vs-FC reproduction of
       the difference map for the five two-condition tasks.

Motor brains pre-rendered by render_motor_brains.py. FC matrices and accuracies are
in a supplement; decoding accuracies are given in the caption.
Outputs: codes/figures/outputs/Figure3.{svg,pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import setup, save_panel, figsize_mm, YEO7_COLORS, DIV_CMAP

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "codes/HCP/results"
M5 = Path(__file__).resolve().parent / "outputs" / "_motor5"
OUT = Path(__file__).resolve().parent / "outputs" / "Figure3"

BLAB = {"LANGUAGE": ("Language", "Default"), "SOCIAL": ("Social", "Default"),
        "EMOTION": ("Emotion", "Vis"), "RELATIONAL": ("Relational", "Cont"),
        "GAMBLING": ("Gambling", "Default")}
BRAINS = [(0, "L hand"), (1, "R hand"), (4, "tongue"), (2, "L foot"), (3, "R foot")]


def short_seed(name):
    return str(name).replace("RH ", "R ").replace("LH ", "L ")


def heatmap(ax, M, conds, title, ylabel):
    vm = np.abs(M).max()
    im = ax.imshow(M, cmap=DIV_CMAP, vmin=-vm, vmax=vm, aspect="equal")
    for i in range(len(conds)):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor="black", lw=1.2, zorder=5))
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=6)
    ax.set_xlabel("Virtually stimulated region (rest)", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=3)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("evoked-map match  ($r$)", fontsize=6); cb.ax.tick_params(labelsize=5.5)


def main():
    setup()
    dM = np.load(RES / "decoding_MOTOR.npz", allow_pickle=True)
    dW = np.load(RES / "decoding_WM.npz", allow_pickle=True)
    seednames = {i: short_seed(n) for i, n in enumerate(dM["ec_seed_names"])}
    scal = pd.read_csv(RES / "task_ec_scalars.csv")

    fig = plt.figure(figsize=figsize_mm(150, 152))
    gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.26, height_ratios=[1, 1, 1])

    # ---- motor brains ----
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for (idx, ttl), (r, c) in zip(BRAINS, cells):
        ax = fig.add_subplot(gs[r, c]); ax.imshow(plt.imread(M5 / f"{idx}.png")); ax.axis("off")
        ax.set_title(ttl, fontsize=7.5, pad=0)
        ax.text(0.5, -0.02, seednames[idx], transform=ax.transAxes, ha="center", va="top",
                fontsize=5.6, color="0.35")
    fig.text(0.012, 0.99, "A", fontsize=12, fontweight="bold", va="top")
    fig.text(0.05, 0.99, "Motor", fontsize=8.5, fontweight="bold", va="top")
    fig.text(0.05, 0.955, "task-activation maps", fontsize=6.8, style="italic", color="0.3", va="top")

    # ---- motor decoding matrix ----
    heatmap(fig.add_subplot(gs[1, 2]), dM["M"], [str(c) for c in dM["conds"]],
            "Somatotopy", "Movement performed (task)")

    # ---- WM decoding matrix ----
    axW = fig.add_subplot(gs[2, 0])
    heatmap(axW, dW["M"], [str(c) for c in dW["conds"]], "WM categories", "Category viewed (task)")
    axW.text(-0.55, 1.12, "B", transform=axW.transAxes, fontsize=12, fontweight="bold", va="top")

    # ---- EC vs FC dumbbell, narrowed and pushed right ----
    g2 = gs[2, 1:3].subgridspec(1, 2, width_ratios=[0.32, 1.0], wspace=0)
    axB = fig.add_subplot(g2[1])
    five = ["LANGUAGE", "SOCIAL", "EMOTION", "RELATIONAL", "GAMBLING"]
    st = scal[scal.task.isin(five)].groupby("task").agg(
        ec=("t1_ec", "median"), fc=("t1_fc", "median")).reindex(five).reset_index()
    st = st.sort_values("ec").reset_index(drop=True)
    for i, r in st.iterrows():
        col = YEO7_COLORS[BLAB[r.task][1]]
        axB.plot([r.fc, r.ec], [i, i], color="0.6", lw=1.4, zorder=2)
        axB.scatter(r.fc, i, s=28, facecolor="white", edgecolor="0.45", lw=1.0, zorder=3)
        axB.scatter(r.ec, i, s=42, facecolor=col, edgecolor="0.2", lw=0.6, zorder=4)
    axB.set_yticks(range(len(st))); axB.set_yticklabels([BLAB[t][0] for t in st.task], fontsize=7)
    axB.set_xlabel(r"Reproduction of difference map ($\rho$)", fontsize=7)
    axB.set_xlim(0.44, 0.88); axB.set_xticks([0.5, 0.7]); axB.margins(y=0.18)
    axB.tick_params(labelsize=6.3)
    axB.scatter([], [], s=28, facecolor="white", edgecolor="0.45", lw=1.0, label="FC")
    axB.scatter([], [], s=42, facecolor="0.5", edgecolor="0.2", lw=0.6, label="EC")
    axB.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.2)
    axB.set_title("Other tasks: EC vs FC", fontsize=8, fontweight="bold", loc="left")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.text(-0.22, 1.08, "C", transform=axB.transAxes, fontsize=12, fontweight="bold", va="bottom")

    save_panel(fig, OUT)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
