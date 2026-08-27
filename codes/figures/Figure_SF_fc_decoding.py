"""
Supplementary figure: the same within-task decoding as Figure 3A,B but using
resting FUNCTIONAL connectivity instead of effective connectivity. The diagonal is
much weaker for motor (FC 42% vs EC 91%) and only modestly weaker for working
memory (FC 58% vs EC 68%).

Output: codes/figures/outputs/Figure_SF_fc_decoding.{svg,pdf,png}
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import setup, save_panel, figsize_mm, DIV_CMAP

RES = Path(__file__).resolve().parents[2] / "codes/HCP/results"
OUT = Path(__file__).resolve().parent / "outputs" / "Figure_SF_fc_decoding"


def heatmap(ax, M, conds, title, acc, chance, ylabel):
    vm = np.abs(M).max()
    im = ax.imshow(M, cmap=DIV_CMAP, vmin=-vm, vmax=vm, aspect="equal")
    for i in range(len(conds)):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor="black", lw=1.2, zorder=5))
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=6.5)
    ax.set_xlabel("Virtually stimulated region (rest)", fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{title}  (FC {acc*100:.0f}%, chance {chance:.0f}%)",
                 fontsize=8, fontweight="bold", loc="left", pad=4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("evoked-map match  ($r$)", fontsize=6.5); cb.ax.tick_params(labelsize=6)


def main():
    setup()
    dM = np.load(RES / "decoding_MOTOR.npz", allow_pickle=True)
    dW = np.load(RES / "decoding_WM.npz", allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize_mm(150, 62))
    heatmap(axes[0], dM["M_fc"], [str(c) for c in dM["conds"]], "Motor",
            float(np.mean(dM["fc_acc"])), 20, "Movement performed (task)")
    heatmap(axes[1], dW["M_fc"], [str(c) for c in dW["conds"]], "Working memory",
            float(np.mean(dW["fc_acc"])), 25, "Category viewed (task)")
    fig.tight_layout()
    save_panel(fig, OUT)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
