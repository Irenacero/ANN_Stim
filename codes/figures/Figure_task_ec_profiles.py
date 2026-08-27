"""
Supplementary figure S3.1 — per-task EC-vs-activation profile (task-specific).

For each task (one row): a left label (task, canonical network, peak region) and,
to its right, every cortical region (grouped by Yeo-7 network, unimodal ->
transmodal, colored by network) plotted against its task-specific reproduction
score dm_EC = m_EC(task) - mean_over_tasks(m_EC). Subtracting each region's
across-task average removes the connector-hub baseline and isolates which regions
reproduce THIS task's activation more than tasks in general. Per-network means are
overlaid; the peak region (best task-specific target) is circled.

Inputs : codes/HCP/results/task_ec_profiles.npz
Outputs: codes/figures/outputs/Figure_task_ec_profiles.{svg,pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from style import setup, save_panel, figsize_mm, YEO7_COLORS

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "codes/HCP/results"
NPZ = RES / "task_ec_profiles.npz"
NETCSV = RES / "dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.csv"
OUT = Path(__file__).resolve().parent / "outputs" / "Figure_task_ec_profiles"

YEO7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Cont", "Default", "Limbic"]
PRETTY = {"Vis": "VIS", "SomMot": "SMN", "DorsAttn": "DAN", "SalVentAttn": "SN",
          "Cont": "FPN", "Default": "DMN", "Limbic": "LN"}
# key, name, contrast, canonical network acronym
TASKS = [("MOTOR_HF", "Motor", "hand − foot", "SMN"),
         ("WM", "Working memory", "2bk − 0bk", "FPN"),
         ("RELATIONAL", "Relational", "relation − match", "FPN"),
         ("LANGUAGE", "Language", "story − math", "DMN"),
         ("EMOTION", "Emotion", "faces − shapes", "VIS"),
         ("SOCIAL", "Social", "mental − random", "DMN"),
         ("GAMBLING", "Gambling", "win − loss", "DMN")]
RNG = np.random.default_rng(0)


def darken(color, f=0.6):
    r, g, b = mcolors.to_rgb(color)
    return (r * f, g * f, b * f)


def short_name(full):
    p = full.split("_")
    return " ".join(p[1:]) if len(p) > 1 else full


def main():
    setup()
    d = np.load(NPZ, allow_pickle=True)
    names = d["roi_names"]
    net = pd.read_csv(NETCSV, usecols=["roi", "rsn_network"]).drop_duplicates("roi")
    net = net.sort_values("roi")["rsn_network"].to_numpy()
    base = np.mean([d[f"mEC_{k}"] for k, *_ in TASKS], axis=0)

    order = np.concatenate([np.where(net == n)[0] for n in YEO7])
    net_ord = net[order]
    bounds, centers = [0], []
    for n in YEO7:
        c = int(np.sum(net == n))
        centers.append(bounds[-1] + c / 2)
        bounds.append(bounds[-1] + c)

    fig = plt.figure(figsize=figsize_mm(130, 16 * len(TASKS)))
    gs = fig.add_gridspec(len(TASKS), 2, width_ratios=[0.92, 3.3], wspace=0.16, hspace=0.30)
    x = np.arange(len(order))
    jit = RNG.uniform(-0.32, 0.32, len(order))
    colors = [YEO7_COLORS[n] for n in net_ord]
    scat = []

    for r, (key, name, contrast, acr) in enumerate(TASKS):
        ds = (d[f"mEC_{key}"] - base)[order]
        pk = int(np.argmax(ds))

        # ---- left label ----
        axl = fig.add_subplot(gs[r, 0]); axl.axis("off")
        axl.text(0.0, 0.72, name, transform=axl.transAxes,
                 fontsize=8, fontweight="bold", va="center")
        axl.text(0.0, 0.47, f"{contrast}  ·  {acr}", transform=axl.transAxes,
                 fontsize=6.8, color="0.35", va="center")
        axl.text(0.0, 0.22, f"peak: {short_name(names[order][pk])}", transform=axl.transAxes,
                 fontsize=6.3, color="0.15", va="center")

        # ---- right scatter ----
        ax = fig.add_subplot(gs[r, 1], sharex=scat[0] if scat else None); scat.append(ax)
        ax.set_facecolor("white")
        ax.axhline(0, color="0.75", lw=0.6, zorder=1)
        for b in bounds[1:-1]:
            ax.axvline(b - 0.5, color="0.9", lw=0.6, zorder=0)
        ax.scatter(x + jit, ds, c=colors, s=8, lw=0.15, edgecolor="0.3", zorder=3)
        for b0, b1, n in zip(bounds[:-1], bounds[1:], YEO7):
            ax.hlines(ds[b0:b1].mean(), b0 - 0.5, b1 - 0.5, color=darken(YEO7_COLORS[n]),
                      lw=2.4, zorder=5)
        ax.scatter([x[pk] + jit[pk]], [ds[pk]], s=36, facecolor="none",
                   edgecolor="0.05", lw=1.0, zorder=6)
        ax.set_ylabel(r"$\Delta m_{EC}$", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.margins(x=0.01, y=0.28)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r < len(TASKS) - 1:
            ax.tick_params(labelbottom=False)

    scat[-1].set_xticks(centers)
    scat[-1].set_xticklabels([PRETTY[n] for n in YEO7], fontsize=7)
    scat[-1].set_xlabel("Stimulated cortical region (grouped by network, unimodal → transmodal)",
                        fontsize=8)
    fig.align_ylabels(scat)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.99, bottom=0.085)
    save_panel(fig, OUT)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
