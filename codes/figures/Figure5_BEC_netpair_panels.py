"""
Two standalone Yeo-7 network-pair matrices for the F6 bifocal last row (next to
the CV win-win scatter, Figure5_BEC_scatter_GESvsCV):

    larger_effect      : % of region pairs whose bifocal mean(GES) exceeds the
                         stronger single site (eff_incr > 0), by network pair.
    more_reproducible  : % of region pairs whose bifocal CV(GES) is below the
                         better single site (impr_naive < 0), by network pair.
                         (CV counterpart of the old "less variable" Var panel.)

Each is saved as its own SVG/PDF panel (separable for Inkscape). Reuses the
cohort-summary logic in Figure5_BEC_suppl_var.

Output:
    codes/figures/outputs/Figure5_BEC_netpair_larger_effect.{svg,pdf,png}
    codes/figures/outputs/Figure5_BEC_netpair_more_reproducible.{svg,pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
from Figure5_BEC_suppl_var import (cortical_network_index, network_fraction,
                                   _netmat, LABELS, NPZ, SUB)
setup()

OUT = Path(__file__).resolve().parents[2] / "codes/figures/outputs"


def main():
    d = np.load(NPZ)
    eff = d["eff_incr"].astype(float)[SUB:, SUB:]
    imp = d["impr_naive"].astype(float)[SUB:, SUB:]   # CV % change vs better single
    net_idx = cortical_network_index()

    Feff = network_fraction(eff, net_idx, positive=True)    # bifocal larger effect
    Fcv = network_fraction(imp, net_idx, positive=False)    # bifocal lower CV (more reproducible)

    for F, stem, title, cbar in [
        (Feff, "Figure5_BEC_netpair_larger_effect",
         "Larger effect by network pair", "% pairs larger"),
        (Fcv, "Figure5_BEC_netpair_more_reproducible",
         "More reproducible by network pair", "% pairs more reproducible"),
    ]:
        fig, ax = plt.subplots(figsize=figsize_mm(62, 58), constrained_layout=True)
        _netmat(ax, F, title, cbar)
        save_panel(fig, OUT / stem)
        plt.close(fig)
        diag = np.nanmedian(np.diag(F))
        off = np.nanmedian(F[~np.eye(7, dtype=bool)])
        print(f"{stem}: within-network median={diag:.0f}%  cross-network median={off:.0f}%")
        top = sorted([(i, j) for i in range(7) for j in range(i, 7)],
                     key=lambda t: -F[t[0], t[1]])[:4]
        print("   top pairs: " + ", ".join(f"{LABELS[i]}x{LABELS[j]}={F[i,j]:.0f}%" for i, j in top))


if __name__ == "__main__":
    main()
