"""
Figure 4 -- Manifold-zoom schematics of the gating mechanism (surfaces only).

Two oblique wireframe zooms onto the Figure-1 manifold:
  flat : a near-flat stretch of the manifold (high-energy / out-of-well state).
  well : a single potential well (low-energy / in-attractor state).

Only the surfaces are drawn here -- the trajectories / kick arrows are added by
hand afterwards. Same wireframe look and viewing angle as the Figure-1 manifold
(front edge horizontal, low oblique view, a touch more from the top-front).

Output
    codes/figures/outputs/Figure4_manifold_flat.{svg,pdf,png}
    codes/figures/outputs/Figure4_manifold_well.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
setup()

OUT_DIR = HERE / "outputs"
MESH = "#4d4d4d"           # wireframe (matches Figure-1 manifold)

X_RANGE = (-6.0, 6.0)
Y_RANGE = (-4.5, 4.5)


def surf_flat(x, y):
    """Gentle ripples -- essentially flat compared with the well depth."""
    return 0.12 * np.sin(0.7 * x + 0.3) + 0.10 * np.cos(0.6 * y)


def surf_well(x, y):
    """A single deep Gaussian basin centered at the origin."""
    return -4.0 * np.exp(-((x ** 2) / (2 * 1.7 ** 2) + (y ** 2) / (2 * 1.6 ** 2)))


def build(kind: str):
    surf = surf_flat if kind == "flat" else surf_well

    fig = plt.figure(figsize=figsize_mm(70, 56))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    nx, ny = 121, 71
    xs = np.linspace(*X_RANGE, nx)
    ys = np.linspace(*Y_RANGE, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = surf(Xg, Yg)
    ax.plot_wireframe(Xg, Yg, Zg, rstride=2, cstride=2, color=MESH,
                      linewidth=0.4, alpha=0.6)

    # Same perspective as the Figure-1 manifold: azim=180 keeps the front
    # (lowest) edge horizontal; elev a little higher for a more top-front view.
    ax.view_init(elev=37, azim=180)
    ax.set_proj_type("persp")
    ax.set_box_aspect((X_RANGE[1] - X_RANGE[0],
                       (Y_RANGE[1] - Y_RANGE[0]) * 1.75, 7.0))
    ax.set_zlim(-5.6, 2.9)
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_panel(fig, OUT_DIR / f"Figure4_manifold_{kind}")


def main():
    build("flat")
    build("well")


if __name__ == "__main__":
    main()
