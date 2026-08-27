"""
Conceptual attractor / energy landscape with a single trajectory.

A flat sheet punched by several Gaussian basins of varying depth and width,
drawn as a fine blue wireframe and viewed at a low oblique angle. A black
trajectory rides over the surface, weaving between basins and ending in an
arrowhead; a dashed arrow feeds in from the front, and dotted guides drop
from above onto selected basins (schematic "state -> basin" pointers).

Style: project INK for the trajectory, GUIDE for dashed/dotted guides, a soft
blue for the mesh (this is a schematic, so the data colormaps don't apply).

Output
    codes/figures/outputs/manifold_landscape.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, INK, GUIDE  # noqa: E402

setup()

OUT_DIR = HERE / "outputs"
MESH_BLUE = "#4d4d4d"            # dark gray wireframe

# ----------------------------------------------------------------------------
# 1. The landscape: flat sheet (z=0) minus a set of Gaussian basins.
#    Each well: (center_x, center_y, depth, sigma_x, sigma_y).
# ----------------------------------------------------------------------------
WELLS = [
    (-4.5, -1.6, 5.2, 0.75, 0.70),   # deep, narrow  (front-left)
    (-1.6,  0.5, 2.2, 1.10, 1.10),   # medium
    ( 0.9, -1.1, 1.5, 1.45, 1.30),   # shallow, wide
    ( 3.1,  0.2, 3.0, 0.90, 0.95),   # medium-deep   (right)
    ( 5.0, -0.7, 1.7, 0.80, 0.80),   # small
]
X_RANGE = (-6.0, 6.0)
Y_RANGE = (-4.6, 4.6)


def surface(x, y):
    """Energy surface z(x, y): zero plane minus the Gaussian basins."""
    z = np.zeros(np.broadcast(x, y).shape)
    for cx, cy, amp, sx, sy in WELLS:
        z = z - amp * np.exp(-(((x - cx) ** 2) / (2 * sx ** 2)
                               + ((y - cy) ** 2) / (2 * sy ** 2)))
    return z


# ----------------------------------------------------------------------------
# 2. Trajectory: smooth Catmull-Rom path through waypoints, riding the surface.
# ----------------------------------------------------------------------------
def catmull_rom(points, n_per_seg=60):
    pts = np.asarray(points, float)
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t = np.linspace(0, 1, n_per_seg)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


# Start in the flat top-left corner (data x~+5.8, y~-3.8 == screen top-left),
# wander there, then sweep diagonally down through the sheet, dive through the
# largest basin W0 (-4.5,-1.6) and end as it climbs back out the far side.
WAYPOINTS = [(5.8, -3.8), (5.0, -3.0), (4.2, -3.8), (3.0, -3.2), (1.8, -2.6),
             (0.0, -1.4), (-1.8, -1.9), (-3.2, -1.7), (-4.5, -1.6), (-5.7, -3.3)]
RIDE_HEIGHT = 0.22               # how far the path floats above the surface


# ----------------------------------------------------------------------------
# 3. A 3D arrow patch (clean 2D-style arrowhead that respects the projection).
# ----------------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._xyz
        x2, y2, z2 = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((x2[0], y2[0]), (x2[1], y2[1]))
        return float(np.min(z2))


def add_arrow3d(ax, p0, p1, *, color, lw, mutation_scale=11, ls="-", zorder=10):
    a = Arrow3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>",
                color=color, linestyle=ls, zorder=zorder)
    ax.add_artist(a)
    return a


# ----------------------------------------------------------------------------
# 4. Build the panel.
# ----------------------------------------------------------------------------
def build_panel(fig):
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # --- surface wireframe ---
    nx, ny = 141, 81
    xs = np.linspace(*X_RANGE, nx)
    ys = np.linspace(*Y_RANGE, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    Zg = surface(Xg, Yg)
    ax.plot_wireframe(Xg, Yg, Zg, rstride=2, cstride=2,
                      color=MESH_BLUE, linewidth=0.4, alpha=0.6)

    # --- trajectory on the surface ---
    path = catmull_rom(WAYPOINTS, n_per_seg=70)
    px, py = path[:, 0], path[:, 1]
    pz = surface(px, py) + RIDE_HEIGHT
    ax.plot(px[:-4], py[:-4], pz[:-4], color=INK, linewidth=1.4,
            solid_capstyle="round", zorder=9)
    # arrowhead on the final segment (exiting the largest basin)
    add_arrow3d(ax, (px[-5], py[-5], pz[-5]), (px[-1], py[-1], pz[-1]),
                color=INK, lw=1.4, mutation_scale=10)

    # --- dotted guide lines dropping onto selected basins ---
    z_top = 2.6
    for cx, cy, *_ in (WELLS[0], WELLS[1], WELLS[3]):
        zb = surface(np.array(cx), np.array(cy))
        ax.plot([cx, cx], [cy, cy], [z_top, float(zb)], color=GUIDE,
                linewidth=0.9, linestyle=(0, (1, 2)), zorder=7)

    # --- camera & framing ---
    ax.view_init(elev=34, azim=180)   # x-axis horizontal -> front edge level; still a bit from the top
    ax.set_proj_type("persp")
    ax.set_box_aspect((X_RANGE[1] - X_RANGE[0],
                       (Y_RANGE[1] - Y_RANGE[0]) * 1.75,   # 75% deeper (y), same width & relief
                       7.0))
    ax.set_zlim(-5.6, z_top + 0.3)
    ax.set_axis_off()
    # transparent panes (belt-and-braces; save_panel is transparent anyway)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    return ax


def main():
    fig = plt.figure(figsize=figsize_mm(76, 90))   # taller canvas to hold the deeper plane
    build_panel(fig)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_panel(fig, OUT_DIR / "manifold_landscape")


if __name__ == "__main__":
    main()
