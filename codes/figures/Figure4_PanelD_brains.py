"""
Figure 4 Panel D -- Whole-brain state vs local activity (topographies).

Two cortical maps of the variance in the global effect size explained, per
target region, by:
  Left  "global -> global": the brain's global ongoing energy E_t.
  Right "local -> global":  the target region's own local energy e_j(t).

Each map has its *own* plasma color scale (the two R2 ranges barely overlap --
global ~0.16-0.20, local ~0.03-0.07 -- so a shared scale would flatten the
local map), letting the within-map topography show. The absolute difference
(global predicts the response, local barely does) is stated by the colorbar
ranges. Per-region values are averaged across HCP subjects.

Input
    codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl

Output
    codes/figures/outputs/Figure4_PanelD_brains.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm
from brain_render import schaefer_parcels_to_vertices, render_hemi_lateral_array
setup()

ROOT = Path(__file__).resolve().parents[2]
HCP_DF = ROOT / "codes/HCP/results/dataframes/HCP_5_df_spatial_network_receptors_ECts_cortical400.pkl"
OUT_DIR = ROOT / "codes/figures/outputs"

CMAP = "plasma"
MAPS = [
    ("r2_global_energy_to_global_effect_size", r"global $\rightarrow$ global"),
    ("r2_local_energy_to_global_effect_size",  r"local $\rightarrow$ global"),
]


def crop_transparent(arr, pad=4):
    if arr.shape[-1] != 4:
        return arr
    a = arr[..., 3] > 0
    rows, cols = np.any(a, axis=1), np.any(a, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return arr[max(0, r0-pad):min(arr.shape[0], r1+pad),
              max(0, c0-pad):min(arr.shape[1], c1+pad)]


def main():
    df = pd.read_pickle(HCP_DF)

    fig = plt.figure(figsize=figsize_mm(90, 50))
    # Each column: a brain (top) above its own colorbar (bottom).
    cols_x = [0.03, 0.53]
    brain_w = 0.44
    for x0, (col, title) in zip(cols_x, MAPS):
        vals = df.groupby("roi")[col].mean().reindex(range(400)).to_numpy()
        # Per-map scale (robust min/max) so the within-map topography shows.
        vmin = float(np.nanpercentile(vals, 3))
        vmax = float(np.nanpercentile(vals, 97))

        vmaps = schaefer_parcels_to_vertices(vals)
        arr = crop_transparent(render_hemi_lateral_array(
            "lh", vmaps["lh"], cmap=CMAP, color_range=(vmin, vmax),
            zoom=1.3, scale=(3, 3)))
        ax = fig.add_axes([x0, 0.28, brain_w, 0.66])
        ax.imshow(arr); ax.set_axis_off()
        ax.set_title(title, pad=3)

        cax = fig.add_axes([x0 + 0.04, 0.13, brain_w - 0.08, 0.05])
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(vmin, vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(r"Variance explained $R^2$", labelpad=2)
        cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cb.ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        cb.outline.set_visible(False)
        cb.ax.tick_params(width=0.7, length=2.5)
        print(f"  {col}: median R2 = {np.nanmedian(vals):.3f}  "
              f"scale [{vmin:.3f}, {vmax:.3f}]")

    save_panel(fig, OUT_DIR / "Figure4_PanelD_brains")


if __name__ == "__main__":
    main()
