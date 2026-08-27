"""
Figure 1 panel A, right-most subpanel -- Seed-based EC.

Plots one row of the time-resolved effective connectivity tensor on the
cortical surface for a single HCP subject and one initial time t. The seed
parcel is marked with a lightning bolt + dark dot.

Inputs
    codes/HCP/results/ECts/id_100206_ECt.npy   (500, 450, 450) float64

Output
    codes/figures/outputs/Figure1_PanelA_seedEC.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel
from brain_render import (
    N_SUBCORTICAL,
    plot_lh_lateral,
    schaefer_parcels_to_vertices,
)
setup()

ROOT = Path(__file__).resolve().parents[2]
EC_NPY = ROOT / "codes/HCP/results/ECts/id_100206_ECt.npy"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_FIXED = 200
SEED_NAME = "7Networks_LH_Default_Par_2"  # left TPJ / angular gyrus (~MNI -46 -58 20)


def parcel_id_by_name(label_txt: Path, name: str) -> int:
    lines = label_txt.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == name:
            return int(lines[i + 1].split()[0])
    raise ValueError(f"{name!r} not in {label_txt}")


EC_CACHE = ROOT / "codes/HCP/results/ECts_cache/id_100206_ECt_cache.npz"


def main():
    seed_pid = parcel_id_by_name(LABEL_TXT, SEED_NAME)
    seed_idx = seed_pid - 1
    if EC_NPY.exists():
        ec = np.load(EC_NPY, mmap_mode="r")
        row_cort = np.asarray(ec[T_FIXED, seed_idx])[N_SUBCORTICAL:]
    else:  # seed-row cache shipped with the figure bundle (same values)
        d = np.load(EC_CACHE)
        assert int(d["seed_idx"]) == seed_idx
        row_cort = d["seed_rows_tpj"][T_FIXED][N_SUBCORTICAL:]

    # vmax slightly above the strongest off-target effect: seed saturates at
    # the cmap extreme without flattening everything else.
    seed_cort_idx = seed_idx - N_SUBCORTICAL
    off_mask = np.ones_like(row_cort, dtype=bool)
    off_mask[seed_cort_idx] = False
    vlim = float(np.max(np.abs(row_cort[off_mask]))) * 1.15
    print(
        f"id_100206  t={T_FIXED}  seed={SEED_NAME}  idx0={seed_idx}  vlim={vlim:.4g}"
    )

    vmaps = schaefer_parcels_to_vertices(row_cort)
    # cmap defaults to brightRdBu (brain_render); explicit for clarity.
    fig = plot_lh_lateral(vmaps["lh"], color_range=(-vlim, vlim), cmap="brightRdBu")
    save_panel(fig, OUT_DIR / "Figure1_PanelA_seedEC")


if __name__ == "__main__":
    main()
