"""
Render the 5 motor body-part brains for Figure 3: dorsal (top-down) view, each
hemisphere rendered separately and stacked with almost no gap so the two
hemispheres sit close together. Best EC seed highlighted (green fill + outline),
tightly cropped on transparent background. Cache to outputs/_motor5/.
"""
import io, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sys
HCP = Path(__file__).resolve().parents[1] / "HCP" / "src"
sys.path.insert(0, str(HCP))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import brain_render as br
from surfplot import Plot
from spin_test import load_centroids

RES = Path(__file__).resolve().parents[2] / "codes/HCP/results"
CACHE = Path(__file__).resolve().parent / "outputs" / "_motor5"
CACHE.mkdir(exist_ok=True)
_, _, NAMES = load_centroids()
GAP = 3   # px between hemispheres


def _crop(arr, pad=1):
    m = arr[..., 3] > 0.05
    ys, xs = np.where(m)
    return arr[max(ys.min()-pad, 0):ys.max()+pad+1, max(xs.min()-pad, 0):xs.max()+pad+1] if ys.size else arr


def _hemi(h, a, seedpos):
    av = br.schaefer_parcels_to_vertices(a)
    vmax = np.nanpercentile(np.abs(a), 97)
    side = "left" if h == "lh" else "right"
    p = Plot(surf_lh=br.lh_inflated_surface() if h == "lh" else None,
             surf_rh=br.rh_inflated_surface() if h == "rh" else None,
             views="dorsal", size=(360, 540), zoom=1.7)
    p.add_layer({side: av[h]}, cmap="brightRdBu", color_range=(-vmax, vmax), cbar=False)
    if (h == "lh") == ("_LH_" in str(NAMES[seedpos])):
        fill = np.full(400, np.nan); fill[seedpos] = 1
        fv = br.schaefer_parcels_to_vertices(fill)
        p.add_layer({side: fv[h]}, cmap=ListedColormap(["#15ff00"]), color_range=(0, 1), cbar=False)
        p.add_layer({side: fv[h]}, cmap="binary", color_range=(0, 1), as_outline=True, cbar=False)
    fig = p.build()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return _crop(plt.imread(buf))


def render_motor(a, seedpos):
    lh, rh = _hemi("lh", a, seedpos), _hemi("rh", a, seedpos)
    H = max(lh.shape[0], rh.shape[0])

    def pad(im):
        o = np.zeros((H, im.shape[1], 4), dtype=im.dtype)
        o[(H - im.shape[0]) // 2:(H - im.shape[0]) // 2 + im.shape[0]] = im
        return o
    return np.concatenate([pad(lh), np.zeros((H, GAP, 4), np.float32), pad(rh)], axis=1)


def main():
    d = np.load(RES / "decoding_MOTOR.npz", allow_pickle=True)
    A, seeds, conds = d["activations"], d["ec_seeds"], d["conds"]
    for i in range(5):
        plt.imsave(CACHE / f"{i}.png", np.clip(render_motor(A[i], int(seeds[i])), 0, 1))
        print(f"  rendered {conds[i]}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
