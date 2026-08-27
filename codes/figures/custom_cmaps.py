"""Custom colormaps for publication figures (self-contained).

Two coherent colormap families, baked into bundled ``.npy`` colour tables so
this module is fully portable -- no reference images or network access needed:

  - ``brightRdBu`` : diverging blue -> white -> red, with bright saturated ends.
        Use for anything centred on zero: brain maps, effective-connectivity
        matrices, difference/contrast maps.
  - ``brightPMY``  : sequential purple -> magenta -> orange -> yellow.
        Use for one-sided magnitudes: responsiveness / global effect size,
        and as the gradient source for ordered line stacks.

Each is exposed as a faithful 256-colour map plus a ``_lin`` 9-anchor variant
(same colours, evenly spaced). All four are registered with matplotlib (with
``_r`` reversed variants) so libraries that resolve colormaps by string name
(e.g. surfplot / VTK) can use them too.

Usage
-----
    from custom_cmaps import brightRdBu, brightPMY
    import matplotlib.pyplot as plt
    plt.imshow(data, cmap=brightRdBu)          # or cmap="brightRdBu"
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent
# Colour tables: (256, 3) float arrays sampled across each reference colorbar.
_TABLES = {
    "brightRdBu": _HERE / "brightRdBu_256.npy",
    "brightPMY":  _HERE / "brightPMY_256.npy",
}


def _load(name, n=256):
    """Return ``n`` RGB colors (0-1) for ``name``, resampled from its table."""
    cols = np.load(_TABLES[name]).astype(float)
    idx = np.linspace(0, len(cols) - 1, n).round().astype(int)
    return cols[idx]


def make_cmap(name, n_full=256, n_anchors=9):
    """Build (faithful, linear) LinearSegmentedColormaps from a colour table."""
    faithful = LinearSegmentedColormap.from_list(name, _load(name, n_full))
    anchors = _load(name, n_anchors)
    linear = LinearSegmentedColormap.from_list(name + "_lin", anchors)
    return faithful, linear, anchors


brightRdBu, brightRdBu_lin, _rdbu_anchors = make_cmap("brightRdBu")
brightPMY,  brightPMY_lin,  _pmy_anchors  = make_cmap("brightPMY")

# Register (with reversed _r variants) so they resolve by string name too.
for _cm in (brightRdBu, brightRdBu_lin, brightPMY, brightPMY_lin):
    for _c in (_cm, _cm.reversed()):
        if _c.name not in mpl.colormaps:
            mpl.colormaps.register(_c)


def anchor_hex(name):
    """Hex codes of the evenly spaced anchor colors for a given colormap."""
    from matplotlib.colors import to_hex
    anchors = _rdbu_anchors if name == "brightRdBu" else _pmy_anchors
    return [to_hex(c) for c in anchors]


if __name__ == "__main__":
    for nm in ("brightRdBu", "brightPMY"):
        print(nm, "anchors:", anchor_hex(nm))
