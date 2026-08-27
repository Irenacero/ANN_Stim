"""
Shared cortical-surface rendering for every paper figure.

Importing this module patches brainspace's VTK lighting defaults so all
brain plots in the project share the same soft, near-white look. Importers
should `from brain_render import plot_lh_lateral, schaefer_parcels_to_vertices`
and avoid configuring surfplot/brainspace directly elsewhere.

The standard look:
    - fsLR-32k inflated surface (smoother than midthickness, keeps anatomy)
    - ambient 0.55, diffuse 0.45, specular 0.0 (soft sulcal shading)
    - lateral LH view, near-white background
    - colored parcels via Schaefer-400 7Networks fsLR-32k dlabel
"""
from __future__ import annotations

from pathlib import Path
import urllib.request
from typing import Iterable

import numpy as np
import nibabel as nib

# IMPORTANT: patch brainspace defaults *before* surfplot imports.
import brainspace.plotting.defaults_plotting as _bs_defaults
_bs_defaults.actor_kwds.update({"ambient": 0.55, "diffuse": 0.45, "specular": 0.0})

from brainspace.mesh.mesh_io import read_surface  # noqa: E402
from neuromaps.datasets import fetch_fslr  # noqa: E402
from surfplot import Plot  # noqa: E402

# Registers brightRdBu / brightPMY with matplotlib so they are available by
# string name to surfplot/VTK. Wrapped because brain_render may be imported
# before the figures dir is on sys.path in some entry points.
try:
    import custom_cmaps  # noqa: F401,E402
except Exception:  # pragma: no cover
    pass


# Layout defaults used by every figure unless overridden.
N_SUBCORTICAL = 50
N_LH = 200
N_RH = 200
ATLAS_CACHE = Path(__file__).resolve().parent / "_atlases"
ATLAS_CACHE.mkdir(parents=True, exist_ok=True)


# Yeo-7 ordering by cortical hierarchy (unimodal -> transmodal, Limbic last).
# Cont = Control/Frontoparietal (FPN); SalVentAttn = Salience/Ventral Attention.
# Limbic is placed last as an SNR outlier (see Fig. 5B, "limbic clipped").
YEO7_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                 "Cont", "Default", "Limbic"]


def cortical_network_only_ordering(label_txt: Path):
    """Permutation that orders the 400 cortical parcels by network only:
    within each Yeo-7 network, LH parcels come first followed by RH parcels.
    Returns (perm, boundaries, network_labels):
      - perm: array of length 400, perm[k] = original cortical index at new pos k
      - boundaries: cumulative end indices (one per network, last == 400)
      - network_labels: list of 7 network names in plotted order
    """
    lines = label_txt.read_text().splitlines()
    # Collect (cort_idx, hemi, network) for every cortical parcel
    buckets: dict[str, list[int]] = {n: [] for n in YEO7_NETWORKS}
    cort_idx = -1
    for i in range(0, len(lines), 2):
        name = lines[i].strip()
        if not name.startswith("7Networks_"):
            continue
        cort_idx += 1
        parts = name.split("_")
        net = parts[2]
        buckets[net].append(cort_idx)
    perm: list[int] = []
    boundaries: list[int] = []
    for net in YEO7_NETWORKS:
        perm.extend(buckets[net])
        boundaries.append(len(perm))
    return np.asarray(perm, dtype=int), boundaries, list(YEO7_NETWORKS)


def parse_cortical_network_runs(label_txt: Path):
    """Return list of (hemi, network, start_idx, end_idx_exclusive) in the
    400-element cortical ordering (subcortical excluded)."""
    lines = label_txt.read_text().splitlines()
    runs = []
    last_key = None
    cort_idx = -1
    for i in range(0, len(lines), 2):
        name = lines[i].strip()
        if not name.startswith("7Networks_"):
            continue
        cort_idx += 1
        parts = name.split("_")
        hemi, net = parts[1], parts[2]
        key = (hemi, net)
        if key != last_key:
            if runs:
                runs[-1] = (*runs[-1][:3], cort_idx)
            runs.append((hemi, net, cort_idx, None))
            last_key = key
    runs[-1] = (*runs[-1][:3], cort_idx + 1)
    return runs


# ---- surfaces -----------------------------------------------------------
_FSLR_CACHE: dict[str, object] = {}


def _fetch_fslr_inflated():
    if "inflated" not in _FSLR_CACHE:
        fs = fetch_fslr(density="32k")
        _FSLR_CACHE["inflated"] = (
            read_surface(str(fs["inflated"].L)),
            read_surface(str(fs["inflated"].R)),
        )
    return _FSLR_CACHE["inflated"]


def lh_inflated_surface():
    return _fetch_fslr_inflated()[0]


def rh_inflated_surface():
    return _fetch_fslr_inflated()[1]


# ---- atlas --------------------------------------------------------------
def fetch_schaefer_fslr32k_dlabel() -> Path:
    """Download (once) the Schaefer-400 7Networks fsLR-32k CIFTI dlabel."""
    base = (
        "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/"
        "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/"
        "cifti"
    )
    fname = "Schaefer2018_400Parcels_7Networks_order.dlabel.nii"
    dst = ATLAS_CACHE / fname
    if not dst.exists():
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(f"{base}/{fname}", dst)
    return dst


def schaefer_parcels_to_vertices(
    values_cort: np.ndarray,
    n_vertices_per_hemi: int = 32492,
    dlabel_path: Path | None = None,
):
    """Project a (400,) Schaefer-400 cortical vector onto fsLR-32k vertices.

    Returns {"lh": (32492,), "rh": (32492,)} with NaN on the medial wall.
    """
    if dlabel_path is None:
        dlabel_path = fetch_schaefer_fslr32k_dlabel()
    cifti = nib.load(str(dlabel_path))
    data = np.asarray(cifti.get_fdata(), dtype=int).ravel()
    ax = cifti.header.get_axis(1)

    lh = np.full(n_vertices_per_hemi, np.nan, dtype=float)
    rh = np.full(n_vertices_per_hemi, np.nan, dtype=float)
    for name, slc, model in ax.iter_structures():
        target = lh if "CORTEX_LEFT" in name else rh if "CORTEX_RIGHT" in name else None
        if target is None:
            continue
        for pid_local, vidx in zip(data[slc], model.vertex):
            if pid_local == 0:
                continue
            target[vidx] = values_cort[pid_local - 1]
    return {"lh": lh, "rh": rh}


# ---- rendering ----------------------------------------------------------
def _build_plot(hemi: str, stat: np.ndarray, *, cmap: str, color_range, cbar: bool,
                size: tuple[int, int], zoom: float, brightness: float) -> Plot:
    if hemi == "lh":
        p = Plot(surf_lh=lh_inflated_surface(), surf_rh=None, views="lateral",
                 size=size, zoom=zoom, layout="row", brightness=brightness)
        p.add_layer({"left": stat}, cmap=cmap, color_range=color_range, cbar=cbar)
    elif hemi == "rh":
        p = Plot(surf_lh=None, surf_rh=rh_inflated_surface(), views="lateral",
                 size=size, zoom=zoom, layout="row", brightness=brightness)
        p.add_layer({"right": stat}, cmap=cmap, color_range=color_range, cbar=cbar)
    else:
        raise ValueError(f"hemi must be 'lh' or 'rh', got {hemi!r}")
    return p


def plot_hemi_lateral(
    hemi: str,
    stat: np.ndarray,
    *,
    cmap: str = "brightRdBu",
    color_range: tuple[float, float] | None = None,
    cbar: bool = False,
    size: tuple[int, int] = (600, 480),
    zoom: float = 1.25,
    brightness: float = 1.0,
):
    """Render a lateral hemisphere with the project-standard look."""
    p = _build_plot(hemi, stat, cmap=cmap, color_range=color_range, cbar=cbar,
                    size=size, zoom=zoom, brightness=brightness)
    return p.build(colorbar=False)


def render_hemi_lateral_array(
    hemi: str,
    stat: np.ndarray,
    *,
    cmap: str = "brightRdBu",
    color_range: tuple[float, float] | None = None,
    size: tuple[int, int] = (600, 480),
    zoom: float = 1.25,
    brightness: float = 1.0,
    scale: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Render a lateral hemisphere and return an (H, W, 4) RGBA numpy array.

    Useful for composing multiple brain views into a single matplotlib figure
    without round-tripping through PNG.
    """
    p = _build_plot(hemi, stat, cmap=cmap, color_range=color_range, cbar=False,
                    size=size, zoom=zoom, brightness=brightness)
    plotter = p.render()
    return plotter.to_numpy(transparent_bg=True, scale=scale)


# Backwards-compatible alias for the LH-only helper used in earlier scripts.
def plot_lh_lateral(stat_lh, **kw):
    return plot_hemi_lateral("lh", stat_lh, **kw)


def fig_size_inches(fig) -> tuple[float, float]:
    """Return (width, height) in inches for a Figure or SubFigure.

    Matplotlib's SubFigure has no `get_size_inches()`; this helper falls back
    to the parent figure size scaled by the subfigure's relative bbox.
    """
    if hasattr(fig, "get_size_inches"):
        return tuple(fig.get_size_inches())
    parent = fig.figure
    pw, ph = parent.get_size_inches()
    bb = fig.bbox_relative
    return pw * bb.width, ph * bb.height


def render_neutral_hemi_array(hemi: str, n_vertices: int = 32492,
                              size: tuple[int, int] = (600, 480),
                              zoom: float = 1.25,
                              brightness: float = 1.0,
                              scale: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Render a lateral hemisphere with no data overlay -- a uniformly
    gray cortex used for schematic panels (e.g., observational-signals).
    """
    stat = np.full(n_vertices, np.nan, dtype=float)
    return render_hemi_lateral_array(
        hemi, stat, cmap="Greys", color_range=(0.0, 1.0),
        size=size, zoom=zoom, brightness=brightness, scale=scale,
    )


_BOLT_SHAPE = np.array([
    (0.55, 1.00),  # top tip
    (0.18, 0.50),  # mid-left dent
    (0.42, 0.50),  # mid-inner-left
    (0.18, 0.00),  # bottom tip
    (0.78, 0.55),  # upper-right
    (0.55, 0.55),  # upper-inner-right
    (0.78, 1.00),  # back near top right
])


def add_stim_marker(
    fig,
    xy_axes: tuple[float, float] = (0.18, 0.55),
    dot_radius: float = 0.018,
    bolt_size: float = 0.10,
    bolt_offset: tuple[float, float] = (-0.08, 0.06),
    bolt_color: str = "#f08a1e",
    dot_color: str = "#1a1a1a",
):
    """Overlay a stimulation-site marker (lightning bolt + dot) on a Plot figure.

    `xy_axes` is in axes coordinates of the single brain panel (0,0 lower-left
    to 1,1 upper-right). Hand-tune per seed location.
    """
    from matplotlib.patches import Circle, Polygon
    ax = fig.axes[0]

    # Dark dot at the stimulation site
    dot = Circle(
        xy_axes, dot_radius, transform=ax.transAxes,
        facecolor=dot_color, edgecolor="white", linewidth=0.8, zorder=6,
    )
    ax.add_patch(dot)

    # Lightning bolt above-left of the dot
    bx, by = xy_axes[0] + bolt_offset[0], xy_axes[1] + bolt_offset[1]
    bolt = _BOLT_SHAPE.copy()
    bolt[:, 0] = bx + (bolt[:, 0] - 0.5) * bolt_size
    bolt[:, 1] = by + (bolt[:, 1] - 0.5) * bolt_size
    poly = Polygon(
        bolt, closed=True, transform=ax.transAxes,
        facecolor=bolt_color, edgecolor="#a04a00", linewidth=0.6, zorder=7,
    )
    ax.add_patch(poly)
    return fig
