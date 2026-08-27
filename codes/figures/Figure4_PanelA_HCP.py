"""
Figure 4 Panel A (HCP) -- Single-subject, single-region example of the
background-dependence of stimulation.

Two scatter subplots for one (subject, region), each dot a time point t:
  Left : baseline energy E_t = sum_i x_i(t)^2  (x, log)
         vs global effect size Sigma^(j)_{t+1} = sum_i EC_t^2  (y, linear).
         A negative correlation -- the lower the brain's ongoing activity, the
         larger the response it produces when stimulated.
  Right: baseline energy (x, log) vs global evoked energy
         E^(j)_{t+1} = sum_i (x^(j)_i(t+1))^2  (y, log).
         A strong positive correlation -- the post-stimulation energy tracks
         the pre-stimulation energy almost one-to-one.

Pearson r and p are computed on the raw (subject, region) time series (matching
HCP_4.2_Background_dependence_summary) and annotated inside each subplot.

`draw_panel` does the plotting and is reused by Figure4_PanelA_TMS.py.

Input
    codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv
        (3.8 GB; streamed in chunks and filtered to one subject + region)

Output
    codes/figures/outputs/Figure4_PanelA_HCP.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, INK
setup()

ROOT = Path(__file__).resolve().parents[2]
HCP_CSV = ROOT / "codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv"
# Small cache of the example (subject, roi) slice so the panel reproduces
# without the full 3.8 GB dataframe.
EXAMPLE_CACHE = ROOT / "codes/HCP/results/Figure4_PanelA_example_HCP.csv"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Example shown in the panel (matches the exploratory notebook example).
# roi is 0-indexed (0..49 subcortical Tian, 50..449 cortical Schaefer); roi=100
# is the cortical parcel 7Networks_LH_SomMot_20 -> displayed as "L SomMot 20".
HCP_SUBJECT = "id_100206"
HCP_REGION = 100
HCP_REGION_LABEL = "L SomMot 20"

CHUNK_ROWS = 2_000_000
COLS = ["sub_id", "roi", "time",
        "global_baseline_energy", "global_effect_size", "global_evoked_energy"]

# Single line/marker color: matplotlib tab:blue (the standard tableau blue).
TAB_BLUE = "#1f77b4"

# Axis labels (shared HCP/TMS): plain words, no equations (full definitions are
# in the paper glossary). "GES" = global effect size.
XLABEL = "Baseline energy"
YLABEL_EFFECT = "GES"
YLABEL_EVOKED = "Evoked energy"


def darken(color, f: float = 0.6) -> tuple[float, float, float]:
    """A darker shade of `color` for soft marker edges."""
    r, g, b = mcolors.to_rgb(color)
    return (r * f, g * f, b * f)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return r"$p<10^{-300}$"
    exp = int(np.floor(np.log10(p)))
    if exp >= -2:
        return rf"$p={p:.2g}$"
    mant = p / 10.0 ** exp
    return rf"$p={mant:.1f}\times10^{{{exp}}}$"


def _scatter(ax, x, y, color, *, ylog: bool):
    """One scatter (dot per time point) with log x, optional log y."""
    ax.scatter(x, y, s=9, color=color, alpha=0.55,
               edgecolor=darken(color), linewidths=0.2, zorder=3)
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="major", linestyle=(0, (4, 3)), linewidth=0.4,
            color="#c9c9c9", alpha=0.6, zorder=0)
    ax.tick_params(width=0.8, length=3, direction="out")


def _annotate_corr(ax, x, y, *, loc: str):
    """Spearman rho + p on the raw time series, in a light box inside the axes."""
    r, p = spearmanr(x, y)
    pos = {"upper right": (0.96, 0.96, "right", "top"),
           "upper left": (0.04, 0.96, "left", "top")}[loc]
    ax.annotate(rf"Spearman $\rho={r:+.2f}$" + "\n" + _fmt_p(p),
                xy=pos[:2], xycoords="axes fraction",
                ha=pos[2], va=pos[3], fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", lw=0.6, alpha=0.85))
    return r, p


def plot_effect(ax, baseline, effect, color=TAB_BLUE):
    """Draw the baseline-vs-GES scatter on `ax` (negative gating). The y-axis
    order of magnitude is folded into the label. Returns (rho, p)."""
    exp_e = int(np.floor(np.log10(np.nanmedian(effect))))
    scale = 10.0 ** (-exp_e)
    _scatter(ax, baseline, effect * scale, color, ylog=False)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL_EFFECT + rf"   ($\times10^{{{exp_e}}}$)")
    return _annotate_corr(ax, baseline, effect, loc="upper right")


def plot_evoked(ax, baseline, evoked, color=TAB_BLUE):
    """Draw the baseline-vs-evoked-energy scatter on `ax` (positive, log-log).
    Returns (rho, p)."""
    _scatter(ax, baseline, evoked, color, ylog=True)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL_EVOKED)
    return _annotate_corr(ax, baseline, evoked, loc="upper left")


def draw_effect_panel(baseline, effect, sid, region_label, out_stem,
                      color=TAB_BLUE):
    """Standalone scatter: baseline energy vs global effect size (GES).

    Negative gating -- lower ongoing activity -> larger response.
    """
    fig, ax = plt.subplots(figsize=figsize_mm(52, 50), constrained_layout=True)
    r, p = plot_effect(ax, baseline, effect, color)
    ax.set_title(f"{sid}  ·  {region_label}", fontsize=8)
    save_panel(fig, OUT_DIR / out_stem)
    return r, p


def draw_evoked_panel(baseline, evoked, sid, region_label, out_stem,
                      color=TAB_BLUE):
    """Standalone scatter: baseline energy vs evoked energy (positive, log-log).

    The post-stimulation energy tracks the pre-stimulation energy almost
    one-to-one.
    """
    fig, ax = plt.subplots(figsize=figsize_mm(52, 50), constrained_layout=True)
    r, p = plot_evoked(ax, baseline, evoked, color)
    ax.set_title(f"{sid}  ·  {region_label}", fontsize=8)
    save_panel(fig, OUT_DIR / out_stem)
    return r, p


def draw_panel(baseline, effect, evoked, sid, region_label,
               out_stem_effect, out_stem_evoked):
    """Draw the example as two standalone panels (effect scatter + evoked
    scatter), each with its own title. Reused by Figure4_PanelA_TMS.py."""
    re, pe = draw_effect_panel(baseline, effect, sid, region_label,
                               out_stem_effect)
    rv, pv = draw_evoked_panel(baseline, evoked, sid, region_label,
                               out_stem_evoked)
    print(f"[{out_stem_effect} / {out_stem_evoked}] {sid} "
          f"{region_label} n={len(baseline)}  "
          f"r(effect)={re:+.3f} (p={pe:.1e})  r(evoked)={rv:+.3f} (p={pv:.1e})")


def load_hcp_subject_roi(sid: str, roi: int) -> pd.DataFrame:
    """Stream the 3.8 GB HCP CSV and return the (sid, roi) rows, time-sorted.

    Rows are written subject-contiguous, so we stop once we have passed `sid`.
    Falls back to the small example cache when the full dataframe is absent.
    """
    if not HCP_CSV.exists() and EXAMPLE_CACHE.exists():
        t = pd.read_csv(EXAMPLE_CACHE)
        sel = t[(t.sub_id == sid) & (t.roi == roi)]
        if not len(sel):
            raise ValueError(
                f"{EXAMPLE_CACHE.name} only holds the paper example "
                f"(id_100206, roi 100); requested ({sid}, roi {roi}) needs the "
                f"full HCP_4_df_background_dependence_ECts.csv")
        return sel.sort_values("time").reset_index(drop=True)
    parts, seen = [], False
    for chunk in pd.read_csv(HCP_CSV, usecols=COLS, chunksize=CHUNK_ROWS):
        in_sub = chunk["sub_id"] == sid
        if in_sub.any():
            seen = True
            sel = chunk[in_sub & (chunk["roi"] == roi)]
            if len(sel):
                parts.append(sel)
        elif seen:
            break
    if not parts:
        raise ValueError(f"No rows for subject {sid}, roi {roi}")
    return pd.concat(parts, ignore_index=True).sort_values("time")


def main():
    df = load_hcp_subject_roi(HCP_SUBJECT, HCP_REGION)
    draw_panel(df["global_baseline_energy"].to_numpy(),
               df["global_effect_size"].to_numpy(),
               df["global_evoked_energy"].to_numpy(),
               HCP_SUBJECT, HCP_REGION_LABEL,
               "Figure4_PanelA_HCP", "Figure4_PanelB_evoked_HCP")


if __name__ == "__main__":
    main()
