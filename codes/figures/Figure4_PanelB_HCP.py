"""
Figure 4 Panel B (HCP) -- Node-wise energy-perturbation correlation profile.

For every (subject, region) we compute the Pearson correlation, across time, of
the brain's ongoing energy with three response measures:

    1. global baseline energy  E_t            vs  global effect size  Sigma^(j)
    2. local  baseline energy  e_j(t)          vs  local  effect size  sigma_j^(j)
    3. global baseline energy  E_t            vs  global evoked energy E^(j)_{t+1}

Each region (node index on x) then gets the mean +/- SD of that correlation
across subjects. The two effect-size lines sit clearly below zero (low ongoing
activity -> larger response), while the evoked-energy line sits near +1 (post-
stimulation energy tracks pre-stimulation energy almost one-to-one).

The 3.8 GB HCP CSV is streamed once; per-(subject, region) correlation
sufficient statistics are accumulated, the per-(subject, region) r table is
cached, and the plot reads the cache on re-runs.

`draw_panel` / `node_corr_table` / `accumulate_stats` are reused by
Figure4_PanelB_TMS.py.

Input
    codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv
Outputs
    codes/HCP/results/Figure4_PanelB_node_corr_HCP.csv   (cache)
    codes/figures/outputs/Figure4_PanelB_HCP.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, GUIDE, YEO7_COLORS, INK
from brain_render import cortical_network_only_ordering
setup()

ROOT = Path(__file__).resolve().parents[2]
HCP_CSV = ROOT / "codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv"
LABEL_TXT = ROOT / "codes/HCP/data/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_label.txt"
# Short network acronyms for the x-axis bands.
PRETTY = {"Vis": "VIS", "SomMot": "SMN", "DorsAttn": "DAN",
          "SalVentAttn": "SN", "Cont": "FPN", "Default": "DMN", "Limbic": "LN"}
CACHE_CSV = ROOT / "codes/HCP/results/Figure4_PanelB_node_corr_HCP.csv"           # Pearson (diagnostics)
CACHE_SPEARMAN = ROOT / "codes/HCP/results/Figure4_PanelB_node_corr_HCP_spearman.csv"
OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_ROWS = 2_000_000
SUBCORT_CORT_BOUNDARY = 50      # rois 0..49 subcortical, 50..449 cortical

# Sufficient-statistic columns accumulated per (sub_id, roi).
SUM_COLS = ["n", "gb", "gb2", "ge", "ge2", "gb_ge",
            "gev", "gev2", "gb_gev", "lb", "lb2", "le", "le2", "lb_le"]
READ_COLS = ["sub_id", "roi", "time", "global_baseline_energy",
             "global_effect_size", "global_evoked_energy",
             "local_baseline_energy", "local_effect_size"]
# Columns needed for the Spearman (rank) correlations actually plotted.
SPEAR_COLS = ["sub_id", "roi", "time", "global_baseline_energy",
              "global_effect_size", "local_baseline_energy", "local_effect_size"]

# (r-column, legend label, color). Neutral tones (dark + gray): the muted
# palette is reserved for the Yeo-7 network bands, so the lines stay achromatic
# and the only color in the panel marks the networks. Only the two effect-size
# lines are shown; the evoked-energy correlation (~+1) would dominate the range.
LINES = [
    ("r_global_effect", "baseline energy vs global effect size", INK),
    ("r_local_effect",  "local baseline vs local effect size", "#9aa0a8"),
]


def _chunk_sums(chunk: pd.DataFrame) -> pd.DataFrame:
    """Per-(sub_id, roi) sums of the products needed for three correlations."""
    gb = chunk["global_baseline_energy"].to_numpy()
    ge = chunk["global_effect_size"].to_numpy()
    gev = chunk["global_evoked_energy"].to_numpy()
    lb = chunk["local_baseline_energy"].to_numpy()
    le = chunk["local_effect_size"].to_numpy()
    d = pd.DataFrame({
        "sub_id": chunk["sub_id"].to_numpy(), "roi": chunk["roi"].to_numpy(),
        "n": 1.0, "gb": gb, "gb2": gb * gb, "ge": ge, "ge2": ge * ge,
        "gb_ge": gb * ge, "gev": gev, "gev2": gev * gev, "gb_gev": gb * gev,
        "lb": lb, "lb2": lb * lb, "le": le, "le2": le * le, "lb_le": lb * le,
    })
    return d.groupby(["sub_id", "roi"])[SUM_COLS].sum()


def accumulate_stats(chunks, time_max: int | None = None) -> pd.DataFrame:
    """Sum `_chunk_sums` over an iterable of dataframe chunks.

    If `time_max` is given, only time points with ``time < time_max`` are used
    (per subject/run), so correlations are computed on that initial window.
    """
    total = None
    for i, chunk in enumerate(chunks):
        if time_max is not None:
            chunk = chunk[chunk["time"] < time_max]
            if len(chunk) == 0:
                continue
        s = _chunk_sums(chunk)
        total = s if total is None else total.add(s, fill_value=0.0)
        print(f"  chunk {i}: groups so far = {len(total)}", flush=True)
    return total


def _pearson_from_sums(sxy, sx, sy, sxx, syy, n):
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def node_corr_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Per-(sub_id, roi) Pearson r for the three energy-response pairs."""
    s = stats.reset_index()
    g = lambda c: s[c].to_numpy()   # bracket access: "ge"/"le" shadow df methods
    n = g("n")
    return pd.DataFrame({
        "sub_id": s["sub_id"], "roi": s["roi"].astype(int),
        "r_global_effect": _pearson_from_sums(
            g("gb_ge"), g("gb"), g("ge"), g("gb2"), g("ge2"), n),
        "r_local_effect": _pearson_from_sums(
            g("lb_le"), g("lb"), g("le"), g("lb2"), g("le2"), n),
        "r_global_evoked": _pearson_from_sums(
            g("gb_gev"), g("gb"), g("gev"), g("gb2"), g("gev2"), n),
    })


def draw_panel(rtab: pd.DataFrame, out_stem: str, title: str = ""):
    """Per-region mean +/- SD across subjects, one line per response measure.

    Cortical regions only, grouped along the x-axis by Yeo-7 network
    (unimodal -> transmodal), with a colored band marking each network.
    """
    # Cortical only; map roi (50..449) -> cortical index (0..399).
    cort = rtab[rtab.roi >= SUBCORT_CORT_BOUNDARY].copy()
    cort["cidx"] = cort.roi.astype(int) - SUBCORT_CORT_BOUNDARY
    g = cort.groupby("cidx")

    # Network grouping order (LH+RH within each network, unimodal -> transmodal).
    perm, boundaries, net_labels = cortical_network_only_ordering(LABEL_TXT)
    x = np.arange(len(perm))

    fig, ax = plt.subplots(figsize=figsize_mm(88, 60), constrained_layout=True)

    for col, label, color in LINES:
        m = g[col].mean().reindex(range(400)).to_numpy()[perm]
        sd = g[col].std(ddof=1).reindex(range(400)).to_numpy()[perm]
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, m, color=color, lw=1.2, label=label, zorder=3)

    ax.axhline(0.0, color=GUIDE, ls=(0, (3, 2)), lw=0.8, zorder=1)

    YMIN, YMAX = -0.65, 0.35
    band_h = 0.035
    starts = [0] + list(boundaries[:-1])
    centers = []
    for s0, e0, net in zip(starts, boundaries, net_labels):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((s0 - 0.5, YMIN), e0 - s0, band_h,
                               facecolor=YEO7_COLORS[net], edgecolor="none",
                               zorder=5))
        if e0 != boundaries[-1]:
            ax.axvline(e0 - 0.5, color="#e2e2e2", lw=0.5, zorder=0)
        centers.append((s0 + e0) / 2 - 0.5)

    ax.set_xlim(-0.5, len(perm) - 0.5)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(np.arange(-0.6, 0.36, 0.2))
    ax.set_xticks(centers)
    ax.set_xticklabels([PRETTY[n] for n in net_labels], fontsize=6.5)
    ax.set_ylabel(r"Correlation $r$")
    ax.grid(True, axis="y", linestyle=(0, (4, 3)), linewidth=0.4,
            color="#d8d8d8", alpha=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.8, length=3, direction="out")
    if title:
        ax.set_title(title, pad=4)
    leg = ax.legend(loc="upper center", ncol=1, frameon=False, fontsize=6.5,
                    handlelength=1.4, borderaxespad=0.3, labelspacing=0.3)
    for ln in leg.get_lines():
        ln.set_linewidth(1.6)

    for col, _, _ in LINES:
        vals = g[col].mean()
        print(f"  {col:18s} median across nodes = {vals.median():+.3f}")
    save_panel(fig, OUT_DIR / out_stem)


# ---------------------------------------------------------------------------
# Spearman (rank) correlations -- the method actually plotted.
# Rank-based, so robust to the heavy-tailed energies and invariant to a log
# transform. Ranks need all of a group's time points at once, so we work
# per (subject) rather than streaming sufficient statistics.
# ---------------------------------------------------------------------------
def process_subject(sdf: pd.DataFrame) -> pd.DataFrame:
    """Per-region Spearman r (across time) for one subject's rows.

    global effect : corr(global baseline energy E_t , global effect size)
    local  effect : corr(local  baseline energy e_j , local  effect size)
    """
    ge = sdf.pivot(index="time", columns="roi", values="global_effect_size")
    gb = (sdf.groupby("time")["global_baseline_energy"].first()
             .reindex(ge.index))                      # E_t (same across roi)
    r_global = ge.corrwith(gb, method="spearman")     # per roi

    lb = sdf.pivot(index="time", columns="roi", values="local_baseline_energy")
    le = sdf.pivot(index="time", columns="roi", values="local_effect_size")
    r_local = lb.corrwith(le, method="spearman")      # column-wise (per roi)

    rois = r_global.index
    return pd.DataFrame({
        "sub_id": sdf["sub_id"].iloc[0],
        "roi": rois.astype(int),
        "r_global_effect": r_global.to_numpy(),
        "r_local_effect": r_local.reindex(rois).to_numpy(),
    })


def spearman_table(subject_iter) -> pd.DataFrame:
    """Concatenate per-subject Spearman tables from an iterable of subject DFs."""
    parts = []
    for sdf in subject_iter:
        parts.append(process_subject(sdf))
        print(f"  subject {parts[-1]['sub_id'].iloc[0]} done "
              f"({len(parts)})", flush=True)
    return pd.concat(parts, ignore_index=True)


def iter_hcp_subjects():
    """Yield one DataFrame per subject by streaming the big CSV.

    Rows are subject-contiguous, so we buffer until a subject is complete
    (a new subject id appears) and never hold more than ~one subject in memory.
    """
    buf = []
    for chunk in pd.read_csv(HCP_CSV, usecols=SPEAR_COLS, chunksize=CHUNK_ROWS):
        buf.append(chunk)
        cat = pd.concat(buf, ignore_index=True)
        last = cat["sub_id"].iloc[-1]
        for s in cat["sub_id"].unique():
            if s != last:
                yield cat[cat["sub_id"] == s]
        buf = [cat[cat["sub_id"] == last]]
    cat = pd.concat(buf, ignore_index=True)
    for s in cat["sub_id"].unique():
        yield cat[cat["sub_id"] == s]


def main():
    if CACHE_SPEARMAN.exists():
        rtab = pd.read_csv(CACHE_SPEARMAN)
        print(f"Loaded cache {CACHE_SPEARMAN}  ({rtab.sub_id.nunique()} subjects)")
    else:
        print("Streaming HCP CSV and computing per-subject Spearman r ...")
        rtab = spearman_table(iter_hcp_subjects())
        rtab.to_csv(CACHE_SPEARMAN, index=False)
        print(f"Cached -> {CACHE_SPEARMAN}")

    n_sub = rtab.sub_id.nunique()
    print(f"{n_sub} subjects")
    draw_panel(rtab, "Figure4_PanelB_HCP")


if __name__ == "__main__":
    main()
