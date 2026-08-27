"""
Figure 5 Panel A (HCP) -- Stimulation-effect variability across energy regimes.

For each (subject, cortical region) we take the global effect size Sigma^(j)
across all stimulation trials (time points) and measure its trial-to-trial
variability as the coefficient of variation (CV = std / mean) within three
equally sized (5% of trials) regimes:

    low    : the 5% of trials with the lowest  baseline energy E_t
    high   : the 5% of trials with the highest baseline energy E_t
    random : 5% of trials drawn at random (averaged over many draws)

Conditioning on a narrow energy band (low or high) yields consistent effects
(low CV); random trials span the whole energy range (high CV). The per-(subject,
region) CV table is cached; two aggregation views are rendered from it:

    by-subject : mean CV across cortical regions -> distribution over subjects
    by-region  : mean CV across subjects        -> distribution over regions

Input
    codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv
Outputs
    codes/HCP/results/Figure5_PanelA_regime_cv_HCP.csv               (cache)
    codes/figures/outputs/Figure5_PanelA_HCP_{bySubject,byRegion}.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import setup, save_panel, figsize_mm, ACCENT_COOL, ACCENT_WARM
from Figure4_PanelB_HCP import iter_hcp_subjects, ROOT
setup()

OUT_DIR = ROOT / "codes/figures/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "codes/HCP/results/Figure5_PanelA_regime_cv_HCP.csv"

CORTICAL_MIN_ROI = 50
FRAC = 0.05            # fraction of trials per regime
N_RAND = 200           # random draws averaged for the random regime
SEED = 0

# Random (open-loop, no state selection) first, then the two closed-loop
# regimes. Colors follow Figure 1's stimulation scheme: blue = open-loop,
# orange = closed-loop. The two closed-loop regimes share the orange hue,
# distinguished by tint -- pale (more "blurred") for high energy, vivid for low.
REGIMES = ["rand", "high", "low"]
REGIME_LABEL = {"low": "Low energy", "high": "High energy", "rand": "Random"}


def _tint(c, f):
    """Blend color `c` toward white by fraction `f` (0 = c, 1 = white)."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


REGIME_COLOR = {
    "rand": ACCENT_COOL,             # open-loop / random  -> blue (as in Fig 1)
    "high": _tint(ACCENT_WARM, 0.5),  # closed-loop, high energy -> pale orange
    "low":  ACCENT_WARM,             # closed-loop, low energy  -> vivid orange
}


def subject_regime_cv(sdf: pd.DataFrame, frac=FRAC, n_rand=N_RAND,
                      seed=SEED) -> pd.DataFrame:
    """Per-cortical-region effect-size statistics in each energy regime.

    Returns, per region, the coefficient of variation (cv_*), the mean effect
    size (mean_*) and the variance (var_*) over the selected trials.
    """
    et = sdf.groupby("time")["global_baseline_energy"].first().sort_index()
    T = len(et)
    k = max(2, int(round(frac * T)))
    order = np.argsort(et.to_numpy())             # ascending energy
    idx_low, idx_high = order[:k], order[-k:]

    piv = (sdf.pivot(index="time", columns="roi", values="global_effect_size")
              .reindex(et.index))
    cort = [c for c in piv.columns if c >= CORTICAL_MIN_ROI]
    E = piv[cort].to_numpy()                       # (T, n_cort)

    def stats(rows):
        s = E[rows]
        m = s.mean(0)
        v = s.var(0, ddof=1)
        return s.std(0, ddof=1) / np.where(m != 0, m, np.nan), m, v

    rng = np.random.default_rng(seed)
    rand_cv = np.zeros(len(cort))
    rand_mean = np.zeros(len(cort))
    rand_var = np.zeros(len(cort))
    for _ in range(n_rand):
        cv_r, m_r, v_r = stats(rng.choice(T, size=k, replace=False))
        rand_cv += cv_r
        rand_mean += m_r
        rand_var += v_r
    rand_cv /= n_rand
    rand_mean /= n_rand
    rand_var /= n_rand

    cv_low, mean_low, var_low = stats(idx_low)
    cv_high, mean_high, var_high = stats(idx_high)
    return pd.DataFrame({"sub_id": sdf["sub_id"].iloc[0], "roi": cort,
                         "cv_low": cv_low, "cv_high": cv_high, "cv_rand": rand_cv,
                         "mean_low": mean_low, "mean_high": mean_high,
                         "mean_rand": rand_mean,
                         "var_low": var_low, "var_high": var_high,
                         "var_rand": rand_var})


def build_table() -> pd.DataFrame:
    parts = []
    for sdf in iter_hcp_subjects():
        parts.append(subject_regime_cv(sdf))
        print(f"  {parts[-1]['sub_id'].iloc[0]} done ({len(parts)})", flush=True)
    return pd.concat(parts, ignore_index=True)


def _stars(p: float) -> str:
    return ("***" if p < 1e-3 else "**" if p < 1e-2 else
            "*" if p < 5e-2 else f"n.s. (p={p:.2f})")


def _bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="#444444")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=7)


METRIC_TITLE = {"cv": "Stimulation-effect variability across energy regimes",
                "mean": "Stimulation effect size across energy regimes",
                "var": "Stimulation-response variance across energy regimes"}


def draw(table: pd.DataFrame, agg: str, out_stem: str, metric: str = "cv",
         dataset: str = "", fig_w_mm: float = 85, fig_h_mm: float = 78,
         show_title: bool = True, regime_labels: dict | None = None,
         regime_colors: dict | None = None, xtick_rotation: float = 0,
         cv_metric_name: str = "effect size", suffix_inline: bool = False,
         mean_metric_name: str | None = None, var_metric_name: str | None = None):
    """metric = 'cv' (CV of effect), 'mean' (mean response), or 'var' (variance
    of response). agg = 'sub_id' (per subject) or 'roi' (per region).
    fig_w_mm/fig_h_mm size the panel; show_title toggles the title.

    Styling overrides (defaults reproduce the original Figure-5 look):
      regime_labels  : {regime: x-tick label}        (default REGIME_LABEL)
      regime_colors  : {regime: box facecolor}       (default REGIME_COLOR)
      xtick_rotation : x-tick label rotation in deg  (right-anchored if != 0)
      cv_metric_name : noun in the CV y-label, "Mean CV of <name> (per ...)"
      suffix_inline  : put the mean's x10^k factor before "(per unit)"
      mean_metric_name : noun in the mean y-label, "Mean <name> (per ...)";
                         default None keeps the formula "effect size <Sigma>"
      var_metric_name  : noun in the variance y-label, "Mean variance of <name>"
    """
    labels = regime_labels or REGIME_LABEL
    colors = regime_colors or REGIME_COLOR
    cols = [f"{metric}_{r}" for r in REGIMES]
    d = table.groupby(agg)[cols].mean()
    data = [d[f"{metric}_{r}"].to_numpy() for r in REGIMES]
    unit = "subject" if agg == "sub_id" else "region"

    # The mean effect (~1e-4) and variances (tiny/large) get their order of
    # magnitude folded into the label, just like the original mean panel.
    exp = 0
    if metric in ("mean", "var"):
        exp = int(np.floor(np.log10(np.median(np.concatenate(data)))))
        data = [x * 10.0 ** (-exp) for x in data]

    fig, ax = plt.subplots(figsize=figsize_mm(fig_w_mm, fig_h_mm),
                           constrained_layout=True)
    pos = np.arange(1, len(REGIMES) + 1)
    bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False,
                    patch_artist=True, zorder=2)
    for patch, r in zip(bp["boxes"], REGIMES):
        patch.set(facecolor=colors[r], edgecolor="#3a3a3a", linewidth=1.0,
                  alpha=0.9)
    for elem in ("whiskers", "caps"):
        for ln in bp[elem]:
            ln.set(color="#3a3a3a", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#1a1a1a", linewidth=1.6)

    # Pairwise Wilcoxon signed-rank tests (same units across regimes).
    pairs = [(0, 1), (0, 2), (1, 2)]
    pvals = {pair: wilcoxon(data[pair[0]], data[pair[1]]).pvalue
             for pair in pairs}
    # Anchor brackets just above the visible whisker caps (fliers are hidden,
    # so np.max would float them up to an off-plot outlier).
    caps = bp["caps"]
    top = max(caps[2 * i + 1].get_ydata()[0] for i in range(len(data)))
    lo = min(caps[2 * i].get_ydata()[0] for i in range(len(data)))
    span = top - lo
    step = span * 0.085
    for lvl, pair in enumerate([(0, 1), (0, 2), (1, 2)]):
        y = top + span * 0.04 + lvl * step
        _bracket(ax, pos[pair[0]], pos[pair[1]], y, step * 0.3,
                 _stars(pvals[pair]))
    ax.set_ylim(lo - span * 0.06, top + span * 0.34)

    ax.set_xticks(pos)
    ax.set_xticklabels([labels[r] for r in REGIMES], rotation=xtick_rotation,
                       ha="right" if xtick_rotation else "center")
    if metric == "cv":
        ax.set_ylabel(f"Mean CV of {cv_metric_name} (per {unit})")
    elif metric == "var":
        factor = rf"($\times 10^{{{exp}}}$)" if exp else ""
        noun = rf"Mean variance of {var_metric_name or 'response'}"
        if suffix_inline and factor:
            ax.set_ylabel(rf"{noun} {factor} (per {unit})")
        else:
            suff = rf"   {factor}" if factor else ""
            ax.set_ylabel(rf"{noun} (per {unit}){suff}")
    else:
        noun = (rf"Mean {mean_metric_name}" if mean_metric_name
                else r"Mean effect size $\langle\Sigma^{(j)}\rangle$")
        factor = rf"($\times 10^{{{exp}}}$)" if exp else ""
        if suffix_inline and factor:
            ax.set_ylabel(rf"{noun} {factor} (per {unit})")
        else:
            suff = rf"   {factor}" if factor else ""
            ax.set_ylabel(rf"{noun} (per {unit}){suff}")
    if show_title:
        ttl = METRIC_TITLE[metric]
        ax.set_title(f"{ttl}\n{dataset}" if dataset else ttl, pad=6)
    ax.tick_params(width=0.8, length=3, direction="out")

    print(f"[{out_stem}] {metric} per-{unit} (n={len(d)})  medians: " +
          "  ".join(f"{REGIME_LABEL[r]}={np.median(d[f'{metric}_{r}']):.4g}"
                    for r in REGIMES))
    for pair in pairs:
        print(f"    {REGIME_LABEL[REGIMES[pair[0]]]} vs "
              f"{REGIME_LABEL[REGIMES[pair[1]]]}: p={pvals[pair]:.2e}")
    save_panel(fig, OUT_DIR / out_stem)


REQUIRED_COLS = [f"{m}_{r}" for m in ("cv", "mean", "var") for r in REGIMES]


def load_or_build() -> pd.DataFrame:
    """Load the cached per-(subject, region) regime table, rebuilding if it
    predates the mean_* columns."""
    if CACHE.exists():
        table = pd.read_csv(CACHE)
        if all(c in table.columns for c in REQUIRED_COLS):
            print(f"Loaded cache {CACHE}  ({table.sub_id.nunique()} subjects)")
            return table
        print("Cache missing mean_* columns; rebuilding ...")
    table = build_table()
    table.to_csv(CACHE, index=False)
    print(f"Cached -> {CACHE}")
    return table


def main():
    # Per-subject is the canonical Panel A (subjects = inferential unit).
    draw(load_or_build(), "sub_id", "Figure5_PanelA_HCP", metric="cv",
         dataset="HCP")


if __name__ == "__main__":
    main()
