"""
Figure 4 Panels C/D (HCP) -- mean & variance of the stimulation response across
baseline-energy regimes, for BOTH the global effect size (GES) and the global
evoked energy.

Reuses Figure5_PanelA_HCP.draw (identical box styling, size, regime colours and
pairwise Wilcoxon statistics as the existing Figure-4 Panel C). For each
(subject, cortical region) the 500 model time steps are split by baseline energy
into three 5%-of-trials regimes -- random (open-loop, blue), high E(t) and
low E(t) (closed-loop, orange) -- and we take, per regime:

    mean : size of the response       (GES: low E(t) evokes the largest effect)
    var  : absolute spread            (GES: low E(t) is the least variable)
    cv   : relative spread, std/mean  (GES: low E(t) is the most reproducible)

The variability panel uses CV(GES) = std/mean (the relative, magnitude-independent
variability) rather than the absolute Var(GES); CV is derived from the cached
mean/var as sqrt(var)/mean. GES isolates the stimulation effect; evoked energy
mostly inherits the baseline magnitude -- the energy-scaling control for the GES
result.

Input
    codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv
Outputs
    codes/HCP/results/Figure4_PanelCD_meanvar_HCP.csv                       (cache)
    codes/figures/outputs/Figure4_PanelC_GES_{mean,var,cv}_HCP.{svg,pdf,png}
    codes/figures/outputs/Figure4_PanelD_evoked_{mean,var}_HCP.{svg,pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Figure5_PanelA_HCP import draw, _tint, REGIMES
from Figure4_PanelB_HCP import ROOT, HCP_CSV

# --- parameters: match the canonical Figure 4 / Figure 5 box plots ----------
FRAC = 0.05            # 5% of trials per regime (as in Figure5_PanelA_HCP)
N_RAND = 200           # random draws averaged for the random regime
CORT = 50              # cortical ROIs only (0..49 = Tian subcortex)
SEED = 0

RESP = {"ges": "global_effect_size", "evo": "global_evoked_energy"}
CACHE = ROOT / "codes/HCP/results/Figure4_PanelCD_meanvar_HCP.csv"
OUT_DIR = ROOT / "codes/figures/outputs"

# tab:blue / tab:orange, exactly as Figure4_PanelC_HCP.
TAB_BLUE, TAB_ORANGE = "#1f77b4", "#ff7f0e"
REGIME_LABELS = {"rand": "Random", "high": "High E(t)", "low": "Low E(t)"}
REGIME_COLORS = {"rand": TAB_BLUE, "high": _tint(TAB_ORANGE, 0.5),
                 "low": TAB_ORANGE}


def build_table() -> pd.DataFrame:
    """Per-(subject, cortical roi) mean & variance of each response in each
    regime. Long format with a `resp` column ('ges' / 'evo')."""
    df = pd.read_csv(HCP_CSV, usecols=["sub_id", "roi", "time",
                     "global_baseline_energy", *RESP.values()],
                     dtype={"roi": "int16", "time": "int16",
                            "global_baseline_energy": "float32",
                            "global_effect_size": "float32",
                            "global_evoked_energy": "float32"})
    parts = []
    for sub, sdf in df.groupby("sub_id", sort=False):
        et = (sdf.groupby("time", sort=True)["global_baseline_energy"]
                 .first().to_numpy())
        T = len(et); k = max(2, int(round(FRAC * T)))
        order = np.argsort(et)
        idx = {"low": order[:k], "high": order[-k:]}
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(T, size=k, replace=False) for _ in range(N_RAND)]
        rois = np.sort(sdf["roi"].unique())
        cmask = rois >= CORT
        for key, col in RESP.items():
            M = sdf.pivot(index="time", columns="roi", values=col).to_numpy()[:, cmask]
            rec = {"mean_low": M[idx["low"]].mean(0),
                   "mean_high": M[idx["high"]].mean(0),
                   "var_low": M[idx["low"]].var(0, ddof=1),
                   "var_high": M[idx["high"]].var(0, ddof=1)}
            mr = np.zeros(M.shape[1]); vr = np.zeros(M.shape[1])
            for dd in draws:
                mr += M[dd].mean(0); vr += M[dd].var(0, ddof=1)
            rec["mean_rand"] = mr / N_RAND; rec["var_rand"] = vr / N_RAND
            p = pd.DataFrame(rec)
            p.insert(0, "roi", rois[cmask]); p.insert(0, "resp", key)
            p.insert(0, "sub_id", sub)
            parts.append(p)
        print(f"  {sub} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def load_or_build() -> pd.DataFrame:
    if CACHE.exists():
        t = pd.read_csv(CACHE)
        print(f"Loaded cache {CACHE}  ({t.sub_id.nunique()} subjects)")
        return t
    print("Building per-(subject, roi) mean/variance table from HCP CSV ...")
    t = build_table()
    t.to_csv(CACHE, index=False)
    print(f"Cached -> {CACHE}")
    return t


def main():
    table = load_or_build()
    ges = table[table.resp == "ges"].copy()
    evo = table[table.resp == "evo"]
    # CV(GES) = std/mean per (subject, roi, regime), derived from the cached
    # mean/var (sqrt(var)/mean = std/mean), matching Figure5_PanelA_HCP's CV.
    for r in REGIMES:
        ges[f"cv_{r}"] = np.sqrt(ges[f"var_{r}"]) / ges[f"mean_{r}"]
    # Same panel geometry / colours / rotation as Figure4_PanelC_HCP.
    kw = dict(fig_w_mm=58, fig_h_mm=54, show_title=False,
              regime_labels=REGIME_LABELS, regime_colors=REGIME_COLORS,
              xtick_rotation=45)
    draw(ges, "sub_id", "Figure4_PanelC_GES_mean_HCP", metric="mean",
         suffix_inline=True, mean_metric_name="GES", **kw)
    draw(ges, "sub_id", "Figure4_PanelC_GES_var_HCP", metric="var",
         suffix_inline=True, var_metric_name="GES", **kw)
    draw(ges, "sub_id", "Figure4_PanelC_GES_cv_HCP", metric="cv",
         cv_metric_name="GES", **kw)
    draw(evo, "sub_id", "Figure4_PanelD_evoked_mean_HCP", metric="mean",
         suffix_inline=True, mean_metric_name="evoked energy", **kw)
    draw(evo, "sub_id", "Figure4_PanelD_evoked_var_HCP", metric="var",
         suffix_inline=True, var_metric_name="evoked energy", **kw)


if __name__ == "__main__":
    main()
