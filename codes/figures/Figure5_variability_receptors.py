"""
Can a region's focal stimulation VARIABILITY be predicted without stimulating it?

Figure 3 showed that regional responsiveness (focal effect size, <Sigma^(j)>) has a
macroscale gradient and (suggestive) receptor associations. Here we run the SAME
pipeline on the focal across-time VARIABILITY -- the CV of the effect -- to ask
whether it, too, is organized by Yeo-7 network and PET receptor density.

Node-level targets (400 cortical regions, group mean):
    responsiveness : mean_global_effect_size from the HCP_5 spatial df (= <Sigma^(j)>)
    focal CV       : diag(cv_bec) from Figure5_BEC_matrices.npz (across-time CV of
                     the focal effect), aligned by roi (df roi r <-> BEC roi 50+r)

For each of the 19 PET maps: Spearman rho and two spatial-autocorrelation-preserving
nulls -- PRIMARY: Alexander-Bloch spin test (1000 rotations; columns p_*/q_*),
ROBUSTNESS: Moran spectral randomization (2000 surrogates precomputed with
neuromaps.nulls.moran from the Hansen PET images, codes/HCP/data/permutation_receptors.py;
columns p_*_moran/q_*_moran). Also the Yeo-7 network breakdown of the focal CV.

Output: codes/HCP/results/Figure5_variability_receptor_correlations.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Figure3_PanelB import compute_spins, bh, per_region, NAME_MAP, HCP_DF, VALUE_COL

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "codes/HCP/results/Figure5_BEC_matrices.npz"
BECTS = ROOT / "codes/HCP/results/BECts_reduced"
OUT_DIR = ROOT / "codes/figures/outputs"
SUB = 50

# --- Moran spectral-randomization spatial nulls (replace the Alexander-Bloch spin
# test). One (n_perm, 400) surrogate set per receptor, roi 0..399 aligned to the
# dataframe (generator: codes/HCP/data/permutation_receptors.py). ---
PERM_DIR = ROOT / "codes/HCP/data/permutations_processed"
# dataframe receptor column (without 'receptor_' prefix) -> Moran null short name.
# 'raw_VAChT_feobv_hc3_spreng' is intentionally absent: it is a single PET source of
# the normalised VAChT map ('Vacht') and is not a separate map -> 19 maps.
RECEPTOR_NULL = {
    "5HT1a": "5HT1a", "5HT1b": "5HT1b", "5HT2a": "5HT2a", "5HTT": "5HTT",
    "CB1": "CB1", "D1": "D1", "D2": "D2", "DAT": "DAT", "GABA": "GABA",
    "MU": "MU", "NAT": "NAT", "Vacht": "VAChT", "mGluR5": "mGluR5",
    "raw_5HT4_sb20_hc59_beliveau": "5HT4",
    "raw_5HT6_gsk_hc30_radhakrishnan": "5HT6",
    "raw_A4B2_flubatine_hc30_hillmer": "A4B2",
    "raw_H3_cban_hc8_gallezot": "H3",
    "raw_M1_lsn_hc24_naganawa": "M1",
    "raw_NMDA_ge179_hc29_galovic": "NMDA",
}


def load_moran_null(short):
    """(n_perm, 400) Moran surrogate maps for receptor `short`, roi 0..399."""
    return np.load(next(PERM_DIR.glob(f"{short}_null_*perm.npy")))


def _spearman_rows(maps, y):
    """Spearman corr of each row of `maps` (P,400) with `y` (Spearman = Pearson of
    ranks), vectorised."""
    ry = rankdata(y); ry = (ry - ry.mean()) / ry.std()
    R = np.apply_along_axis(rankdata, 1, maps).astype(float)
    R = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    return (R @ ry) / y.shape[0]


def moran_p(rho, short, y):
    """Two-tailed Moran-null p: fraction of surrogates with |rho_null| >= |rho|."""
    null = _spearman_rows(load_moran_null(short), y)
    return (np.sum(np.abs(null) >= abs(rho)) + 1) / (null.shape[0] + 1)


def focal_ges_variance():
    """Group-mean focal Var(GES) per cortical region (roi 0..399).

    ssq_all is the across-stimulation std of the global effect size; Var(GES) is
    its square, averaged across participants (matching how cv_bec averages the CV).
    """
    files = sorted(BECTS.glob("id_*.npz"))
    acc = np.zeros(400)
    for f in files:
        ss = np.diag(np.load(f)["ssq_all"].astype(float))[SUB:]
        acc += ss ** 2
    return acc / len(files)


def focal_ges_cv():
    """Group-mean focal CV(GES) per cortical region (roi 0..399).

    GES = BEC^2, so over trials its mean is msq_all and its std is ssq_all; the
    coefficient of variation CV = ssq_all/msq_all, averaged across participants
    (matches diag(cv_bec)). Unlike Var(GES) (= ssq_all^2 ~ mean^2), the CV divides
    out the magnitude, so it measures *relative* trial-to-trial variability."""
    files = sorted(BECTS.glob("id_*.npz"))
    acc = np.zeros(400)
    for f in files:
        d = np.load(f)
        acc += np.diag(d["ssq_all"].astype(float))[SUB:] / np.diag(d["msq_all"].astype(float))[SUB:]
    return acc / len(files)


def corr_table(df, targets, spins):
    """Spearman rho + two spatial nulls for each of the 19 receptor maps against each
    target (dict name -> (400,) array, roi 0..399). PRIMARY: Alexander-Bloch spin
    (columns p_/q_). ROBUSTNESS: Moran surrogates (columns p_*_moran/q_*_moran).
    Observed rho uses the dataframe map; the redundant raw-VAChT source is dropped."""
    rows = []
    for col_full, short in RECEPTOR_NULL.items():
        x = per_region(df, f"receptor_{col_full}")
        rec = {"receptor": col_full}
        for tname, y in targets.items():
            rho = spearmanr(x, y).correlation
            rec[f"rho_{tname}"] = rho
            # primary: spin null (reindex the observed map under each rotation)
            snull = np.array([spearmanr(x[spins[:, i]], y).correlation
                              for i in range(spins.shape[1])])
            rec[f"p_{tname}"] = (np.sum(np.abs(snull) >= abs(rho)) + 1) / (spins.shape[1] + 1)
            # robustness: Moran spectral-randomization surrogates
            rec[f"p_{tname}_moran"] = moran_p(rho, short, y)
        rows.append(rec)
    t = pd.DataFrame(rows)
    for tname in targets:
        t[f"q_{tname}"] = bh(t[f"p_{tname}"].to_numpy())
        t[f"q_{tname}_moran"] = bh(t[f"p_{tname}_moran"].to_numpy())
    t["name"] = t.receptor.map(lambda r: NAME_MAP.get(r, r))
    return t


def main():
    df = pd.read_pickle(HCP_DF)
    resp = per_region(df, VALUE_COL)                 # responsiveness, roi 0..399

    g = np.load(NPZ)
    focal_ges_bec = np.diag(g["mean_bec"].astype(float))[SUB:]   # roi 0..399
    focal_var = focal_ges_variance()                             # roi 0..399, Var(GES)
    focal_cv = focal_ges_cv()                                    # roi 0..399, CV(GES)

    # Alignment check: df responsiveness vs BEC focal GES should be ~identical.
    r_align = spearmanr(resp, focal_ges_bec).correlation
    print(f"roi-alignment check  Spearman(df responsiveness, BEC focal GES) = {r_align:+.3f}")
    print(f"focal Var(GES) across regions: mean={focal_var.mean():.3g}  std={focal_var.std():.3g}  "
          f"range {focal_var.min():.3g}-{focal_var.max():.3g}")
    print(f"Spearman(responsiveness, Var(GES)) = {spearmanr(resp, focal_var).correlation:+.3f}  "
          f"(are effect size and variability spatially redundant?)")

    # Yeo-7 network breakdown of Var(GES) vs responsiveness.
    net = df.groupby("roi")["rsn_network"].first().to_numpy()
    print("\nYeo-7 network medians (Var(GES) x1e-8 | responsiveness x1e-4):")
    for nw in pd.unique(net):
        msk = net == nw
        print(f"  {nw:14s} Var={np.median(focal_var[msk])*1e8:.2f}   "
              f"resp={np.median(resp[msk])*1e4:.2f}  (n={msk.sum()})")

    print("\nSpin null (Alexander-Bloch, 1000 rotations) + Moran robustness...", flush=True)
    spins = compute_spins()
    # resp = mean(GES); cv = CV(GES) (the SF6 measures); var kept for back-compat
    # with the F4 DAT scatter (Figure3_PanelC_receptor_scatter_brains.py).
    t = corr_table(df, {"resp": resp, "cv": focal_cv, "var": focal_var}, spins)
    t = t.reindex(t.rho_resp.abs().sort_values(ascending=False).index).reset_index(drop=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3g}")
    print("\n=== mean(GES) and CV(GES) vs 19 PET receptor maps "
          "[rho | spin p | Moran p] ===")
    print(t[["name", "rho_resp", "p_resp", "p_resp_moran",
             "rho_cv", "p_cv", "p_cv_moran"]].to_string(index=False))
    for tn, lab in [("resp", "mean(GES)"), ("cv", "CV(GES)")]:
        print(f"{lab:10s}: spin {(t[f'p_{tn}']<0.05).sum()} nominal / {(t[f'q_{tn}']<0.05).sum()} FDR  |  "
              f"Moran {(t[f'p_{tn}_moran']<0.05).sum()} nominal / {(t[f'q_{tn}_moran']<0.05).sum()} FDR")

    csv = ROOT / "codes/HCP/results/Figure5_variability_receptor_correlations.csv"
    t.to_csv(csv, index=False)
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
