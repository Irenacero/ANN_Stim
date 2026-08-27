"""
Two controls for the "sparse high-energy sampling could fake the gating" worry.

Test B (decisive): does the gating survive when restricted to the densely
sampled middle of the baseline-energy distribution? For each (subject, cortical
target) recompute Spearman(E(t), GES(t)) using only time points with E(t) in the
[20,80] (and [10,90]) percentile range, and compare to the full-range value.
Source: the precomputed background-dependence CSV (GES and E per subj/roi/time).

Test A: is the MLP accurate enough at high energy? For each subject, one-step
prediction error on the held-out test split (last 20%), binned by E(t) decile,
normalized to each subject's mean error. Source: saved MLP models + inputs.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "codes/HCP/results/dataframes/HCP_4_df_background_dependence_ECts.csv"
PROC = ROOT / "codes/HCP/results/processed"
MODELS = ROOT / "codes/HCP/results/ANN_model"
N = 450

# ---------------------------------------------------------------------------
# Test B  — gating in the dense energy range (streamed, all subjects)
# ---------------------------------------------------------------------------
def test_B():
    COLS = ["sub_id", "roi", "time", "global_baseline_energy", "global_effect_size"]
    CHUNK = 3_000_000
    full, mid, wide = [], [], []     # per-(subject,roi) spearman: full / [20,80] / [10,90]

    def flush(df):
        et = df.drop_duplicates("time").set_index("time")["global_baseline_energy"]
        lo2, hi2 = et.quantile(0.20), et.quantile(0.80)
        lo1, hi1 = et.quantile(0.10), et.quantile(0.90)
        mid_t = set(et[(et >= lo2) & (et <= hi2)].index)
        wide_t = set(et[(et >= lo1) & (et <= hi1)].index)
        for roi, g in df[df.roi >= 50].groupby("roi"):
            E = g["global_baseline_energy"].to_numpy()
            G = g["global_effect_size"].to_numpy()
            t = g["time"].to_numpy()
            full.append(spearmanr(E, G)[0])
            m = np.array([x in mid_t for x in t])
            w = np.array([x in wide_t for x in t])
            mid.append(spearmanr(E[m], G[m])[0])
            wide.append(spearmanr(E[w], G[w])[0])

    buf = []
    cur = None
    nread = 0
    for chunk in pd.read_csv(CSV, usecols=COLS, chunksize=CHUNK):
        nread += len(chunk)
        for sid, sub in chunk.groupby("sub_id", sort=False):
            if cur is None:
                cur = sid
            if sid != cur:
                flush(pd.concat(buf, ignore_index=True))
                buf = []
                cur = sid
            buf.append(sub)
        print(f"  ...streamed {nread/1e6:.1f}M rows, {len(full)} targets done")
    if buf:
        flush(pd.concat(buf, ignore_index=True))

    full = np.array(full); mid = np.array(mid); wide = np.array(wide)
    print("\n=== TEST B: gating restricted to dense energy range ===")
    for name, a in [("full range", full), ("[10,90] pct", wide), ("[20,80] pct", mid)]:
        print(f"  {name:12s}: median rho = {np.nanmedian(a):+.3f}   "
              f"frac negative = {np.nanmean(a < 0):.3f}   n = {np.sum(~np.isnan(a))}")
    np.savez(ROOT / "codes/HCP/results/test_sparsity_B.npz",
             full=full, mid=mid, wide=wide)


# ---------------------------------------------------------------------------
# Test A  — held-out prediction accuracy vs energy decile
# ---------------------------------------------------------------------------
def test_A(n_subjects=100):
    import types
    import torch
    import torch.serialization
    sys.path.insert(0, str(ROOT / "codes/HCP/src"))
    import NPI
    # models were pickled referencing a `src` package -> alias it
    src_mod = types.ModuleType("src")
    src_mod.NPI = NPI
    sys.modules.setdefault("src", src_mod)
    sys.modules.setdefault("src.NPI", NPI)
    torch.serialization.add_safe_globals(
        [NPI.ANN_MLP, NPI.ANN_CNN, NPI.ANN_RNN, NPI.ANN_VAR])
    dev = torch.device("cpu")

    sids = sorted(p.name.split("_inputs.npy")[0] for p in PROC.glob("*_inputs.npy"))[:n_subjects]
    n_dec = 10
    abs_by_dec = np.zeros((len(sids), n_dec))   # absolute squared error
    rel_by_dec = np.zeros((len(sids), n_dec))   # error / ||true||^2  (magnitude-free)
    cos_by_dec = np.zeros((len(sids), n_dec))   # cosine(pred, true): scale-invariant accuracy
    for k, sid in enumerate(sids):
        inp = np.load(PROC / f"{sid}_inputs.npy")
        tgt = np.load(PROC / f"{sid}_targets.npy")
        ckpt = torch.load(MODELS / f"{sid}_MLP.pt", map_location=dev, weights_only=False)
        model = ckpt if hasattr(ckpt, "eval") else None
        if model is None:
            m = NPI.build_model("MLP", N, 3); m.load_state_dict(ckpt["model_state_dict"]); model = m
        model.eval()
        split = int(0.8 * inp.shape[0])
        Xte, Yte = inp[split:], tgt[split:]
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy()
        sq = np.sum((pred - Yte) ** 2, axis=1)            # (n_te,) absolute
        rel = sq / (np.sum(Yte ** 2, axis=1) + 1e-12)     # (n_te,) relative to target norm
        cos = (np.sum(pred * Yte, axis=1) /
               (np.linalg.norm(pred, axis=1) * np.linalg.norm(Yte, axis=1) + 1e-12))
        E = np.sum(Xte[:, -N:] ** 2, axis=1)              # (n_te,)
        dec = np.clip((np.argsort(np.argsort(E)) * n_dec // len(E)), 0, n_dec - 1)
        for d in range(n_dec):
            abs_by_dec[k, d] = sq[dec == d].mean()
            rel_by_dec[k, d] = rel[dec == d].mean()
            cos_by_dec[k, d] = cos[dec == d].mean()
        abs_by_dec[k] /= abs_by_dec[k].mean()
        rel_by_dec[k] /= rel_by_dec[k].mean()
        if k % 25 == 0:
            print(f"  Test A {k+1}/{len(sids)}  {sid}")
    print("\n=== TEST A: 1-step test error by energy decile (low->high), per-subject normalized ===")
    ma, sa = abs_by_dec.mean(0), abs_by_dec.std(0)
    mr, sr = rel_by_dec.mean(0), rel_by_dec.std(0)
    mc, sc = cos_by_dec.mean(0), cos_by_dec.std(0)
    print("  decile :  abs err (norm) |  rel err (norm) |  cosine(pred,true) [scale-free]")
    for d in range(n_dec):
        print(f"   {d+1:2d}    :  {ma[d]:.3f}        |  {mr[d]:.3f}       |  {mc[d]:.3f} +/-{sc[d]:.2f}")
    print(f"  high/low ratio:  abs = {ma[-1]/ma[0]:.2f}   rel = {mr[-1]/mr[0]:.2f}   "
          f"cosine low={mc[0]:.3f} high={mc[-1]:.3f}")
    np.savez(ROOT / "codes/HCP/results/test_sparsity_A.npz",
             abs_by_dec=abs_by_dec, rel_by_dec=rel_by_dec, cos_by_dec=cos_by_dec)


def panel_sparsity():
    """Combined supplementary draft (SF6): (A) gating survives in the densely
    sampled energy range; (B) model accuracy does not degrade at high energy."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import setup, figsize_mm, INK
    setup()
    import matplotlib.pyplot as plt

    B = np.load(ROOT / "codes/HCP/results/test_sparsity_B.npz")
    A = np.load(ROOT / "codes/HCP/results/test_sparsity_A.npz")
    full, wide, mid = B["full"], B["wide"], B["mid"]
    cos = A["cos_by_dec"]; absd = A["abs_by_dec"]
    MLP_BLUE = "#1f77b4"; VAR_ORANGE = "#e8853a"; GREY = "#9a9a9a"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_mm(150, 58),
                                   constrained_layout=True)

    # Panel A: gating per target, restricted to energy ranges
    data = [full[~np.isnan(full)], wide[~np.isnan(wide)], mid[~np.isnan(mid)]]
    labels = ["Full\nrange", "Central\n80%", "Central\n60%"]
    bp = ax1.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                     medianprops=dict(color=INK, lw=1.3))
    for patch in bp["boxes"]:
        patch.set(facecolor=MLP_BLUE, alpha=0.65, edgecolor=(0.12, 0.27, 0.42))
    ax1.axhline(0.0, color=VAR_ORANGE, lw=1.6, zorder=0)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Baseline-energy range used")
    ax1.set_ylabel(r"Gating correlation $\rho(E,$ GES$)$")
    for i, a in enumerate(data, 1):
        ax1.annotate(f"{np.median(a):+.2f}", (i, np.median(a)), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax1.set_title("Gating survives in well-sampled range", fontsize=8)
    ax1.text(-0.16, 1.02, "A", transform=ax1.transAxes, fontsize=11, fontweight="bold")

    # Panel B: accuracy vs energy decile
    dec = np.arange(1, 11)
    mc, sc = cos.mean(0), cos.std(0)
    ax2.plot(dec, mc, "-o", color=MLP_BLUE, ms=3.5, lw=1.6, label="cosine(pred, true)")
    ax2.fill_between(dec, mc - sc, mc + sc, color=MLP_BLUE, alpha=0.18, lw=0)
    ax2.set_ylim(0.80, 1.0)
    ax2.set_xlabel("Baseline-energy decile (low $\\rightarrow$ high)")
    ax2.set_ylabel("Cosine(pred, true)", color=MLP_BLUE)
    ax2.tick_params(axis="y", labelcolor=MLP_BLUE)
    axr = ax2.twinx()
    axr.plot(dec, absd.mean(0), "--s", color=GREY, ms=3, lw=1.2,
             label="abs. error (magnitude effect)")
    axr.set_ylabel("Abs. 1-step error (norm.)", color=GREY)
    axr.tick_params(axis="y", labelcolor=GREY)
    ax2.set_title("Model accuracy does not drop at high energy", fontsize=8)
    ax2.text(-0.16, 1.02, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower center", framealpha=0.85)

    pdf_dir = ROOT / "paper/ANN_fMRI_HCP/Figures/Inkscape/PDF"
    if not pdf_dir.parent.parent.exists():   # figure bundle: no paper/ tree
        pdf_dir = ROOT / "codes/figures/outputs"
    fig.savefig(pdf_dir / "SF6.pdf", dpi=600)
    fig.savefig(ROOT / "codes/figures/outputs/FigureS_sparsity_combined.png", dpi=300)
    plt.close(fig)
    print(f"  -> {pdf_dir/'SF6.pdf'}")


if __name__ == "__main__":
    # cache-first: the two analysis passes need the full 3.8 GB dataframe and
    # the 100 trained models; skip them when their outputs are already saved.
    if not (ROOT / "codes/HCP/results/test_sparsity_B.npz").exists():
        test_B()
    if not (ROOT / "codes/HCP/results/test_sparsity_A.npz").exists():
        test_A()
    panel_sparsity()
