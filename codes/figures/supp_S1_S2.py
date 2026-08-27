"""
Two new supplementary controls.

S1  Subject specificity: each twin predicts its OWN participant's held-out data
    better than it predicts other participants. Self vs cross prediction.

S2  Perturbation-strength robustness: the global effect size scales as the
    square of the perturbation amplitude (linear regime) and the state-gating is
    unchanged across amplitudes; the spatial map of responsiveness is preserved.
"""
from __future__ import annotations
from pathlib import Path
import sys
import types
import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "codes/HCP/results/processed"
MODELS = ROOT / "codes/HCP/results/ANN_model"
OUT = ROOT / "codes/figures/outputs"
PDFDIR = ROOT / "paper/ANN_fMRI_HCP/Figures/Inkscape/PDF"
if not PDFDIR.parent.exists():          # figure bundle: no paper/ tree
    PDFDIR = OUT
S1_NPZ = ROOT / "codes/HCP/results/supp_S1_specificity.npz"
S2_NPZ = ROOT / "codes/HCP/results/supp_S2_pertstrength.npz"
N = 450; S = 3; LAST = (S - 1) * N
MLP_BLUE = "#1f77b4"; ORANGE = "#e8853a"; INKC = "#1a1a1a"


def _torch():
    import torch, torch.serialization
    sys.path.insert(0, str(ROOT / "codes/HCP/src"))
    import NPI
    src = types.ModuleType("src"); src.NPI = NPI
    sys.modules.setdefault("src", src); sys.modules.setdefault("src.NPI", NPI)
    torch.serialization.add_safe_globals([NPI.ANN_MLP, NPI.ANN_CNN, NPI.ANN_RNN, NPI.ANN_VAR])
    return torch, NPI


def load_model(sid, torch, NPI):
    ckpt = torch.load(MODELS / f"{sid}_MLP.pt", map_location="cpu", weights_only=False)
    if hasattr(ckpt, "eval"):
        return ckpt.eval()
    m = NPI.build_model("MLP", N, S); m.load_state_dict(ckpt["model_state_dict"]); return m.eval()


# ---------------------------------------------------------------------------
# S1  subject specificity
# ---------------------------------------------------------------------------
def s1_specificity(n_test=400):
    if S1_NPZ.exists():   # cache-first: plot from the saved analysis outputs
        d = np.load(S1_NPZ, allow_pickle=True)
        C, self_acc, cross_mean = d["C"], d["self_acc"], d["cross_mean"]
        rank_self_best = np.mean([C[i].argmax() == i for i in range(C.shape[0])])
        w = wilcoxon(self_acc, cross_mean)
        print(f"Loaded cache {S1_NPZ}")
        print(f"  self mean = {self_acc.mean():.4f}  cross mean = {cross_mean.mean():.4f}")
        print(f"  self is best-matching model: {rank_self_best*100:.0f}% of participants")
        print(f"  Wilcoxon self>cross: stat={w.statistic:.0f} p={w.pvalue:.2e}")
        _plot_s1(C, self_acc, cross_mean, rank_self_best)
        return
    import torch
    torch, NPI = _torch()
    sids = sorted(p.name.split("_inputs.npy")[0] for p in PROC.glob("*_inputs.npy"))
    n = len(sids)
    # cache each subject's held-out test slice (subsampled)
    Xte, Yte = {}, {}
    for sid in sids:
        inp = np.load(PROC / f"{sid}_inputs.npy"); tgt = np.load(PROC / f"{sid}_targets.npy")
        sp = int(0.8 * inp.shape[0])
        xi, yi = inp[sp:], tgt[sp:]
        idx = np.linspace(0, xi.shape[0] - 1, min(n_test, xi.shape[0])).astype(int)
        Xte[sid] = xi[idx]; Yte[sid] = yi[idx]
    C = np.zeros((n, n))   # C[i,j] = cosine accuracy of model j on subject i
    for j, sj in enumerate(sids):
        mj = load_model(sj, torch, NPI)
        with torch.no_grad():
            for i, si in enumerate(sids):
                pred = mj(torch.tensor(Xte[si], dtype=torch.float32)).cpu().numpy()
                Y = Yte[si]
                cos = (np.sum(pred * Y, axis=1) /
                       (np.linalg.norm(pred, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12))
                C[i, j] = cos.mean()
        if j % 25 == 0:
            print(f"  S1 model {j+1}/{n}")
    self_acc = np.diag(C)
    cross = C.copy(); np.fill_diagonal(cross, np.nan)
    cross_mean = np.nanmean(cross, axis=1)
    rank_self_best = np.mean([C[i].argmax() == i for i in range(n)])
    w = wilcoxon(self_acc, cross_mean)
    print(f"\n=== S1 subject specificity (cosine, n={n}) ===")
    print(f"  self mean   = {self_acc.mean():.4f}")
    print(f"  cross mean  = {cross_mean.mean():.4f}")
    print(f"  self is best-matching model: {rank_self_best*100:.0f}% of participants")
    print(f"  Wilcoxon self>cross: stat={w.statistic:.0f} p={w.pvalue:.2e}")
    np.savez(ROOT / "codes/HCP/results/supp_S1_specificity.npz",
             C=C, self_acc=self_acc, cross_mean=cross_mean, sids=np.array(sids))
    _plot_s1(C, self_acc, cross_mean, rank_self_best)


def _plot_s1(C, self_acc, cross_mean, rank_self_best):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import setup, figsize_mm
    setup()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_mm(150, 60), constrained_layout=True)
    im = ax1.imshow(C, cmap="magma", aspect="auto")
    ax1.set_xlabel("Model (participant $j$)"); ax1.set_ylabel("Data (participant $i$)")
    ax1.set_title("One-step accuracy: model $j$ on data $i$", fontsize=8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046); cb.set_label("cosine(pred, true)", fontsize=7)
    ax1.text(-0.18, 1.02, "A", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    ax2.hist(cross_mean, bins=30, color="#b9c9e0", alpha=0.9, label="cross-subject (mean)")
    ax2.hist(self_acc, bins=30, color=MLP_BLUE, alpha=0.85, label="self")
    ax2.axvline(self_acc.mean(), color=MLP_BLUE, lw=1.2, ls=(0, (4, 2)))
    ax2.axvline(cross_mean.mean(), color="#5a78a8", lw=1.2, ls=(0, (4, 2)))
    ax2.set_xlabel("cosine(pred, true)"); ax2.set_ylabel("Participants")
    ax2.set_title(f"Self best for {rank_self_best*100:.0f}% of participants", fontsize=8)
    ax2.legend(fontsize=6.5, framealpha=0.85)
    ax2.text(-0.18, 1.02, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold")
    fig.savefig(PDFDIR / "SF1.pdf", dpi=600)
    fig.savefig(OUT / "supp_S1_specificity.png", dpi=200)
    plt.close(fig); print(f"  -> {PDFDIR/'SF1.pdf'}")


# ---------------------------------------------------------------------------
# S2  perturbation-strength robustness
# ---------------------------------------------------------------------------
def s2_pert_strength(n_subj=8, n_targets=60, n_time=200,
                     amps=(0.025, 0.05, 0.1, 0.2, 0.4)):
    if S2_NPZ.exists():   # cache-first: plot from the saved analysis outputs
        d = np.load(S2_NPZ)
        meanGES, rho = d["meanGES"], d["rho"]
        amps, spatial_r = d["amps"], d["spatial_r"]
        gpooled = meanGES.mean(axis=(0, 2))
        slope = np.polyfit(np.log(amps), np.log(gpooled), 1)[0]
        print(f"Loaded cache {S2_NPZ}")
        print(f"  log-log slope of mean GES vs amplitude = {slope:.2f}")
        _plot_s2(amps, gpooled, slope, spatial_r, rho)
        return
    import torch
    torch, NPI = _torch()
    sids = sorted(p.name.split("_inputs.npy")[0] for p in PROC.glob("*_inputs.npy"))[:n_subj]
    targets = np.linspace(50, N - 1, n_targets).astype(int)
    amps = np.array(amps)
    meanGES = np.zeros((len(sids), len(amps), n_targets))      # spatial map per amp
    rho = np.zeros((len(sids), len(amps), n_targets))          # gating per amp
    for s, sid in enumerate(sids):
        inp = np.load(PROC / f"{sid}_inputs.npy")
        idx = np.linspace(0, min(500, inp.shape[0]) - 1, n_time).astype(int)
        X = inp[idx]
        E = np.sum(X[:, -N:] ** 2, axis=1)
        model = load_model(sid, torch, NPI)
        with torch.no_grad():
            base = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
            for a, amp in enumerate(amps):
                for ti, j in enumerate(targets):
                    Xp = X.copy(); Xp[:, LAST + j] += amp
                    pert = model(torch.tensor(Xp, dtype=torch.float32)).cpu().numpy()
                    ges = np.sum((pert - base) ** 2, axis=1)        # (n_time,)
                    meanGES[s, a, ti] = ges.mean()
                    rho[s, a, ti] = spearmanr(E, ges)[0]
        print(f"  S2 subj {s+1}/{len(sids)}  {sid}")
    # scaling: log mean GES vs log amp (pooled), slope ~ 2 in linear regime
    ref = list(amps).index(0.1)
    spatial_r = np.zeros((len(sids), len(amps)))
    for s in range(len(sids)):
        for a in range(len(amps)):
            spatial_r[s, a] = spearmanr(meanGES[s, a], meanGES[s, ref])[0]
    gpooled = meanGES.mean(axis=(0, 2))                              # mean GES per amp
    slope = np.polyfit(np.log(amps), np.log(gpooled), 1)[0]
    print(f"\n=== S2 perturbation-strength robustness ===")
    print(f"  log-log slope of mean GES vs amplitude = {slope:.2f}  (2 = linear regime)")
    print(f"  spatial map corr vs amp=0.1: " +
          "  ".join(f"{amps[a]:.3f}->{np.median(spatial_r[:,a]):.3f}" for a in range(len(amps))))
    print(f"  gating rho (median over targets/subj) per amp: " +
          "  ".join(f"{amps[a]:.3f}->{np.median(rho[:,a,:]):+.3f}" for a in range(len(amps))))
    np.savez(ROOT / "codes/HCP/results/supp_S2_pertstrength.npz",
             meanGES=meanGES, rho=rho, amps=amps, spatial_r=spatial_r)
    _plot_s2(amps, gpooled, slope, spatial_r, rho)


def _plot_s2(amps, gpooled, slope, spatial_r, rho):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import setup, figsize_mm
    setup()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_mm(150, 60), constrained_layout=True)
    # A: GES vs amplitude (log-log) + slope-2 guide
    ax1.loglog(amps, gpooled, "o-", color=MLP_BLUE, ms=4, lw=1.5, label="mean GES")
    g2 = gpooled[list(amps).index(0.1)] * (amps / 0.1) ** 2
    ax1.loglog(amps, g2, "--", color=ORANGE, lw=1.4, label=r"$\propto$ amplitude$^2$")
    ax1.set_xlabel("Perturbation amplitude"); ax1.set_ylabel("mean GES")
    ax1.set_title(f"Linear regime (slope = {slope:.2f})", fontsize=8)
    ax1.legend(fontsize=6.5, framealpha=0.85)
    ax1.text(-0.2, 1.02, "A", transform=ax1.transAxes, fontsize=11, fontweight="bold")
    # B: gating rho per amplitude (boxplots over targets/subjects)
    data = [rho[:, a, :].ravel() for a in range(len(amps))]
    bp = ax2.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                     medianprops=dict(color=INKC, lw=1.2))
    for p in bp["boxes"]:
        p.set(facecolor=MLP_BLUE, alpha=0.6, edgecolor=(0.12, 0.27, 0.42))
    ax2.axhline(0, color=ORANGE, lw=1.4, zorder=0)
    ax2.set_xticklabels([f"{a:.3f}" for a in amps])
    ax2.set_xlabel("Perturbation amplitude"); ax2.set_ylabel(r"Gating $\rho(E,$ GES$)$")
    ax2.set_title("Gating unchanged across amplitudes", fontsize=8)
    ax2.text(-0.2, 1.02, "B", transform=ax2.transAxes, fontsize=11, fontweight="bold")
    fig.savefig(PDFDIR / "SF2.pdf", dpi=600)
    fig.savefig(OUT / "supp_S2_pertstrength.png", dpi=200)
    plt.close(fig); print(f"  -> {PDFDIR/'SF2.pdf'}")


if __name__ == "__main__":
    s1_specificity()
    s2_pert_strength()
