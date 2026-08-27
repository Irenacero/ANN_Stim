"""
task_gating.py — Does the resting-state gating law generalize to task states?

For each participant we load the rest-trained twin, feed it the LANGUAGE task
states, and compute the global effect size (GES) of a virtual perturbation at
every cortical target and every task volume. We then ask:

  (1) Gating law on task data: across all task volumes, is GES lower when the
      baseline energy E(t) is higher (Spearman rho < 0, as at rest)?
  (2) During-vs-rest contrast (within-block sampling): is GES smaller for states
      sampled *inside* the lagged task blocks (high energy) than for states
      *outside* them, i.e. is the brain transiently less stimulable during task?

GES convention matches HCP_4.1: GES[t, j] = sum_i (x^(j)_i(t+1) - x_i(t+1))^2,
summed over all 450 downstream regions; targets j restricted to the 400 cortical
parcels (indices 50..449); perturbation amplitude 0.1.
"""

from __future__ import annotations

import os
import sys

# Make both `import NPI` (local) and `import src.NPI` (the pickled class path of
# the saved twins) resolve, regardless of where this script is launched from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_HCP_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _HCP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

import NPI
import task_state as tsm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ANN_model")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PERT = 0.1
CORTICAL = np.arange(50, 450)   # 400 Schaefer cortical targets


def load_twin(subject: str):
    """Load a full serialized MLP twin and put it in eval mode."""
    path = os.path.join(MODEL_DIR, f"id_{subject}_MLP.pt")
    torch.serialization.add_safe_globals([NPI.ANN_MLP, NPI.ANN_CNN, NPI.ANN_RNN, NPI.ANN_VAR])
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ckpt.to(device) if hasattr(ckpt, "eval") else ckpt
    model.eval()
    return model


def ges_for_windows(model, input_X: np.ndarray, targets=CORTICAL, pert=PERT, n_roi=tsm.N_ROI):
    """
    GES[m, k] = sum_i (perturbed - unperturbed)^2 for window m, target targets[k].

    Lean: computes the unperturbed forward pass once and never stores the full
    (M, N, N) tensor. Returns (M, len(targets)).
    """
    S = input_X.shape[1] // n_roi
    Xt = torch.tensor(input_X, dtype=torch.float32, device=device)
    with torch.no_grad():
        unpert = model(Xt).cpu().numpy()                 # (M, N)
    M = input_X.shape[0]
    GES = np.empty((M, len(targets)), dtype=float)
    for k, j in enumerate(targets):
        pert_flat = np.zeros(S * n_roi, dtype=np.float32)
        pert_flat[(S - 1) * n_roi + j] = pert            # last frame, region j
        with torch.no_grad():
            pert_out = model(Xt + torch.tensor(pert_flat, device=device)).cpu().numpy()
        EC = pert_out - unpert                           # (M, N)
        GES[:, k] = np.sum(EC ** 2, axis=1)
    return GES


def analyze_subject(subject: str, lag_s: float = tsm.DEFAULT_LAG_S):
    """
    Compute the gating result for one subject using within-block sampling.

    Returns a dict of per-subject summary statistics.
    """
    ts = tsm.load_language_bold(subject, preprocess=True)        # (T, N)
    T, N = ts.shape
    blocks = tsm.load_block_onsets(subject)
    anchors = tsm.block_window_anchors(blocks, T, lag_s=lag_s)   # during / rest

    # All valid task volumes, each labelled during (1) or rest (0)
    vols, label = [], []
    for it in anchors["during"]:
        vols.append(it["vol"]); label.append(1)
    for it in anchors["rest"]:
        vols.append(it["vol"]); label.append(0)
    vols = np.array(vols); label = np.array(label)
    order = np.argsort(vols)
    vols, label = vols[order], label[order]

    X = tsm.windows_at(ts, vols)                                 # (M, S*N)
    E = tsm.baseline_energy(X, N)                                # (M,) baseline energy
    GES = ges_for_windows(model=TW[subject], input_X=X)         # (M, 400)
    ges_w = GES.mean(axis=1)                                     # mean over cortical targets -> (M,)

    dur = label == 1
    rst = label == 0
    # (1) gating law on task states
    rho_task, _ = spearmanr(E, ges_w)
    # (2) during vs rest contrast
    return {
        "subject": subject,
        "n_during": int(dur.sum()),
        "n_rest": int(rst.sum()),
        "E_during": float(E[dur].mean()),
        "E_rest": float(E[rst].mean()),
        "GES_during": float(ges_w[dur].mean()),
        "GES_rest": float(ges_w[rst].mean()),
        "varGES_during": float(ges_w[dur].var()),
        "varGES_rest": float(ges_w[rst].var()),
        "rho_E_GES_task": float(rho_task),
    }


# module-level cache so analyze_subject can reuse a loaded model
TW: dict = {}


def run(subjects, lag_s: float = tsm.DEFAULT_LAG_S, verbose: bool = True):
    import pandas as pd
    rows = []
    for n, s in enumerate(subjects, 1):
        TW[s] = load_twin(s)
        rows.append(analyze_subject(s, lag_s=lag_s))
        del TW[s]
        if verbose and (n % 10 == 0 or n == len(subjects)):
            print(f"  {n}/{len(subjects)} done", flush=True)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import sys, pandas as pd
    subs = [l.strip() for l in open(os.path.join(tsm.DATA_ROOT, "language_subjects_paper100.txt"))]
    if len(sys.argv) > 1:
        subs = subs[: int(sys.argv[1])]
    print(f"Computing task-state gating for {len(subs)} subjects (lag={tsm.DEFAULT_LAG_S}s)...")
    df = run(subs)

    out = os.path.join(RESULTS_DIR, "task_state_gating_language.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # ---- group-level summary ----
    print("\n=== Gating law on task states ===")
    rho = df["rho_E_GES_task"].values
    w = wilcoxon(rho)
    print(f"  per-subject Spearman rho(E, GES): median {np.median(rho):+.3f}, "
          f"negative in {(rho<0).mean()*100:.0f}% of subjects (Wilcoxon p={w.pvalue:.2e})")

    print("\n=== During (high-E) vs Rest (low-E) task states ===")
    gd, gr = df["GES_during"].values, df["GES_rest"].values
    wge = wilcoxon(gd, gr)
    print(f"  E:   during {df['E_during'].mean():.1f}  vs rest {df['E_rest'].mean():.1f}")
    print(f"  GES: during {gd.mean():.4g}  vs rest {gr.mean():.4g}  "
          f"(during<rest in {(gd<gr).mean()*100:.0f}% of subjects, Wilcoxon p={wge.pvalue:.2e})")
    vd, vr = df["varGES_during"].values, df["varGES_rest"].values
    wv = wilcoxon(vd, vr)
    print(f"  Var(GES): during {vd.mean():.4g} vs rest {vr.mean():.4g} "
          f"(Wilcoxon p={wv.pvalue:.2e})")
