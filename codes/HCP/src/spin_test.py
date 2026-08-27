"""
spin_test.py — Alexander-Bloch / Vazquez-Rodriguez spatial spin null for the
Schaefer-400 cortical parcellation, built from MNI parcel centroids.

Generates, for each of n_rotate random rotations of the cortical sphere, a
reassignment index so that a map `a` can be spun (`a[idx]`) while preserving its
spatial autocorrelation. Left and right hemispheres are rotated with mirrored
rotations to respect bilateral symmetry (Alexander-Bloch et al. 2018).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

CENTROIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figures", "_atlases", "Schaefer400_centroids_MNI.csv")


def load_centroids(path: str = CENTROIDS):
    """Return (coords (400,3), hemi (400,) of 'L'/'R', names (400,)) in atlas order."""
    df = pd.read_csv(path)
    coords = df[["R", "A", "S"]].to_numpy(dtype=float)
    names = df["ROI Name"].to_numpy()
    hemi = np.array(["L" if "_LH_" in n else "R" for n in names])
    return coords, hemi, names


def _haar_rotation(rng):
    """A uniformly random 3x3 rotation/reflection (orthogonal) matrix via QR."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))   # fix QR sign ambiguity -> Haar measure
    return Q


def gen_spin_indices(coords, hemi, n_rotate=1000, seed=0):
    """
    Returns spin_idx (N, n_rotate): column r is a permutation-like reassignment of
    parcels under rotation r. `a[spin_idx[:, r]]` is the r-th spun map.

    Vazquez-Rodriguez assignment: each original parcel takes the value of the
    nearest rotated parcel (nearest-neighbour, within hemisphere).
    """
    rng = np.random.default_rng(seed)
    N = coords.shape[0]
    L = hemi == "L"
    R = hemi == "R"
    # Project each hemisphere's centroids onto the unit sphere (center, normalize)
    sph = np.zeros_like(coords)
    for m in (L, R):
        c = coords[m] - coords[m].mean(0)
        sph[m] = c / np.linalg.norm(c, axis=1, keepdims=True)

    refl = np.diag([-1.0, 1.0, 1.0])       # mirror across the YZ (L<->R) plane
    idx = np.empty((N, n_rotate), dtype=int)
    iL = np.where(L)[0]
    iR = np.where(R)[0]
    for r in range(n_rotate):
        rot_l = _haar_rotation(rng)
        rot_r = refl @ rot_l @ refl        # mirrored rotation for right hemisphere
        for m_idx, rot in ((iL, rot_l), (iR, rot_r)):
            orig = sph[m_idx]
            rotated = orig @ rot.T
            # each original parcel <- nearest rotated parcel (both within hemisphere)
            nn = cdist(orig, rotated).argmin(axis=1)
            idx[m_idx, r] = m_idx[nn]
    return idx


if __name__ == "__main__":
    # sanity check: spun maps preserve spatial autocorrelation, destroy alignment
    from scipy.stats import spearmanr
    coords, hemi, names = load_centroids()
    print(f"loaded {len(names)} centroids: L={np.sum(hemi=='L')} R={np.sum(hemi=='R')}")
    idx = gen_spin_indices(coords, hemi, n_rotate=200, seed=0)
    rng = np.random.default_rng(1)
    a = rng.normal(size=len(names))
    # smooth the map over the sphere so it has autocorrelation
    D = cdist(coords, coords)
    a = (np.exp(-(D / 30) ** 2) @ a)
    # neighbour-similarity preserved under spin?
    near = D + np.eye(len(names)) * 1e9
    nnb = near.argmin(1)
    real = spearmanr(a, a[nnb]).statistic
    spun = np.mean([spearmanr(a[idx[:, r]], a[idx[:, r]][nnb]).statistic for r in range(50)])
    print(f"neighbour autocorr: real {real:.2f} vs spun {spun:.2f} (should both be high)")
