"""Why the tree-block assumption failed, and what survives.

Finding: on the deterministic ultrametric matrix the level-1 wavelet (the mode
antisymmetric between i and its xor-1 sibling) has eigenvalue exactly -1. A
negative eigenvalue contributes NEGATIVELY to that pair's covariance, so the
most strongly CONNECTED pair is not the most CORRELATED pair. Greedy ICG
therefore crosses the first tree boundary at K=2, and -- because all level-1
partners are exactly tied -- which one it picks is decided by rounding noise.

From K=4 upward the clusters coincide with tree blocks again, so the closed form
should be exact there. Tested.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import linear_theory as lt
from hierarchical_closed_form import (A_deterministic, spectrum_closed_form,
                                      levels_closed_form)

np.seterr(all="ignore")
N = 512
PHIS = [1.5, 2.0, 2.83, 3.5, 5.0]

print("=" * 98)
print("1  the level-1 wavelet eigenvalue is exactly -1, for every Phi")
print("=" * 98)
print(f"{'Phi':<8}{'lambda_1 formula':<20}{'closest eigh eigenvalue':<26}{'corr(0,1) level 0':<20}{'corr(0,2) level 1'}")
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev, V = eigh(A, check_finite=False)
    lam1 = spectrum_closed_form(N, phi)[1][0]
    S, _ = lt.stationary_covariances(ev, V, 0.99)
    sd = np.sqrt(np.diag(S)); C = S / np.outer(sd, sd)
    print(f"{phi:<8}{lam1:<20.12f}{ev[np.argmin(np.abs(ev-lam1))]:<26.12f}"
          f"{C[0,1]:<20.8f}{C[0,2]:.8f}")
print("  => the nearest neighbour is the LEAST correlated of the close partners.")

print()
print("=" * 98)
print("2  all level-1 partners are exactly tied, so the K=2 pairing is noise-decided")
print("=" * 98)
phi = 2.83
A = A_deterministic(N, phi)
ev, V = eigh(A, check_finite=False)
S, _ = lt.stationary_covariances(ev, V, 0.99)
sd = np.sqrt(np.diag(S)); C = S / np.outer(sd, sd)
print(f"  corr(8,10) - corr(8,11) = {C[8,10]-C[8,11]:.3e}   (exactly 0 in theory)")
base = lt.greedy_pairs(S)
flips = []
for eps_seed in range(6):
    rng = np.random.default_rng(eps_seed)
    P = S * (1.0 + 1e-14 * rng.standard_normal(S.shape))
    P = (P + P.T) / 2
    pr = lt.greedy_pairs(P)
    same = set(map(tuple, map(sorted, base))) == set(map(tuple, map(sorted, pr)))
    flips.append(same)
print(f"  K=2 pairing unchanged under 1e-14 relative perturbation: {flips}")
print(f"  first 6 pairs unperturbed: {base[:6].tolist()}")

print()
print("=" * 98)
print("3  from K=4 upward, are the clusters exactly the tree blocks?")
print("=" * 98)
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev, V = eigh(A, check_finite=False)
    res = []
    for m in (0.5, 0.9, 0.99, 0.999, 0.9999):
        S, L_ = lt.stationary_covariances(ev, V, m)
        memb, K, ok = np.arange(N), 1, True
        while len(S) >= 4:
            if K >= 4:
                for c in range(memb.max() + 1):
                    idx = np.flatnonzero(memb == c)
                    if not (len(idx) == K and idx[0] % K == 0
                            and np.all(np.diff(idx) == 1)):
                        ok = False
            pairs = lt.greedy_pairs(S)
            a, b = pairs.T
            rl = np.empty(len(S), int)
            rl[a] = np.arange(len(pairs)); rl[b] = np.arange(len(pairs))
            memb = rl[memb]
            S, L_ = lt.pool(S, pairs), lt.pool(L_, pairs)
            K *= 2
        res.append(ok)
    print(f"  Phi={phi:<6} tree blocks for all K>=4 at m=0.5..0.9999 -> {res}")

print()
print("=" * 98)
print("4  overlap formula  abar_s(K) = K^2/n  if 2^s > K else 0,  checked at K>=4")
print("=" * 98)
phi = 2.83
A = A_deterministic(N, phi)
ev, V = eigh(A, check_finite=False)
S, L_ = lt.stationary_covariances(ev, V, 0.99)
memb, K = np.arange(N), 1
L_LEV = int(round(np.log2(N)))
while len(S) >= 4:
    if K in (4, 16, 64):
        nc = memb.max() + 1
        ind = np.zeros((N, nc)); ind[np.arange(N), memb] = 1.0
        ab = np.mean((V.T @ ind) ** 2, axis=1)
        # group measured overlaps by predicted class
        spec = spectrum_closed_form(N, phi)
        # map each eigenvector to its wavelet level by matching eigenvalue
        lev = np.empty(N, int)
        for i, e in enumerate(ev):
            k = int(np.argmin([abs(e - v) for v, _ in spec]))
            lev[i] = 0 if k == 0 else k            # 0 = uniform, else wavelet level s
        above = np.array([(l == 0) or (2 ** l > K) for l in lev])
        print(f"  K={K:<4} predicted K^2/n = {K**2/N:.6f}   "
              f"measured mean over 'support > K' modes = {ab[above].mean():.6f}   "
              f"max abs dev = {np.abs(ab[above]-K**2/N).max():.2e}")
        print(f"         measured max over 'support <= K' modes = {ab[~above].max():.2e}  (predicted 0)")
    pairs = lt.greedy_pairs(S)
    a, b = pairs.T
    rl = np.empty(len(S), int); rl[a] = np.arange(len(pairs)); rl[b] = np.arange(len(pairs))
    memb = rl[memb]
    S, L_ = lt.pool(S, pairs), lt.pool(L_, pairs)
    K *= 2

print()
print("=" * 98)
print("5  closed form vs module, FITTING K=4..64 only (drop_first=2)")
print("=" * 98)
print(f"{'Phi':<8}{'m':<10}{'alpha module':<15}{'alpha formula':<16}{'beta module':<14}"
      f"{'beta formula':<15}{'|d alpha|':<12}{'|d beta|'}")
worst = 0.0
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev, V = eigh(A, check_finite=False)
    for m in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999):
        S, L_ = lt.stationary_covariances(ev, V, m)
        lv = lt.coarse_grain(S, L_, 4)
        cf = levels_closed_form(N, phi, m, lv.K.tolist())
        fm = lt.fit_scaling(lv, drop_first=2, drop_last=1)
        fc = lt.fit_scaling(cf, drop_first=2, drop_last=1)
        da, db = abs(fm["alpha"]-fc["alpha"]), abs(fm["beta"]-fc["beta"])
        worst = max(worst, da, db)
        print(f"{phi:<8}{m:<10}{fm['alpha']:<15.8f}{fc['alpha']:<16.8f}"
              f"{fm['beta']:<14.8f}{fc['beta']:<15.8f}{da:<12.2e}{db:.2e}")
print(f"\n  worst discrepancy with K=2 excluded: {worst:.3e}")
