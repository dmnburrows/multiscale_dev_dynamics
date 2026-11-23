"""Is there a genuine closed form alpha(m), beta(m)?

For the ring there is. Three facts close the loop:

  (i)   the adjacency is circulant, so eigenvalues are known exactly
              lambda_j = 2 sum_{d=1..k/2} cos(2 pi j d / n),   rho = k
  (ii)  eigenvectors are Fourier modes, known exactly
  (iii) greedy pairing on a ring yields CONTIGUOUS blocks of K consecutive
        sites (tested below), and the squared overlap of Fourier mode j with a
        contiguous block of length K is the Dirichlet kernel, independent of
        where the block starts:
              abar_j(K) = D_K(j) / n,  D_K(j) = sin^2(pi j K/n)/sin^2(pi j/n)
        with D_K(0) = K^2.

Hence, with s_j = lambda_j / k:

   V(K)   = (1/n) sum_j  D_K(j) / (1 - m^2 s_j^2)
   tau(K) = [ sum_j D_K(j) / ((1-m^2 s_j^2)(1-m s_j)) ] / [ sum_j D_K(j)/(1-m^2 s_j^2) ]

No eigendecomposition, no covariance matrix, no cascade, no partition search:
elementary functions of m, n, k, K only. alpha and beta are its log-slopes.
Tested against linear_theory.py below.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import linear_theory as lt

np.seterr(all="ignore")


# ----------------------------------------------------------------- test (iii)
def partition_is_contiguous(n, k):
    """Run lt's own cascade and check every cluster is a run of consecutive sites."""
    A = lt.make_network(n, "ring", k, 7)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, 0.99)
    memb, K, report = np.arange(n), 1, []
    while len(S) >= 4:
        ok = True
        for c in range(memb.max() + 1):
            idx = np.flatnonzero(memb == c)
            # contiguous modulo n
            d = np.diff(idx)
            if not (np.all(d == 1) or (np.sum(d != 1) == 1 and idx[0] == 0)):
                ok = False
        report.append((K, memb.max() + 1, ok))
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs)); relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return report


# ------------------------------------------------------------- the closed form
def ring_spectrum(n, k):
    j = np.arange(n)
    d = np.arange(1, k // 2 + 1)[:, None]
    lam = 2.0 * np.cos(2.0 * np.pi * j[None, :] * d / n).sum(axis=0)
    return lam / k                                  # s_j, rho = k


def dirichlet(n, K):
    j = np.arange(n)
    out = np.empty(n)
    out[0] = K ** 2
    jj = j[1:]
    out[1:] = np.sin(np.pi * jj * K / n) ** 2 / np.sin(np.pi * jj / n) ** 2
    return out


def closed_form_levels(n, k, m, Ks):
    s = ring_spectrum(n, k)
    w = 1.0 / (1.0 - m ** 2 * s ** 2)
    t = 1.0 / (1.0 - m * s)
    rows = []
    for K in Ks:
        D = dirichlet(n, K)
        num = float(np.sum(w * D)) / n
        rows.append({"K": K, "variance": num,
                     "tau": float(np.sum(w * t * D)) / n / num})
    return pd.DataFrame(rows)


def module_levels(n, k, m):
    A = lt.make_network(n, "ring", k, 7)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, m)
    return lt.coarse_grain(S, L, 4)


N = 512
print("=" * 94)
print("STEP 1  does lt.greedy_pairs give contiguous blocks on a ring?")
print("=" * 94)
for k in (2, 4, 6):
    rep = partition_is_contiguous(N, k)
    print(f"  ring k={k}: contiguous at every level -> "
          f"{all(ok for _, _, ok in rep)}   (levels K=1..{rep[-1][0]})")

print()
print("=" * 94)
print("STEP 2  closed form vs linear_theory.py, level by level   (ring k=4, N=512)")
print("=" * 94)
print(f"{'m':<9}{'K':<6}{'V module':<16}{'V closed form':<16}{'rel err':<12}"
      f"{'tau module':<14}{'tau closed':<14}{'rel err'}")
for m in (0.5, 0.9, 0.99, 0.999, 0.9999):
    lv = module_levels(N, 4, m)
    cf = closed_form_levels(N, 4, m, lv.K.tolist())
    for i in (0, 3, 6, len(lv) - 1):
        print(f"{m:<9}{int(lv.K[i]):<6}{lv.variance[i]:<16.8g}{cf.variance[i]:<16.8g}"
              f"{abs(cf.variance[i]/lv.variance[i]-1):<12.2e}"
              f"{lv.tau[i]:<14.8g}{cf.tau[i]:<14.8g}"
              f"{abs(cf.tau[i]/lv.tau[i]-1):.2e}")

print()
print("=" * 94)
print("STEP 3  alpha and beta from the closed form alone, no matrices at all")
print("=" * 94)
print(f"{'k':<5}{'m':<10}{'alpha module':<15}{'alpha closed':<15}{'beta module':<14}"
      f"{'beta closed':<14}{'|d alpha|':<12}{'|d beta|'}")
for k in (2, 4, 6):
    for m in (0.5, 0.9, 0.99, 0.995, 0.999, 0.9999, 0.99999):
        lv = module_levels(N, k, m)
        cf = closed_form_levels(N, k, m, lv.K.tolist())
        fm, fc = lt.fit_scaling(lv), lt.fit_scaling(cf)
        print(f"{k:<5}{m:<10}{fm['alpha']:<15.6f}{fc['alpha']:<15.6f}"
              f"{fm['beta']:<14.6f}{fc['beta']:<14.6f}"
              f"{abs(fm['alpha']-fc['alpha']):<12.2e}{abs(fm['beta']-fc['beta']):.2e}")

print()
print("=" * 94)
print("STEP 4  the closed form evaluated where the module cannot go")
print("        (n up to 2^20, and 1-m down to 1e-9: no 512x512 eigendecomposition)")
print("=" * 94)
print(f"{'n':<10}{'k':<4}{'m':<12}{'alpha':<11}{'beta':<11}{'1-m'}")
for n in (512, 4096, 65536, 1048576):
    for m in (0.99, 0.999999, 1 - 1e-9):
        Ks = [2 ** i for i in range(0, 12)]
        cf = closed_form_levels(n, 4, m, Ks)
        f = lt.fit_scaling(cf)
        print(f"{n:<10}{4:<4}{m:<12.9g}{f['alpha']:<11.4f}{f['beta']:<11.4f}{1-m:.1e}")
