"""What IS a "pair one step apart" after coarse-graining -- and is beta_b ~ 0 a
finding or a tautology?

band_cascade sums the MICROSCOPIC covariance matrices S and L over the original
neurons inside each cluster:

    V(K) = (1/n_c) sum_c sum_{i,j in cluster c} S_ij

S, L and the band index B are computed once and never recomputed. So a "pair" is
always a pair of ORIGINAL single neurons, never a pair of clusters, and "one step
apart at K=64" means two originally adjacent neurons that both happen to sit
inside the same cluster of 64. The band split is a split of that double sum.

THE WORRY. For a ring, A is circulant, so S and L are circulant too: S_{i,i+1}
is the SAME number for every i. Then

    tau_1(K) = sum_{adjacent co-clustered pairs} L_ij / sum S_ij
             = L_{i,i+1} / S_{i,i+1}

for every K, whatever the clustering does, because it is the ratio of two fixed
numbers times the same count. beta_1 = 0 would then be an algebraic identity, not
a property of coarse-graining -- and the claim "nothing in the ring gets slower
as you coarse-grain" would be vacuous for the single-distance bands.

More generally, for a band b spanning several distances,

    tau_b(K) = sum_{d in b} n_d(K) l(d) / sum_{d in b} n_d(K) s(d)

a weighted average of the FIXED menu {l(d)/s(d) : d in b}, so tau_b(K) is pinned
between the smallest and largest tau(d) in the band however the weights move.
A dyadic band spans a factor 2 in d, so that interval is narrow by construction.

WHAT THIS SCRIPT CHECKS
  1  Is S actually circulant for the ring, and NOT for HM?
  2  Is tau_1(K) exactly L_01/S_01, i.e. is beta_1 = 0 a tautology?
  3  For contiguous clusters V(K) must equal a triangular-kernel sum over the
     microscopic correlation function, with NO coarse-graining needed:
         n_d(K) = K (d=0), 2(K-d) (1<=d<K), 0 otherwise
         V(K) = sum_d n_d(K) s(d)
     If that holds exactly, the ring's whole ICG cascade is block-variance of a
     stationary 1-D process and alpha, beta follow from the tails of s and l.
  4  THE REAL DISCRIMINATOR. Within a band, how much spread is there in the
     per-pair timescale tau_ij = L_ij/S_ij? Ring: zero by symmetry. HM: wide, so
     correlation-greedy ICG can select its slow end -- which is where HM's
     beta_b must come from. This is measurable in data without any model.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u ring_block_variance.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
from range_composition import N, band_cascade, band_index, band_label, best_m, net

np.seterr(all="ignore")


def circulant_defect(M):
    """Max deviation of M from the circulant matrix with the same first row."""
    n = len(M)
    i = np.arange(n)
    ref = M[0][(i[None, :] - i[:, None]) % n]
    return float(np.abs(M - ref).max() / np.abs(M).max())


def report(label, kind, par, metric):
    A = net(kind, par, 1)
    ev, V = eigh(A, check_finite=False)
    dist, lm, f = best_m(ev, V)
    S, L = lt.stationary_covariances(ev, V, 1 - lm)
    print("\n" + "=" * 100)
    print(f"{label}   1-m={lm:.2e}   alpha={f['alpha']:.3f}  beta={f['beta']:.3f}")
    print("=" * 100)
    print(f"  1  circulant defect of S: {circulant_defect(S):.2e}   "
          f"of L: {circulant_defect(L):.2e}   (0 = exactly circulant)")

    B = band_index(N, metric)
    nb = int(B.max()) + 1
    rows = band_cascade(S, L, B, nb)

    # 2  is band 1 exactly the single-pair ratio?
    ratio = L[0, 1] / S[0, 1]
    got = [r["T_b"][1] / r["V_b"][1] for r in rows[1:]]
    print(f"  2  L_01/S_01 = {ratio:.4f};  tau_1(K) over K=2..{rows[-1]['K']}: "
          f"{', '.join(f'{g:.4f}' for g in got[:5])}{' ...' if len(got) > 5 else ''}")
    print(f"     max |tau_1(K) - L_01/S_01| / tau = "
          f"{max(abs(g - ratio) for g in got) / abs(ratio):.2e}")

    # 3  triangular-kernel closed form (ring geometry only)
    if metric == "ring":
        s = np.array([S[0, d] for d in range(N)])
        l = np.array([L[0, d] for d in range(N)])
        print("  3  triangular-kernel closed form  V(K) = sum_d n_d(K) s(d),"
              " n_d = 2(K-d):")
        print(f"     {'K':<6}{'V cascade':<16}{'V closed form':<16}{'rel err':<12}"
              f"{'tau cascade':<14}{'tau closed':<14}{'rel err'}")
        for r in rows:
            K = r["K"]
            d = np.arange(K)
            nd = np.where(d == 0, K, 2.0 * (K - d))
            # V(K) = sum over the cluster's ORDERED pairs, no 1/K:
            # pooling sums, so Sc_cc = sum_{i,j in c} S_ij directly
            Vc, Tc = float(nd @ s[:K]), float(nd @ l[:K])
            print(f"     {K:<6}{r['V']:<16.6f}{Vc:<16.6f}"
                  f"{abs(Vc - r['V']) / abs(r['V']):<12.1e}"
                  f"{r['tau']:<14.4f}{Tc / Vc:<14.4f}"
                  f"{abs(Tc / Vc - r['tau']) / r['tau']:.1e}")

    # 4  within-band spread of the per-pair timescale
    print("  4  per-pair timescale tau_ij = L_ij/S_ij, WITHIN each band")
    print(f"     {'band':<10}{'pairs':<10}{'S>0 frac':<11}{'tau_b (weighted)':<19}"
          f"{'weighted CV':<14}{'min':<11}{'max'}")
    for b in range(nb):
        mask = (B == b) & ~np.eye(N, dtype=bool) if b else np.eye(N, dtype=bool)
        sv, lv = S[mask], L[mask]
        pos = sv > 0
        if pos.sum() < 2:
            continue
        t = lv[pos] / sv[pos]
        w = sv[pos] / sv[pos].sum()
        mu = float(w @ t)
        cv = float(np.sqrt(w @ (t - mu) ** 2) / abs(mu))
        print(f"     {band_label(b, metric):<10}{int(mask.sum()):<10}"
              f"{pos.mean():<11.3f}{mu:<19.2f}{cv:<14.4f}"
              f"{t.min():<11.2f}{t.max():.2f}")


def main():
    for args in (("ring k=4", "ring", 4, "ring"),
                 ("hm phi=3.0", "hm", 3.0, "module"),
                 ("geo sigma=4", "geo", 4.0, "ring")):
        report(*args)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
