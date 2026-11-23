"""Derive and test the mechanism identities.

Writing pi_k(K) = w_k abar_k(K) / sum_j w_j abar_j(K)  (mode k's share of V(K))
and  gamma_k = d log abar_k / d log K  (pure topology, m-free given the partition):

  EXACT, discrete:   alpha_loc = log2 sum_k pi_k 2^(gamma_k)
                     beta_loc  = log2 [ <tau 2^gamma>_pi / <tau>_pi ] - log2 <2^gamma>_pi
  LINEARISED:        alpha_loc ~ <gamma>_pi
                     beta_loc  ~ Cov_pi(gamma, tau) / <tau>_pi

Both are tested below against linear_theory.py's own numbers.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np
from scipy.linalg import eigh
import linear_theory as lt

np.seterr(all="ignore")
N = 512


def cascade(S, L, min_clusters=4):
    memb, levels, K = np.arange(len(S)), [], 1
    while len(S) >= min_clusters:
        levels.append(dict(K=K, variance=np.trace(S) / len(S),
                           tau=np.trace(L) / np.trace(S), memb=memb.copy()))
        if len(S) == 1:
            break
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs)); relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return levels


def overlaps(memb, V):
    nc = memb.max() + 1
    ind = np.zeros((len(memb), nc))
    ind[np.arange(len(memb)), memb] = 1.0
    return np.mean((V.T @ ind) ** 2, axis=1)


NETWORKS = [("hm2.83", "hm", 2.83), ("er6.2", "er", 6.2), ("ring4", "ring", 4)]
M_VALUES = [0.5, 0.9, 0.99, 0.995, 0.999, 0.9999]

print("=" * 104)
print("TEST A   exact discrete identities for the LOCAL slopes (K -> 2K), HM Phi=2.83")
print("=" * 104)
print(f"{'m':<9}{'K':<6}{'alpha_loc true':<17}{'alpha_loc exact id':<21}"
      f"{'alpha_loc ~<g>':<17}{'beta_loc true':<16}{'beta_loc exact id':<20}{'beta_loc ~Cov/<t>'}")
A = lt.make_network(N, "hm", 2.83, 7)
ev, V = eigh(A, check_finite=False)
for m in (0.9, 0.99, 0.999):
    S, L = lt.stationary_covariances(ev, V, m)
    lv = cascade(S, L)
    lam = m * ev / np.abs(ev).max()
    w, tk = 1 / ((1 - lam) * (1 + lam)), 1 / (1 - lam)
    ab = [overlaps(d["memb"], V) for d in lv]
    for i in range(len(lv) - 1):
        if lv[i]["K"] not in (2, 8, 32):
            continue
        g = np.log2(ab[i + 1] / ab[i])
        pi = w * ab[i] / np.sum(w * ab[i])
        a_true = np.log2(lv[i + 1]["variance"] / lv[i]["variance"])
        b_true = np.log2(lv[i + 1]["tau"] / lv[i]["tau"])
        a_exact = np.log2(np.sum(pi * 2.0 ** g))
        b_exact = (np.log2(np.sum(pi * tk * 2.0 ** g) / np.sum(pi * tk))
                   - np.log2(np.sum(pi * 2.0 ** g)))
        a_lin = float(np.sum(pi * g))
        mt = np.sum(pi * tk)
        b_lin = float((np.sum(pi * g * tk) - np.sum(pi * g) * mt) / mt)
        print(f"{m:<9}{lv[i]['K']:<6}{a_true:<17.6f}{a_exact:<21.6f}{a_lin:<17.6f}"
              f"{b_true:<16.6f}{b_exact:<20.6f}{b_lin:.6f}")

print()
print("=" * 104)
print("TEST B   mode concentration vs beta:  does beta die when one mode owns V(K)?")
print("  p        = Perron share of V(K) at K=32")
print("  N_eff    = participation of the pi distribution, 1/sum pi^2  (effective # amplified modes)")
print("  sd_tau/mean_tau = pi-weighted spread of mode timescales (the raw material of Cov(g,tau))")
print("=" * 104)
print(f"{'network':<9}{'m':<9}{'alpha':<9}{'beta':<9}{'p':<9}{'N_eff':<9}"
      f"{'sd_tau/<tau>':<15}{'Cov_pi(g,tau)/<tau>':<21}{'corr_pi(g,tau)'}")
for name, fam, par in NETWORKS:
    A = lt.make_network(N, fam, par, 7)
    ev, V = eigh(A, check_finite=False)
    for m in M_VALUES:
        S, L = lt.stationary_covariances(ev, V, m)
        lv = cascade(S, L)
        import pandas as pd
        fit = lt.fit_scaling(pd.DataFrame({"K": [d["K"] for d in lv],
                                           "variance": [d["variance"] for d in lv],
                                           "tau": [d["tau"] for d in lv]}))
        lam = m * ev / np.abs(ev).max()
        w, tk = 1 / ((1 - lam) * (1 + lam)), 1 / (1 - lam)
        i = [j for j, d in enumerate(lv) if d["K"] == 32][0]
        ab_i, ab_j = overlaps(lv[i]["memb"], V), overlaps(lv[i + 1]["memb"], V)
        g = np.log2(ab_j / ab_i)
        pi = w * ab_i / np.sum(w * ab_i)
        mt = np.sum(pi * tk)
        sd_t = np.sqrt(max(np.sum(pi * tk ** 2) - mt ** 2, 0.0))
        mg = np.sum(pi * g)
        sd_g = np.sqrt(max(np.sum(pi * g ** 2) - mg ** 2, 0.0))
        cov = np.sum(pi * g * tk) - mg * mt
        corr = cov / (sd_g * sd_t) if sd_g * sd_t > 0 else np.nan
        print(f"{name:<9}{m:<9}{fit['alpha']:<9.4f}{fit['beta']:<9.4f}{pi[-1]:<9.4f}"
              f"{1/np.sum(pi**2):<9.2f}{sd_t/mt:<15.4f}{cov/mt:<21.4f}{corr:.4f}")

print()
print("=" * 104)
print("TEST C   gamma_k is m-free:  per-mode overlap growth exponents at K=32->64, across m")
print("         (if gamma is topology only, these columns are identical)")
print("=" * 104)
for name, fam, par in NETWORKS:
    A = lt.make_network(N, fam, par, 7)
    ev, V = eigh(A, check_finite=False)
    rows = {}
    for m in (0.5, 0.9, 0.99, 0.999):
        S, L = lt.stationary_covariances(ev, V, m)
        lv = cascade(S, L)
        i = [j for j, d in enumerate(lv) if d["K"] == 32][0]
        g = np.log2(overlaps(lv[i + 1]["memb"], V) / overlaps(lv[i]["memb"], V))
        rows[m] = g
    print(f"\n{name}:   gamma for the 3 top modes and the bulk median")
    print(f"{'m':<9}{'gamma_1 (Perron)':<20}{'gamma_2':<12}{'gamma_3':<12}"
          f"{'median gamma (bulk)':<22}{'max gamma'}")
    for m, g in rows.items():
        print(f"{m:<9}{g[-1]:<20.4f}{g[-2]:<12.4f}{g[-3]:<12.4f}"
              f"{np.median(g[:-8]):<22.4f}{g.max():.4f}")
