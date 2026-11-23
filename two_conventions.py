"""The two exponent conventions, side by side on the same cascade.

SHARED PIPELINE (identical for both)

  1. adjacency          A, symmetric binary, from lt.make_network
  2. eigendecomposition A = V diag(e) V^T;  s = e/rho(A);  lambda = m s
  3. stationary moments  for x(t+1) = W x(t) + xi, W = (m/rho)A, Q = I:
         Sigma  = sum_k w_k      v_k v_k^T ,  w_k   = 1/(1 - lambda_k^2)
         Lambda = sum_k w_k t_k  v_k v_k^T ,  t_k   = 1/(1 - lambda_k)
     (lt.stationary_covariances; closed form, no simulation)
  4. ICG cascade        repeatedly merge the most-correlated pair and pool,
     recording at each level K = 1, 2, 4, ... (lt.coarse_grain):
         V(K)   = trace(Sigma_K) / (number of clusters)
         tau(K) = trace(Lambda_K) / trace(Sigma_K)

Both conventions consume exactly the same table of (K, V, tau). They differ
only in how a slope is taken from it.


CONVENTION A -- LOCAL SLOPE   (used in the first version of the figure)

  Two-point finite difference between consecutive levels. Because K doubles,
  d log V / d log K between K and 2K is simply a base-2 log ratio:

      alpha_loc(K) = log2[ V(2K) / V(K) ]
      beta_loc(K)  = log2[ tau(2K) / tau(K) ]

  This yields one value PER LEVEL, so cluster size stays a free axis and the
  plot carries ~8 points per (network, m). It measures whether the scaling is
  locally a power law, but it is not "the exponent", and taking its minimum
  distance to a target lets K be cherry-picked.


CONVENTION B -- FITTED / GLOBAL SLOPE   (used in the recent figures; what the
Munn measurement does)

  One ordinary least squares slope of log V on log K across the whole cascade,
  with the first and last levels dropped (lt.fit_scaling defaults, which for
  n = 512 and min_clusters = 4 fits K = 2..64):

      alpha = sum_i (x_i - xbar)(y_i - ybar) / sum_i (x_i - xbar)^2,
      x_i = log K_i,  y_i = log V(K_i)

  and likewise beta with y_i = log tau(K_i). One value per (network, m).

RELATION BETWEEN THEM
  Since the K_i are evenly spaced in log, the OLS slope is an exactly weighted
  average of the local slopes, with weights that are largest in the middle of
  the window. That is why the fitted beta is always smaller than the peak local
  beta: the fit averages across the peak instead of sitting on it. Verified
  numerically below.

    python two_conventions.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scipy.linalg import eigh
import linear_theory as lt


def cascade_table(family, parameter, m, n=512, seed=1):
    """Steps 1-4: the shared (K, variance, tau) table."""
    A = lt.make_network(n, family, parameter, seed)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, m)
    return lt.coarse_grain(S, L, 4)


def local_slopes(levels):
    """CONVENTION A: alpha_loc, beta_loc between each K and 2K."""
    V = levels.variance.to_numpy()
    T = levels.tau.to_numpy()
    K = levels.K.to_numpy()
    return pd.DataFrame({"K": K[:-1],
                         "alpha_loc": np.log2(V[1:] / V[:-1]),
                         "beta_loc": np.log2(T[1:] / T[:-1])})


def fitted_slopes(levels, drop_first=1, drop_last=1):
    """CONVENTION B: one OLS slope each, over the whole window."""
    return lt.fit_scaling(levels, drop_first=drop_first, drop_last=drop_last)


def ols_weights_on_local(n_levels):
    """The weights that turn local slopes into the OLS slope.

    With x evenly spaced, write y_i - y_0 as a cumulative sum of the local
    steps. Collecting terms gives  slope*dx = sum_j c_j * step_j  with
    c_j = sum_{i>j} (x_i - xbar) / sum_i (x_i - xbar)^2 * dx.
    """
    x = np.arange(n_levels, dtype=float)
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    return np.array([np.sum(xc[j + 1:]) / denom for j in range(n_levels - 1)])


def main():
    cases = [("ring", 4, 1 - 2.7e-2), ("hm", 3.03, 1 - 1.1e-3),
             ("er", 4.4, 1 - 1.6e-3)]
    for fam, par, m in cases:
        levels = cascade_table(fam, par, m)
        loc = local_slopes(levels)
        fit = fitted_slopes(levels)
        print("=" * 78)
        print(f"{fam}  parameter={par}  m={m:.6g}   (1-m = {1-m:.1e})")
        print("=" * 78)
        print(f"  {'K':<8}{'V(K)':<16}{'tau(K)':<14}{'alpha_loc':<12}{'beta_loc'}")
        for i, r in levels.iterrows():
            a = f"{loc.alpha_loc[i]:.4f}" if i < len(loc) else ""
            b = f"{loc.beta_loc[i]:.4f}" if i < len(loc) else ""
            print(f"  {int(r.K):<8}{r.variance:<16.6g}{r.tau:<14.6g}{a:<12}{b}")
        print(f"\n  CONVENTION A (local) spans alpha {loc.alpha_loc.min():.4f} "
              f"to {loc.alpha_loc.max():.4f},  beta {loc.beta_loc.min():.4f} "
              f"to {loc.beta_loc.max():.4f}")
        print(f"  CONVENTION B (fitted, K=2..64) alpha = {fit['alpha']:.4f} "
              f"(R2 {fit['alpha_r2']:.4f}),  beta = {fit['beta']:.4f} "
              f"(R2 {fit['beta_r2']:.4f})")
        print(f"  peak local beta = {loc.beta_loc.max():.4f}  vs fitted beta = "
              f"{fit['beta']:.4f}   ->  fit sits {loc.beta_loc.max()-fit['beta']:.4f} below the peak")

        # the fitted slope IS a weighted average of the local slopes
        used = levels.iloc[1:len(levels) - 1]
        lu = local_slopes(used)
        w = ols_weights_on_local(len(used))
        for name, col, got in (("alpha", "alpha_loc", fit["alpha"]),
                               ("beta", "beta_loc", fit["beta"])):
            recon = float(np.sum(w * lu[col].to_numpy()) * np.log(2) / np.log(2))
            print(f"  check: sum(w * {name}_loc) = {recon:.6f}   "
                  f"fit_scaling = {got:.6f}   diff = {abs(recon-got):.2e}")
        print(f"  weights: {np.round(w, 4).tolist()}")
        print()


if __name__ == "__main__":
    main()
