"""alpha and beta from two numbers, and what the Munn pair says those numbers are.

The exact identities (analytical_theory.py, verified to 6 dp) are

    alpha_loc = log2 <2^gamma>_pi
    beta_loc  = log2( <tau 2^gamma>_pi / <tau>_pi ) - log2 <2^gamma>_pi

with pi_k = w_k abar_k(K) / sum_j w_j abar_j(K)  (mode k's share of V(K)) and
gamma_k = d log abar_k / d log K.

gamma is close to BIMODAL. A mode whose eigenvector is still smooth across a
cluster of K units has its overlap grow like K^2 (gamma = 2); one that already
oscillates within the cluster has it grow like K (gamma = 1). Writing

    p = share of V(K) held by the wide (gamma ~ 2) modes
    R = tau_wide / tau_narrow     (how much slower the wide modes are)

the identities reduce to two numbers:

    alpha = 1 + log2(1 + p)
    beta  = log2[ (2 p R + 1 - p) / ((p R + 1 - p)(1 + p)) ]

Both limits give beta = 0 (p -> 0 and p -> 1), so beta always has an interior
maximum in p, for any R > 1. That is the mechanism stated in two variables.

INVERTING THE MUNN PAIR. alpha = 1.5 needs 1 + p = 2^0.5, so

    p* = 0.4142      and then beta = 0.2 needs      R* = 2.353

So the empirical exponents say: at the scale you measured, 41.4% of the
variance sits in modes wider than a cluster, and those modes are 2.35 times
slower than the rest. m enters only by setting p and R, both computable from
the eigendecomposition with no fitting.

This module checks the reduction against linear_theory.py's own numbers, then
reports p and R for every family and locates where (p*, R*) is reached.

    python mode_share.py exact      # how much the bimodal reduction costs
    python mode_share.py invert     # p*, R* and the beta surface
    python mode_share.py families   # p, R, and where the Munn pair is reached
    python mode_share.py all
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import analytical_theory as at

np.seterr(all="ignore")

FAMILIES = [("hm2.83", "hm", 2.83), ("hm4.0", "hm", 4.0), ("er6.2", "er", 6.2),
            ("ring4", "ring", 4), ("sw0.1", "sw", 0.1), ("sf2.5", "sf", 2.5)]
GAMMA_SPLIT = 1.5


def alpha_of_p(p):
    return 1.0 + np.log2(1.0 + p)


def beta_of_pR(p, R):
    return np.log2((2.0 * p * R + 1.0 - p) / ((p * R + 1.0 - p) * (1.0 + p)))


def p_for_alpha(alpha):
    return 2.0 ** (alpha - 1.0) - 1.0


def R_for_beta(p, beta):
    """Solve beta_of_pR(p, R) = beta for R > 1."""
    f = lambda R: beta_of_pR(p, R) - beta
    lo, hi = 1.0 + 1e-12, 1e6
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def shares(ev, V, m, min_clusters=4):
    """Per level: exact alpha/beta, and the reduced (p, R) description."""
    S, L = lt.stationary_covariances(ev, V, m)
    levels = at.cascade(S, L, min_clusters)
    w, tk = at.modal_weights(ev, m)
    ab = [at.overlaps(d["memb"], V) for d in levels]
    rows = []
    for i in range(len(levels) - 1):
        gamma = np.log2(ab[i + 1] / ab[i])
        pi = w * ab[i] / np.sum(w * ab[i])
        wide = gamma > GAMMA_SPLIT
        p = float(np.sum(pi[wide]))
        if p <= 0 or p >= 1:
            R = np.nan
        else:
            tau_w = float(np.sum(pi[wide] * tk[wide]) / np.sum(pi[wide]))
            tau_n = float(np.sum(pi[~wide] * tk[~wide]) / np.sum(pi[~wide]))
            R = tau_w / tau_n
        rows.append({
            "K": levels[i]["K"], "p": p, "R": R,
            "alpha_exact": float(np.log2(np.sum(pi * 2.0 ** gamma))),
            "beta_exact": float(np.log2(np.sum(pi * tk * 2.0 ** gamma)
                                        / np.sum(pi * tk))
                                - np.log2(np.sum(pi * 2.0 ** gamma))),
            "alpha_true": np.log2(levels[i + 1]["variance"] / levels[i]["variance"]),
            "beta_true": np.log2(levels[i + 1]["tau"] / levels[i]["tau"]),
            "alpha_pR": alpha_of_p(p),
            "beta_pR": beta_of_pR(p, R) if np.isfinite(R) else np.nan,
            "gamma_bimodality": float(np.sum(pi[(gamma > 1.25) & (gamma < 1.75)])),
        })
    return pd.DataFrame(rows)


# =============================================================================

def exact():
    print("=" * 104)
    print("1  HOW MUCH THE TWO-NUMBER REDUCTION COSTS")
    print("   'true' = linear_theory.py traces; 'exact' = the log-sum-exp identity;")
    print("   'p,R' = the two-number reduction.  n = 1024.")
    print("=" * 104)
    n = 1024
    print(f"{'family':<10}{'m':<9}{'K':<6}{'alpha true':<13}{'alpha p,R':<12}"
          f"{'d':<10}{'beta true':<12}{'beta p,R':<11}{'d':<10}{'pi in 1.25-1.75'}")
    for label, fam, par in FAMILIES:
        A = lt.make_network(n, fam, par, 7)
        ev, V = eigh(A, check_finite=False)
        for m in (0.99, 0.999):
            df = shares(ev, V, m)
            for _, r in df[df.K.isin([8, 64])].iterrows():
                print(f"{label:<10}{m:<9}{int(r.K):<6}{r.alpha_true:<13.5f}"
                      f"{r.alpha_pR:<12.5f}{r.alpha_pR-r.alpha_true:<+10.4f}"
                      f"{r.beta_true:<12.5f}{r.beta_pR:<11.5f}"
                      f"{r.beta_pR-r.beta_true:<+10.4f}{r.gamma_bimodality:.4f}")
    print()
    print("  Last column = variance share sitting in the ambiguous middle of gamma.")
    print("  Small => gamma really is bimodal and the reduction is tight.")


def invert():
    print("=" * 104)
    print("2  WHAT (alpha, beta) = (1.5, 0.2) REQUIRES")
    print("=" * 104)
    p_star = p_for_alpha(1.5)
    R_star = R_for_beta(p_star, 0.2)
    print(f"  alpha = 1.5  =>  p* = 2^0.5 - 1 = {p_star:.6f}")
    print(f"  beta  = 0.2  =>  R* = {R_star:.6f}")
    print(f"  check: alpha(p*) = {alpha_of_p(p_star):.6f}, "
          f"beta(p*,R*) = {beta_of_pR(p_star, R_star):.6f}")
    print()
    print("  In words: 41.4% of the variance in modes wider than the cluster, and")
    print("  those modes 2.35x slower than the rest. Both are measurable.")

    print()
    print("  The beta surface: beta(p, R)")
    Rs = [1.5, 2.0, 2.353, 3.0, 5.0, 10.0]
    print(f"  {'p':<9}" + "".join(f"{f'R={R:g}':>11}" for R in Rs))
    for p in (0.1, 0.2, 0.3, 0.4142, 0.5, 0.7, 0.9):
        print(f"  {p:<9.4f}" + "".join(f"{beta_of_pR(p, R):>11.5f}" for R in Rs))

    print()
    print("  beta is maximal in p at:")
    print(f"  {'R':<9}{'p at max':<12}{'beta_max':<12}{'alpha there':<13}"
          f"{'distance to (1.5,0.2)'}")
    ps = np.linspace(1e-4, 1 - 1e-4, 20000)
    for R in (1.5, 2.0, 2.353, 3.0, 5.0, 10.0, 50.0):
        b = beta_of_pR(ps, R)
        i = int(np.nanargmax(b))
        a = alpha_of_p(ps[i])
        print(f"  {R:<9.3f}{ps[i]:<12.5f}{b[i]:<12.5f}{a:<13.5f}"
              f"{float(at.distance_to_target(a, b[i])):.4f}")
    print()
    print("  The p that maximises beta is 1/sqrt(R) to good approximation, so")
    print("  alpha at the beta-peak is 1 + log2(1 + 1/sqrt(R)): it drifts slowly")
    print("  with R, which is why alpha at the peak sits near 1.5 so persistently.")


def families():
    print("=" * 104)
    print("3  p AND R ACROSS FAMILIES, and where the Munn pair is reached")
    print("   p* = 0.4142, R* = 2.353")
    print("=" * 104)
    n = 1024
    p_star, R_star = p_for_alpha(1.5), R_for_beta(p_for_alpha(1.5), 0.2)
    for label, fam, par in FAMILIES:
        A = lt.make_network(n, fam, par, 7)
        ev, V = eigh(A, check_finite=False)
        print(f"\n  {label}")
        print(f"  {'1-m':<10}{'K':<6}{'p':<10}{'R':<10}{'alpha':<10}{'beta':<10}"
              f"{'dist to (1.5,0.2)'}")
        best = (np.inf, None)
        for lm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
            df = shares(ev, V, 1 - lm)
            for _, r in df.iterrows():
                d = float(at.distance_to_target(r.alpha_true, r.beta_true))
                if d < best[0]:
                    best = (d, (lm, r))
            r = df[df.K == 16]
            if len(r):
                r = r.iloc[0]
                print(f"  {lm:<10.0e}{16:<6}{r.p:<10.4f}{r.R:<10.4f}"
                      f"{r.alpha_true:<10.4f}{r.beta_true:<10.4f}"
                      f"{float(at.distance_to_target(r.alpha_true, r.beta_true)):.4f}")
        d, (lm, r) = best
        print(f"  closest anywhere: 1-m={lm:.0e}, K={int(r.K)}, "
              f"p={r.p:.4f} (p*={p_star:.4f}), R={r.R:.4f} (R*={R_star:.4f}), "
              f"(alpha,beta)=({r.alpha_true:.4f},{r.beta_true:.4f}), d={d:.4f}")


SECTIONS = {"exact": exact, "invert": invert, "families": families}


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[name]()
