"""Brute-force search for the variable m acts through, family by family.

universal_variable.py tested ONE candidate, x = K M(m)/n, derived from the two
solvable cases. It collapses the ring (0.95) and not the stochastic families
(0.35-0.65). Before concluding that no single variable exists for them, search
the family properly.

Candidates, all of the form  x = K * g(m):

    g = M^a            a free            (M = participation ratio of w)
    g = (1-m)^nu       nu free           (a correlation-length ansatz)
    g = P^a            P = Perron share of the total weight, a free
    g = (sum w)^a      a free            (total amplification)

For the ring the first should pick out a = +1, since M/n ~ sqrt(1-m) and the
known variable is K sqrt(1-m). That is the control: if the search does not
recover it, the search is wrong, not the family.

Collapse quality is measured per seed (different realisations are different
networks, and their scatter is not m-dependence).

    python variable_search.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import universal_variable as uv

np.seterr(all="ignore")

FAMILIES = uv.FAMILIES
SEEDS = [1, 2, 3]
LMS = [10.0 ** -e for e in np.arange(1.0, 5.01, 0.25)]


def enriched_sweep(n, family, parameter, seeds, one_minus_m):
    """Local slopes plus every candidate m-statistic, per (seed, m)."""
    rows = []
    for seed in seeds:
        A = lt.make_network(n, family, parameter, seed)
        ev, V = eigh(A, check_finite=False)
        s = ev / np.max(np.abs(ev))
        for lm in one_minus_m:
            m = 1.0 - lm
            w = 1.0 / (1.0 - m ** 2 * s ** 2)
            M = float(np.sum(w) ** 2 / np.sum(w ** 2))
            P = float(np.max(w) / np.sum(w))
            W = float(np.sum(w))
            S, L = lt.stationary_covariances(ev, V, m)
            loc = uv.local_slopes(lt.coarse_grain(S, L, 4))
            for _, r in loc.iterrows():
                rows.append({"seed": seed, "one_minus_m": lm, "K": r.K,
                             "M": M, "P": P, "W": W,
                             "alpha": r.alpha, "beta": r.beta})
    return pd.DataFrame(rows)


def best_exponent(df, stat, exps):
    """The exponent a maximising collapse of alpha and beta against K*stat^a."""
    best = (np.nan, -np.inf, np.nan, np.nan)
    for a in exps:
        d = df.assign(xx=df.K * df[stat] ** a)
        qa, qb, _ = uv.collapse_quality(d, "xx")
        score = np.nanmean([qa, qb])
        if np.isfinite(score) and score > best[1]:
            best = (a, score, qa, qb)
    return best


def main():
    n = 2048
    exps = np.arange(-2.0, 2.001, 0.05)
    nus = np.arange(0.05, 3.001, 0.05)
    print("=" * 108)
    print(f"BRUTE-FORCE VARIABLE SEARCH   n = {n}, seeds {SEEDS}, "
          f"{len(LMS)} values of 1-m")
    print("  score = mean collapse quality of alpha and beta (1.0 = perfect)")
    print("=" * 108)
    print(f"{'family':<10}{'K*M^a: a':<11}{'score':<9}{'alpha':<9}{'beta':<9}"
          f"{'K*(1-m)^nu: nu':<17}{'score':<9}{'K*P^a: a':<11}{'score':<9}"
          f"{'best overall'}")
    for label, fam, par in FAMILIES:
        df = enriched_sweep(n, fam, par, SEEDS, LMS)
        aM, sM, qaM, qbM = best_exponent(df, "M", exps)
        results = [("K*M^a", sM)]
        bestnu, snu = (np.nan, -np.inf)
        for nu in nus:
            d = df.assign(xx=df.K * df.one_minus_m ** nu)
            qa, qb, _ = uv.collapse_quality(d, "xx")
            sc = np.nanmean([qa, qb])
            if np.isfinite(sc) and sc > snu:
                bestnu, snu = nu, sc
        results.append(("K*(1-m)^nu", snu))
        aP, sP, _, _ = best_exponent(df, "P", exps)
        results.append(("K*P^a", sP))
        aW, sW, _, _ = best_exponent(df, "W", exps)
        results.append(("K*W^a", sW))
        winner = max(results, key=lambda t: t[1])
        print(f"{label:<10}{aM:<11.2f}{sM:<9.4f}{qaM:<9.4f}{qbM:<9.4f}"
              f"{bestnu:<17.2f}{snu:<9.4f}{aP:<11.2f}{sP:<9.4f}"
              f"{winner[0]} ({winner[1]:.4f})")
        df.to_csv(f"/tmp/enriched_{label}.csv", index=False)

    print()
    print("  CONTROL: the ring must pick a = +1 on K*M^a (since M/n ~ sqrt(1-m),")
    print("  and the verified variable is K sqrt(1-m)). If it does, the search is")
    print("  sound and a low score elsewhere means the family genuinely has no")
    print("  single variable -- not that the search missed one.")


if __name__ == "__main__":
    main()
