"""Is DISTANCE a sufficient coordinate for the pair covariance?

This is the only non-trivial sense in which "the Munn pair comes from composing
local with long-range structure" is exactly expressible. The composition law
over range bands is exact for ANY partition of pairs, so exactness alone carries
no content. What carries content is whether range is the CANONICAL coordinate:

    S_ij = s(d_ij) exactly   =>  grouping pairs by distance loses nothing, the
                                 range decomposition is complete, and
                                 V(K) = sum_d n_d(K) s(d) is the whole model
    S_ij independent of d    =>  range is an arbitrary grouping and the
                                 decomposition says nothing about the network

The natural measure is the fraction of the variation in S_ij explained by
distance alone -- the correlation ratio

    eta^2 = sum_g n_g (mean_g - mean)^2 / sum_ij (S_ij - mean)^2

with one group g per exact distance value (so this is binning-free: it uses the
finest grouping distance admits). eta^2 = 1 means S is a deterministic function
of distance; eta^2 at the permutation null means distance is uninformative.

This quantity needs no model, no coarse-graining and no bin width, and it is
computable from real data: pairwise correlations plus unit positions. It is the
honest replacement for the within-band spread statistic, which was withdrawn
because tau_ij = L_ij/S_ij divides by an S_ij that crosses zero.

NOTE eta^2 is biased upward when groups are small. Group sizes here are ~1000
(ring metric) and 512 (module metric), and the permutation null quantifies the
bias directly.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u distance_sufficiency.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
from range_composition import N, best_m, net

np.seterr(all="ignore")
RNG = np.random.default_rng(0)


def eta2(vals, groups, n_perm=12):
    """Correlation ratio of vals against a categorical grouping, and its null."""
    vals = np.asarray(vals, dtype=float)
    g = np.asarray(groups)
    tot = float(((vals - vals.mean()) ** 2).sum())
    if tot <= 0:
        return np.nan, np.nan

    def ratio(gg):
        cnt = np.bincount(gg)
        sm = np.bincount(gg, weights=vals)
        ok = cnt > 0
        mu = sm[ok] / cnt[ok]
        return float((cnt[ok] * (mu - vals.mean()) ** 2).sum() / tot)

    null = np.mean([ratio(RNG.permutation(g)) for _ in range(n_perm)])
    return ratio(g), null


def metrics(n):
    i = np.arange(n)
    ring = np.minimum(np.abs(i[:, None] - i[None, :]),
                      n - np.abs(i[:, None] - i[None, :]))
    return {"ring distance": ring, "xor distance": i[:, None] ^ i[None, :]}


def main():
    M = metrics(N)
    off = ~np.eye(N, dtype=bool)
    print("eta^2 = fraction of the variation in the pair covariance explained by")
    print("distance alone, finest possible grouping (one group per exact distance).")
    print("'null' is the same statistic after permuting the distance labels, so it")
    print("is the level reachable by an uninformative coordinate.\n")
    print(f"{'network':<14}{'metric':<16}{'eta2(S|d)':<12}{'null':<9}"
          f"{'eta2(L|d)':<12}{'null':<9}{'groups':<9}{'pairs/group'}")

    for label, kind, par in (("ring k=4", "ring", 4), ("ring k=8", "ring", 8),
                             ("hm phi=3.0", "hm", 3.0), ("hm phi=2.6", "hm", 2.6),
                             ("geo sigma=4", "geo", 4.0), ("geo sigma=2", "geo", 2.0),
                             ("er degree 6", "er", 6.0)):
        A = (lt.make_network(N, "er", 6.0, 1) if kind == "er" else net(kind, par, 1))
        ev, V = eigh(A, check_finite=False)
        _, lm, f = best_m(ev, V)
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        for mname, D in M.items():
            d = D[off].astype(np.int64)
            es, ns = eta2(S[off], d)
            el, nl = eta2(L[off], d)
            ng = len(np.unique(d))
            print(f"{label:<14}{mname:<16}{es:<12.4f}{ns:<9.4f}"
                  f"{el:<12.4f}{nl:<9.4f}{ng:<9}{len(d) // ng}")
        print(f"{'':<14}{'(alpha,beta)':<16}"
              f"({f['alpha']:.3f}, {f['beta']:.3f})   1-m={lm:.2e}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
