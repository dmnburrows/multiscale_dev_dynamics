"""Does ICG have to follow CORRELATION, or will STRUCTURE do?

THE QUESTION.  Every result in this repository coarse-grains by correlation:
lt.greedy_pairs normalises S to a correlation matrix and gives each node its
most-correlated free partner. That is Munn et al.'s procedure, and it is a
measurement rule -- it uses only recorded activity. But the exponents are
supposed to be telling us something about the NETWORK. So: if we pair by the
wiring instead of by the activity, do alpha and beta survive?

Only the PAIRING rule changes. V(K) and tau(K) are still measured from the same
S and L, by the same pooling, so the exponents remain comparable line for line.

THE RULES.

  corr     cosine of S                     the baseline, bit-for-bit lt.greedy_pairs
  shared   cosine of (B + I) profiles      "share the most structural connections":
                                           M = BB + 2B, i.e. shared partners plus
                                           twice the direct weight, normalised the
                                           same way the correlation rule normalises
  nbr      cosine of BB                    shared partners ONLY, direct edge removed
  adj      raw B_ij                        strongest direct connection, nothing else
  dyadic   (0,1),(2,3),(4,5),...           the ground truth: HM's own module tree
                                           (i and i+1 are siblings by construction)
                                           and contiguous blocks on a ring
  random   a random perfect matching       the null: how much of the scaling is just
                                           "pool things in a binary tree"

B is the cluster-level adjacency, B_cc' = sum of edge weights between clusters c
and c'. It is pooled with lt.pool exactly as S and L are, so "share the most
connections" is re-evaluated AT EACH SCALE on the coarse-grained wiring, which is
what was asked for. Weighted graphs need no special case: B is already a weighted
sum, and A happens to be binary here.

WHAT TO WATCH.  On a ring, correlation falls off with distance and the wiring is
local, so the two rules should agree and nothing should move. On HM, correlation
pairing was measured to have module purity ~0.50 -- it does NOT recover the tree.
dyadic does, by construction. If HM's (1.5, 0.2) match depends on which of those
two cascades you run, the match is a property of the measurement, not the network.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u structural_icg.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spectral_classes as sc

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3)
LMS = np.logspace(-5.0, -0.6, 30)
LG = np.log10(LMS)
REF_K = 64
MIN_CLUSTERS = 4

NETWORKS = [
    ("ring k=4",    "ring", 4),
    ("ring k=6",    "ring", 6),
    ("ring k=8",    "ring", 8),
    ("sw p=0.004",  "sw",   0.004),
    ("hm phi=3.4",  "hm",   3.4),
    ("hm phi=3.0",  "hm",   3.0),
    ("hm phi=2.6",  "hm",   2.6),
]
RULE_ORDER = ["corr", "shared", "nbr", "adj", "dyadic", "random"]


# --------------------------------------------------------------------------
# pairing rules
# --------------------------------------------------------------------------
def greedy_from_score(C):
    """lt.greedy_pairs's own index-order greedy rule, on an arbitrary score.

    Copied in structure, not in content, so the comparison differs ONLY in the
    matrix that is being maximised. Ties resolve to the lowest index, as there.
    """
    C = np.array(C, dtype=float, copy=True)
    np.fill_diagonal(C, -np.inf)
    available = np.ones(len(C), dtype=bool)
    pairs = []
    for i in range(len(C)):
        if not available[i]:
            continue
        available[i] = False
        j = int(np.argmax(np.where(available, C[i], -np.inf)))
        if not available[j]:
            raise ValueError("Pairing requires an even number of clusters.")
        available[j] = False
        pairs.append((i, j))
    return np.asarray(pairs)


def cosine(M):
    """M_ij / sqrt(M_ii M_jj): the same normalisation the correlation rule uses."""
    d = np.sqrt(np.clip(np.diag(M), 0.0, None))
    d[d <= 0] = 1.0
    return M / np.outer(d, d)


def pairs_corr(Sc, B, rng):
    return lt.greedy_pairs(Sc)


def pairs_shared(Sc, B, rng):
    """Direct weight AND shared partners: (B+I)(B+I)^T = BB + 2B + I."""
    Ba = B + np.eye(len(B))
    return greedy_from_score(cosine(Ba @ Ba.T))


def pairs_nbr(Sc, B, rng):
    """Shared partners only."""
    return greedy_from_score(cosine(B @ B.T))


def pairs_adj(Sc, B, rng):
    return greedy_from_score(B)


def pairs_dyadic(Sc, B, rng):
    i = np.arange(0, len(Sc), 2)
    return np.stack([i, i + 1], axis=1)


def pairs_random(Sc, B, rng):
    p = rng.permutation(len(Sc))
    return np.stack([p[0::2], p[1::2]], axis=1)


RULES = {"corr": pairs_corr, "shared": pairs_shared, "nbr": pairs_nbr,
         "adj": pairs_adj, "dyadic": pairs_dyadic, "random": pairs_random}


# --------------------------------------------------------------------------
# cascade + cluster diagnostics
# --------------------------------------------------------------------------
def cascade(S, L, A, rule, rseed=0, min_clusters=MIN_CLUSTERS, ref_K=REF_K):
    """lt.coarse_grain with a swappable pairing rule; also return the clusters.

    V(K) and tau(K) are computed from the pooled S and L exactly as in
    lt.coarse_grain, so only the pairing differs.
    """
    rng = np.random.default_rng(90_000 + rseed)
    fn = RULES[rule]
    Sc, Lc, B = S.copy(), L.copy(), A.copy()
    P = np.eye(len(S))
    Pref = None
    records, K, first = [], 1, None
    while len(Sc) >= min_clusters:
        records.append({"K": K, "clusters": len(Sc),
                        "variance": np.trace(Sc) / len(Sc),
                        "tau": np.trace(Lc) / np.trace(Sc)})
        if K == ref_K:
            Pref = P.copy()
        if len(Sc) == 1:
            break
        pr = fn(Sc, B, rng)
        if first is None:
            first = pr
        a, b = pr.T
        Sc, Lc, B = lt.pool(Sc, pr), lt.pool(Lc, pr), lt.pool(B, pr)
        P = P[a] + P[b]
        K *= 2
    return pd.DataFrame(records), first, Pref


def module_purity(P, K):
    """Mean fraction of a cluster lying in one dyadic module of size K.

    HM's modules of size K are exactly {i : i >> log2(K) = const}, so
    arange(n)//K labels them. 1.0 = the cascade recovered the tree.
    """
    mod = np.arange(P.shape[1]) // K
    out = []
    for row in P:
        idx = np.flatnonzero(row > 0)
        if idx.size:
            out.append(np.bincount(mod[idx]).max() / idx.size)
    return float(np.mean(out)) if out else np.nan


def ring_compactness(P):
    """K / (shortest circular arc containing the cluster).  1.0 = contiguous."""
    n = P.shape[1]
    out = []
    for row in P:
        idx = np.sort(np.flatnonzero(row > 0))
        if idx.size < 2:
            continue
        gaps = np.diff(np.concatenate([idx, [idx[0] + n]]))
        out.append(idx.size / (n - gaps.max() + 1))
    return float(np.mean(out)) if out else np.nan


def pair_agreement(p, q):
    sp = {tuple(sorted(x)) for x in p}
    sq = {tuple(sorted(x)) for x in q}
    return len(sp & sq) / len(sp)


# --------------------------------------------------------------------------
def selfcheck():
    """cascade(..., 'corr') must reproduce lt.coarse_grain bit-for-bit."""
    A = lt.make_network(64, "hm", 3.0, 1)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, 1 - 1e-3)
    ref = lt.coarse_grain(S, L, MIN_CLUSTERS)
    got, _, _ = cascade(S, L, A, "corr")
    e = max(np.abs(ref[c].to_numpy() - got[c].to_numpy()).max()
            for c in ("variance", "tau"))
    print(f"  self-check: corr cascade vs lt.coarse_grain, max abs diff {e:.2e}")
    assert e == 0.0, "the baseline rule is not reproducing lt.coarse_grain"


def main():
    print("=" * 110)
    print("0  SELF-CHECK")
    print("=" * 110)
    selfcheck()

    rows, diag = [], []
    for name, fam, par in NETWORKS:
        for seed in SEEDS:
            A = lt.make_network(N, fam, par, seed)
            ev, V = eigh(A, check_finite=False)
            firsts = {}
            for il, lm in enumerate(LMS):
                S, L = lt.stationary_covariances(ev, V, 1 - lm)
                for rule in RULE_ORDER:
                    lv, first, Pref = cascade(S, L, A, rule, rseed=seed)
                    f = lt.fit_scaling(lv)
                    rows.append({"network": name, "seed": seed, "rule": rule,
                                 "one_minus_m": lm,
                                 "alpha": f["alpha"], "beta": f["beta"],
                                 "alpha_r2": f["alpha_r2"],
                                 "beta_r2": f["beta_r2"],
                                 "dist": float(sc.distance(f["alpha"], f["beta"]))})
                    # cluster diagnostics once per (network, seed, rule), at a
                    # mid-range m where the cascade is well conditioned
                    if il == len(LMS) // 2:
                        firsts[rule] = first
                        diag.append({
                            "network": name, "seed": seed, "rule": rule,
                            "purity": module_purity(Pref, REF_K),
                            "compact": ring_compactness(Pref)})
            for d in diag:
                if d["network"] == name and d["seed"] == seed and "agree" not in d:
                    d["agree"] = pair_agreement(firsts[d["rule"]], firsts["corr"])
            print(f"  swept {name}, seed {seed}", flush=True)

    out = pd.DataFrame(rows)
    dg = pd.DataFrame(diag)
    os.makedirs(os.path.join(HERE, "sweep_data"), exist_ok=True)
    pc = os.path.join(HERE, "sweep_data", "structural_icg_curves.csv")
    out.to_csv(pc, index=False)

    # ---------------------------------------------------------------- tables
    agg = (out.groupby(["network", "rule", "one_minus_m"], sort=False)
              .agg(alpha=("alpha", "mean"), beta=("beta", "mean"),
                   dist=("dist", "mean"), ar2=("alpha_r2", "mean"),
                   br2=("beta_r2", "mean")).reset_index())

    print("\n" + "=" * 110)
    print("1  DOES THE CASCADE EVEN FIND THE SAME CLUSTERS?")
    print("   agree  = fraction of level-1 pairs shared with the correlation rule")
    print("   purity = cluster's share in one dyadic module of size 64 (HM's own tree)")
    print("   compact= 64 / shortest ring arc containing the cluster (1.0 = contiguous)")
    print("=" * 110)
    dsum = dg.groupby(["network", "rule"], sort=False).mean(numeric_only=True).reset_index()
    print(f"  {'network':<14}{'rule':<10}{'agree':<9}{'purity':<9}{'compact'}")
    for _, r in dsum.iterrows():
        print(f"  {r.network:<14}{r.rule:<10}{r.agree:<9.3f}{r.purity:<9.3f}{r.compact:.3f}")

    print("\n" + "=" * 110)
    print("2  BEST APPROACH TO THE MUNN PAIR (1.5, 0.2), m optimised per rule")
    print("   ceiling = max beta over the whole m sweep;  residual = range-weighted")
    print("   distance at the best m;  d_corr = change in residual vs the baseline")
    print("=" * 110)
    best = []
    for (name, rule), g in agg.groupby(["network", "rule"], sort=False):
        g = g.sort_values("one_minus_m")
        j = int(np.nanargmin(g.dist.to_numpy()))
        r = g.iloc[j]
        best.append({"network": name, "rule": rule, "alpha": r.alpha,
                     "beta": r.beta, "lm_best": r.one_minus_m,
                     "residual": r.dist,
                     "ceiling": float(np.nanmax(g.beta.to_numpy())),
                     "alpha_r2": r.ar2, "beta_r2": r.br2})
    bf = pd.DataFrame(best)
    base = bf[bf.rule == "corr"].set_index("network").residual
    bf["d_corr"] = bf.residual - bf.network.map(base)
    pb = os.path.join(HERE, "sweep_data", "structural_icg_best.csv")
    bf.to_csv(pb, index=False)

    print(f"  {'network':<14}{'rule':<10}{'1-m best':<11}{'alpha':<9}{'beta':<9}"
          f"{'residual':<11}{'d_corr':<10}{'ceiling':<10}{'a_r2':<8}{'b_r2'}")
    for name, _, _ in NETWORKS:
        for _, r in bf[bf.network == name].iterrows():
            print(f"  {r.network:<14}{r.rule:<10}{r.lm_best:<11.2e}{r.alpha:<9.3f}"
                  f"{r.beta:<9.3f}{r.residual:<11.4f}{r.d_corr:<+10.4f}"
                  f"{r.ceiling:<10.4f}{r.alpha_r2:<8.3f}{r.beta_r2:.3f}")
        print()

    print("=" * 110)
    print("3  SUMMARY BY RULE, averaged over all 7 networks")
    print("=" * 110)
    print(f"  {'rule':<10}{'mean residual':<17}{'mean d_corr':<15}{'mean ceiling':<15}"
          f"{'mean beta_r2'}")
    for rule in RULE_ORDER:
        g = bf[bf.rule == rule]
        print(f"  {rule:<10}{g.residual.mean():<17.4f}{g.d_corr.mean():<+15.4f}"
              f"{g.ceiling.mean():<15.4f}{g.beta_r2.mean():.3f}")

    print(f"\nwrote {pc}")
    print(f"wrote {pb}")
    return agg, bf


if __name__ == "__main__":
    main()
