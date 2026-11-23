"""Tune every topology to (1.5, 0.2) on GLOBAL exponents, then compare covariance structure.

The exponent pair is degenerate: ring, small-world and hierarchical-modular all
reach within 0.03 of it. So the question is what else differs, and whether any
of it is measurable. Munn et al. publish exactly the right observables -- the
pairwise correlation distribution at each coarse-graining level -- so each model
can be scored against them directly.

Their zebrafish numbers (Figure 1C, 2A-F):
    l = 1   mean  1e-2   sd 0.06   kurtosis heavy-tailed (>3, up to >20 in mammals)
    l = 13  mean -1e-2   sd 0.32   kurtosis approaching 3 from above
plus: anticorrelated populations are present at coarse scales.

So the data show the correlation distribution BROADENING while staying centred
near zero. Those are two separate constraints and they turn out to pull in
opposite directions for our models.

Selection: for each family, the configurations whose FITTED exponents are
closest to (1.5, 0.2), restricted to networks that stay connected.

    python covariance_signatures.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import extended_networks as xn

np.seterr(all="ignore")
N = 512
FAMILIES = ["ring", "sw", "hm", "er", "sf"]
LABEL = {"ring": "Ring", "sw": "Small-world", "hm": "Hierarchical modular",
         "er": "Erdős–Rényi", "sf": "Scale-free"}


def build(fam, p1, p2):
    if fam == "ring":
        return lt.make_network(N, "ring", int(p1), 1)
    if fam == "hm":
        return lt.make_network(N, "hm", float(p1), 1)
    if fam == "er":
        return lt.make_network(N, "er", float(p1), 1)
    if fam == "sw":
        return xn.small_world(N, int(p1), float(p2), 1)
    return xn.scale_free(N, float(p1), float(p2), 1)


def connected(A):
    _, lab = connected_components(csr_matrix(A), directed=False)
    return np.bincount(lab).max() / N > 0.98


def profile(A, m):
    """Correlation-distribution statistics at every coarse-graining level."""
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, m)
    rows, K = [], 1
    while len(S) >= 4:
        sd = np.sqrt(np.diag(S))
        C = S / np.outer(sd, sd)
        c = C[np.triu_indices(len(S), 1)]
        c = c[np.isfinite(c)]
        rows.append({"K": K, "mean": float(c.mean()), "sd": float(c.std()),
                     "kurtosis": float(np.mean((c - c.mean()) ** 4) / c.var() ** 2),
                     "frac_neg": float(np.mean(c < 0))})
        pairs = lt.greedy_pairs(S)
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "matched_fitted.csv"))
    print("Selected by FITTED (global) exponents closest to (1.5, 0.2), connected only.")
    print("Up to 6 configurations per family; statistics are mean +- sd over them.\n")

    summary = []
    for fam in FAMILIES:
        d = df[df.family == fam]
        cfg = (d.groupby(["param", "param2"], dropna=False).dist.min()
                 .reset_index().sort_values("dist"))
        picked = []
        for _, r in cfg.iterrows():
            if len(picked) >= 6:
                break
            A = build(fam, r.param, r.param2)
            if not connected(A):
                continue
            sub = d[(d.param == r.param) &
                    (d.param2.isna() if not np.isfinite(r.param2)
                     else d.param2 == r.param2)]
            b = sub.loc[sub.dist.idxmin()]
            picked.append((b, A))
        if not picked:
            print(f"{LABEL[fam]}: no connected configuration\n")
            continue

        profs = [profile(A, 1 - b.one_minus_m) for b, A in picked]
        dists = [b.dist for b, _ in picked]
        alphas = [b.alpha_fit for b, _ in picked]
        betas = [b.beta_fit for b, _ in picked]
        Ks = profs[0].K.values
        n_lev = min(len(p) for p in profs)

        def agg(col, i):
            return np.array([p[col].values[i] for p in profs])

        print(f"=== {LABEL[fam]} ===  n={len(picked)} configs, "
              f"fitted alpha {np.mean(alphas):.3f}, beta {np.mean(betas):.3f}, "
              f"distance {np.mean(dists):.3f}")
        print(f"  {'K':<8}" + "".join(f"{k:>9}" for k in Ks[:min(8, n_lev)]))
        for col, lab in (("mean", "mean corr"), ("sd", "sd corr"),
                         ("kurtosis", "kurtosis"), ("frac_neg", "frac neg")):
            vals = [agg(col, i).mean() for i in range(min(8, n_lev))]
            print(f"  {lab:<8}" + "".join(f"{v:>9.3f}" for v in vals))
        i0, i1 = 0, min(7, n_lev - 1)
        summary.append({
            "family": LABEL[fam], "dist": np.mean(dists),
            "mean_lo": agg("mean", i0).mean(), "mean_hi": agg("mean", i1).mean(),
            "sd_lo": agg("sd", i0).mean(), "sd_hi": agg("sd", i1).mean(),
            "kurt_lo": agg("kurtosis", i0).mean(),
            "kurt_hi": agg("kurtosis", i1).mean(),
            "neg_hi": agg("frac_neg", i1).mean()})
        print()

    s = pd.DataFrame(summary)
    print("=" * 104)
    print("SCORED AGAINST MUNN'S ZEBRAFISH NUMBERS")
    print("=" * 104)
    print(f"  {'family':<22}{'mean K=1':<11}{'mean coarse':<14}{'sd K=1':<10}"
          f"{'sd coarse':<12}{'sd growth':<12}{'kurt K=1':<11}{'frac neg coarse'}")
    for _, r in s.iterrows():
        print(f"  {r.family:<22}{r.mean_lo:<11.3f}{r.mean_hi:<14.3f}{r.sd_lo:<10.3f}"
              f"{r.sd_hi:<12.3f}{r.sd_hi/r.sd_lo:<12.2f}{r.kurt_lo:<11.1f}"
              f"{r.neg_hi:.3f}")
    print(f"  {'ZEBRAFISH (Munn)':<22}{0.010:<11.3f}{-0.010:<14.3f}{0.060:<10.3f}"
          f"{0.320:<12.3f}{5.33:<12.2f}{'>3':<11}{'present'}")
    print()
    print("  Munn require BOTH: the distribution broadens (sd grows ~5x) AND stays")
    print("  centred near zero, with anticorrelated populations appearing at coarse")
    print("  scales. Those are separate constraints.")


if __name__ == "__main__":
    main()
