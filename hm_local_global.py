"""Local vs long-range decomposition for HIERARCHICAL MODULAR networks.

WHY THE EARLIER ATTEMPT FAILED.  local_global_topologies.py split pairs by RING
distance and found HM clusters have compactness 0.20-0.23, i.e. at the
no-information baseline, so the split was uninterpretable (1 of 60 valid rows).
That was the wrong metric, not a real obstacle: ring distance is the natural
coordinate for a ring, not for a tree.

THE NATIVE METRIC.  make_network's hierarchy sets a pair's level from the
highest differing bit of their indices, so the module of size 2^l containing i
is just i >> l. Hence

    i and j share a module of size K = 2^l   <=>   (i >> l) == (j >> l)
                                             <=>   (i XOR j) < K

So passing the XOR matrix as the "distance" to the existing `decompose`
reproduces the module split exactly, with no change to that code:

    local  = the pair sits inside the same module of size K  (within-module)
    global = the pair spans two different modules at that scale (between-module)

Crucially this needs NO assumption that ICG clusters are contiguous or compact.
"Same module" is a structural fact about the pair, however the cluster is shaped.
The relevant diagnostic is therefore not compactness but module PURITY: what
fraction of a cluster's members fall in its most-represented size-K module.

WHAT THIS TESTS.  Ring-based networks show alpha_glob > alpha_loc (long-range
pairs pool coherently); disordered local ones show the reverse. HM is neither:
it is ordered in the tree sense but has no 1-D geometry. Since HM also reaches
the Munn pair, it is the decisive third case.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u hm_local_global.py
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
import spectral_classes as sc
from local_global import decompose, ring_distance, slope

np.seterr(all="ignore")

N = 512
REF_K = 64
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
# Configurations within this range-weighted distance of (1.5, 0.2) count as
# "HM reaching the Munn pair". 0.08 admitted only phi=3.00, giving n=4 and a
# verdict of +0.200+-0.234 -- no statistical content. This band spans
# phi=2.6-3.1, and every row's own distance is printed so the reader can see
# how close each configuration actually is.
MUNN_TOL = 0.25
MAX_CONFIGS = 6


def xor_distance(n):
    """D[i,j] = i XOR j.  (i XOR j) < K  iff  i and j share a module of size K."""
    i = np.arange(n)
    return (i[:, None] ^ i[None, :]).astype(float)


def module_purity(P, K):
    """Mean fraction of a cluster's members lying in one module of size K."""
    n = P.shape[1]
    mod = np.arange(n) // K
    out = []
    for row in P:
        idx = np.flatnonzero(row > 0)
        if idx.size == 0:
            continue
        counts = np.bincount(mod[idx])
        out.append(counts.max() / idx.size)
    return float(np.mean(out)) if out else np.nan


def cascade_purity(S, L, K_target=REF_K, min_clusters=4):
    """Module purity of the ICG clusters at cluster size K_target."""
    n = len(S)
    P = np.eye(n)
    Sc, Lc = S.copy(), L.copy()
    K = 1
    while len(Sc) >= min_clusters:
        if K == K_target:
            return module_purity(P, K_target)
        if len(Sc) == 1:
            break
        pr = lt.greedy_pairs(Sc)
        Sc, Lc = lt.pool(Sc, pr), lt.pool(Lc, pr)
        a, b = pr.T
        P = P[a] + P[b]
        K *= 2
    return np.nan


# ---------------------------------------------------------------------------
def find_munn_configs():
    """Dense search for the (phi, m) where stochastic HM reaches (1.5, 0.2)."""
    phis = np.round(np.linspace(2.0, 4.0, 21), 3)
    lms = np.logspace(-6, -0.5, 26)
    rows = []
    for phi in phis:
        for seed in SEEDS[:3]:
            A = lt.make_network(N, "hm", float(phi), seed)
            _, lab = connected_components(csr_matrix(A), directed=False)
            conn = np.bincount(lab).max() / N
            ev, V = eigh(A, check_finite=False)
            for lm in lms:
                S, L = lt.stationary_covariances(ev, V, 1 - lm)
                f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
                rows.append({"phi": float(phi), "seed": seed, "one_minus_m": lm,
                             "alpha": f["alpha"], "beta": f["beta"],
                             "conn": conn,
                             "dist": float(sc.distance(f["alpha"], f["beta"]))})
        print(f"  searched phi={phi}", flush=True)
    return pd.DataFrame(rows)


def main():
    print("=" * 104)
    print("1  WHERE DOES STOCHASTIC HM REACH THE MUNN PAIR?   n=512, 3 seeds")
    print("=" * 104)
    srch = find_munn_configs()
    srch.to_csv(os.path.join(HERE, "sweep_data", "hm_munn_search.csv"), index=False)
    good = srch[srch.conn > 0.98]
    agg = (good.groupby(["phi", "one_minus_m"])
               .agg(alpha=("alpha", "mean"), beta=("beta", "mean"),
                    dist=("dist", "mean"), conn=("conn", "mean"))
               .reset_index().sort_values("dist"))
    print(f"\n  closest connected configurations:")
    print(f"  {'phi':<8}{'1-m':<11}{'alpha':<9}{'beta':<9}{'distance'}")
    for _, r in agg.head(12).iterrows():
        print(f"  {r.phi:<8.2f}{r.one_minus_m:<11.2e}{r.alpha:<9.3f}"
              f"{r.beta:<9.3f}{r.dist:.4f}")

    # pick a spread of distinct phi values that reach the pair
    picks, seen = [], set()
    for _, r in agg.iterrows():
        if r.dist > MUNN_TOL:
            break
        if r.phi in seen:
            continue
        seen.add(r.phi)
        picks.append((r.phi, r.one_minus_m, r.alpha, r.beta, r.dist))
        if len(picks) >= MAX_CONFIGS:
            break
    if not picks:
        r = agg.iloc[0]
        picks = [(r.phi, r.one_minus_m, r.alpha, r.beta, r.dist)]

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("2  DECOMPOSITION AT THOSE CONFIGURATIONS, TWO METRICS")
    print("   module  = (i XOR j) < K    the hierarchy's own metric")
    print("   ring    = |i-j| < K        the metric used for ring/geometric nets")
    print("=" * 104)
    Dx, Dr = xor_distance(N), ring_distance(N)
    rows = []
    print(f"  testing {len(picks)} configurations x {len(SEEDS)} seeds:")
    for phi, lm, a_t, b_t, dist in picks:
        print(f"    phi={phi:.2f}  1-m={lm:.2e}  alpha={a_t:.3f}  "
              f"beta={b_t:.3f}  distance={dist:.3f}")
    for phi, lm, a_t, b_t, dist in picks:
        for seed in SEEDS:
            A = lt.make_network(N, "hm", float(phi), seed)
            ev, V = eigh(A, check_finite=False)
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            pur = cascade_purity(S, L)
            for metric, D in (("module", Dx), ("ring", Dr)):
                df = decompose(S, L, D, thresh=1.0)
                r = df[df.K == REF_K]
                rows.append({
                    "phi": phi, "one_minus_m": lm, "seed": seed,
                    "metric": metric, "purity": pur,
                    "alpha": slope(df, "V")[0], "beta": slope(df, "tau")[0],
                    "alpha_loc": slope(df, "V_loc")[0],
                    "beta_loc": slope(df, "tau_loc")[0],
                    "alpha_glob": slope(df, "V_glob")[0],
                    "beta_glob": slope(df, "tau_glob")[0],
                    "glob_share": float(r.V_glob.iloc[0] / r.V.iloc[0]),
                    "compact": float(r.compactness.iloc[0]),
                })
        print(f"  done phi={phi:.2f}, 1-m={lm:.2e}", flush=True)

    out = pd.DataFrame(rows)
    path = os.path.join(HERE, "sweep_data", "hm_local_global.csv")
    out.to_csv(path, index=False)

    def ms(g, c):
        v = g[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return f"{v[0]:.3f}".ljust(14) if v.size else "   --         "
        return f"{v.mean():.3f}+-{v.std(ddof=1):.3f}".ljust(14)

    print(f"\n  {'phi':<7}{'1-m':<10}{'metric':<9}{'purity':<9}{'glob sh':<9}"
          f"{'alpha':<8}{'alpha_loc':<15}{'alpha_glob':<15}"
          f"{'beta':<8}{'beta_loc':<15}{'beta_glob'}")
    for (phi, lm, metric), g in out.groupby(["phi", "one_minus_m", "metric"],
                                            sort=False):
        print(f"  {phi:<7.2f}{lm:<10.1e}{metric:<9}{g.purity.mean():<9.3f}"
              f"{g.glob_share.mean():<9.3f}{g.alpha.mean():<8.3f}"
              f"{ms(g,'alpha_loc'):<15}{ms(g,'alpha_glob'):<15}"
              f"{g.beta.mean():<8.3f}{ms(g,'beta_loc'):<15}{ms(g,'beta_glob')}")

    print("\n" + "=" * 104)
    print("3  THE VERDICT:  is the long-range component more coherent?")
    print("   ring-based networks gave +0.80 (39/39); disordered local -0.87 (0/21)")
    print("=" * 104)
    out["diff"] = out.alpha_glob - out.alpha_loc
    for metric, g in out.groupby("metric", sort=False):
        h = g[np.isfinite(g["diff"]) & (g.glob_share > 0.02)]
        if not len(h):
            print(f"  {metric:<9} no rows with a measurable long-range part")
            continue
        d = h["diff"]
        print(f"  {metric:<9} n={len(h):<4} alpha_loc {h.alpha_loc.mean():.3f}"
              f"   alpha_glob {h.alpha_glob.mean():.3f}"
              f"   diff {d.mean():+.3f}+-{d.std(ddof=1):.3f}"
              f"   positive in {int((d>0).sum())}/{len(d)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
