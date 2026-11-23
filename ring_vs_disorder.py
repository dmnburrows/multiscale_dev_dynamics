"""Is the local/global result about LOCALITY, or about RING STRUCTURE?

local_global_topologies.py found alpha_glob > alpha_loc in small-world networks
and the reverse in distance-graded ones. But small-world is not merely local --
it is built on an exact ring, which is additionally:

    circulant / translation invariant   exact Fourier eigenmodes, with a
                                        monotone eigenvalue <-> wavenumber map
    exactly regular                     every node has degree k, no fluctuation
    a 1-D chain                         edges only to the k/2 nearest neighbours

A geometric network with p ~ d^(-sigma) is local only in a probabilistic,
DISORDERED sense: fluctuating degree, no translation symmetry, Anderson-like
eigenvector localisation. So the earlier contrast confounded two things:

    locality of the background   (both families have it)
    order of the background      (only the ring has it)
    form of the long-range part  (sw: a few discrete shortcuts;
                                  geometric: a smooth graded tail)

THE 2x2.  Hold locality fixed and vary the other two independently, by ADDING
a fixed number of discrete long-range edges to each background rather than
rewiring (rewiring also removes local edges, which changes locality itself):

    background        long-range part        configuration
    ordered ring      S discrete shortcuts   ring_plus
    disordered local  S discrete shortcuts   geo_plus        <- THE KEY CELL
    disordered local  graded d^(-sigma) tail geometric sigma=2  (already run)
    ordered ring      none                   ring            (degenerate)

If geo_plus reproduces alpha_glob > alpha_loc, the ingredient is sparse
discrete shortcuts and ring order is irrelevant. If it does not, ring order is
the ingredient and the whole effect is a property of ordered 1-D backgrounds.

Degree is matched between the two backgrounds, and shortcuts are constrained to
ring distance >= n/8 so they are unambiguously long-range.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u ring_vs_disorder.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spatial_networks as sn
from local_global import decompose, ring_distance, slope

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3, 4, 5, 6)
ONE_MINUS_M = (3e-2, 1e-2, 3e-3, 1e-3)
N_SHORT = (2, 4, 8, 16, 32)
REF_K = 64
MIN_SHORT_DIST = N // 8


def add_shortcuts(A, n_short, seed, min_dist=MIN_SHORT_DIST):
    """Add n_short undirected edges, each spanning ring distance >= min_dist.

    Edges are ADDED, never rewired, so the local background is untouched and
    locality is held fixed across the comparison.
    """
    n = len(A)
    A = A.copy()
    D = ring_distance(n)
    rng = np.random.default_rng(seed + 50_000)
    added, tries = 0, 0
    while added < n_short and tries < 200_000:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and A[i, j] == 0 and D[i, j] >= min_dist:
            A[i, j] = A[j, i] = 1.0
            added += 1
        tries += 1
    if added < n_short:
        raise RuntimeError("could not place the requested shortcuts")
    return A


def ring_plus(n, k, n_short, seed):
    """Ordered background: exact ring of degree k, plus discrete shortcuts."""
    return add_shortcuts(lt.make_network(n, "ring", k, seed), n_short, seed)


def geo_plus(n, sigma, degree, n_short, seed):
    """Disordered background: local geometric graph, plus discrete shortcuts."""
    return add_shortcuts(sn.geometric(n, sigma, degree, seed), n_short, seed)


def main():
    D = ring_distance(N)
    rows = []

    # degree-matched local backgrounds: ring k=4 (degree 4) and geometric
    # sigma=4 tuned to the same degree
    backgrounds = [
        ("ring_plus (ordered)",      lambda s, S: ring_plus(N, 4, S, s)),
        ("geo_plus (disordered)",    lambda s, S: geo_plus(N, 4.0, 4.0, S, s)),
    ]

    print("Background properties with no shortcuts added:")
    for lab, _ in backgrounds:
        A = (lt.make_network(N, "ring", 4, 1) if "ring" in lab
             else sn.geometric(N, 4.0, 4.0, 1))
        el = D[A > 0]
        dg = A.sum(axis=1)
        print(f"  {lab:<26} degree {dg.mean():.2f}+-{dg.std():.2f}   "
              f"mean edge length {el.mean():.2f}")

    for lab, build in backgrounds:
        for S in N_SHORT:
            for seed in SEEDS:
                A = build(seed, S)
                ev, V = eigh(A, check_finite=False)
                for lm in ONE_MINUS_M:
                    Sm, L = lt.stationary_covariances(ev, V, 1 - lm)
                    df = decompose(Sm, L, D, thresh=1.0)
                    r = df[df.K == REF_K]
                    rows.append({
                        "background": lab, "n_short": S, "seed": seed,
                        "one_minus_m": lm,
                        "alpha": slope(df, "V")[0], "beta": slope(df, "tau")[0],
                        "alpha_loc": slope(df, "V_loc")[0],
                        "beta_loc": slope(df, "tau_loc")[0],
                        "alpha_glob": slope(df, "V_glob")[0],
                        "beta_glob": slope(df, "tau_glob")[0],
                        "glob_share": float(r.V_glob.iloc[0] / r.V.iloc[0]),
                        "compact": float(r.compactness.iloc[0]),
                    })
            print(f"  done {lab}, S={S}", flush=True)

    out = pd.DataFrame(rows)
    path = os.path.join(HERE, "sweep_data", "ring_vs_disorder.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)

    out["diff"] = out.alpha_glob - out.alpha_loc
    ok = out[(out.compact > 0.6) & (out.glob_share > 0.02)
             & np.isfinite(out["diff"])]

    print("\n" + "=" * 104)
    print("THE KEY CELL.  Same locality, same degree, same discrete shortcuts.")
    print("Only the ORDER of the background differs. Valid rows only")
    print("(compactness > 0.6 and global share > 0.02).")
    print("=" * 104)
    print(f"  {'background':<26}{'rows':<7}{'alpha_loc':<15}{'alpha_glob':<15}"
          f"{'difference':<16}{'glob>loc'}")
    for lab, g in ok.groupby("background", sort=False):
        d = g["diff"]
        print(f"  {lab:<26}{len(g):<7}"
              f"{g.alpha_loc.mean():.3f}+-{g.alpha_loc.std(ddof=1):.3f}   "
              f"{g.alpha_glob.mean():.3f}+-{g.alpha_glob.std(ddof=1):.3f}   "
              f"{d.mean():+.3f}+-{d.std(ddof=1):.3f}    "
              f"{int((d > 0).sum())}/{len(d)}")

    print("\n" + "=" * 104)
    print("BY NUMBER OF SHORTCUTS  (does the effect need them to be sparse?)")
    print("=" * 104)
    print(f"  {'S':<6}{'ordered: diff':<28}{'disordered: diff':<28}"
          f"{'ordered compact':<18}{'disord compact'}")
    for S in N_SHORT:
        cells = []
        for lab, _ in backgrounds:
            g = ok[(ok.background == lab) & (ok.n_short == S)]
            cells.append(f"{g['diff'].mean():+.3f} ({int((g['diff']>0).sum())}/{len(g)})"
                         if len(g) else "(no valid rows)")
        c = [out[(out.background == lab) & (out.n_short == S)].compact.mean()
             for lab, _ in backgrounds]
        print(f"  {S:<6}{cells[0]:<28}{cells[1]:<28}{c[0]:<18.3f}{c[1]:.3f}")

    print("\n" + "=" * 104)
    print("BY m")
    print("=" * 104)
    print(f"  {'1-m':<9}{'ordered: diff':<28}{'disordered: diff':<28}")
    for lm in ONE_MINUS_M:
        cells = []
        for lab, _ in backgrounds:
            g = ok[(ok.background == lab) & (ok.one_minus_m == lm)]
            cells.append(f"{g['diff'].mean():+.3f} ({int((g['diff']>0).sum())}/{len(g)})"
                         if len(g) else "(no valid rows)")
        print(f"  {lm:<9.0e}{cells[0]:<28}{cells[1]:<28}")

    print("\n" + "=" * 104)
    print("WHERE EACH BACKGROUND SITS IN (alpha, beta), for context")
    print("=" * 104)
    print(f"  {'background':<26}{'S':<6}{'1-m':<9}{'alpha':<9}{'beta':<9}"
          f"{'alpha_loc':<11}{'beta_loc':<11}{'compact':<10}{'glob share'}")
    for lab, _ in backgrounds:
        for S in (4, 32):
            g = out[(out.background == lab) & (out.n_short == S)]
            for lm in ONE_MINUS_M:
                h = g[g.one_minus_m == lm]
                print(f"  {lab:<26}{S:<6}{lm:<9.0e}{h.alpha.mean():<9.3f}"
                      f"{h.beta.mean():<9.3f}{h.alpha_loc.mean():<11.3f}"
                      f"{h.beta_loc.mean():<11.3f}{h.compact.mean():<10.3f}"
                      f"{h.glob_share.mean():.3f}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
