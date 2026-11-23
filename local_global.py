"""Decompose small-world ICG scaling into local and long-range contributions.

WHY A PAIR DECOMPOSITION AND NOT AN EDGE DECOMPOSITION
Sigma = sum over all walks: Sigma_ij receives contributions from every path
between i and j, so there is no "variance that arrived via the ring" versus
"via a shortcut". Sigma cannot be partitioned by edge type.

What CAN be partitioned is the pair sum. A cluster's variance is exactly

    V_S = sum_{i,j in S} Sigma_ij

and that sum splits cleanly by the RING DISTANCE between i and j,

    d(i,j) = min(|i-j|, n-|i-j|)

A contiguous arc of K units contains only pairs with d <= K-1. So any pair
inside an ICG cluster with d >= K is there because rewiring made the cluster
non-contiguous. Define

    V_loc(K)  = mean_c sum_{i,j in S_c, d(i,j) <  K} Sigma_ij
    V_glob(K) = mean_c sum_{i,j in S_c, d(i,j) >= K} Sigma_ij

so that V_loc + V_glob = V(K) identically (asserted below). The same split of
Lambda gives T_loc, T_glob and hence tau_loc = T_loc/V_loc, tau_glob =
T_glob/V_glob. The i=j terms carry d=0 and so sit in the local part.

WHAT THIS FILE FIXES relative to the first draft of this analysis
  1. Compactness is now a well-defined ratio in (0,1]: K divided by the number
     of ring positions spanned by the cluster's minimal covering arc. A
     contiguous cluster is exactly 1.0. The earlier draft reported 1.02, which
     is impossible for this quantity, so its normalisation was wrong.
  2. The threshold K in "d < K" is arbitrary. Three thresholds are run
     (K/2, K, 2K) and the exponents compared, so threshold sensitivity is
     measured rather than assumed.
  3. Several seeds (default 8) with spread reported, because the earlier
     single-seed alpha_glob rested on 3-6% of the variance.
  4. V_glob can be negative (Sigma is PSD but has negative off-diagonals), in
     which case log-slopes are undefined. Levels with V_glob <= 0 are counted
     and excluded, and the count is reported rather than hidden.
  5. The surprising beta result (beta_loc rising with p while beta_glob stays
     flat) is re-tested across seeds, since the draft flagged it as unverified.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u local_global.py
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
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
P_VALUES = (0.0, 0.001, 0.004, 0.02, 0.1, 1.0)
M_GRID = 1.0 - np.logspace(-6, -0.5, 18)
REF_K = 64


# ---------------------------------------------------------------------------
def ring_distance(n):
    """d(i,j) = min(|i-j|, n-|i-j|) on a ring of n positions."""
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    return np.minimum(d, n - d)


def compactness(idx, n):
    """|S| / (ring positions spanned by the minimal covering arc). 1 iff contiguous.

    On a circle the minimal covering arc is the complement of the largest gap,
    so it spans n - max_gap + 1 positions.
    """
    p = np.sort(np.asarray(idx))
    gaps = np.diff(np.concatenate([p, [p[0] + n]]))
    span = n - gaps.max() + 1
    return len(p) / span


def members(P):
    """Row index lists of a 0/1 membership matrix."""
    return [np.flatnonzero(row > 0) for row in P]


def decompose(S, L, D, min_clusters=4, thresh=1.0):
    """Coarse-grain, splitting every level's pair sums by ring distance.

    `thresh` multiplies K: local means d(i,j) < thresh*K. Because pooling a
    covariance is exactly summing its blocks, the split is computed on the
    ORIGINAL S and L using each cluster's member list, which is identical to
    pooling but lets the pairs be separated.
    """
    n = len(S)
    P = np.eye(n)
    Sc, Lc = S.copy(), L.copy()
    rows, K = [], 1
    while len(Sc) >= min_clusters:
        ncl = len(Sc)
        vl = vg = tl = tg = 0.0
        comp, nloc, nglob = [], 0, 0
        for idx in members(P):
            sS = S[np.ix_(idx, idx)]
            sL = L[np.ix_(idx, idx)]
            mask = D[np.ix_(idx, idx)] < thresh * K
            vl += sS[mask].sum()
            vg += sS[~mask].sum()
            tl += sL[mask].sum()
            tg += sL[~mask].sum()
            nloc += int(mask.sum())
            nglob += int((~mask).sum())
            comp.append(compactness(idx, n))
        rows.append({
            "K": K, "clusters": ncl,
            "V": np.trace(Sc) / ncl, "tau": np.trace(Lc) / np.trace(Sc),
            "V_loc": vl / ncl, "V_glob": vg / ncl,
            "tau_loc": tl / vl if vl != 0 else np.nan,
            "tau_glob": tg / vg if vg != 0 else np.nan,
            "compactness": float(np.mean(comp)),
            "frac_pairs_glob": nglob / max(nloc + nglob, 1),
        })
        if ncl == 1:
            break
        pairs = lt.greedy_pairs(Sc)
        Sc, Lc = lt.pool(Sc, pairs), lt.pool(Lc, pairs)
        a, b = pairs.T
        P = P[a] + P[b]
        K *= 2
    df = pd.DataFrame(rows)
    # the decomposition must be exact
    err = np.max(np.abs(df.V_loc + df.V_glob - df.V) / np.abs(df.V))
    assert err < 1e-10, f"decomposition not exact: {err:.2e}"
    return df


def slope(df, col, drop_first=1, drop_last=1):
    """OLS log-log slope over the same window linear_theory.fit_scaling uses.

    Returns (slope, n_used, n_dropped) and skips non-positive values, which
    occur for the global component when it is a small signed residue.
    """
    s, _, nu, nd = slope_r2(df, col, drop_first, drop_last)
    return s, nu, nd


def slope_r2(df, col, drop_first=1, drop_last=1):
    """As `slope`, additionally returning R^2 of the log-log fit.

    R^2 matters here: a component that switches on at one level rather than
    scaling will still return a slope, but a poor R^2 exposes it.
    """
    end = len(df) - drop_last if drop_last else len(df)
    d = df.iloc[drop_first:end]
    y = d[col].to_numpy(dtype=float)
    x = np.log(d.K.to_numpy(dtype=float))
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum()), int((~ok).sum())
    xx, yy = x[ok], np.log(y[ok])
    s, c = np.polyfit(xx, yy, 1)
    resid = np.sum((yy - (c + s * xx)) ** 2)
    total = np.sum((yy - yy.mean()) ** 2)
    r2 = float(1 - resid / total) if total > 1e-20 else np.nan
    return float(s), r2, int(ok.sum()), int((~ok).sum())


def local_slopes(df, col):
    """Per-level log2 slopes, to expose a step masquerading as a power law."""
    y = df[col].to_numpy(dtype=float)
    out = np.full(len(y) - 1, np.nan)
    for i in range(len(y) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] > 0 and y[i + 1] > 0:
            out[i] = np.log2(y[i + 1] / y[i])
    return out


def best_m(ev, V):
    """The m whose FITTED exponents sit closest to (1.5, 0.2)."""
    best = (np.inf, None, None)
    for m in M_GRID:
        S, L = lt.stationary_covariances(ev, V, m)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        d = float(sc.distance(f["alpha"], f["beta"]))
        if d < best[0]:
            best = (d, m, f)
    return best


# ---------------------------------------------------------------------------
def main():
    D = ring_distance(N)
    print("=" * 100)
    print(f"LOCAL / LONG-RANGE DECOMPOSITION OF ICG SCALING     n={N}, "
          f"small-world (k=4 ring + rewiring p), {len(SEEDS)} seeds")
    print("  local = pairs with ring distance d(i,j) < K;  global = the rest")
    print("  m chosen per (p, seed) to minimise range-weighted distance to (1.5, 0.2)")
    print("=" * 100)

    store = {}
    for p in P_VALUES:
        per_seed = []
        for seed in SEEDS:
            A = lt.make_network(N, "sw", p, seed)
            ev, Vv = eigh(A, check_finite=False)
            dist, m, f = best_m(ev, Vv)
            S, L = lt.stationary_covariances(ev, Vv, m)
            df = decompose(S, L, D, thresh=1.0)
            a_t, _, _ = slope(df, "V")
            b_t, _, _ = slope(df, "tau")
            a_l, _, _ = slope(df, "V_loc")
            b_l, _, _ = slope(df, "tau_loc")
            a_g, n_g, drop_g = slope(df, "V_glob")
            b_g, _, _ = slope(df, "tau_glob")
            ref = df[df.K == REF_K]
            gs = float(ref.V_glob.iloc[0] / ref.V.iloc[0]) if len(ref) else np.nan
            per_seed.append({
                "seed": seed, "m": m, "one_minus_m": 1 - m, "dist": dist,
                "alpha": a_t, "beta": b_t,
                "alpha_loc": a_l, "beta_loc": b_l,
                "alpha_glob": a_g, "beta_glob": b_g,
                "glob_share": gs,
                "compact": float(ref.compactness.iloc[0]) if len(ref) else np.nan,
                "frac_pairs_glob": float(ref.frac_pairs_glob.iloc[0]) if len(ref) else np.nan,
                "n_glob_used": n_g, "n_glob_dropped": drop_g,
            })
        store[p] = pd.DataFrame(per_seed)

    def ms(d, c):
        v = d[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "      --     "
        if v.size == 1:
            return f"{v[0]:6.3f}        "
        return f"{v.mean():6.3f}+-{v.std(ddof=1):5.3f}"

    print(f"\n{'p':<8}{'1-m (med)':<12}{'alpha total':<16}{'alpha_loc':<16}"
          f"{'alpha_glob':<16}{'glob share V':<14}{'compact K=64'}")
    for p in P_VALUES:
        d = store[p]
        print(f"{p:<8.4g}{np.median(d.one_minus_m):<12.1e}{ms(d,'alpha'):<16}"
              f"{ms(d,'alpha_loc'):<16}{ms(d,'alpha_glob'):<16}"
              f"{ms(d,'glob_share'):<14}{ms(d,'compact')}")

    print(f"\n{'p':<8}{'beta total':<16}{'beta_loc':<16}{'beta_glob':<16}"
          f"{'global pairs %':<16}{'levels dropped (V_glob<=0)'}")
    for p in P_VALUES:
        d = store[p]
        fp = d.frac_pairs_glob.to_numpy(dtype=float) * 100
        print(f"{p:<8.4g}{ms(d,'beta'):<16}{ms(d,'beta_loc'):<16}"
              f"{ms(d,'beta_glob'):<16}{fp.mean():<16.2f}"
              f"{d.n_glob_dropped.sum()} of {(d.n_glob_dropped+d.n_glob_used).sum()}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("THRESHOLD SENSITIVITY   the 'd < K' boundary is a choice; vary it")
    print("=" * 100)
    print(f"{'p':<8}{'threshold':<13}{'alpha_loc':<14}{'alpha_glob':<14}"
          f"{'beta_loc':<14}{'beta_glob':<14}{'glob share'}")
    for p in (0.004, 0.02, 0.1):
        for th, lab in ((0.5, "d < K/2"), (1.0, "d < K"), (2.0, "d < 2K")):
            al, bl, ag, bg, gsv = [], [], [], [], []
            for seed in SEEDS[:4]:
                A = lt.make_network(N, "sw", p, seed)
                ev, Vv = eigh(A, check_finite=False)
                _, m, _ = best_m(ev, Vv)
                S, L = lt.stationary_covariances(ev, Vv, m)
                df = decompose(S, L, D, thresh=th)
                al.append(slope(df, "V_loc")[0]); bl.append(slope(df, "tau_loc")[0])
                ag.append(slope(df, "V_glob")[0]); bg.append(slope(df, "tau_glob")[0])
                r = df[df.K == REF_K]
                gsv.append(float(r.V_glob.iloc[0] / r.V.iloc[0]))
            f = lambda v: np.nanmean(np.array(v, dtype=float))
            print(f"{p:<8.4g}{lab:<13}{f(al):<14.3f}{f(ag):<14.3f}"
                  f"{f(bl):<14.3f}{f(bg):<14.3f}{f(gsv):.3f}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("LEVEL-BY-LEVEL at the best-matching p (seed 1): where the split is real")
    print("=" * 100)
    A = lt.make_network(N, "sw", 0.004, 1)
    ev, Vv = eigh(A, check_finite=False)
    _, m, _ = best_m(ev, Vv)
    S, L = lt.stationary_covariances(ev, Vv, m)
    df = decompose(S, L, D, thresh=1.0)
    print(f"1-m = {1-m:.2e}")
    print(f"{'K':<7}{'V':<13}{'V_loc':<13}{'V_glob':<13}{'glob share':<13}"
          f"{'tau':<10}{'tau_loc':<10}{'tau_glob':<11}{'compact':<10}{'glob pairs %'}")
    for _, r in df.iterrows():
        print(f"{int(r.K):<7}{r.V:<13.4e}{r.V_loc:<13.4e}{r.V_glob:<13.4e}"
              f"{r.V_glob/r.V:<13.4f}{r.tau:<10.3f}{r.tau_loc:<10.3f}"
              f"{r.tau_glob:<11.3f}{r.compactness:<10.3f}{100*r.frac_pairs_glob:.2f}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("IS THE GLOBAL COMPONENT ACTUALLY SCALING?  R^2 of each fit, and the")
    print("per-level slopes. A component that switches on at one K still yields")
    print("a slope; only R^2 and the per-level profile reveal that.")
    print("=" * 100)
    print(f"{'p':<8}{'R2 V_loc':<12}{'R2 V_glob':<12}{'R2 tau_loc':<13}"
          f"{'R2 tau_glob':<13}{'per-level alpha_glob (K=2,4,...,64)'}")
    for p in (0.004, 0.02, 0.1, 1.0):
        r2l, r2g, r2tl, r2tg, prof = [], [], [], [], []
        for seed in SEEDS[:4]:
            A = lt.make_network(N, "sw", p, seed)
            ev, Vv = eigh(A, check_finite=False)
            _, m, _ = best_m(ev, Vv)
            S, L = lt.stationary_covariances(ev, Vv, m)
            df = decompose(S, L, D, thresh=1.0)
            r2l.append(slope_r2(df, "V_loc")[1])
            r2g.append(slope_r2(df, "V_glob")[1])
            r2tl.append(slope_r2(df, "tau_loc")[1])
            r2tg.append(slope_r2(df, "tau_glob")[1])
            prof.append(local_slopes(df, "V_glob")[1:7])
        pm = np.nanmean(np.array(prof, dtype=float), axis=0)
        f = lambda v: np.nanmean(np.array(v, dtype=float))
        print(f"{p:<8.4g}{f(r2l):<12.3f}{f(r2g):<12.3f}{f(r2tl):<13.3f}"
              f"{f(r2tg):<13.3f}" + " ".join(f"{v:5.2f}" for v in pm))

    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIXED-m CONTROL.  Above, m was re-optimised per p, and it moved by 20x")
    print("(1-m: 3.4e-2 -> 1.7e-3). So the p-dependence there is confounded with m.")
    print("Here m is held fixed, so every column varies with p ALONE.")
    print("=" * 100)
    for lm_fix in (1.6e-2, 1.7e-3):
        print(f"\n  1-m = {lm_fix:.1e} for all p")
        print(f"  {'p':<8}{'alpha':<9}{'alpha_loc':<16}{'alpha_glob':<16}"
              f"{'beta':<9}{'beta_loc':<16}{'beta_glob':<16}{'glob share':<14}{'compact'}")
        for p in P_VALUES:
            rec = []
            for seed in SEEDS:
                A = lt.make_network(N, "sw", p, seed)
                ev, Vv = eigh(A, check_finite=False)
                S, L = lt.stationary_covariances(ev, Vv, 1 - lm_fix)
                df = decompose(S, L, D, thresh=1.0)
                r = df[df.K == REF_K]
                rec.append({
                    "alpha": slope(df, "V")[0], "beta": slope(df, "tau")[0],
                    "alpha_loc": slope(df, "V_loc")[0],
                    "beta_loc": slope(df, "tau_loc")[0],
                    "alpha_glob": slope(df, "V_glob")[0],
                    "beta_glob": slope(df, "tau_glob")[0],
                    "glob_share": float(r.V_glob.iloc[0] / r.V.iloc[0]),
                    "compact": float(r.compactness.iloc[0])})
            d = pd.DataFrame(rec)
            print(f"  {p:<8.4g}{d.alpha.mean():<9.3f}{ms(d,'alpha_loc'):<16}"
                  f"{ms(d,'alpha_glob'):<16}{d.beta.mean():<9.3f}"
                  f"{ms(d,'beta_loc'):<16}{ms(d,'beta_glob'):<16}"
                  f"{ms(d,'glob_share'):<14}{d.compact.mean():.3f}")

    out = pd.concat([d.assign(p=p) for p, d in store.items()], ignore_index=True)
    path = os.path.join(HERE, "sweep_data", "local_global.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
