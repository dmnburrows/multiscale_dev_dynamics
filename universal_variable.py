"""The link between alpha, beta and m, with nothing fitted.

CLAIM. m acts on the exponents only by setting how many eigenmodes carry the
variance. Count them with the participation ratio of the modal weights

    w_k = 1/(1 - m^2 s_k^2),     s_k = eig_k / rho(A)

    M(m) = (sum_k w_k)^2 / sum_k w_k^2          [# of amplified modes]

and the exponents depend on cluster size only through

    x = K M(m) / n

so that, for any fitting window,  alpha(m) = A(x),  beta(m) = B(x).

WHY THIS VARIABLE AND NOT ANOTHER. It reduces to the right thing in both cases
that can be solved by hand:

  ring / lattice: w is Lorentzian of width q_c ~ sqrt(1-m) about q = 0, so the
      modes inside the peak number ~ n q_c, giving M ~ n sqrt(1-m) and
      x ~ K sqrt(1-m) -- the collapse variable verified to 5 dp in
      analytical_theory.py.

  deterministic hierarchy: w_s ~ 1/((1-m) + (2-r) r^(s-1)) with multiplicity
      n/2^s, so the amplified modes are those below the correlation level s*,
      numbering ~ n/2^(s*), giving M ~ n/2^(s*) and x ~ K/2^(s*) -- the
      collapse variable verified by the parameter-free shift test in
      collapse_law.py (nu = 1/log2(Phi/2)).

Two unrelated derivations, one formula. M(m) needs no fit, no topology
argument, and no assumption that the partition is congruent -- which matters,
because for stochastic networks none of those are available.

So this is the family-free version of the question "how do alpha and beta
depend on m", and it is testable on exactly the networks in Figs 1-2, where no
closed form exists. That is what this module does, using linear_theory.py
verbatim for every network, covariance and coarse-graining.

    python universal_variable.py sanity     # M(m) on the two solvable cases
    python universal_variable.py collapse   # all families, collapse quality
    python universal_variable.py master     # the master curves A(x), B(x)
    python universal_variable.py all
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import analytical_theory as at

np.seterr(all="ignore")

FAMILIES = [("hm2.83", "hm", 2.83), ("hm4.0", "hm", 4.0), ("er6.2", "er", 6.2),
            ("ring4", "ring", 4), ("sw0.1", "sw", 0.1), ("sf2.5", "sf", 2.5)]


def participation(s, m):
    """M(m) = (sum w)^2 / sum w^2, the number of amplified modes."""
    w = 1.0 / (1.0 - m ** 2 * s ** 2)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def local_slopes(levels):
    """alpha_loc, beta_loc per K -> 2K doubling, from a coarse_grain table."""
    V, T, K = levels.variance.values, levels.tau.values, levels.K.values
    return pd.DataFrame({"K": K[:-1],
                         "alpha": np.log2(V[1:] / V[:-1]),
                         "beta": np.log2(T[1:] / T[:-1])})


def sweep(n, family, parameter, seeds, one_minus_m):
    """Every (K, m) local slope for one family, with x = K M(m)/n attached."""
    rows = []
    for seed in seeds:
        A = lt.make_network(n, family, parameter, seed)
        ev, V = eigh(A, check_finite=False)
        s = ev / np.max(np.abs(ev))
        for lm in one_minus_m:
            m = 1.0 - lm
            M = participation(s, m)
            S, L = lt.stationary_covariances(ev, V, m)
            loc = local_slopes(lt.coarse_grain(S, L, 4))
            for _, r in loc.iterrows():
                rows.append({"seed": seed, "one_minus_m": lm, "K": r.K,
                             "M": M, "x": r.K * M / n,
                             "alpha": r.alpha, "beta": r.beta})
    return pd.DataFrame(rows)


def collapse_quality(df, variable="x", n_bins=14, lo=None, hi=None):
    """Within-bin scatter of alpha and beta when binned by log(variable).

    A good collapse means points from different m that share x also share the
    exponents, so the within-bin spread is small compared with the overall
    spread. Reported as 1 - within/total (1.0 = perfect collapse).

    Run this PER SEED. Different realisations of a random family are different
    networks, so their scatter is not m-dependence; pooling seeds here charges
    the collapse for network-to-network variation it never claimed to explain.
    """
    if "seed" in df.columns and df.seed.nunique() > 1:
        per = [collapse_quality(d, variable, n_bins, lo, hi)
               for _, d in df.groupby("seed")]
        return (float(np.nanmean([p[0] for p in per])),
                float(np.nanmean([p[1] for p in per])),
                int(np.sum([p[2] for p in per])))
    d = df[(df[variable] > 0) & np.isfinite(df.alpha) & np.isfinite(df.beta)]
    if lo is None:
        lo, hi = np.percentile(np.log(d[variable]), [2, 98])
    d = d[(np.log(d[variable]) >= lo) & (np.log(d[variable]) <= hi)]
    if len(d) < 3 * n_bins:
        return np.nan, np.nan, len(d)
    bins = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(np.log(d[variable]), bins) - 1, 0, n_bins - 1)
    out = []
    for col in ("alpha", "beta"):
        v = d[col].values
        within = np.nanmean([np.var(v[idx == b]) for b in range(n_bins)
                             if np.sum(idx == b) >= 2])
        out.append(1.0 - within / np.var(v))
    return out[0], out[1], len(d)


def fitted_nu(df, nus=None):
    """The exponent nu that best collapses the data against K (1-m)^nu.

    Reported for comparison with x = K M/n, which has no free exponent.
    """
    if nus is None:
        nus = np.arange(0.1, 3.01, 0.05)
    best = (None, -np.inf)
    for nu in nus:
        d = df.assign(xn=df.K * df.one_minus_m ** nu)
        qa, qb, _ = collapse_quality(d, "xn")
        score = np.nanmean([qa, qb])
        if np.isfinite(score) and score > best[1]:
            best = (nu, score)
    return best


# =============================================================================

def sanity():
    print("=" * 100)
    print("1  M(m) on the two solvable cases: does it reproduce the known variable?")
    print("=" * 100)
    n = 2 ** 16
    print("  ring k=4:  predicted  M/n ~ sqrt(1-m)")
    print(f"  {'1-m':<11}{'M/n':<14}{'sqrt(1-m)':<14}{'ratio':<11}{'local slope d log(M/n)/d log(1-m)'}")
    s_ring = at.ring_spectrum(n, 4)
    prev = None
    for lm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        M = participation(s_ring, 1 - lm) / n
        slope = ""
        if prev is not None:
            slope = f"{np.log(M/prev[0])/np.log(lm/prev[1]):.5f}"
        print(f"  {lm:<11.0e}{M:<14.6e}{np.sqrt(lm):<14.6e}{M/np.sqrt(lm):<11.5f}{slope}")
        prev = (M, lm)

    print()
    print("  deterministic hierarchy:  predicted  M/n ~ (1-m)^nu, nu = 1/log2(Phi/2)")
    print(f"  {'Phi':<8}{'nu predicted':<15}{'nu measured from M(m)':<24}{'ratio'}")
    for phi in (3.2, 3.5, 4.0, 5.0, 6.0):
        spec = at.hier_spectrum(n, phi)
        rho = max(abs(v) for v, _ in spec)
        s = np.concatenate([np.full(mult, val / rho) for val, mult in spec])
        lms = np.array([1e-3, 1e-4, 1e-5])
        Ms = np.array([participation(s, 1 - lm) / n for lm in lms])
        nu_meas = np.polyfit(np.log(lms), np.log(Ms), 1)[0]
        nu_pred = 1.0 / np.log2(phi / 2.0)
        print(f"  {phi:<8}{nu_pred:<15.5f}{nu_meas:<24.5f}{nu_meas/nu_pred:.5f}")


def collapse():
    print("=" * 100)
    print("2  COLLAPSE QUALITY, all families   (1.0 = perfect; linear_theory.py verbatim)")
    print("   x = K M(m)/n has NO free parameter. nu is fitted, for comparison only.")
    print("=" * 100)
    n = 2048
    seeds = [1, 2, 3]
    lms = [10.0 ** -e for e in np.arange(1.0, 5.01, 0.25)]
    print(f"{'family':<10}{'points':<9}{'x=KM/n: alpha':<16}{'beta':<10}"
          f"{'best nu':<10}{'nu: alpha':<12}{'beta':<10}{'raw K: alpha':<14}{'beta'}")
    store = {}
    for label, fam, par in FAMILIES:
        df = sweep(n, fam, par, seeds, lms)
        store[label] = df
        qa, qb, npts = collapse_quality(df, "x")
        nu, _ = fitted_nu(df)
        dn = df.assign(xn=df.K * df.one_minus_m ** nu)
        na, nb, _ = collapse_quality(dn, "xn")
        ra, rb, _ = collapse_quality(df, "K")
        print(f"{label:<10}{npts:<9}{qa:<16.4f}{qb:<10.4f}{nu:<10.2f}"
              f"{na:<12.4f}{nb:<10.4f}{ra:<14.4f}{rb:.4f}")
    print()
    print("  'raw K' is the control: no collapse at all if m matters independently.")
    for label, df in store.items():
        df.to_csv(f"/tmp/sweep_{label}.csv", index=False)
    return store


def master():
    print("=" * 100)
    print("3  THE MASTER CURVES  A(x) and B(x)")
    print("=" * 100)
    n = 1024
    seeds = [1, 2, 3, 4, 5]
    lms = [10.0 ** -e for e in np.arange(1.0, 5.01, 0.25)]
    for label, fam, par in FAMILIES:
        df = sweep(n, fam, par, seeds, lms)
        d = df[(df.x > 0) & np.isfinite(df.alpha)]
        lo, hi = np.percentile(np.log(d.x), [2, 98])
        bins = np.linspace(lo, hi, 13)
        idx = np.clip(np.digitize(np.log(d.x), bins) - 1, 0, 11)
        print(f"\n  {label}")
        print(f"  {'x (bin centre)':<18}{'alpha':<11}{'sd':<10}{'beta':<11}{'sd':<10}{'n'}")
        for b in range(12):
            sel = d[idx == b]
            if len(sel) < 2:
                continue
            xc = np.exp((bins[b] + bins[b + 1]) / 2)
            print(f"  {xc:<18.5f}{sel.alpha.mean():<11.4f}{sel.alpha.std():<10.4f}"
                  f"{sel.beta.mean():<11.4f}{sel.beta.std():<10.4f}{len(sel)}")
        i = int(np.nanargmax([d[idx == b].beta.mean() if np.sum(idx == b) >= 2
                              else -9 for b in range(12)]))
        sel = d[idx == i]
        print(f"  beta peaks at x = {np.exp((bins[i]+bins[i+1])/2):.4f}: "
              f"beta = {sel.beta.mean():.4f}, alpha = {sel.alpha.mean():.4f}")


SECTIONS = {"sanity": sanity, "collapse": collapse, "master": master}


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[name]()
