"""What decides which architectures match, and what this model class cannot do.

Consolidates the analyses that were run ad hoc late in the session. Each
section is one finding; run them by name or all at once.

    python limits_and_dimension.py powerlaw   # Munn's r criterion on our nets
    python limits_and_dimension.py window     # exponents drift with system size
    python limits_and_dimension.py peak       # where beta peaks, by architecture
    python limits_and_dimension.py invariant  # what is fixed at alpha = 1.5
    python limits_and_dimension.py modes      # how few modes are ever active
    python limits_and_dimension.py disorder   # coupling disorder does not help
    python limits_and_dimension.py dimension  # spectral dimension sets beta_max
    python limits_and_dimension.py all
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import analytical_theory as at
import extended_networks as xn
import universality as un
import continuum as co

np.seterr(all="ignore")
N = 1024
LMS = np.logspace(-7, -0.3, 250)


# ---------------------------------------------------------------- helpers

def modular(n, msize, pin, pout, seed=1):
    """Munn's modular family: pin inside a module, pout between."""
    rng = np.random.default_rng(seed)
    lab = np.arange(n) // msize
    P = np.where(lab[:, None] == lab[None, :], pin, pout).astype(float)
    np.fill_diagonal(P, 0)
    return lt.sample_edges(P, rng)


def zoo(n=N):
    """The architectures compared throughout."""
    return [("ring k=4", lt.make_network(n, "ring", 4, 1)),
            ("ring k=16", lt.make_network(n, "ring", 16, 1)),
            ("torus 2D", un.torus_adjacency(32, 2, 4)),
            ("small-world p=0.004", xn.small_world(n, 4, 0.004, 1)),
            ("small-world p=0.1", xn.small_world(n, 4, 0.1, 1)),
            ("modular 32 10/0.2", modular(n, 32, 0.10, 0.002)),
            ("modular 32 95/5", modular(n, 32, 0.95, 0.05)),
            ("hm Phi=2.83", lt.make_network(n, "hm", 2.83, 1)),
            ("ER degree 6", lt.make_network(n, "er", 6.0, 1))]


def local_at(ev, V, lm, K):
    """Local slopes between K and 2K at coupling 1 - lm."""
    S, L = lt.stationary_covariances(ev, V, 1 - lm)
    lv = lt.coarse_grain(S, L, 4)
    Ks = lv.K.values
    if K not in Ks:
        return None
    i = int(np.flatnonzero(Ks == K)[0])
    return (np.log2(lv.variance.values[i + 1] / lv.variance.values[i]),
            np.log2(lv.tau.values[i + 1] / lv.tau.values[i]))


def beta_peak(ev, V, K=64, lms=LMS):
    """(beta_max, alpha there, 1-m there) over a fine m sweep."""
    A, B = [], []
    for lm in lms:
        r = local_at(ev, V, lm, K)
        if r is None:
            return None
        A.append(r[0]); B.append(r[1])
    A, B = np.array(A), np.array(B)
    j = int(np.nanargmax(B))
    return B[j], A[j], lms[j]


def graph_dim(A, nsrc=40):
    """d from neighbourhood growth N(r) ~ r^d, r = hops."""
    rng = np.random.default_rng(0)
    src = rng.choice(len(A), min(nsrc, len(A)), replace=False)
    D = shortest_path(csr_matrix(A), unweighted=True, indices=src)
    fin = D[np.isfinite(D)]
    if fin.size == 0:
        return np.nan
    rs = np.arange(1, min(int(fin.max()), 12) + 1)
    Nr = np.array([np.mean(np.sum((D <= r) & np.isfinite(D), axis=1)) for r in rs])
    ok = (Nr > 2) & (Nr < 0.4 * len(A))
    return (float(np.polyfit(np.log(rs[ok]), np.log(Nr[ok]), 1)[0])
            if ok.sum() >= 3 else np.nan)


def spectral_dim(ev):
    """d_s from the edge counting function N(delta) ~ n delta^(d_s/2)."""
    s = ev / np.abs(ev).max()
    g = np.sort(1 - s)
    g = g[g > 0]
    ds = np.logspace(np.log10(max(g[1], 1e-6)), np.log10(0.5), 30)
    Nd = np.array([np.sum(g <= d) for d in ds], float)
    ok = (Nd >= 3) & (Nd <= 0.3 * len(s))
    return (2 * float(np.polyfit(np.log(ds[ok]), np.log(Nd[ok]), 1)[0])
            if ok.sum() >= 4 else np.nan)


# ---------------------------------------------------------------- sections

def powerlaw():
    """Munn assess power laws by r = corr(log stat, log K), r>0.9 passing.

    Our crossovers pass everywhere, including where the local slope varies by
    half an exponent unit, so the criterion cannot separate a power law from a
    smooth crossover on a window this short.
    """
    print("=" * 96)
    print("MUNN'S r CRITERION APPLIED TO OUR NETWORKS (ring k=4)")
    print("=" * 96)
    A = lt.make_network(512, "ring", 4, 1)
    ev, V = eigh(A, check_finite=False)
    print(f"{'1-m':<10}{'alpha':<9}{'r(alpha)':<11}{'beta':<9}{'r(beta)':<11}"
          f"{'alpha_loc spread':<19}{'verdict'}")
    for lm in (1e-1, 3e-2, 2.7e-2, 1e-2, 1e-3, 1e-5):
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        lv = lt.coarse_grain(S, L, 4)
        f = lt.fit_scaling(lv)
        a = np.log2(lv.variance.values[1:] / lv.variance.values[:-1])[1:6]
        ra, rb = np.sqrt(max(f["alpha_r2"], 0)), np.sqrt(max(f["beta_r2"], 0))
        print(f"{lm:<10.1e}{f['alpha']:<9.3f}{ra:<11.4f}{f['beta']:<9.3f}"
              f"{rb:<11.4f}{a.max()-a.min():<19.3f}"
              f"{'passes' if min(ra, rb) > 0.9 else 'fails'}")
    print("\n  The criterion passes at every m. It is not diagnostic here.")


def window():
    """Exponents drift with system size at fixed parameters: no scale-invariance.

    Munn's five species agree to +-0.02 across three orders of magnitude in N.
    Reproducing that with a crossover needs each species tuned to about +-30%,
    at couplings spanning 700-fold.
    """
    print("=" * 96)
    print("WINDOW INDEPENDENCE: fixed parameters, growing n")
    print("=" * 96)
    s = at.ring_spectrum(2 ** 20, 4)
    print("  ring k=4, 1-m = 2.7e-2")
    print(f"    {'n':<10}{'levels':<9}{'alpha':<10}{'beta'}")
    for p in (7, 9, 11, 13, 15, 17):
        n = 2 ** p
        f = lt.fit_scaling(at.ring_levels(2 ** 20, 4, 1 - 2.7e-2,
                                          [2 ** t for t in range(1, p - 1)], s=s))
        print(f"    {n:<10}{p-3:<9}{f['alpha']:<10.3f}{f['beta']:.3f}")
    print("\n  deterministic hierarchy, Phi=3.03, 1-m = 1e-3")
    print(f"    {'n':<10}{'levels':<9}{'alpha':<10}{'beta'}")
    for p in range(10, 29, 3):
        Ks = [2 ** t for t in range(2, p - 1)]
        if len(Ks) < 4:
            continue
        f = lt.fit_scaling(at.hier_levels(2 ** p, 3.03, 1 - 1e-3, Ks))
        print(f"    {2**p:<10}{len(Ks)-2:<9}{f['alpha']:<10.3f}{f['beta']:.3f}")
    print("\n  Neither is scale-invariant: K*(m) is fixed while the window grows.")


def peak():
    """Where beta peaks, architecture by architecture. The ring is alone."""
    print("=" * 96)
    print("LOCATION OF THE beta MAXIMUM (local slopes, K=64, m swept finely)")
    print("=" * 96)
    print(f"{'network':<24}{'degree':<9}{'beta_max':<11}{'alpha at peak':<16}"
          f"{'1-m at peak':<14}{'distance to (1.5,0.2)'}")
    for name, A in zoo():
        if A.shape[0] != N:
            continue
        ev, V = eigh(A, check_finite=False)
        r = beta_peak(ev, V)
        if r is None:
            continue
        b, a, lm = r
        d = float(np.hypot(a - 1.5, 2 * (b - 0.2)))
        print(f"{name:<24}{A.sum(1).mean():<9.1f}{b:<11.4f}{a:<16.4f}"
              f"{lm:<14.2e}{d:.4f}")
    print("\n  Only the ring peaks on the empirical pair, and it does so with no")
    print("  tuning: at the extremum m is not a free choice.")


def invariant():
    """p -- the variance share in modes wider than the cluster -- is what is
    fixed at alpha = 1.5. Everything else varies by orders of magnitude."""
    from scipy.optimize import brentq
    print("=" * 96)
    print("WHAT IS INVARIANT AT alpha = 1.5 (global fit; modal stats at K=16)")
    print("=" * 96)

    def alpha_glob(ev, V, lm):
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        return lt.fit_scaling(lt.coarse_grain(S, L, 4))["alpha"]

    print(f"{'network':<24}{'1-m':<12}{'beta':<10}{'p (wide share)':<17}"
          f"{'R=tau_w/tau_n':<16}{'N_eff'}")
    for name, A in zoo():
        if A.shape[0] != N:
            continue
        ev, V = eigh(A, check_finite=False)
        try:
            lm = brentq(lambda x: alpha_glob(ev, V, x) - 1.5, 1e-6, 0.3, xtol=1e-10)
        except ValueError:
            print(f"{name:<24}no alpha=1.5 crossing")
            continue
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        lv = at.cascade(S, L, 4)
        i = next(j for j, d in enumerate(lv) if d["K"] == 16)
        w, t = at.modal_weights(ev, 1 - lm)
        ab0 = at.overlaps(lv[i]["memb"], V)
        ab1 = at.overlaps(lv[i + 1]["memb"], V)
        ok = (ab0 > 1e-300) & (ab1 > 1e-300)
        g = np.zeros_like(ab0); g[ok] = np.log2(ab1[ok] / ab0[ok])
        pi = w * ab0 / np.sum(w * ab0)
        wide = g > 1.5
        p = float(np.sum(pi[wide]))
        R = np.nan
        if 0 < p < 1:
            R = (float(np.sum(pi[wide] * t[wide]) / np.sum(pi[wide]))
                 / float(np.sum(pi[~wide] * t[~wide]) / np.sum(pi[~wide])))
        print(f"{name:<24}{lm:<12.2e}{f['beta']:<10.4f}{p:<17.4f}{R:<16.1f}"
              f"{1/np.sum(pi**2):.2f}")
    print("\n  p spans 1.4x while 1-m spans 70x and R spans 244x.")


def modes():
    """The hierarchy has a ladder of modes but no m activates more than a few."""
    print("=" * 96)
    print("HOW MANY HIERARCHY LEVELS ARE EVER ACTIVE? (deterministic, Phi=2.83)")
    print("=" * 96)
    n, L, phi = 2 ** 16, 16, 2.83
    spec = at.hier_spectrum(n, phi)
    rho = max(abs(v) for v, _ in spec)
    sP = spec[0][0] / rho
    s = np.array([spec[i][0] / rho for i in range(1, L + 1)])
    mult = np.array([2.0 ** (L - i) for i in range(1, L + 1)])
    cols = [1, 2, 4, 8, 12, 16]
    print(f"{'1-m':<10}{'Perron':>9}" +
          "".join(f"{'s='+str(i):>9}" for i in cols) + f"{'N_eff':>9}")
    for lm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10):
        m = 1 - lm
        wP = 1 / (1 - m ** 2 * sP ** 2)
        w = 1 / (1 - m ** 2 * s ** 2)
        contrib = np.concatenate(([wP], mult * w))
        contrib /= contrib.sum()
        print(f"{lm:<10.0e}{contrib[0]:>9.4f}" +
              "".join(f"{contrib[i]:>9.4f}" for i in cols) +
              f"{1/np.sum(contrib**2):>9.2f}")
    print("\n  N_eff never exceeds ~4 of 16 levels. One Perron mode, one global")
    print("  control parameter, monotone amplification: only one peak at a time.")


def disorder():
    """Coupling disorder flattens the curve but drives alpha to 1, because the
    slow modes it creates are localised and localised modes have gamma -> 0."""
    print("=" * 96)
    print("DOES COUPLING DISORDER RESCUE THE SCALING?  (HM Phi=2.83)")
    print("=" * 96)
    A0 = lt.make_network(N, "hm", 2.83, 1)
    L = int(np.log2(N))

    def block_gain(sigma, seed=7):
        rng = np.random.default_rng(seed)
        g = np.ones(N)
        for lev in range(1, L + 1):
            g *= np.repeat(np.exp(sigma * rng.standard_normal(2 ** (L - lev))),
                           2 ** lev)[:N]
        return g / g.mean()

    print(f"{'disorder':<22}{'sigma':<8}{'mean alpha':<13}{'spread':<10}"
          f"{'PR/n of slowest mode'}")
    rng = np.random.default_rng(7)
    variants = [("none", 0.0, np.ones(N))]
    for sig in (0.5, 1.0):
        g = np.exp(sig * rng.standard_normal(N)); variants.append(
            (f"i.i.d. per node", sig, g / g.mean()))
    for sig in (0.6, 1.5):
        variants.append(("block, all levels", sig, block_gain(sig)))
    for label, sig, g in variants:
        M = np.sqrt(g)[:, None] * A0 * np.sqrt(g)[None, :]
        ev, V = eigh(M, check_finite=False)
        o = np.argsort(ev)[::-1]
        pr = 1.0 / (N * np.sum(V[:, o[0]] ** 4))
        best = None
        for lm in np.logspace(-6, -0.5, 100):
            S, Lm = lt.stationary_covariances(ev, V, 1 - lm)
            lv = lt.coarse_grain(S, Lm, 4)
            Ks = lv.K.values
            a = np.log2(lv.variance.values[1:] / lv.variance.values[:-1])
            sel = (Ks[:-1] >= 2) & (Ks[:-1] <= 256)
            if sel.sum() < 4:
                continue
            aw = a[sel]
            sc = np.hypot(aw.mean() - 1.5, aw.max() - aw.min())
            if best is None or sc < best[0]:
                best = (sc, aw.mean(), aw.max() - aw.min())
        print(f"{label:<22}{sig:<8.1f}{best[1]:<13.3f}{best[2]:<10.3f}{pr:.4f}")
    print("\n  Flat but wrong: disorder localises the slow modes, and a localised")
    print("  mode saturates its cluster overlap, so gamma -> 0 and alpha -> 1.")


def dimension():
    """Spectral dimension sets beta_max, and the continuum law predicts it."""
    print("=" * 96)
    print("SPECTRAL DIMENSION SETS beta_max")
    print("=" * 96)
    print(f"{'network':<24}{'graph d':<10}{'spectral d_s':<15}{'beta_max':<11}"
          f"{'predicted from d_s':<20}{'error'}")
    rows = []
    for name, A in zoo():
        if A.shape[0] != N:
            continue
        ev, V = eigh(A, check_finite=False)
        gd, sd = graph_dim(A), spectral_dim(ev)
        r = beta_peak(ev, V)
        if r is None or not np.isfinite(sd):
            continue
        b = r[0]
        pred = co.beta_max(float(sd))["beta_max"]
        rows.append((gd, sd, b))
        print(f"{name:<24}{gd:<10.2f}{sd:<15.2f}{b:<11.4f}{pred:<20.4f}"
              f"{abs(pred-b):.4f}")
    r = np.array([x for x in rows if np.all(np.isfinite(x))])
    print()
    print(f"  Spearman(graph d, beta_max)    = {spearmanr(r[:,0], r[:,2])[0]:+.3f}")
    print(f"  Spearman(spectral d, beta_max) = {spearmanr(r[:,1], r[:,2])[0]:+.3f}")
    print()
    print("  The analytic relation beta_max(d) from continuum.py:")
    print(f"    {'d':<8}{'beta_max':<12}{'alpha at peak'}")
    for d in (0.5, 1.0, 1.5, 2.0, 2.5):
        q = co.beta_max(d)
        print(f"    {d:<8}{q['beta_max']:<12.4f}{q['alpha_at_peak']:.4f}")
    print("\n  Accurate to 0.005 for rings (d ~ 1); it assumes a quadratic")
    print("  spectral edge and extended modes, and fails once those break.")


SECTIONS = {"powerlaw": powerlaw, "window": window, "peak": peak,
            "invariant": invariant, "modes": modes, "disorder": disorder,
            "dimension": dimension}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, nm in enumerate(names):
        if nm not in SECTIONS:
            raise SystemExit(f"unknown section {nm!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[nm]()
