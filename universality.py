"""Is the exponent pair a universal point on a one-parameter curve?

Panel C of Fig 3. analytical_theory.py established that for the ring and for the
deterministic hierarchy, alpha and beta collapse onto a SINGLE curve and beta has
exactly one maximum on it:

    ring k=4:   beta_max = 0.21324 at alpha = 1.527   (closest to (1.5,0.2): 0.038)
    hierarchy:  best grid point   (1.497, 0.228)      (distance 0.056)

Two architectures with nothing in common peaking next to the Munn exponents is
either a coincidence or a law. This module tests which, on structurally different
exactly-solvable families, none used to build the explanation:

  ring       d = 1, k-nearest on a cycle        van Hove edge, z = 2
  torus      d = 2, 3, k-nearest per axis       same edge, different dimension
  hypercube  degree log2(n), binomial spectrum  ultrametric-style overlap rule
  hierarchy  Phi^(-level) weights               geometric multiplicity ladder


WHAT IS COARSE-GRAINED, AND WHY IT IS NOT GREEDY ICG HERE
---------------------------------------------------------
Greedy ICG does NOT produce congruent clusters outside 1D and the tree. Measured
on a 32x32 torus at m = 0.99: at K = 2, 490/512 clusters are dominoes; by K = 16,
0/64 are rectangles and 63 of the 64 clusters have distinct shapes. Greedy
matching in d >= 2 is geometrically frustrated, so the cluster average in the
modal identity stops being a single term and no closed form exists.

So the closed forms below are for IDEAL BOX coarse-graining (congruent boxes,
sides doubled round-robin across axes), which is what ICG approximates. The cost
of that idealisation is measured, not assumed:

    torus d=2:  |d alpha| <= 0.019,  |d beta| <= 0.011   vs greedy
    torus d=3:  |d alpha| <= 0.017,  |d beta| <= 0.005   and shrinking as m -> 1

The hypercube is a different story and a real finding. It is BIPARTITE, so its
spectrum is symmetric: the radius is attained at -rho as well as +rho, and the
+lambda / -lambda terms in Sigma cancel EXACTLY for any pair of nodes in
different bipartition classes. Measured corr(0, neighbour) = 0.000000 at every m
and every dim. Nearest neighbours in a bipartite network are exactly
uncorrelated, so greedy ICG never merges across an edge -- 511/512 of its pairs
sit at Hamming distance 2 -- and coarse-graining proceeds inside each class
separately. Subcube coarse-graining is therefore a different partition from the
one greedy takes (gap up to 0.12 in alpha), and the hypercube row below is a
statement about box coarse-graining, flagged as such.


OVERLAP RULES (the only family-specific input)
----------------------------------------------
  ring / torus:  Fourier modes and box clusters, so abar is a PRODUCT of
                 Dirichlet kernels over the axes:
                     abar_q(K) = prod_i D_{K_i}(j_i) / n

  hypercube:     characters chi_x, clusters are subcubes with t varying
                 coordinates. The sum over a subcube vanishes unless x is
                 supported on the FIXED coordinates, giving
                     abar_x(K) = K^2/n  if supp(x) subset fixed  else 0
                 so with rem = dim - t,
                     V(K) = (K^2/n) sum_j C(rem,j) w_j,   s_j = 1 - 2j/dim
                 Structurally the hierarchy's rule with binomial instead of
                 geometric multiplicities: a clean test of what the multiplicity
                 ladder does on its own.

Local slopes for d >= 2 use ISOTROPIC steps (K -> 2^d K, one side doubling),
divided by d. Per-axis doublings alternate the aspect ratio and make the local
slope oscillate, which is an artefact of the step, not of the dynamics.

    python universality.py validate
    python universality.py curves
    python universality.py collapse
    python universality.py all
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import analytical_theory as at


# =============================================================================
# torus:  d-dimensional, k nearest neighbours per axis (d = 1 is the ring)
# =============================================================================

def torus_adjacency(L, d, k):
    """A for the d-torus of side L, each axis coupled to +-1..+-k/2, row-major."""
    n = L ** d
    coords = np.stack(np.unravel_index(np.arange(n), (L,) * d), axis=1)
    A = np.zeros((n, n))
    for axis in range(d):
        for step in range(1, k // 2 + 1):
            for sign in (+1, -1):
                shifted = coords.copy()
                shifted[:, axis] = (shifted[:, axis] + sign * step) % L
                j = np.ravel_multi_index(tuple(shifted.T), (L,) * d)
                A[np.arange(n), j] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def torus_spectrum(L, d, k):
    """s = lambda/rho over the whole q grid, flattened row-major. rho = d*k."""
    per_axis = np.zeros(L)
    for step in range(1, k // 2 + 1):
        per_axis += 2.0 * np.cos(2.0 * np.pi * step * np.arange(L) / L)
    s = np.zeros((L,) * d)
    for axis in range(d):
        shape = [1] * d
        shape[axis] = L
        s = s + per_axis.reshape(shape)
    return (s / (d * k)).ravel()


def torus_sides(d, t, mode="box"):
    """Box sides after t doublings: round-robin across axes, or all on one axis."""
    if mode == "stripe":
        return [2 ** t] + [1] * (d - 1)
    return [2 ** (t // d + (1 if i < t % d else 0)) for i in range(d)]


def torus_overlap(L, d, sides):
    """prod_i D_{K_i}(j_i), flattened row-major to match torus_spectrum."""
    out = np.ones((L,) * d)
    for axis, K_i in enumerate(sides):
        shape = [1] * d
        shape[axis] = L
        out = out * at.dirichlet(L, K_i).reshape(shape)
    return out.ravel()


def torus_levels(L, d, k, m, ts, mode="box", s=None):
    """V(K), tau(K) for box coarse-graining at K = 2^t. No matrix, no ICG."""
    if s is None:
        s = torus_spectrum(L, d, k)
    n = L ** d
    w = 1.0 / (1.0 - m ** 2 * s ** 2)
    tk = 1.0 / (1.0 - m * s)
    rows = []
    for t in ts:
        D = torus_overlap(L, d, torus_sides(d, t, mode))
        V = float(np.sum(w * D)) / n
        rows.append({"K": 2 ** t, "variance": V,
                     "tau": float(np.sum(w * tk * D)) / n / V})
    return pd.DataFrame(rows)


def box_membership(L, d, t):
    """The ideal box partition itself, for validating the closed form."""
    sides = np.array(torus_sides(d, t))
    coords = np.stack(np.unravel_index(np.arange(L ** d), (L,) * d), axis=1)
    return np.ravel_multi_index(tuple((coords // sides).T),
                                tuple(L // s for s in sides))


# =============================================================================
# hypercube:  degree dim, binomial spectrum, subcube clusters
# =============================================================================

def hypercube_adjacency(dim):
    n = 2 ** dim
    labels = np.arange(n)
    A = np.zeros((n, n))
    for b in range(dim):
        A[labels, labels ^ (1 << b)] = 1.0
    return A


def hypercube_levels(dim, m, ts):
    """V(K), tau(K) for subcube coarse-graining: O(dim) work, no matrix."""
    n = 2 ** dim
    rows = []
    for t in ts:
        rem = dim - t                       # coordinates x may be supported on
        j = np.arange(rem + 1)
        s = 1.0 - 2.0 * j / dim
        w = 1.0 / (1.0 - m ** 2 * s ** 2)
        tk = 1.0 / (1.0 - m * s)
        c = comb(rem, j, exact=False)
        num_w = float(np.sum(c * w))
        rows.append({"K": 2 ** t, "variance": (4 ** t / n) * num_w,
                     "tau": float(np.sum(c * w * tk)) / num_w})
    return pd.DataFrame(rows)


def subcube_membership(dim, t):
    """The ideal subcube partition: the t lowest bits vary within a cluster."""
    return np.arange(2 ** dim) >> t


def exact_levels_from_membership(ev, V, m, memberships):
    """V(K), tau(K) via the exact modal identity for an IMPOSED partition."""
    w, tk = at.modal_weights(ev, m)
    rows = []
    for K, memb in memberships:
        ab = at.overlaps(memb, V)
        num = float(np.sum(w * ab))
        rows.append({"K": K, "variance": num,
                     "tau": float(np.sum(w * tk * ab)) / num})
    return pd.DataFrame(rows)


# =============================================================================
# curves, maxima, collapse
# =============================================================================

def local_curve(levels, step=1):
    """Local slopes over K -> 2^step K, divided by step. Exact, no fitting."""
    V, T, K = levels.variance.values, levels.tau.values, levels.K.values
    return pd.DataFrame({"K": K[:-step],
                         "alpha": np.log2(V[step:] / V[:-step]) / step,
                         "beta": np.log2(T[step:] / T[:-step]) / step})


def curve_summary(curve, label, target=(1.5, 0.2), beta_weight=2):
    """beta_max, alpha there, and the closest approach of the curve to the target."""
    a, b, K = curve.alpha.values, curve.beta.values, curve.K.values
    i = int(np.nanargmax(b))
    dist = at.distance_to_target(a, b, target, beta_weight)
    jm = int(np.nanargmin(dist))
    return {"family": label, "beta_max": b[i], "alpha_at_beta_max": a[i],
            "K_at_beta_max": int(K[i]), "closest_alpha": a[jm],
            "closest_beta": b[jm], "closest_distance": dist[jm],
            "K_closest": int(K[jm])}


def production_curves(one_minus_m=1e-4):
    """Exact (alpha, beta) curves at large n, one per family."""
    m = 1.0 - one_minus_m
    out = {}

    for k in (4, 6):
        s = at.ring_spectrum(2 ** 20, k)
        out[f"ring d=1 k={k}"] = local_curve(
            at.ring_levels(2 ** 20, k, m, [2 ** t for t in range(1, 20)], s=s), step=1)

    for L, d in ((1024, 2), (128, 3)):
        s = torus_spectrum(L, d, 4)
        out[f"torus d={d} k=4"] = local_curve(
            torus_levels(L, d, 4, m, list(range(1, d * int(np.log2(L)) - 1)), s=s),
            step=d)

    out["hypercube dim=20"] = local_curve(
        hypercube_levels(20, m, list(range(1, 20))), step=1)

    for phi in (2.83, 3.5, 4.0, 5.0):
        out[f"hierarchy Phi={phi}"] = local_curve(
            at.hier_levels(2 ** 20, phi, m, [2 ** t for t in range(2, 20)]), step=1)

    return out


# =============================================================================
# 1  VALIDATION
# =============================================================================

def validate():
    print("=" * 100)
    print("1a  CLOSED FORMS vs the exact modal identity on the SAME imposed partition")
    print("    (this checks the algebra; the partition question is 1b)")
    print("=" * 100)
    print(f"{'family':<20}{'n':<8}{'m':<9}{'max rel err V':<17}{'max rel err tau'}")
    for L, d in ((512, 1), (32, 2), (8, 3)):
        A = torus_adjacency(L, d, 4)
        ev, V = eigh(A, check_finite=False)
        ts = list(range(0, int(np.log2(L ** d)) - 1))
        for m in (0.9, 0.99, 0.999):
            num = exact_levels_from_membership(
                ev, V, m, [(2 ** t, box_membership(L, d, t)) for t in ts])
            cf = torus_levels(L, d, 4, m, ts)
            print(f"{f'torus d={d} L={L}':<20}{L**d:<8}{m:<9}"
                  f"{np.abs(cf.variance.values/num.variance.values-1).max():<17.2e}"
                  f"{np.abs(cf.tau.values/num.tau.values-1).max():.2e}")
    for dim in (9, 10):
        ev, V = eigh(hypercube_adjacency(dim), check_finite=False)
        ts = list(range(0, dim - 1))
        for m in (0.9, 0.99, 0.999):
            num = exact_levels_from_membership(
                ev, V, m, [(2 ** t, subcube_membership(dim, t)) for t in ts])
            cf = hypercube_levels(dim, m, ts)
            print(f"{f'hypercube dim={dim}':<20}{2**dim:<8}{m:<9}"
                  f"{np.abs(cf.variance.values/num.variance.values-1).max():<17.2e}"
                  f"{np.abs(cf.tau.values/num.tau.values-1).max():.2e}")

    print()
    print("=" * 100)
    print("1b  COST OF IDEALISATION: greedy ICG (linear_theory.py) vs box coarse-graining")
    print("=" * 100)
    print(f"{'family':<20}{'m':<9}{'alpha greedy':<14}{'alpha box':<12}{'d alpha':<11}"
          f"{'beta greedy':<13}{'beta box':<11}{'d beta'}")
    for L, d, label in ((512, 1, "ring d=1 L=512"), (32, 2, "torus d=2 L=32"),
                        (8, 3, "torus d=3 L=8")):
        ev, V = eigh(torus_adjacency(L, d, 4), check_finite=False)
        for m in (0.9, 0.99, 0.999):
            S, Lm = lt.stationary_covariances(ev, V, m)
            greedy = lt.coarse_grain(S, Lm, 4)
            ts = [int(round(np.log2(K))) for K in greedy.K]
            box = torus_levels(L, d, 4, m, ts)
            fg, fb = lt.fit_scaling(greedy), lt.fit_scaling(box)
            print(f"{label:<20}{m:<9}{fg['alpha']:<14.5f}{fb['alpha']:<12.5f}"
                  f"{fg['alpha']-fb['alpha']:<+11.5f}{fg['beta']:<13.5f}"
                  f"{fb['beta']:<11.5f}{fg['beta']-fb['beta']:+.5f}")
    for dim in (9, 10):
        ev, V = eigh(hypercube_adjacency(dim), check_finite=False)
        for m in (0.9, 0.99, 0.999):
            S, Lm = lt.stationary_covariances(ev, V, m)
            greedy = lt.coarse_grain(S, Lm, 4)
            ts = [int(round(np.log2(K))) for K in greedy.K]
            sub = hypercube_levels(dim, m, ts)
            fg, fs = lt.fit_scaling(greedy), lt.fit_scaling(sub)
            print(f"{f'hypercube dim={dim}':<20}{m:<9}{fg['alpha']:<14.5f}{fs['alpha']:<12.5f}"
                  f"{fg['alpha']-fs['alpha']:<+11.5f}{fg['beta']:<13.5f}"
                  f"{fs['beta']:<11.5f}{fg['beta']-fs['beta']:+.5f}")
    print("  d >= 2 lattices: greedy is frustrated but stays within ~0.02 of boxes.")
    print("  hypercube: greedy coarse-grains INSIDE one bipartition class, a different")
    print("  partition entirely, so its row is not an idealisation error (see 1c).")

    print()
    print("=" * 100)
    print("1c  WHY: bipartite networks have exactly zero cross-class covariance")
    print("=" * 100)
    print(f"{'dim':<7}{'s_min':<12}{'corr(0,nbr)':<15}{'corr(0,dist 2)':<17}"
          f"{'greedy pair Hamming distances'}")
    for dim in (8, 9, 10):
        ev, V = eigh(hypercube_adjacency(dim), check_finite=False)
        for m in (0.9, 0.99):
            S, _ = lt.stationary_covariances(ev, V, m)
            sd = np.sqrt(np.diag(S))
            C = S / np.outer(sd, sd)
            hd = np.array([bin(int(a) ^ int(b)).count("1")
                           for a, b in lt.greedy_pairs(S)])
            u, c = np.unique(hd, return_counts=True)
            print(f"{dim:<7}{ev[0]/np.abs(ev).max():<+12.6f}{C[0,1]:<15.6f}"
                  f"{C[0,3]:<17.6f}{dict(zip(u.tolist(), c.tolist()))}")
    print("  The +lambda / -lambda eigenvector pairs cancel identically in Sigma for")
    print("  nodes in different classes, for EVERY m. Not an approximation.")

    print()
    print("  Shapes greedy actually makes on a 32x32 torus at m=0.99:")
    ev, V = eigh(torus_adjacency(32, 2, 4), check_finite=False)
    S, Lm = lt.stationary_covariances(ev, V, 0.99)
    coords = np.stack(np.unravel_index(np.arange(1024), (32, 32)), axis=1)
    for lv in at.cascade(S, Lm, 4):
        K, memb = lv["K"], lv["memb"]
        if K == 1:
            continue
        shapes, rect = set(), 0
        for c in range(memb.max() + 1):
            pts = coords[np.flatnonzero(memb == c)]
            shapes.add(min(tuple(sorted(map(tuple, (pts - sh) % 32))) for sh in pts))
            ext = (pts - pts.min(0)) % 32
            if np.prod(ext.max(0) - ext.min(0) + 1) == len(pts):
                rect += 1
        print(f"    K={K:<5} clusters={memb.max()+1:<6} distinct shapes={len(shapes):<6}"
              f" rectangular={rect}/{memb.max()+1}")


# =============================================================================
# 2  THE CURVES, and where beta peaks
# =============================================================================

def curves():
    print("=" * 104)
    print("2  WHERE beta PEAKS, family by family   (1-m = 1e-4, exact local slopes)")
    print("=" * 104)
    df = pd.DataFrame([curve_summary(c, n) for n, c in production_curves().items()])
    print(f"{'family':<20}{'beta_max':<11}{'alpha there':<13}{'K there':<11}"
          f"{'closest (alpha,beta)':<24}{'distance':<11}{'K'}")
    for _, r in df.iterrows():
        print(f"{r.family:<20}{r.beta_max:<11.5f}{r.alpha_at_beta_max:<13.5f}"
              f"{r.K_at_beta_max:<11}"
              f"{f'({r.closest_alpha:.4f}, {r.closest_beta:.4f})':<24}"
              f"{r.closest_distance:<11.4f}{r.K_closest}")
    print()
    print(f"  beta_max spread:            {df.beta_max.min():.4f} .. {df.beta_max.max():.4f}")
    print(f"  alpha at beta_max spread:   {df.alpha_at_beta_max.min():.4f} .. "
          f"{df.alpha_at_beta_max.max():.4f}")
    print(f"  distance to (1.5, 0.2):     {df.closest_distance.min():.4f} .. "
          f"{df.closest_distance.max():.4f}")

    print()
    print("=" * 104)
    print("  Is the peak a property of the architecture or of the distance from")
    print("  instability? (alpha, beta) AT the beta-maximum, across 1-m")
    print("=" * 104)
    table, names = {}, None
    for lm in (1e-2, 1e-4, 1e-6):
        cs = production_curves(lm)
        names = list(cs)
        table[lm] = {n: curve_summary(c, n) for n, c in cs.items()}
    print(f"  {'family':<20}" + "".join(f"{'1-m='+f'{v:.0e}':>22}" for v in (1e-2, 1e-4, 1e-6)))
    for name in names:
        cells = ["({:.4f}, {:.4f})".format(table[lm][name]["alpha_at_beta_max"],
                                           table[lm][name]["beta_max"])
                 for lm in (1e-2, 1e-4, 1e-6)]
        print(f"  {name:<20}" + "".join(f"{c:>22}" for c in cells))
    print("  Columns agreeing => the peak belongs to the architecture, and m only")
    print("  sets WHERE ON the curve a given cluster size K lands.")


# =============================================================================
# 3  COLLAPSE
# =============================================================================

def collapse():
    print("=" * 100)
    print("3  COLLAPSE   predicted variable: x = K^(1/d) sqrt(1-m)  (linear size / xi)")
    print("=" * 100)
    for L, d, k, label in ((2 ** 20, 1, 4, "ring d=1 k=4"),
                           (1024, 2, 4, "torus d=2 k=4"),
                           (128, 3, 4, "torus d=3 k=4")):
        s = at.ring_spectrum(L, k) if d == 1 else torus_spectrum(L, d, k)
        print(f"\n  {label}:   local alpha (top) and beta (bottom) at matched x")
        sides = [2 ** p for p in range(1, 7)]
        for which in (0, 1):
            print(f"  {'1-m':<10}" + "".join(f"{'side='+str(v):>12}" for v in sides))
            for lm in (1e-2, 1e-4, 1e-6):
                m = 1 - lm
                vals = []
                for side in sides:
                    t = d * int(round(np.log2(side)))
                    if d == 1:
                        lv = at.ring_levels(L, k, m, [2 ** t, 2 ** (t + d)], s=s)
                    else:
                        lv = torus_levels(L, d, k, m, [t, t + d], s=s)
                    col = "variance" if which == 0 else "tau"
                    vals.append(float(np.log2(lv[col][1] / lv[col][0]) / d))
                print(f"  {lm:<10.0e}" + "".join(f"{v:>12.5f}" for v in vals))
            print()
        print("  x = side * sqrt(1-m); rows are at matched SIDE, so a collapse shows up")
        print("  as the same value appearing at side scaled by sqrt(1-m) ratios.")

    print()
    print("=" * 100)
    print("  Non-lattice families: does the peak drift with 1-m? (no collapse => drift)")
    print("=" * 100)
    print(f"  {'family':<20}{'1-m':<10}{'alpha at peak':<16}{'beta_max':<12}{'K at peak'}")
    for lm in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        r = curve_summary(local_curve(hypercube_levels(20, 1 - lm, list(range(1, 20)))),
                          "hypercube dim=20")
        print(f"  {'hypercube dim=20':<20}{lm:<10.0e}{r['alpha_at_beta_max']:<16.5f}"
              f"{r['beta_max']:<12.5f}{r['K_at_beta_max']}")
    print()
    for lm in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        r = curve_summary(local_curve(at.hier_levels(
            2 ** 20, 4.0, 1 - lm, [2 ** t for t in range(2, 20)])), "hierarchy Phi=4")
        print(f"  {'hierarchy Phi=4.0':<20}{lm:<10.0e}{r['alpha_at_beta_max']:<16.5f}"
              f"{r['beta_max']:<12.5f}{r['K_at_peak' if False else 'K_at_beta_max']}")


SECTIONS = {"validate": validate, "curves": curves, "collapse": collapse}


if __name__ == "__main__":
    np.seterr(all="ignore")
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[name]()
