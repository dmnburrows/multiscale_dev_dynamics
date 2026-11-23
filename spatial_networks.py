"""Spatially embedded network generators, for the local/global decomposition.

Every network here lives on the SAME 1-D ring of n positions, so the pair
decomposition by ring distance in local_global.py applies unchanged to all of
them. That shared coordinate is the whole point: it is what makes "local" and
"long-range" mean the same thing across families.

    geometric(n, sigma, degree, seed)
        p(i~j) proportional to d(i,j)^(-sigma), rescaled to a target mean
        degree. One knob interpolating all the way from non-spatial to local:
            sigma = 0   -> uniform, i.e. exactly Erdos-Renyi
            sigma large -> edges only between near neighbours
        This is the long-range-percolation / Kleinberg form, chosen over a hard
        radius because the exponent is a continuous locality knob.

    spatial_sf(n, gamma, sigma, degree, seed, permute=True)
        Chung-Lu scale-free weights TIMES the same distance decay, so hubs
        exist but connect preferentially nearby. sigma=0 recovers the canonical
        scale-free generator.

Both are built with linear_theory.sample_edges, so the random draw order is the
canonical one, and both are verified byte-identical to linear_theory's own
generators in the limits where they coincide:

    geometric(sigma=0, degree=z)        == make_network(er, z)
    spatial_sf(sigma=0, permute=False)  == make_network(sf, gamma)   [degree 6]

ON HM AND THE RING COORDINATE.  make_network's hierarchy sets the level of a
pair from the highest differing bit of their indices, so a level-l module is
the contiguous index block of size 2^l. Two nodes in the same module of size
2^l therefore satisfy |i-j| < 2^l. The ring-distance split "d < K" is thus the
same thing as "in the same module of size K" for the hierarchy -- the shared
coordinate is not an imposition on it.

ON HUB PLACEMENT.  Chung-Lu weights decay with index, so without `permute` the
hubs are all neighbours at one end of the ring, which is a spatial artifact.
`permute=True` (the default) shuffles which position receives which weight, so
degree is independent of position. The equivalence check uses permute=False.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u spatial_networks.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt


def ring_distance(n):
    """d(i,j) = min(|i-j|, n-|i-j|)."""
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    return np.minimum(d, n - d)


def _decay(n, sigma):
    """d^(-sigma) off the diagonal, zero on it."""
    D = ring_distance(n)
    F = np.zeros((n, n))
    nz = D > 0
    F[nz] = D[nz].astype(float) ** (-float(sigma))
    return F


def _scale_to_degree(F, degree, n, iters=80):
    """Find c with mean_i sum_j min(c F_ij, 1) == degree, by bisection.

    A plain linear rescale c = degree*n/F.sum() undershoots badly once the
    decay is steep, because the near-diagonal probabilities saturate at 1 and
    the clipped mass is lost: that route gave mean degree 2.4 for a target of
    6 at sigma=4. Clipping is monotone in c, so bisection hits the target.
    """
    lo, hi = 1e-14, 1e14
    for _ in range(iters):
        c = np.sqrt(lo * hi)
        if np.minimum(c * F, 1.0).sum() / n < degree:
            lo = c
        else:
            hi = c
    return np.minimum(np.sqrt(lo * hi) * F, 1.0)


def geometric(n, sigma, degree, seed):
    """Distance-decaying random graph; sigma=0 is exactly Erdos-Renyi."""
    if degree <= 0 or degree > n - 1:
        raise ValueError("mean degree must be in (0, n-1].")
    rng = np.random.default_rng(seed)
    if sigma == 0:
        # identical expression to make_network("er"), including the diagonal
        P = np.full((n, n), degree / (n - 1))
    else:
        P = _scale_to_degree(_decay(n, sigma), degree, n)
    return lt.sample_edges(P, rng)


def spatial_sf(n, gamma, sigma, degree, seed, permute=True):
    """Scale-free weights with a distance preference; sigma=0 is canonical sf."""
    if gamma <= 1:
        raise ValueError("Scale-free gamma must exceed 1.")
    rng = np.random.default_rng(seed)
    w = np.arange(1, n + 1, dtype=float) ** (-1 / (gamma - 1))
    w *= degree / w.mean()
    if permute:
        w = w[np.random.default_rng(seed + 10_000).permutation(n)]
    base = np.outer(w, w) / w.sum()
    if sigma == 0:
        P = np.minimum(base, 1.0)
    else:
        P = _scale_to_degree(base * _decay(n, sigma), degree, n)
    return lt.sample_edges(P, rng)


def mean_degree(A):
    return float(A.sum(axis=1).mean())


def _check():
    """Verify the limits against linear_theory, byte for byte."""
    n = 512
    ok = True
    for z in (2.0, 6.0, 20.0):
        for seed in (1, 2, 3):
            a = geometric(n, 0, z, seed)
            b = lt.make_network(n, "er", z, seed)
            same = np.array_equal(a, b)
            ok &= same
            print(f"  geometric(sigma=0, degree={z:<5}) == er({z:<5}) seed {seed}: "
                  f"{'identical' if same else 'DIFFERS'}")
    for g in (2.0, 2.5, 3.5):
        for seed in (1, 2):
            a = spatial_sf(n, g, 0, 6.0, seed, permute=False)
            b = lt.make_network(n, "sf", g, seed)
            same = np.array_equal(a, b)
            ok &= same
            print(f"  spatial_sf(sigma=0, gamma={g}, no permute) == sf({g}) "
                  f"seed {seed}: {'identical' if same else 'DIFFERS'}")
    return ok


def main():
    print("=" * 88)
    print("EQUIVALENCE CHECKS against linear_theory.make_network")
    print("=" * 88)
    ok = _check()
    print(f"\n  {'all limits reproduce the canonical generators'if ok else 'MISMATCH'}")

    print("\n" + "=" * 88)
    print("REALISED PROPERTIES  (n=512, seed 1)")
    print("=" * 88)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = 512
    print(f"  {'family':<34}{'mean degree':<14}{'largest component':<20}"
          f"{'mean edge length'}")
    D = ring_distance(n)

    def describe(name, A):
        _, lab = connected_components(csr_matrix(A), directed=False)
        frac = np.bincount(lab).max() / n
        el = D[A > 0]
        print(f"  {name:<34}{mean_degree(A):<14.2f}{frac:<20.3f}"
              f"{el.mean() if el.size else float('nan'):.1f}")

    describe("er (degree 6)", lt.make_network(n, "er", 6.0, 1))
    for s in (0.5, 1.0, 2.0, 3.0, 4.0):
        describe(f"geometric sigma={s} (degree 6)", geometric(n, s, 6.0, 1))
    for s in (0.0, 1.0, 2.0, 3.0):
        describe(f"spatial_sf gamma=2.5 sigma={s}", spatial_sf(n, 2.5, s, 6.0, 1))
    for phi in (2.2, 2.4, 2.8, 3.5):
        describe(f"hm phi={phi}", lt.make_network(n, "hm", phi, 1))
    describe("ring k=4", lt.make_network(n, "ring", 4, 1))
    print("\n  mean edge length is the mean ring distance spanned by an edge;")
    print("  n/4 = 128 is the value for a spatially unstructured graph.")


if __name__ == "__main__":
    main()
