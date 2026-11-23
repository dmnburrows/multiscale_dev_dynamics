"""Small-world and scale-free generators with the density freed.

linear_theory.make_network hardcodes small-world degree to 4 and scale-free
mean degree to 6, so neither can be swept over density. These two functions
reproduce those recipes EXACTLY, with the hardcoded constant replaced by an
argument. Nothing else differs: the same ring-lattice start, the same rewiring
rule and draw order for sw; the same Chung-Lu rank weights and the same
lt.sample_edges for sf.

Only the network construction is reimplemented here. Everything downstream --
covariances, coarse-graining, fitting -- still goes through linear_theory
unchanged.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt


def small_world(n, k, p, seed):
    """Watts-Strogatz: ring lattice of degree k, each local edge rewired w.p. p.

    Identical to linear_theory.make_network(..., 'sw', p, seed) when k = 4.
    """
    if k < 2 or k >= n or k % 2:
        raise ValueError("Small-world degree must be an even integer in [2, n).")
    if not 0 <= p <= 1:
        raise ValueError("Rewiring probability must be in [0, 1].")
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n, n))
    for offset in range(1, k // 2 + 1):
        for i in range(n):
            j = (i + offset) % n
            adjacency[i, j] = adjacency[j, i] = 1
    if p:
        for offset in range(1, k // 2 + 1):
            for i in range(n):
                j = (i + offset) % n
                if rng.random() < p:
                    candidates = np.flatnonzero(
                        (adjacency[i] == 0) & (np.arange(n) != i))
                    if candidates.size:
                        other = rng.choice(candidates)
                        adjacency[i, j] = adjacency[j, i] = 0
                        adjacency[i, other] = adjacency[other, i] = 1
    return adjacency


def scale_free(n, degree, gamma, seed):
    """Chung-Lu with rank weights r^(-1/(gamma-1)), rescaled to `degree`.

    Identical to linear_theory.make_network(..., 'sf', gamma, seed) when
    degree = 6. Note the min(.,1) clip: at high target degree or low gamma the
    realised mean degree falls below the target, so it is recorded separately.
    """
    if gamma <= 1:
        raise ValueError("Scale-free gamma must exceed 1.")
    rng = np.random.default_rng(seed)
    weights = np.arange(1, n + 1, dtype=float) ** (-1 / (gamma - 1))
    weights *= degree / weights.mean()
    probability = np.minimum(np.outer(weights, weights) / weights.sum(), 1)
    return lt.sample_edges(probability, rng)


def _check():
    """The k=4 / degree=6 cases must reproduce linear_theory exactly."""
    n = 256
    for p in (0.0, 0.05, 0.3, 1.0):
        a = small_world(n, 4, p, 7)
        b = lt.make_network(n, "sw", p, 7)
        print(f"  sw p={p:<6g} identical to make_network: {np.array_equal(a, b)}")
    for g in (2.1, 2.5, 3.5):
        a = scale_free(n, 6, g, 7)
        b = lt.make_network(n, "sf", g, 7)
        print(f"  sf gamma={g:<6g} identical to make_network: {np.array_equal(a, b)}")


if __name__ == "__main__":
    _check()
