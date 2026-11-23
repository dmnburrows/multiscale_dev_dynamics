"""Is "low effective dimension" a different axis from "proximity to criticality"?

In the continuum law (alpha, beta) = F(x, d) with x = K sqrt(1-m) = K/xi and d
the effective dimension. x says where you sit relative to criticality; d is a
property of the wiring that does not move when m moves. So they are two
arguments of one function, not one thing twice.

THE OPERATIONAL TEST. Optimise m away. For each network scan 1-m over six
decades, keep the best distance to (1.5, 0.2), and then ask whether that
RESIDUAL -- the part of the mismatch m cannot fix -- is predicted by the
measured spectral dimension. If it is, "low d" is doing independent work. If
the residuals are flat in d_s, then d was a relabelling of m after all.

d_s is measured exactly as `limits_and_dimension.py` does it, from the spectral
edge counting function N(delta) ~ n delta^(d_s/2), so these numbers are
comparable with the ones already in the index (ring 1.05, torus 2.16, HM 4.08,
ER 12.4). The gap this closes: geometric sigma=4 reaches distance 0.035, nearly
the ring's 0.028, and its d_s has never been measured. If it is high, the story
that low d_s is what matters breaks.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u dimension_vs_m.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spectral_classes as sc
import spatial_networks as sn
from limits_and_dimension import graph_dim, spectral_dim

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3)
LMS = np.logspace(-6, -0.6, 70)

CONFIGS = [
    ("ring k=4",        lambda s: lt.make_network(N, "ring", 4, s)),
    ("ring k=6",        lambda s: lt.make_network(N, "ring", 6, s)),
    ("ring k=8",        lambda s: lt.make_network(N, "ring", 8, s)),
    ("sw p=0.004",      lambda s: lt.make_network(N, "sw", 0.004, s)),
    ("geo sigma=4",     lambda s: sn.geometric(N, 4.0, 4.0, s)),
    ("geo sigma=3",     lambda s: sn.geometric(N, 3.0, 6.0, s)),
    ("geo sigma=2",     lambda s: sn.geometric(N, 2.0, 6.0, s)),
    ("geo sigma=1",     lambda s: sn.geometric(N, 1.0, 6.0, s)),
    ("hm phi=2.6",      lambda s: lt.make_network(N, "hm", 2.6, s)),
    ("hm phi=3.0",      lambda s: lt.make_network(N, "hm", 3.0, s)),
    ("hm phi=3.4",      lambda s: lt.make_network(N, "hm", 3.4, s)),
    ("er degree 6",     lambda s: lt.make_network(N, "er", 6.0, s)),
    ("sf gamma=2.5",    lambda s: lt.make_network(N, "sf", 2.5, s)),
]


def best_over_m(ev, V):
    """Scan 1-m over six decades; return the best distance and where it sits."""
    best = (np.inf, None, None, None)
    for lm in LMS:
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        d = float(sc.distance(f["alpha"], f["beta"]))
        if d < best[0]:
            best = (d, lm, f["alpha"], f["beta"])
    return best


def main():
    print("m is SCANNED OVER and discarded in every row: 'residual' is the best")
    print("distance to (1.5, 0.2) achievable at ANY m. The question is whether")
    print("that residual is explained by d_s, which m cannot change.\n")
    print(f"{'network':<16}{'graph d':<10}{'spectral d_s':<14}{'best 1-m':<11}"
          f"{'alpha':<9}{'beta':<8}{'residual'}")
    rows = []
    for label, build in CONFIGS:
        acc = []
        for seed in SEEDS:
            A = build(seed)
            ev, V = eigh(A, check_finite=False)
            d, lm, a, b = best_over_m(ev, V)
            acc.append((graph_dim(A), spectral_dim(ev), lm, a, b, d))
        r = np.nanmean(np.array(acc, dtype=float), axis=0)
        rows.append(r)
        print(f"{label:<16}{r[0]:<10.2f}{r[1]:<14.2f}{r[2]:<11.1e}"
              f"{r[3]:<9.3f}{r[4]:<8.3f}{r[5]:.3f}")
        sys.stdout.flush()

    R = np.array(rows)
    ok = np.all(np.isfinite(R[:, [1, 5]]), axis=1)
    print(f"\n  Spearman(spectral d_s, residual) = "
          f"{spearmanr(R[ok, 1], R[ok, 5])[0]:+.3f}   n={int(ok.sum())}")
    ok2 = np.all(np.isfinite(R[:, [0, 5]]), axis=1)
    print(f"  Spearman(graph d,      residual) = "
          f"{spearmanr(R[ok2, 0], R[ok2, 5])[0]:+.3f}   n={int(ok2.sum())}")
    print(f"  Spearman(best 1-m,     residual) = "
          f"{spearmanr(R[:, 2], R[:, 5])[0]:+.3f}   n={len(R)}")
    print("\n  If d_s correlates with the residual but best 1-m does not, then")
    print("  dimension is the independent axis and m is not a substitute for it.")


if __name__ == "__main__":
    main()
