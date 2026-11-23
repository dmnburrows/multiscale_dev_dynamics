"""The range-composition law verifies just as well under a COIN FLIP.

range_composition.py verified

    2^alpha = sum_b w_b rho_b,        w_b = V_b(K)/V(K),  rho_b = V_b(2K)/V_b(K)
    2^beta  = (sum_b u_b sigma_b) / (sum_b w_b rho_b)

to 6e-15 with b indexing dyadic DISTANCE bands, and that was read as support for
"the exponents compose over connection ranges". But the derivation only ever
multiplies and divides by V_b(K), so it holds for ANY partition of the pairs.
If a random partition verifies equally well, the exactness carries no
information about distance -- it is the portfolio identity, not a result.

CONTROL FOR WHAT A REAL CLAIM LOOKS LIKE. For the ring, w_b and rho_b are
PREDICTABLE from the adjacency matrix alone, with no coarse-graining run:
contiguous clusters give n_d(K) = K (d=0), 2(K-d) (1<=d<K), and S is circulant
so V_b(K) = sum_{d in b} n_d(K) s(d). That is a prediction and it can fail.
`ring_block_variance.py` verifies it to 7e-15. The identity below cannot fail,
which is exactly the difference.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u composition_is_trivial.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
from range_composition import N, band_cascade, band_index, best_m, net

np.seterr(all="ignore")


def coin_flip_partition(n, nb, seed):
    """Assign every unordered pair to a group at random. No structure at all."""
    rng = np.random.default_rng(seed)
    B = rng.integers(0, nb, size=(n, n))
    B = np.triu(B) + np.triu(B, 1).T          # symmetric: a real partition
    return B.astype(np.int64)


def check(rows, nb):
    """Max relative error of the composition law across the K -> 2K doublings."""
    ea, eb = 0.0, 0.0
    for r, q in zip(rows[:-1], rows[1:]):
        w = r["V_b"] / r["V"]
        u = r["T_b"] / r["T"]
        live_v = r["V_b"] != 0
        live_t = r["T_b"] != 0
        rho = np.where(live_v, q["V_b"] / np.where(live_v, r["V_b"], 1.0), 0.0)
        sig = np.where(live_t, q["T_b"] / np.where(live_t, r["T_b"], 1.0), 0.0)
        # bands switching on from exactly zero are added directly
        add_v = q["V_b"][~live_v].sum() / r["V"]
        add_t = q["T_b"][~live_t].sum() / r["T"]
        lhs_a, rhs_a = q["V"] / r["V"], float(w @ rho) + add_v
        lhs_b = (q["T"] / q["V"]) / (r["T"] / r["V"])
        rhs_b = (float(u @ sig) + add_t) / rhs_a
        ea = max(ea, abs(lhs_a - rhs_a) / abs(lhs_a))
        eb = max(eb, abs(lhs_b - rhs_b) / abs(lhs_b))
    return ea, eb


def main():
    print("Max relative error of the composition law, over all K -> 2K doublings.")
    print("If the coin flip matches the distance bands, exactness says nothing")
    print("about distance.\n")
    print(f"{'network':<14}{'partition':<26}{'groups':<9}"
          f"{'alpha law err':<16}{'beta law err'}")
    for label, kind, par, metric in (("ring k=4", "ring", 4, "ring"),
                                     ("hm phi=3.0", "hm", 3.0, "module"),
                                     ("geo sigma=4", "geo", 4.0, "ring")):
        A = net(kind, par, 1)
        ev, V = eigh(A, check_finite=False)
        _, lm, _ = best_m(ev, V)
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        parts = [("dyadic distance bands", band_index(N, metric))]
        for s_ in (1, 2):
            parts.append((f"coin flip, 10 groups (seed {s_})",
                          coin_flip_partition(N, 10, s_)))
        parts.append(("coin flip, 2 groups", coin_flip_partition(N, 2, 3)))
        parts.append(("one group (no split)", np.zeros((N, N), dtype=np.int64)))
        for pname, B in parts:
            nb = int(B.max()) + 1
            ea, eb = check(band_cascade(S, L, B, nb), nb)
            print(f"{label:<14}{pname:<26}{nb:<9}{ea:<16.1e}{eb:.1e}")
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
