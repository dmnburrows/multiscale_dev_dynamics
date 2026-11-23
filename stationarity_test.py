"""Is the Munn pair a STATIONARY POINT or a TUNED point, family by family?

continuum.py shows that for effective dimension d <~ 2.5 beta has an INTERIOR
maximum in x = K sqrt(1-m), and that at that maximum alpha is pinned near 1.5
(1.507-1.614 over d = 0.5-2.5). The ring (d_s = 1.05) therefore reaches (1.5,
0.2) by sitting on an extremum, where dbeta/dx = 0 makes beta insensitive to the
one free parameter. That is what "parameter-free" means for the ring.

For d >~ 3 the maximum is at x = 0, so there is no interior peak to sit on. HM's
measured d_s is 4.08, yet HM empirically reaches (1.527, 0.194) -- distance
0.029, as close as the ring. Either HM's effective d is not its spectral
dimension, or HM reaches the pair OFF peak, i.e. by tuning.

THE TEST, which needs no partition, no binning and no closed form. Sweep m
finely through each family's best fit and measure

    dbeta / dlog10(1-m)      at the best point: ~0 if stationary
    the width in log10(1-m) over which the distance to (1.5, 0.2) stays < 0.05

A family sitting on a stationary point has a small derivative and a wide
tolerant window. A tuned family has a large derivative and a narrow window.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u stationarity_test.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spectral_classes as sc
from range_composition import N, net

np.seterr(all="ignore")

SEEDS = (1, 2, 3)
TOL = 0.05


def curve(kind, par, seed, lms):
    A = (lt.make_network(N, "er", 6.0, seed) if kind == "er" else net(kind, par, seed))
    ev, V = eigh(A, check_finite=False)
    out = []
    for lm in lms:
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        out.append((f["alpha"], f["beta"],
                    float(sc.distance(f["alpha"], f["beta"]))))
    return np.array(out)


def main():
    lms = np.logspace(-5.5, -0.6, 90)
    lg = np.log10(lms)
    print(f"Fine sweep of 1-m, {len(lms)} points, {len(SEEDS)} seeds, n={N}.")
    print(f"'window' = width in log10(1-m) where distance to (1.5,0.2) < {TOL}.")
    print(f"'|dbeta/dlog|' is the local slope at the best point: ~0 means beta is")
    print("stationary there, so the pair survives changing m.\n")
    print(f"{'network':<14}{'best 1-m':<12}{'alpha':<9}{'beta':<8}{'dist':<9}"
          f"{'|dbeta/dlog|':<15}{'window':<10}{'beta_max on curve'}")

    for label, kind, par in (("ring k=4", "ring", 4), ("ring k=8", "ring", 8),
                             ("hm phi=3.0", "hm", 3.0), ("hm phi=2.6", "hm", 2.6),
                             ("geo sigma=4", "geo", 4.0), ("geo sigma=2", "geo", 2.0),
                             ("er degree 6", "er", 6.0)):
        rows = []
        for seed in SEEDS:
            c = curve(kind, par, seed, lms)
            i = int(np.nanargmin(c[:, 2]))
            # local slope of beta by central difference in log10(1-m)
            j0, j1 = max(i - 2, 0), min(i + 2, len(lms) - 1)
            db = abs((c[j1, 1] - c[j0, 1]) / (lg[j1] - lg[j0]))
            ok = c[:, 2] < TOL
            width = (lg[ok].max() - lg[ok].min()) if ok.any() else 0.0
            rows.append((lms[i], c[i, 0], c[i, 1], c[i, 2], db, width,
                         np.nanmax(c[:, 1])))
        r = np.array(rows).mean(axis=0)
        print(f"{label:<14}{r[0]:<12.2e}{r[1]:<9.3f}{r[2]:<8.3f}{r[3]:<9.3f}"
              f"{r[4]:<15.3f}{r[5]:<10.2f}{r[6]:.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
