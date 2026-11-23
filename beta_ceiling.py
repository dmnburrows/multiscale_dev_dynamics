"""The beta CEILING and its plateau: the operative quantity, measured directly.

Consolidates three analyses that were previously run inline and so had no
script. Supersedes the dimension-based framing: no dimension estimate is needed
to say which networks reach Munn's exponents without fine tuning.

WHAT IS MEASURED. Sweep 1-m over the whole range and record beta(1-m). Every
family's beta rises, peaks and falls, so each has a CEILING with a plateau. The
question is whether the Munn point coincides with that plateau, which happens
only when the ceiling itself is ~0.2:

    beta_ceiling      max of beta over the sweep
    plateau width     decades of log10(1-m) where beta >= 0.95 * ceiling
    beta at best fit  beta where the distance to (1.5, 0.2) is smallest
    |dbeta/dlog|      local slope there -- small means insensitive to m
    window            decades where that distance stays < 0.05

TERMINOLOGY. A maximum of beta in m is a STATIONARY POINT, not an attractor:
nothing here drives a network toward it, and m is scanned by hand. The claim is
about identifiability -- a ceiling at 0.2 makes beta ~ 0.2 consistent with a
broad range of m -- not about self-organisation. Note the clash with
"stationary process" as used in linear_theory.stationary_covariances.

TWO CAUTIONS THIS SCRIPT EXISTS TO RECORD.
  ring k=8 is ordered, has a 0.5-decade plateau, and still misses, because its
  plateau sits at 0.28 rather than 0.22: stationary at the WRONG VALUE. Neither
  its graph d (0.96) nor its spectral d_s (1.08) predicts that ceiling.
  Even ring k=4's plateau is only ~0.7 decades, so write "no FINE tuning", never
  "no tuning": 1-m is still pinned to within a factor of ~5.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u beta_ceiling.py
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
import spectral_classes as sc
from limits_and_dimension import graph_dim, spectral_dim

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3)
LMS = np.logspace(-5.0, -0.6, 45)
LG = np.log10(LMS)
TOL = 0.05
PLATEAU = 0.95

CONFIGS = [
    ("ring k=4",    lambda s: lt.make_network(N, "ring", 4, s)),
    ("ring k=6",    lambda s: lt.make_network(N, "ring", 6, s)),
    ("ring k=8",    lambda s: lt.make_network(N, "ring", 8, s)),
    ("sw p=0.004",  lambda s: lt.make_network(N, "sw", 0.004, s)),
    ("geo sigma=4", lambda s: sn.geometric(N, 4.0, 4.0, s)),
    ("geo sigma=3", lambda s: sn.geometric(N, 3.0, 6.0, s)),
    ("geo sigma=2", lambda s: sn.geometric(N, 2.0, 6.0, s)),
    ("hm phi=3.4",  lambda s: lt.make_network(N, "hm", 3.4, s)),
    ("hm phi=3.0",  lambda s: lt.make_network(N, "hm", 3.0, s)),
    ("hm phi=2.6",  lambda s: lt.make_network(N, "hm", 2.6, s)),
    ("er degree 6", lambda s: lt.make_network(N, "er", 6.0, s)),
]


def sweep(A):
    """beta, alpha and distance along the whole 1-m sweep, plus the dimensions."""
    ev, V = eigh(A, check_finite=False)
    a, b, d = [], [], []
    for lm in LMS:
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        a.append(f["alpha"])
        b.append(f["beta"])
        d.append(float(sc.distance(f["alpha"], f["beta"])))
    return (np.array(a), np.array(b), np.array(d),
            graph_dim(A), spectral_dim(ev))


def summarise(a, b, d):
    bmax = float(np.nanmax(b))
    i = int(np.nanargmax(b))
    j = int(np.nanargmin(d))
    pl = b >= PLATEAU * bmax
    j0, j1 = max(j - 2, 0), min(j + 2, len(LMS) - 1)
    slope = abs((b[j1] - b[j0]) / (LG[j1] - LG[j0]))
    ok = d < TOL
    return {
        "beta_ceiling": bmax, "lm_at_ceiling": LMS[i],
        "plateau_dec": float(LG[pl].max() - LG[pl].min()) if pl.any() else 0.0,
        "lm_best": LMS[j], "alpha_best": a[j], "beta_best": b[j],
        "residual": d[j], "slope": slope,
        "window_dec": float(LG[ok].max() - LG[ok].min()) if ok.any() else 0.0,
        "beta_frac_of_ceiling": b[j] / bmax,
    }


def main():
    curves, rows = [], []
    for label, build in CONFIGS:
        acc = []
        for seed in SEEDS:
            a, b, d, gd, sd = sweep(build(seed))
            acc.append(dict(summarise(a, b, d), graph_d=gd, spectral_d=sd))
            curves.append(pd.DataFrame({"config": label, "seed": seed,
                                        "one_minus_m": LMS, "alpha": a,
                                        "beta": b, "distance": d}))
        r = {k: float(np.nanmean([x[k] for x in acc])) for k in acc[0]}
        rows.append(dict(config=label, **r))
        print(f"  swept {label}", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "sweep_data"), exist_ok=True)
    p1 = os.path.join(HERE, "sweep_data", "beta_ceiling.csv")
    p2 = os.path.join(HERE, "sweep_data", "beta_curves.csv")
    out.to_csv(p1, index=False)
    pd.concat(curves, ignore_index=True).to_csv(p2, index=False)

    print("\n" + "=" * 118)
    print("THE CEILING AND ITS PLATEAU.  'frac' = beta at best fit / ceiling.")
    print("A family matches without fine tuning iff its CEILING is itself ~0.2,")
    print("so that the Munn point lands on the plateau (frac ~ 1, small slope).")
    print("=" * 118)
    print(f"  {'network':<14}{'ceiling':<10}{'plateau':<10}{'1-m best':<11}"
          f"{'alpha':<8}{'beta':<8}{'frac':<8}{'slope':<9}{'window':<9}"
          f"{'residual':<10}{'graph d':<9}{'d_s'}")
    for _, r in out.sort_values("residual").iterrows():
        print(f"  {r.config:<14}{r.beta_ceiling:<10.4f}{r.plateau_dec:<10.2f}"
              f"{r.lm_best:<11.1e}{r.alpha_best:<8.3f}{r.beta_best:<8.3f}"
              f"{r.beta_frac_of_ceiling:<8.2f}{r.slope:<9.3f}"
              f"{r.window_dec:<9.2f}{r.residual:<10.3f}"
              f"{r.graph_d:<9.2f}{r.spectral_d:.2f}")

    print("\n  Correlations against the residual (1-m already optimised away):")
    for col in ("beta_ceiling", "graph_d", "spectral_d", "lm_best",
                "plateau_dec", "slope"):
        v = out[col].to_numpy(float)
        ok = np.isfinite(v)
        from scipy.stats import spearmanr
        print(f"    {col:<16} Spearman = "
              f"{spearmanr(v[ok], out.residual.to_numpy(float)[ok])[0]:+.3f}")
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
