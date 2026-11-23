"""Dense parameter sweep, fitted exponents only.

Roughly five times the parameter resolution of the earlier grids, with alpha
and beta from linear_theory.fit_scaling on its default window -- one slope per
network across all cluster sizes, as the empirical exponents are measured.

One row per (family, parameter, seed, m).
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
OUT = os.path.join(HERE, "sweep_data", "dense_fitted.csv")

NPAR = 60
N = 512
_L = int(np.log2(N))


def hm_degree(phi):
    """Mean degree of the hierarchical network: sum_l (2/phi)^l over L levels.

    Phi sweeps density as a side effect -- 511 at phi=1 (the complete graph)
    down to 1.67 at phi=5 -- so the ER grid below is matched to this range to
    keep the two comparable on density rather than only on parameter index.
    """
    r = 2.0 / phi
    return _L if abs(r - 1) < 1e-12 else (r ** _L - 1) / (r - 1)


_PHIS = np.linspace(1.0, 5.0, NPAR)
_DEG = np.array([hm_degree(p) for p in _PHIS])

GRIDS = {
    # ring coupling is k-nearest per side, so k must stay even
    "ring": [int(v) for v in np.arange(2, 2 * NPAR + 1, 2)],
    "hm":   [round(v, 4) for v in _PHIS],
    # matched to the HM mean-degree range; capped at n-1 (the complete graph)
    "er":   [float(min(v, N - 1.0)) for v in
             np.logspace(np.log10(_DEG.min()), np.log10(_DEG.max()), NPAR)],
    # small-world degree is hardcoded to 4 and scale-free to ~6 inside
    # linear_theory.make_network, so these sweep topology at fixed density
    "sw":   [round(v, 6) for v in np.logspace(-3, 0, NPAR)],
    "sf":   [round(v, 4) for v in np.linspace(2.1, 3.5, NPAR)],
}
SEEDS = [1]
# m from 0.9999 down to 0.8, log-spaced in the distance from instability
LMS = np.logspace(np.log10(1e-4), np.log10(0.2), 20)


def _flush(rows):
    """Write what we have so far, so a long run is usable before it ends."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[np.isfinite(df.alpha_fit) & np.isfinite(df.beta_fit)].copy()
    df["dist"] = sc.distance(df.alpha_fit.values, df.beta_fit.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)


def main():
    rows = []
    for fam, params in GRIDS.items():
        for par in params:
            for seed in SEEDS:
                try:
                    A = lt.make_network(N, fam, par, seed)
                    ev, V = eigh(A, check_finite=False)
                except Exception as exc:                     # keep going
                    print(f"  skip {fam} {par} seed {seed}: {exc}", flush=True)
                    continue
                if np.max(np.abs(ev)) == 0:
                    continue
                for lm in LMS:
                    S, L = lt.stationary_covariances(ev, V, 1.0 - lm)
                    try:
                        levels = lt.coarse_grain(S, L, 4)
                        f = lt.fit_scaling(levels)
                    except Exception:
                        continue
                    rows.append({"family": fam, "param": par, "seed": seed,
                                 "one_minus_m": lm,
                                 "alpha_fit": f["alpha"], "beta_fit": f["beta"]})
        _flush(rows)
        print(f"  {fam}: {len(params)} parameter values done "
              f"({len(rows)} rows so far)", flush=True)

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df.alpha_fit) & np.isfinite(df.beta_fit)]
    df["dist"] = sc.distance(df.alpha_fit.values, df.beta_fit.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {len(df)} rows to {OUT}")
    print(df.groupby("family").dist.min())


if __name__ == "__main__":
    main()
