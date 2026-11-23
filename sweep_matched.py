"""Degree-matched sweep: every family over the same mean-degree range.

The hierarchical network's Phi sweeps density as a side effect, from the
complete graph at Phi = 1 (degree n-1) down to 1.67 at Phi = 5. Every other
family is matched to that range so the comparison is not confounded by density:

  ring   60 values of k (even), log-spaced across the range; degree = k
  hm     60 values of Phi from 1 to 5
  er     60 values of expected degree, log-spaced across the range
  sw     10 degrees x 20 rewiring probabilities   (2-D)
  sf     10 degrees x 20 exponents                (2-D)

sw and sf need degree freed, which linear_theory.make_network hardcodes, so
their graphs come from extended_networks (verified identical to make_network
at the hardcoded values). Everything downstream is linear_theory unchanged.

Columns: param (primary), param2 (secondary, NaN for 1-D families), degree
(realised mean degree), alpha_fit, beta_fit, dist.
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
import extended_networks as xn

np.seterr(all="ignore")
OUT = os.path.join(HERE, "sweep_data", "matched_fitted.csv")

N = 512
SEED = 1
LMS = np.logspace(np.log10(1e-4), np.log10(0.2), 20)
_L = int(np.log2(N))


def hm_degree(phi):
    r = 2.0 / phi
    return _L if abs(r - 1) < 1e-12 else (r ** _L - 1) / (r - 1)


PHIS = np.linspace(1.0, 5.0, 60)
DEG_LO, DEG_HI = min(hm_degree(p) for p in PHIS), max(hm_degree(p) for p in PHIS)


def even_grid(count, lo, hi, cap):
    """Log-spaced even integers in [2, cap], deduplicated."""
    vals = np.logspace(np.log10(max(lo, 2)), np.log10(min(hi, cap)), count)
    out = sorted({int(min(max(2, 2 * round(v / 2)), cap)) for v in vals})
    return out


DEG10 = [float(v) for v in np.logspace(np.log10(DEG_LO), np.log10(DEG_HI), 10)]
K10 = even_grid(10, DEG_LO, DEG_HI, N - 2)
PS20 = np.logspace(-3, 0, 20)
GAMMAS20 = np.linspace(2.1, 3.5, 20)


def levels_for(A):
    ev, V = eigh(A, check_finite=False)
    if np.max(np.abs(ev)) == 0:
        return None
    return ev, V


def fits(ev, V, rows, **meta):
    for lm in LMS:
        S, L = lt.stationary_covariances(ev, V, 1.0 - lm)
        try:
            f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        except Exception:
            continue
        rows.append(dict(one_minus_m=lm, alpha_fit=f["alpha"],
                         beta_fit=f["beta"], **meta))


def main():
    rows = []
    print(f"matched degree range: {DEG_LO:.3f} to {DEG_HI:.1f}", flush=True)

    for k in even_grid(60, DEG_LO, DEG_HI, N - 2):
        A = lt.make_network(N, "ring", k, SEED)
        r = levels_for(A)
        if r:
            fits(*r, rows, family="ring", param=k, param2=np.nan,
                 degree=A.sum(1).mean())
    print(f"  ring done ({len(rows)} rows)", flush=True)

    for phi in PHIS:
        A = lt.make_network(N, "hm", float(phi), SEED)
        r = levels_for(A)
        if r:
            fits(*r, rows, family="hm", param=round(float(phi), 4),
                 param2=np.nan, degree=A.sum(1).mean())
    print(f"  hm done ({len(rows)} rows)", flush=True)

    for d in np.logspace(np.log10(DEG_LO), np.log10(DEG_HI), 60):
        A = lt.make_network(N, "er", float(min(d, N - 1.0)), SEED)
        r = levels_for(A)
        if r:
            fits(*r, rows, family="er", param=round(float(d), 4),
                 param2=np.nan, degree=A.sum(1).mean())
    print(f"  er done ({len(rows)} rows)", flush=True)

    for k in K10:
        for p in PS20:
            A = xn.small_world(N, k, float(p), SEED)
            r = levels_for(A)
            if r:
                fits(*r, rows, family="sw", param=k,
                     param2=round(float(p), 6), degree=A.sum(1).mean())
        print(f"    sw k={k} done ({len(rows)} rows)", flush=True)
    print(f"  sw done ({len(rows)} rows)", flush=True)

    for d in DEG10:
        for g in GAMMAS20:
            A = xn.scale_free(N, float(d), float(g), SEED)
            r = levels_for(A)
            if r:
                fits(*r, rows, family="sf", param=round(float(d), 4),
                     param2=round(float(g), 4), degree=A.sum(1).mean())
        print(f"    sf degree={d:.2f} done ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df.alpha_fit) & np.isfinite(df.beta_fit)].copy()
    df["dist"] = sc.distance(df.alpha_fit.values, df.beta_fit.values)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {len(df)} rows to {OUT}")
    print(df.groupby("family").agg(best=("dist", "min"),
                                   deg_lo=("degree", "min"),
                                   deg_hi=("degree", "max")))


if __name__ == "__main__":
    main()
