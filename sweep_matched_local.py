"""The degree-matched sweep again, recording LOCAL slopes per level.

Same grids, same networks, same couplings as sweep_matched.py -- only the
exponent definition differs:

    alpha_loc(K) = log2[ V(2K) / V(K) ],   beta_loc(K) = log2[ tau(2K)/tau(K) ]

so there is one row per (family, parameter, m, K) instead of one per
(family, parameter, m).
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
import sweep_matched as sm

np.seterr(all="ignore")
OUT = os.path.join(HERE, "sweep_data", "matched_local.csv")


def local_rows(ev, V, rows, **meta):
    for lm in sm.LMS:
        S, L = lt.stationary_covariances(ev, V, 1.0 - lm)
        try:
            lv = lt.coarse_grain(S, L, 4)
        except Exception:
            continue
        Vv = lv.variance.to_numpy()
        Tt = lv.tau.to_numpy()
        Ks = lv.K.to_numpy()
        a = np.log2(Vv[1:] / Vv[:-1])
        b = np.log2(Tt[1:] / Tt[:-1])
        for i in range(len(a)):
            rows.append(dict(one_minus_m=lm, K=int(Ks[i]),
                             alpha_loc=a[i], beta_loc=b[i], **meta))


def spectrum(A):
    ev, V = eigh(A, check_finite=False)
    return None if np.max(np.abs(ev)) == 0 else (ev, V)


def main():
    rows = []
    for k in sm.even_grid(60, sm.DEG_LO, sm.DEG_HI, sm.N - 2):
        A = lt.make_network(sm.N, "ring", k, sm.SEED)
        r = spectrum(A)
        if r:
            local_rows(*r, rows, family="ring", param=k, param2=np.nan,
                       degree=A.sum(1).mean())
    print(f"  ring done ({len(rows)} rows)", flush=True)

    for phi in sm.PHIS:
        A = lt.make_network(sm.N, "hm", float(phi), sm.SEED)
        r = spectrum(A)
        if r:
            local_rows(*r, rows, family="hm", param=round(float(phi), 4),
                       param2=np.nan, degree=A.sum(1).mean())
    print(f"  hm done ({len(rows)} rows)", flush=True)

    for d in np.logspace(np.log10(sm.DEG_LO), np.log10(sm.DEG_HI), 60):
        A = lt.make_network(sm.N, "er", float(min(d, sm.N - 1.0)), sm.SEED)
        r = spectrum(A)
        if r:
            local_rows(*r, rows, family="er", param=round(float(d), 4),
                       param2=np.nan, degree=A.sum(1).mean())
    print(f"  er done ({len(rows)} rows)", flush=True)

    for k in sm.K10:
        for p in sm.PS20:
            A = xn.small_world(sm.N, k, float(p), sm.SEED)
            r = spectrum(A)
            if r:
                local_rows(*r, rows, family="sw", param=k,
                           param2=round(float(p), 6), degree=A.sum(1).mean())
    print(f"  sw done ({len(rows)} rows)", flush=True)

    for d in sm.DEG10:
        for g in sm.GAMMAS20:
            A = xn.scale_free(sm.N, float(d), float(g), sm.SEED)
            r = spectrum(A)
            if r:
                local_rows(*r, rows, family="sf", param=round(float(d), 4),
                           param2=round(float(g), 4), degree=A.sum(1).mean())
    print(f"  sf done ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df.alpha_loc) & np.isfinite(df.beta_loc)].copy()
    df["dist"] = sc.distance(df.alpha_loc.values, df.beta_loc.values)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {len(df)} rows to {OUT}")
    print(df.groupby("family").dist.min())


if __name__ == "__main__":
    main()
