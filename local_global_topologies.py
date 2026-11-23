"""Is the local/global split of ICG scaling a general result or a ring artifact?

local_global.py found, for small-world networks, that the Munn pair (1.5, 0.2)
is carried by SPATIALLY LOCAL pairs while long-range pairs pool near alpha=2
(the fully coherent limit). That was one family on one topology. This runs the
identical decomposition across five spatially embedded families on the same
ring coordinate:

    er               no spatial structure at all -- the CONTROL. If the split
                     is detecting real spatial organisation rather than
                     manufacturing a difference, ER's clusters should not be
                     spatially compact and the split should be meaningless.
    geometric        p ~ d^(-sigma); sigma is a pure locality knob
    spatial_sf       scale-free hubs that prefer nearby targets
    hm               hierarchical modular (its dyadic modules ARE ring blocks)
    ring / sw        the previous result, carried along for continuity

m IS AN EXPLICIT AXIS AND IS NEVER OPTIMISED. local_global.py established that
re-optimising m per parameter value silently erases parameter dependence,
because m then drifts 20x across the range and compensates. Every m here is
fixed and every row states it.

VALIDITY OF THE SPLIT. The decomposition is always exact, but it is only
*interpretable* where clusters are still spatial objects and the long-range
part is big enough to fit. Both conditions are reported per row rather than
assumed:
    compactness > 0.3   clusters are still recognisable ring neighbourhoods
    glob share  > 0.02  enough long-range variance to fit a slope to
Rows failing either are printed with a flag and excluded from the summary.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u local_global_topologies.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spatial_networks as sn
from local_global import decompose, ring_distance, slope

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3, 4)
ONE_MINUS_M = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
REF_K = 64
DEGREE = 6.0

# (label, builder) -- every builder takes a seed and returns an adjacency matrix
CONFIGS = [
    ("er degree 6",            lambda s: lt.make_network(N, "er", DEGREE, s)),
    ("geometric sigma=0.5",    lambda s: sn.geometric(N, 0.5, DEGREE, s)),
    ("geometric sigma=1",      lambda s: sn.geometric(N, 1.0, DEGREE, s)),
    ("geometric sigma=2",      lambda s: sn.geometric(N, 2.0, DEGREE, s)),
    ("geometric sigma=3",      lambda s: sn.geometric(N, 3.0, DEGREE, s)),
    ("geometric sigma=4",      lambda s: sn.geometric(N, 4.0, DEGREE, s)),
    ("spatial_sf sigma=0",     lambda s: sn.spatial_sf(N, 2.5, 0.0, DEGREE, s)),
    ("spatial_sf sigma=1",     lambda s: sn.spatial_sf(N, 2.5, 1.0, DEGREE, s)),
    ("spatial_sf sigma=2",     lambda s: sn.spatial_sf(N, 2.5, 2.0, DEGREE, s)),
    ("spatial_sf sigma=3",     lambda s: sn.spatial_sf(N, 2.5, 3.0, DEGREE, s)),
    ("hm phi=2.2",             lambda s: lt.make_network(N, "hm", 2.2, s)),
    ("hm phi=2.4",             lambda s: lt.make_network(N, "hm", 2.4, s)),
    ("hm phi=2.8",             lambda s: lt.make_network(N, "hm", 2.8, s)),
    ("sw p=0.004",             lambda s: lt.make_network(N, "sw", 0.004, s)),
    ("ring k=4",               lambda s: lt.make_network(N, "ring", 4, s)),
]


def main():
    D = ring_distance(N)
    rows = []

    for label, build in CONFIGS:
        for seed in SEEDS:
            A = build(seed)
            _, lab = connected_components(csr_matrix(A), directed=False)
            conn = np.bincount(lab).max() / N
            deg = float(A.sum(axis=1).mean())
            el = D[A > 0]
            edge_len = float(el.mean()) if el.size else np.nan
            ev, V = eigh(A, check_finite=False)
            for lm in ONE_MINUS_M:
                S, L = lt.stationary_covariances(ev, V, 1 - lm)
                df = decompose(S, L, D, thresh=1.0)
                r = df[df.K == REF_K]
                a_g, n_g, drop_g = slope(df, "V_glob")
                rows.append({
                    "config": label, "seed": seed, "one_minus_m": lm,
                    "degree": deg, "conn": conn, "edge_len": edge_len,
                    "alpha": slope(df, "V")[0], "beta": slope(df, "tau")[0],
                    "alpha_loc": slope(df, "V_loc")[0],
                    "beta_loc": slope(df, "tau_loc")[0],
                    "alpha_glob": a_g, "beta_glob": slope(df, "tau_glob")[0],
                    "glob_share": float(r.V_glob.iloc[0] / r.V.iloc[0]),
                    "compact": float(r.compactness.iloc[0]),
                    "n_glob_dropped": drop_g,
                })
        print(f"  done {label}", flush=True)

    out = pd.DataFrame(rows)
    path = os.path.join(HERE, "sweep_data", "local_global_topologies.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)

    def ms(d, c, w=13):
        v = d[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "     --      "[:w]
        if v.size == 1:
            return f"{v[0]:.3f}".ljust(w)
        return f"{v.mean():.3f}+-{v.std(ddof=1):.3f}".ljust(w)

    g = out.groupby(["config", "one_minus_m"], sort=False)

    print("\n" + "=" * 118)
    print("PER CONFIGURATION AND m     n=512, 4 seeds, mean degree ~6 except hm")
    print("  'ok' = clusters still spatial (compact>0.3) AND long-range part")
    print("  measurable (glob share>0.02). Only 'ok' rows enter the summary.")
    print("=" * 118)
    hdr = (f"{'configuration':<22}{'1-m':<8}{'compact':<9}{'glob sh':<9}"
           f"{'alpha':<8}{'alpha_loc':<14}{'alpha_glob':<14}"
           f"{'beta':<8}{'beta_loc':<14}{'beta_glob':<14}{'ok'}")
    last = None
    for (label, lm), d in g:
        if label != last:
            print("-" * 118)
            print(hdr)
            last = label
        comp, gs = d.compact.mean(), d.glob_share.mean()
        ok = (comp > 0.3) and (gs > 0.02)
        print(f"{label:<22}{lm:<8.0e}{comp:<9.3f}{gs:<9.3f}"
              f"{d.alpha.mean():<8.3f}{ms(d,'alpha_loc'):<14}"
              f"{ms(d,'alpha_glob'):<14}{d.beta.mean():<8.3f}"
              f"{ms(d,'beta_loc'):<14}{ms(d,'beta_glob'):<14}"
              f"{'yes' if ok else 'no'}")

    # ------------------------------------------------------------------
    out["ok"] = (out.compact > 0.3) & (out.glob_share > 0.02)
    v = out[out.ok & np.isfinite(out.alpha_glob) & np.isfinite(out.alpha_loc)]
    print("\n" + "=" * 118)
    print("TEST 1  Is the long-range component more coherent than the local one?")
    print("        (alpha_glob > alpha_loc). The small-world result says yes.")
    print("=" * 118)
    print(f"  {'configuration':<22}{'rows':<7}{'alpha_loc':<14}{'alpha_glob':<14}"
          f"{'difference':<14}{'glob>loc in'}")
    for label, d in v.groupby("config", sort=False):
        diff = d.alpha_glob - d.alpha_loc
        print(f"  {label:<22}{len(d):<7}{ms(d,'alpha_loc'):<14}"
              f"{ms(d,'alpha_glob'):<14}"
              f"{diff.mean():+.3f}+-{diff.std(ddof=1):.3f}  "
              f"   {int((diff>0).sum())}/{len(d)}")
    allp = v.alpha_glob - v.alpha_loc
    print(f"\n  pooled: alpha_glob - alpha_loc = {allp.mean():+.3f} "
          f"+- {allp.std(ddof=1):.3f},  positive in "
          f"{int((allp>0).sum())}/{len(allp)} valid rows")

    print("\n" + "=" * 118)
    print("TEST 2  Does the LOCAL component carry the Munn pair, across families?")
    print("        Best (1-m) per configuration by distance of (alpha_loc, beta_loc)")
    print("        to (1.5, 0.2), range-weighted as everywhere else.")
    print("=" * 118)
    import spectral_classes as sc
    print(f"  {'configuration':<22}{'edge len':<11}{'best 1-m':<11}"
          f"{'alpha_loc':<12}{'beta_loc':<11}{'dist_loc':<11}"
          f"{'alpha tot':<12}{'beta tot':<11}{'dist tot'}")
    for label, d in out.groupby("config", sort=False):
        agg = d.groupby("one_minus_m").mean(numeric_only=True)
        dl = [float(sc.distance(r.alpha_loc, r.beta_loc)) for _, r in agg.iterrows()]
        i = int(np.nanargmin(dl))
        r = agg.iloc[i]
        dt = float(sc.distance(r.alpha, r.beta))
        print(f"  {label:<22}{r.edge_len:<11.1f}{agg.index[i]:<11.0e}"
              f"{r.alpha_loc:<12.3f}{r.beta_loc:<11.3f}{dl[i]:<11.3f}"
              f"{r.alpha:<12.3f}{r.beta:<11.3f}{dt:.3f}")

    print("\n" + "=" * 118)
    print("TEST 3  The ER control. With no spatial structure, is the split")
    print("        meaningless as it should be?")
    print("=" * 118)
    for label in ("er degree 6", "spatial_sf sigma=0", "ring k=4",
                  "geometric sigma=4"):
        d = out[out.config == label]
        print(f"  {label:<22} compactness {d.compact.mean():.3f}"
              f"   mean edge length {d.edge_len.mean():6.1f}"
              f"   global share {d.glob_share.mean():.3f}"
              f"   valid rows {int(d.ok.sum())}/{len(d)}")
    print("\n  A compactness near K/(n/2) ~ 0.25 would mean clusters are spread")
    print("  over the whole ring, i.e. 'ring distance' carries no information.")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
