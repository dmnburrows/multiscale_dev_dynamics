"""Two crossover scales, two variables, one mechanism.

The variable search (variable_search.py) found the ring collapsing under
K M(m)/n and the gapped families under K P(m)^a with a ~ 1.5, where

    P(m) = max_k w_k / sum_k w_k        the top mode's share of the weight
    M(m) = (sum w)^2 / sum w^2          the number of amplified modes

Those look like different answers. They are the same statement, x = K / K*(m),
with K* set by whichever crossover the spectrum has:

  GAPLESS (ring, lattice, hierarchy). A band sits below the top eigenvalue, so
      the amplified modes form a continuum of width ~ sqrt(1-m) in q. Clusters
      are coherent until they exceed the correlation length:
          K*(m) = n / M(m),   x = K M/n  ~  K sqrt(1-m).

  GAPPED (ER, scale-free, and HM in practice at these n). One isolated top mode
      sits above a bulk. Its eigenvector is near-uniform, so it contributes
      w_1 K^2/n to the cluster variance, while the delocalised bulk contributes
      ~ K/n times its total weight. The quadratic term overtakes the linear one
      at
          K*(m) = (sum_bulk w) / w_1 = (1 - P)/P ~ 1/P,   x = K P.

So the general variable is x = K / K*(m), and which K* applies is decided by
one measurable property of the adjacency spectrum: whether the gap closes with
n. This module tests the gapped prediction directly.

    python two_regimes.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import universal_variable as uv
import variable_search as vs

np.seterr(all="ignore")

GAPPED = [("er6.2", "er", 6.2), ("sf2.5", "sf", 2.5), ("hm2.83", "hm", 2.83),
          ("sw0.1", "sw", 0.1)]
GAPLESS = [("ring4", "ring", 4)]
SEEDS = [1, 2, 3]
LMS = [10.0 ** -e for e in np.arange(1.0, 5.01, 0.25)]


def master_table(df, variable, label, n_bins=10):
    d = df[(df[variable] > 0) & np.isfinite(df.alpha) & np.isfinite(df.beta)]
    lo, hi = np.percentile(np.log(d[variable]), [3, 97])
    d = d[(np.log(d[variable]) >= lo) & (np.log(d[variable]) <= hi)]
    bins = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(np.log(d[variable]), bins) - 1, 0, n_bins - 1)
    print(f"\n  {label}:  master curve against {variable}")
    print(f"  {'x':<13}{'alpha':<10}{'sd':<9}{'beta':<10}{'sd':<9}{'n':<6}"
          f"{'spread of 1-m in bin'}")
    rows = []
    for b in range(n_bins):
        sel = d[idx == b]
        if len(sel) < 3:
            continue
        xc = np.exp((bins[b] + bins[b + 1]) / 2)
        rng = f"{sel.one_minus_m.min():.0e}-{sel.one_minus_m.max():.0e}"
        print(f"  {xc:<13.5f}{sel.alpha.mean():<10.4f}{sel.alpha.std():<9.4f}"
              f"{sel.beta.mean():<10.4f}{sel.beta.std():<9.4f}{len(sel):<6}{rng}")
        rows.append((xc, sel.alpha.mean(), sel.beta.mean()))
    if rows:
        i = int(np.argmax([r[2] for r in rows]))
        print(f"  beta peaks at x = {rows[i][0]:.4f}:  "
              f"(alpha, beta) = ({rows[i][1]:.4f}, {rows[i][2]:.4f})")
    return rows


def main():
    n = 2048
    print("=" * 104)
    print("1  THE GAPPED PREDICTION:  K*(m) = (1-P)/P,  so the crossover sits at K P ~ 1")
    print("=" * 104)
    print(f"{'family':<10}{'1-m':<10}{'P':<11}{'(1-P)/P':<12}"
          f"{'K where alpha=1.5':<20}{'ratio to (1-P)/P'}")
    for label, fam, par in GAPPED:
        A = lt.make_network(n, fam, par, 1)
        ev, V = eigh(A, check_finite=False)
        s = ev / np.max(np.abs(ev))
        for lm in (1e-2, 1e-3, 1e-4):
            m = 1 - lm
            w = 1.0 / (1.0 - m ** 2 * s ** 2)
            P = float(np.max(w) / np.sum(w))
            S, L = lt.stationary_covariances(ev, V, m)
            loc = uv.local_slopes(lt.coarse_grain(S, L, 4))
            a, K = loc.alpha.values, loc.K.values.astype(float)
            cross = np.nan
            for i in range(len(a) - 1):
                if (a[i] - 1.5) * (a[i + 1] - 1.5) < 0:
                    t = (1.5 - a[i]) / (a[i + 1] - a[i])
                    cross = np.exp(np.log(K[i]) + t * (np.log(K[i + 1]) - np.log(K[i])))
                    break
            kstar = (1 - P) / P
            print(f"{label:<10}{lm:<10.0e}{P:<11.5f}{kstar:<12.3f}"
                  f"{cross:<20.3f}{cross/kstar if np.isfinite(cross) else np.nan:.3f}")

    print()
    print("=" * 104)
    print("2  MASTER CURVES under the regime-appropriate variable")
    print("   (sd within a bin is the collapse: small sd across a wide span of 1-m")
    print("    means m acted only through x)")
    print("=" * 104)
    for label, fam, par in GAPLESS:
        df = vs.enriched_sweep(n, fam, par, SEEDS, LMS)
        df = df.assign(x=df.K * df.M / n)
        master_table(df, "x", f"{label} [gapless, x = K M/n]")
    for label, fam, par in GAPPED:
        df = vs.enriched_sweep(n, fam, par, SEEDS, LMS)
        df = df.assign(x=df.K * df.P)
        master_table(df, "x", f"{label} [gapped, x = K P]")

    print()
    print("=" * 104)
    print("3  COLLAPSE SCORES SIDE BY SIDE")
    print("=" * 104)
    print(f"{'family':<10}{'regime':<11}{'x = K M/n':<22}{'x = K P':<22}{'raw K'}")
    for label, fam, par in GAPLESS + GAPPED:
        df = vs.enriched_sweep(n, fam, par, SEEDS, LMS)
        qa1, qb1, _ = uv.collapse_quality(df.assign(x=df.K * df.M / n), "x")
        qa2, qb2, _ = uv.collapse_quality(df.assign(x=df.K * df.P), "x")
        qa3, qb3, _ = uv.collapse_quality(df, "K")
        reg = "gapless" if (label, fam, par) in GAPLESS else "gapped"
        print(f"{label:<10}{reg:<11}{f'{qa1:.3f} / {qb1:.3f}':<22}"
              f"{f'{qa2:.3f} / {qb2:.3f}':<22}{qa3:.3f} / {qb3:.3f}")
    print("  (alpha / beta)")


if __name__ == "__main__":
    main()
