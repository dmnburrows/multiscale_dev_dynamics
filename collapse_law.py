"""alpha(m) and beta(m): finding the single variable that m enters through.

GOAL: an explicit link between alpha, beta and m.

The claim this module tests, family by family, is that m never acts on its own.
It acts only by setting a correlation volume, and the exponents depend on the
cluster size only through

    x = K / Vc(m),        Vc(m) ~ (1-m)^(-nu)

so that alpha and beta are FUNCTIONS OF ONE VARIABLE, and

    alpha(m) = A(K (1-m)^nu),   beta(m) = B(K (1-m)^nu)

for whatever fitting window K you use. One exponent nu per architecture is then
the whole of the m-dependence, and nu maps onto the effective dimension of the
continuum law (continuum.py) by nu = d/2.


WHAT IS DERIVED HERE, NOT FITTED
--------------------------------
Lattices (ring, torus): s(q) = 1 - c q^2 gives xi ~ (1-m)^(-1/2) per axis, so
    Vc = xi^d  =>  nu = d/2.
Verified in continuum.py to 0.003 in alpha across five decades.

Deterministic hierarchy: with r = 2/Phi and large L, the wavelet spectrum gives
    1 - s_s = (2 - r) r^(s-1)                      (checked below)
so mode s is amplified while (2-r) r^(s-1) < 1-m, i.e. up to the level
    s* = ln((2-r)/r / (1-m)) / ln(1/r).
A cluster of K = 2^t units resolves levels up to t, so everything depends on
t - s*, which means the collapse variable is

    x = K (1-m)^(1/log2(Phi/2))      =>   nu_hier = 1 / log2(Phi/2)

and matching nu = d/2 gives the hierarchy an effective dimension

    d_eff = 2 / log2(Phi/2)

Phi = 2.83 -> d_eff = 4.00,  Phi = 4 -> 2.00,  Phi = 5 -> 1.51. At d_eff = 4 the
continuum law gives alpha = (d+2)/d = 1.5 exactly for small clusters, so the
Munn alpha falls out of Phi = 2.83 with nothing fitted.

THE EXACT SHIFT TEST (no interpolation, no fitting)
If alpha depends only on t - s*(1-m), then since s*((1-m) r) = s*(1-m) + 1,

    alpha_loc(t, 1-m)  ==  alpha_loc(t+1, (1-m) r)

for every t. Same for beta. That is a parameter-free prediction with no fitted
constants, and it either holds or it does not.

    python collapse_law.py hier        # the derivation and the shift test
    python collapse_law.py deff        # hierarchy vs the continuum law at d_eff
    python collapse_law.py all
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analytical_theory as at
import continuum as co

np.seterr(all="ignore")

PHIS = [2.25, 2.5, 2.83, 3.2, 3.5, 4.0, 5.0, 6.0]


def nu_hier(phi):
    """Predicted exponent: Vc ~ (1-m)^(-nu)."""
    return 1.0 / np.log2(phi / 2.0)


def d_eff_hier(phi):
    """Effective dimension implied by matching nu = d/2."""
    return 2.0 * nu_hier(phi)


def s_star(phi, one_minus_m):
    """The correlation LEVEL: the deepest wavelet level still amplified."""
    r = 2.0 / phi
    return np.log((2.0 - r) / r / one_minus_m) / np.log(1.0 / r)


def hier_local(n, phi, one_minus_m, ts):
    """Exact local slopes of the closed form, per K -> 2K doubling."""
    m = 1.0 - one_minus_m
    Ks = [2 ** t for t in range(min(ts), max(ts) + 2)]
    lv = at.hier_levels(n, phi, m, Ks)
    V, T = lv.variance.values, lv.tau.values
    out = {}
    for t in ts:
        i = t - min(ts)
        out[t] = (float(np.log2(V[i + 1] / V[i])),
                  float(np.log2(T[i + 1] / T[i])))
    return out


def check_spectrum(n=2 ** 16):
    print("=" * 100)
    print("1  is  1 - s_s = (2 - r) r^(s-1)  ?   (the step the whole derivation rests on)")
    print("=" * 100)
    print(f"{'Phi':<8}{'r=2/Phi':<11}{'s':<5}{'1-s_s exact':<17}{'(2-r)r^(s-1)':<17}{'ratio'}")
    L = int(round(np.log2(n)))
    for phi in (2.83, 4.0, 5.0):
        r = 2.0 / phi
        spec = at.hier_spectrum(n, phi)
        rho = max(abs(v) for v, _ in spec)
        for s in (2, 4, 6, 8):
            exact = 1.0 - spec[s][0] / rho
            pred = (2.0 - r) * r ** (s - 1)
            print(f"{phi:<8}{r:<11.5f}{s:<5}{exact:<17.10f}{pred:<17.10f}{exact/pred:.6f}")
    print("  (ratio -> 1 as L grows; the 1/(1-r^L) correction is the only gap)")


def check_shift():
    print()
    print("=" * 100)
    print("2  THE SHIFT TEST:  alpha_loc(t, delta) == alpha_loc(t+1, delta*r)?")
    print("   parameter-free. delta = 1-m, r = 2/Phi. n = 2^24.")
    print("=" * 100)
    n = 2 ** 24
    for phi in (2.83, 4.0, 5.0):
        r = 2.0 / phi
        print(f"\n  Phi={phi}  r={r:.5f}  predicted nu = {nu_hier(phi):.4f}  "
              f"d_eff = {d_eff_hier(phi):.3f}")
        print(f"  {'delta':<12}{'t':<5}{'alpha(t,d)':<14}{'alpha(t+1,d*r)':<17}"
              f"{'diff':<12}{'beta(t,d)':<13}{'beta(t+1,d*r)':<16}{'diff'}")
        for delta in (1e-2, 1e-4, 1e-6):
            base = hier_local(n, phi, delta, list(range(2, 16)))
            shifted = hier_local(n, phi, delta * r, list(range(2, 17)))
            for t in (4, 8, 12):
                a0, b0 = base[t]
                a1, b1 = shifted[t + 1]
                print(f"  {delta:<12.0e}{t:<5}{a0:<14.6f}{a1:<17.6f}{a1-a0:<+12.2e}"
                      f"{b0:<13.6f}{b1:<16.6f}{b1-b0:+.2e}")


def check_collapse():
    print()
    print("=" * 100)
    print("3  THE COLLAPSE: alpha and beta against x = K (1-m)^nu, nu = 1/log2(Phi/2)")
    print("   rows at matched x should agree across decades of 1-m")
    print("=" * 100)
    n = 2 ** 24
    for phi in (2.83, 4.0, 5.0):
        nu = nu_hier(phi)
        print(f"\n  Phi={phi}  nu={nu:.4f}")
        print(f"  {'1-m':<11}{'t range':<12}" + "".join(f"{f'x={v:g}':>12}"
                                                        for v in (1e-3, 1e-2, 1e-1, 1.0)))
        for delta in (1e-2, 1e-3, 1e-4, 1e-5):
            cells = []
            for xt in (1e-3, 1e-2, 1e-1, 1.0):
                t = int(round(np.log2(xt / delta ** nu)))
                t = max(2, min(t, 20))
                a, b = hier_local(n, phi, delta, [t])[t]
                cells.append(a)
            print(f"  {delta:<11.0e}{'':<12}" + "".join(f"{v:>12.5f}" for v in cells))
        print("  (local alpha; t rounded to the nearest integer doubling, so matching")
        print("   is exact only where x/delta^nu lands near a power of 2)")


def check_deff():
    print("=" * 100)
    print("4  DOES THE HIERARCHY SIT ON THE CONTINUUM CURVE AT d_eff = 2/log2(Phi/2)?")
    print("   hierarchy (exact closed form, n=2^24) vs continuum.alpha_beta(x, d_eff)")
    print("=" * 100)
    n = 2 ** 24
    for phi in PHIS:
        d = d_eff_hier(phi)
        nu = nu_hier(phi)
        print(f"\n  Phi={phi}   nu={nu:.4f}   d_eff={d:.3f}")
        print(f"  {'1-m':<11}{'t':<5}{'alpha hier':<14}{'alpha law':<13}"
              f"{'beta hier':<13}{'beta law':<12}{'d alpha':<11}{'d beta'}")
        for delta in (1e-3, 1e-5):
            loc = hier_local(n, phi, delta, list(range(2, 20)))
            for t in (6, 10, 14):
                if t not in loc:
                    continue
                a_h, b_h = loc[t]
                # the law's x is cluster SIDE / xi; the hierarchy's volume variable
                # is K (1-m)^nu, and side = volume^(1/d), so x = (K delta^nu)^(1/d)
                x = (2 ** t * delta ** nu) ** (1.0 / d)
                a_l, b_l = co.alpha_beta(x, d)
                print(f"  {delta:<11.0e}{t:<5}{a_h:<14.5f}{a_l:<13.5f}"
                      f"{b_h:<13.5f}{b_l:<12.5f}{a_h-a_l:<+11.4f}{b_h-b_l:+.4f}")


def check_peak():
    print()
    print("=" * 100)
    print("5  WHERE beta PEAKS ON THE HIERARCHY, and what the law says it should be")
    print("=" * 100)
    n = 2 ** 24
    print(f"  {'Phi':<8}{'d_eff':<9}{'1-m':<11}{'beta_max hier':<16}{'alpha there':<14}"
          f"{'beta_max law':<15}{'alpha law'}")
    for phi in PHIS:
        d = d_eff_hier(phi)
        law = co.beta_max(d)
        for delta in (1e-3, 1e-5):
            loc = hier_local(n, phi, delta, list(range(2, 21)))
            ts = sorted(loc)
            bs = [loc[t][1] for t in ts]
            i = int(np.nanargmax(bs))
            print(f"  {phi:<8}{d:<9.3f}{delta:<11.0e}{bs[i]:<16.5f}{loc[ts[i]][0]:<14.5f}"
                  f"{law['beta_max']:<15.5f}{law['alpha_at_peak']:.5f}")


SECTIONS = {"hier": lambda: (check_spectrum(), check_shift(), check_collapse()),
            "deff": lambda: (check_deff(), check_peak())}


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[name]()
