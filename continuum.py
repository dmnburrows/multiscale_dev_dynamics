"""The scaling law: alpha and beta are two moments of one measure.

Panel C of Fig 3. Continuum limit of the box closed form (analytical_theory.py,
universality.py). Near the top of the spectrum s(q) = 1 - c q^2 + O(q^4), so

    w(q) = 1/(A + B q^2),   A = 1 - m^2,   B = 2 m^2 c      (variance weight)
    t(q) = 1/(C + E q^2),   C = 1 - m,     E = m c          (timescale weight)

and the box overlap is the continuum Dirichlet kernel D_l(q) -> 4 sin^2(ql/2)/q^2.
Writing 1/(A + Bq^2) = int ds exp(-sA) prod_i exp(-s q_i^2 B) makes the
d-dimensional integral SEPARABLE, and the per-axis integral is

    (l/2pi) g(sB/l^2),   g(w) = int du [4 sin^2(u/2)/u^2] exp(-w u^2)

which has a closed form. With b = 1/(2 sqrt(w)),

    g(w) = 2 pi [ erf(b) - (1 - exp(-b^2)) / (b sqrt(pi)) ]

(g(0) = 2 pi, g(w) -> sqrt(pi/w) as w -> inf: both verified below.)

Collecting terms, with the single scaling variable

    x = l sqrt(A/B) = l sqrt((1-m^2) / (2 m^2 c))     [cluster side / xi]

everything reduces to moments of ONE probability measure on w > 0:

    pi(w)  propto  exp(-w x^2) g(w)^d

    V(l) propto l^(d+2) * Z(x,d),      Z = int dw exp(-w x^2) g(w)^d
    tau(l) propto l^2 * <w>_pi

because the double Laplace integral for tau collapses: the inner variables enter
only through their sum w, whose measure carries a factor w. Hence

    alpha(x,d) = 1 + [ 2 - 2 x^2 <w> ] / d
    beta(x,d)  =     [ 2 - 2 x^2 Var(w)/<w> ] / d

Two observables, two unknowns. Measured (alpha, beta) therefore determine BOTH

    d  -- the effective dimension of the mode spectrum, and
    x  -- the cluster size in units of the correlation length,

so the exponent pair is not a statement about m alone. This is why alpha = 1.5
has two very different explanations that beta tells apart: d = 4 at x -> 0
(alpha = (d+2)/d = 1.5 but beta -> 0), or d = 1 at x ~ 1.6 (alpha = 1.51 AND
beta = 0.21). Only the second reproduces the Munn pair.

Limits, all recovered below:
    x -> 0:   alpha -> min(2, (d+2)/d)
    x -> inf: alpha -> 1,  beta -> 0  (for every d)

beta's small-x limit is NOT zero in general, and this matters: the moments of
pi diverge as x -> 0 once d > 2 (g(w)^d ~ w^(-d/2), so int dw w g^d converges
at large w only for d > 4), leaving

    beta(x -> 0) -> 0      for d <~ 2        (interior maximum in x)
             -> finite     for d > 2         (maximum sits AT x = 0)

measured: 0.000 (d=1), 0.064 (d=2), 0.333 (d=3), 0.466 (d=4), 0.400 (d=5).
So an interior beta-maximum exists only in the low-dimensional regime, and only
there can the exponent pair sit on one.

What IS striking: for d <~ 2 the position of that interior maximum barely moves,

    d = 0.5 -> (1.5068, 0.2059)    d = 1.0 -> (1.5201, 0.2168)
    d = 1.5 -> (1.5402, 0.2302)    d = 2.0 -> (1.5706, 0.2478)

so alpha at the peak is pinned near 1.5 across a fourfold range of d. That is a
real near-universality, but a restricted one.

    python continuum.py verify     # g(w), and the law vs exact lattice sums
    python continuum.py law        # alpha, beta, beta_max and the implied d
    python continuum.py invert     # (alpha, beta) -> (d, x), with the Munn pair
    python continuum.py all
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analytical_theory as at
import universality as un

# log-spaced quadrature grid for the w integrals
_V = np.linspace(np.log(1e-12), np.log(1e14), 24001)
_W = np.exp(_V)


def g(w):
    """int du [4 sin^2(u/2)/u^2] exp(-w u^2), in closed form."""
    w = np.asarray(w, float)
    b = 1.0 / (2.0 * np.sqrt(w))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = 2.0 * np.pi * (erf(b) - (1.0 - np.exp(-b ** 2)) / (b * np.sqrt(np.pi)))
    # b -> inf (w -> 0): g -> 2 pi ; b -> 0 (w -> inf): g -> sqrt(pi/w)
    out = np.where(b > 1e8, 2.0 * np.pi, out)
    out = np.where(b < 1e-8, np.sqrt(np.pi / w), out)
    return out


_LOGG = np.log(g(_W))


def moments(x, d):
    """<w> and Var(w) under pi(w) propto exp(-w x^2) g(w)^d, plus log Z."""
    logf = -_W * x ** 2 + d * _LOGG + _V          # + _V from dw = w dv
    logf -= logf.max()
    f = np.exp(logf)
    Z = np.trapz(f, _V)
    m1 = np.trapz(f * _W, _V) / Z
    m2 = np.trapz(f * _W ** 2, _V) / Z
    return m1, max(m2 - m1 ** 2, 0.0)


def alpha_beta(x, d):
    """The scaling law."""
    m1, var = moments(x, d)
    alpha = 1.0 + (2.0 - 2.0 * x ** 2 * m1) / d
    beta = (2.0 - 2.0 * x ** 2 * var / m1) / d
    return alpha, beta


def curve(d, xs=None):
    if xs is None:
        xs = np.exp(np.linspace(np.log(1e-3), np.log(60.0), 400))
    ab = np.array([alpha_beta(x, d) for x in xs])
    return pd.DataFrame({"x": xs, "alpha": ab[:, 0], "beta": ab[:, 1]})


def beta_max(d):
    """The interior maximum of beta along the curve for this d."""
    c = curve(d)
    i = int(np.nanargmax(c.beta.values))
    return {"d": d, "x_peak": c.x[i], "beta_max": c.beta[i], "alpha_at_peak": c.alpha[i]}


def invert(alpha_target, beta_target, d0=1.5, x0=1.5):
    """Solve alpha(x,d) = alpha*, beta(x,d) = beta* for (d, x)."""
    def residual(p):
        d, x = np.exp(p)
        a, b = alpha_beta(x, d)
        return [a - alpha_target, b - beta_target]
    sol = least_squares(residual, np.log([d0, x0]), xtol=1e-12, ftol=1e-12)
    d, x = np.exp(sol.x)
    a, b = alpha_beta(x, d)
    return {"d": d, "x": x, "alpha": a, "beta": b,
            "residual": float(np.hypot(a - alpha_target, b - beta_target))}


# =============================================================================
# verification
# =============================================================================

def verify():
    print("=" * 100)
    print("1  the closed form for g(w)")
    print("=" * 100)
    U = np.linspace(-600, 600, 1200001)
    kern = np.where(np.abs(U) < 1e-9, 1.0, 4 * np.sin(U / 2) ** 2 / U ** 2)
    print(f"  {'w':<12}{'g closed form':<18}{'g quadrature':<18}{'rel err':<12}{'asymptote'}")
    for w in (1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3):
        num = np.trapz(kern * np.exp(-w * U ** 2), U)
        cf = float(g(w))
        asym = 2 * np.pi if w < 1e-3 else np.sqrt(np.pi / w)
        print(f"  {w:<12.0e}{cf:<18.10f}{num:<18.10f}{abs(cf/num-1):<12.2e}{asym:.6f}")

    print()
    print("=" * 100)
    print("2  THE LAW vs EXACT LATTICE SUMS, in the bulk regime 1 << l << xi << L")
    print("   (my earlier lattice test used xi > L, which forces alpha = 2 by finite")
    print("    size and tests nothing -- these rows keep xi well inside L)")
    print("=" * 100)
    # c from s(q) = 1 - c q^2: ring k=4 -> c = 5/4 ; per-axis k=4 on a d-torus the
    # same c applies per axis, and the isotropic q^2 coefficient is c/d.
    for L, d, k, lm in ((2 ** 20, 1, 4, 1e-6), (1024, 2, 4, 1e-4), (128, 3, 4, 1e-2)):
        m = 1 - lm
        c = 1.25 / d
        A, B = 1 - m ** 2, 2 * m ** 2 * c
        s = at.ring_spectrum(L, k) if d == 1 else un.torus_spectrum(L, d, k)
        xi = np.sqrt(B / A)
        print(f"\n  d={d} L={L} 1-m={lm:.0e}   xi = {xi:.2f}   L/xi = {L/xi:.1f}")
        print(f"  {'side':<8}{'x=side/xi':<13}{'alpha lattice':<16}{'alpha law':<13}"
              f"{'beta lattice':<15}{'beta law'}")
        for p in range(1, int(np.log2(L)) - 1):
            side, t = 2 ** p, d * p
            if side > L / 4:
                break
            if d == 1:
                lv = at.ring_levels(L, k, m, [2 ** t, 2 ** (t + d)], s=s)
            else:
                lv = un.torus_levels(L, d, k, m, [t, t + d], s=s)
            a_lat = float(np.log2(lv.variance[1] / lv.variance[0]) / d)
            b_lat = float(np.log2(lv.tau[1] / lv.tau[0]) / d)
            x = side * np.sqrt(A / B)
            a_law, b_law = alpha_beta(x * np.sqrt(2.0), d)   # midpoint of the doubling
            print(f"  {side:<8}{x:<13.5f}{a_lat:<16.5f}{a_law:<13.5f}"
                  f"{b_lat:<15.5f}{b_law:.5f}")

    print()
    print("=" * 100)
    print("3  limits")
    print("=" * 100)
    print(f"  {'d':<7}{'alpha(x->0)':<15}{'min(2,(d+2)/d)':<18}{'alpha(x->inf)':<16}"
          f"{'beta at both ends'}")
    for d in (1, 2, 3, 4, 5):
        a0, b0 = alpha_beta(1e-3, d)
        a1, b1 = alpha_beta(50.0, d)
        print(f"  {d:<7}{a0:<15.5f}{min(2.0,(d+2)/d):<18.4f}{a1:<16.5f}"
              f"{b0:.2e} / {b1:.2e}")


# =============================================================================
# the law
# =============================================================================

def law():
    print("=" * 100)
    print("1  alpha and beta along the curve, by effective dimension")
    print("=" * 100)
    xs = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for name, fn in (("alpha", 0), ("beta", 1)):
        print(f"\n  {name}")
        print(f"  {'d':<7}" + "".join(f"{'x='+f'{x:g}':>11}" for x in xs))
        for d in (0.5, 1, 1.5, 2, 3, 4, 6):
            vals = [alpha_beta(x, d)[fn] for x in xs]
            print(f"  {d:<7}" + "".join(f"{v:>11.5f}" for v in vals))

    print()
    print("=" * 100)
    print("2  the beta MAXIMUM, dimension by dimension")
    print("=" * 100)
    print(f"  {'d':<8}{'x at peak':<13}{'beta_max':<12}{'alpha at peak':<16}"
          f"{'distance to (1.5,0.2)'}")
    rows = []
    for d in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        r = beta_max(d)
        r["distance"] = float(at.distance_to_target(r["alpha_at_peak"], r["beta_max"]))
        rows.append(r)
        print(f"  {d:<8}{r['x_peak']:<13.4f}{r['beta_max']:<12.5f}"
              f"{r['alpha_at_peak']:<16.5f}{r['distance']:.4f}")
    df = pd.DataFrame(rows)
    interior = df[df.x_peak > 0.01]
    print()
    print("  x at peak = 0.0010 is the grid edge: for d > ~2.5 beta is maximal AT x = 0,")
    print("  so there is no interior maximum to sit on.")
    print(f"  Over the interior-maximum range (d = {interior.d.min()} .. {interior.d.max()}):")
    print(f"    beta_max        {interior.beta_max.min():.4f} .. {interior.beta_max.max():.4f}")
    print(f"    alpha at peak   {interior.alpha_at_peak.min():.4f} .. "
          f"{interior.alpha_at_peak.max():.4f}   <- pinned near 1.5")
    print("  So beta_max is set by d, not universal; but alpha at the peak barely moves.")

    print()
    print("=" * 100)
    print("3  which d can reach beta = 0.2 at all?")
    print("=" * 100)
    print(f"  {'d':<8}{'beta_max':<12}{'reaches 0.2?':<15}{'alpha where beta=0.2'}")
    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        c = curve(d)
        ok = c.beta.max() >= 0.2
        where = ""
        if ok:
            i = int(np.nanargmax(c.beta.values))
            left = c.iloc[:i + 1]
            right = c.iloc[i:]
            a_l = float(np.interp(0.2, left.beta.values, left.alpha.values))
            a_r = float(np.interp(0.2, right.beta.values[::-1], right.alpha.values[::-1]))
            where = f"{a_l:.4f} (small x) and {a_r:.4f} (large x)"
        print(f"  {d:<8}{c.beta.max():<12.5f}{str(ok):<15}{where}")


def invert_section():
    print("=" * 100)
    print("1  INVERSION: (alpha, beta) -> (d, x)")
    print("=" * 100)
    print(f"  {'target (alpha, beta)':<24}{'d':<11}{'x':<11}{'recovered':<24}{'residual'}")
    targets = [(1.5, 0.2), (1.5, 0.15), (1.5, 0.25), (1.45, 0.2), (1.55, 0.2),
               (1.6, 0.2), (1.4, 0.2), (1.5, 0.1), (1.3, 0.3)]
    for a, b in targets:
        for d0, x0 in ((1.0, 1.5), (2.0, 1.0), (0.7, 2.0)):
            r = invert(a, b, d0, x0)
            if r["residual"] < 1e-6:
                break
        got = "({:.5f}, {:.5f})".format(r["alpha"], r["beta"])
        print(f"  {f'({a}, {b})':<24}{r['d']:<11.4f}{r['x']:<11.4f}{got:<24}"
              f"{r['residual']:.2e}")

    print()
    print("=" * 100)
    print("2  SENSITIVITY of the inferred (d, x) to the measured exponents")
    print("=" * 100)
    base = invert(1.5, 0.2)
    print(f"  Munn pair (1.50, 0.20)  ->  d = {base['d']:.3f},  x = {base['x']:.3f}")
    print(f"  {'perturbation':<26}{'d':<11}{'x':<11}{'d d':<11}{'d x'}")
    for da, db in ((0.05, 0), (-0.05, 0), (0, 0.02), (0, -0.02), (0.05, 0.02),
                   (-0.05, -0.02)):
        r = invert(1.5 + da, 0.2 + db)
        print(f"  {f'alpha{da:+.2f}, beta{db:+.2f}':<26}{r['d']:<11.4f}{r['x']:<11.4f}"
              f"{r['d']-base['d']:<+11.4f}{r['x']-base['x']:+.4f}")

    print()
    print("=" * 100)
    print("3  WHAT IT PREDICTS: fix (d, x) from the data, then change the window")
    print("   Subsampling or re-windowing multiplies the cluster side by a factor f,")
    print("   which moves x -> f x along the SAME curve at the SAME d. No refitting.")
    print("=" * 100)
    d_star, x_star = base["d"], base["x"]
    print(f"  d = {d_star:.3f} held fixed")
    print(f"  {'f':<10}{'x = f x*':<13}{'alpha predicted':<18}{'beta predicted'}")
    for f in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        a, b = alpha_beta(f * x_star, d_star)
        print(f"  {f:<10}{f*x_star:<13.4f}{a:<18.5f}{b:.5f}")
    print()
    print("  Falsifiable: halving the recording's spatial extent must move the pair")
    print("  ALONG this curve, not off it. One ICG run on subsampled data tests it.")


SECTIONS = {"verify": verify, "law": law, "invert": invert_section}


if __name__ == "__main__":
    np.seterr(all="ignore")
    args = sys.argv[1:] or ["all"]
    names = list(SECTIONS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}; choose from {list(SECTIONS)}")
        if i:
            print("\n")
        SECTIONS[name]()
