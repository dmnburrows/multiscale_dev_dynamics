"""alpha and beta in closed form for the hierarchy: elementary functions of (m, Phi, n, K).

This is the hierarchical analogue of the ring's Dirichlet-kernel solution, and
it is simpler, because the overlaps are a step function rather than a kernel.

SETUP.  r = 2/Phi, n = 2^L, cluster size K = 2^t. The Haar wavelet basis on the
tree diagonalises the deterministic adjacency A_ij = Phi^(-level(i,j)):

    uniform mode      lambda_P = sum_{b=0}^{L-1} r^b                 (mult 1)
    wavelet level s   lambda_s = sum_{b=0}^{s-2} r^b - r^(s-1)       (mult 2^(L-s))

and a wavelet's mean squared overlap with a cluster of K units is

    abar_s(K) = K^2/n   if 2^s > K,   else 0

i.e. a mode counts if and only if its support is wider than the cluster. So the
modal sums truncate at s = t, and with s_k = lambda_k/rho,

    w_k = 1/(1 - m^2 s_k^2),   u_k = 1/(1 - m s_k)

    S(t) = w_P        + sum_{s=t+1}^{L} 2^(L-s) w_s
    T(t) = w_P u_P    + sum_{s=t+1}^{L} 2^(L-s) w_s u_s

    V(K)   = (K^2/n) S(t)
    tau(K) = T(t) / S(t)

THE EXPONENTS.  Because V carries an exact K^2 prefactor, the local slopes are

    alpha(t) = 2 + log2[ S(t+1) / S(t) ]
    beta(t)  = log2[ T(t+1) S(t) / ( T(t) S(t+1) ) ]

Both are finite sums of elementary terms in m, Phi, n and K. No matrix, no
eigendecomposition, no coarse-graining run.

LARGE-L FORM.  For large L, 1 - s_s = (2 - r) r^(s-1) (checked to 1e-3 or better
for Phi >= 2.83), so writing delta = 1 - m and eps_s = (2-r) r^(s-1),

    w_s ~ 1 / ( 2 [delta + eps_s] ),     u_s ~ 1 / ( delta + eps_s )

    S(t) ~ w_P      + (n/2) sum_{s>t} 2^(-s) / (delta + eps_s)
    T(t) ~ w_P u_P  + (n/2) sum_{s>t} 2^(-s) / (delta + eps_s)^2

which makes the m-dependence transparent: each tree level contributes a
Lorentzian-like term whose width is set by delta, and the sum is cut off at the
correlation level s* where eps_s = delta, i.e.

    s* = ln[(2-r)/(r delta)] / ln(1/r),      K*(m) = 2^(s*) ~ delta^(-nu),
    nu = 1 / log2(Phi/2)

so alpha and beta depend on K and m only through x = K delta^nu. That collapse
was confirmed by a parameter-free test: alpha(t, delta) == alpha(t+1, delta r).

    python hier_closed_form.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analytical_theory as at
import linear_theory as lt


def spectrum(L, phi):
    """(s_P, [s_1..s_L], [multiplicities]) normalised by the spectral radius."""
    r = 2.0 / phi
    lam_P = sum(r ** b for b in range(L))
    lam = np.array([sum(r ** b for b in range(s - 1)) - r ** (s - 1)
                    for s in range(1, L + 1)])
    rho = max(abs(lam_P), np.abs(lam).max())
    mult = np.array([2.0 ** (L - s) for s in range(1, L + 1)])
    return lam_P / rho, lam / rho, mult


def S_T(t, L, phi, m):
    """The two modal sums, exactly."""
    sP, s, mult = spectrum(L, phi)
    wP = 1.0 / (1.0 - m ** 2 * sP ** 2)
    uP = 1.0 / (1.0 - m * sP)
    keep = np.arange(1, L + 1) > t                 # support 2^s > K = 2^t
    w = 1.0 / (1.0 - m ** 2 * s[keep] ** 2)
    u = 1.0 / (1.0 - m * s[keep])
    return (wP + np.sum(mult[keep] * w),
            wP * uP + np.sum(mult[keep] * w * u))


def alpha_beta(t, L, phi, m):
    """Closed-form local slopes at cluster size K = 2^t."""
    S0, T0 = S_T(t, L, phi, m)
    S1, T1 = S_T(t + 1, L, phi, m)
    return 2.0 + np.log2(S1 / S0), np.log2(T1 * S0 / (T0 * S1))


def alpha_beta_largeL(t, L, phi, m):
    """The transparent large-L form, showing the m-dependence explicitly."""
    r = 2.0 / phi
    delta = 1.0 - m
    s = np.arange(1, L + 1)
    eps = (2.0 - r) * r ** (s - 1.0)
    sP = 1.0
    wP = 1.0 / (1.0 - m ** 2 * sP ** 2)
    uP = 1.0 / (1.0 - m * sP)

    def sums(tt):
        keep = s > tt
        d = delta + eps[keep]
        n_over = 2.0 ** (L - s[keep])
        return (wP + np.sum(n_over / (2.0 * d)),
                wP * uP + np.sum(n_over / (2.0 * d ** 2)))

    S0, T0 = sums(t)
    S1, T1 = sums(t + 1)
    return 2.0 + np.log2(S1 / S0), np.log2(T1 * S0 / (T0 * S1))


def main():
    print("=" * 92)
    print("1  CLOSED FORM vs the simulation pipeline (linear_theory, n = 512)")
    print("=" * 92)
    n, L = 512, 9
    print(f"{'Phi':<7}{'1-m':<10}{'K':<7}{'alpha pipeline':<17}{'alpha formula':<16}"
          f"{'beta pipeline':<16}{'beta formula':<15}{'max err'}")
    worst = 0.0
    for phi in (2.83, 3.5, 5.0):
        for lm in (1e-2, 1e-4):
            from scipy.linalg import eigh
            A = at.hier_adjacency(n, phi)
            ev, V = eigh(A, check_finite=False)
            S, Lm = lt.stationary_covariances(ev, V, 1 - lm)
            lv = lt.coarse_grain(S, Lm, 4)
            Vv, Tt = lv.variance.values, lv.tau.values
            ap = np.log2(Vv[1:] / Vv[:-1])
            bp = np.log2(Tt[1:] / Tt[:-1])
            for t in (2, 4):
                af, bf = alpha_beta(t, L, phi, 1 - lm)
                e = max(abs(af - ap[t]), abs(bf - bp[t]))
                worst = max(worst, e)
                print(f"{phi:<7}{lm:<10.0e}{2**t:<7}{ap[t]:<17.8f}{af:<16.8f}"
                      f"{bp[t]:<16.8f}{bf:<15.8f}{e:.2e}")
    print(f"\n  worst discrepancy: {worst:.2e}")

    print()
    print("=" * 92)
    print("2  THE TRANSPARENT LARGE-L FORM vs the exact one (n = 2^20)")
    print("=" * 92)
    print(f"{'Phi':<7}{'1-m':<10}{'K':<9}{'alpha exact':<15}{'alpha large-L':<16}"
          f"{'beta exact':<14}{'beta large-L'}")
    for phi in (2.83, 3.5, 5.0):
        for lm in (1e-3, 1e-5):
            for t in (6, 10):
                ae, be = alpha_beta(t, 20, phi, 1 - lm)
                al, bl = alpha_beta_largeL(t, 20, phi, 1 - lm)
                print(f"{phi:<7}{lm:<10.0e}{2**t:<9}{ae:<15.6f}{al:<16.6f}"
                      f"{be:<14.6f}{bl:.6f}")

    print()
    print("=" * 92)
    print("3  THE COLLAPSE:  alpha(t, delta) == alpha(t+1, delta*r),  r = 2/Phi")
    print("   parameter-free; confirms x = K (1-m)^nu with nu = 1/log2(Phi/2)")
    print("=" * 92)
    print(f"{'Phi':<7}{'nu':<9}{'1-m':<10}{'t':<5}{'alpha(t,d)':<14}"
          f"{'alpha(t+1,dr)':<16}{'diff'}")
    for phi in (3.0, 4.0, 5.0):
        r = 2.0 / phi
        nu = 1.0 / np.log2(phi / 2.0)
        for lm in (1e-3,):
            for t in (6, 9):
                a0, _ = alpha_beta(t, 24, phi, 1 - lm)
                a1, _ = alpha_beta(t + 1, 24, phi, 1 - lm * r)
                print(f"{phi:<7}{nu:<9.4f}{lm:<10.0e}{t:<5}{a0:<14.6f}"
                      f"{a1:<16.6f}{a1-a0:+.2e}")


if __name__ == "__main__":
    main()
