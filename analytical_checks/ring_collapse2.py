"""Scaling collapse of the ring closed form -- integer cluster sizes only.

The Dirichlet kernel  D_K(j) = sin^2(pi j K/n)/sin^2(pi j/n)  is the squared
overlap of Fourier mode j with a contiguous block of K sites ONLY for integer K.
Evaluating it at fractional K is meaningless, so every K below is an integer and
every local slope is taken between two integer levels K -> 2K.

Continuum argument: s(q) = 1 - (5/4)q^2 + ... for k=4, so
    w(q) ~ 1/[(1-m^2) + (5/2)m^2 q^2]      correlation length xi ~ (1-m)^(-1/2)
and D_K ~ K^2 for qK<<1, ~ 1/sin^2(q/2) for qK>>1. Therefore the local slopes
should be functions of  x = K sqrt(1-m)  alone.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np

np.seterr(all="ignore")
N, KRING = 2 ** 20, 4

j = np.arange(N)
d = np.arange(1, KRING // 2 + 1)[:, None]
S_J = 2.0 * np.cos(2.0 * np.pi * j[None, :] * d / N).sum(axis=0) / KRING
SIN2 = np.sin(np.pi * j[1:] / N) ** 2


def V_tau(m, K):
    assert float(K).is_integer(), "K must be an integer"
    K = int(K)
    w = 1.0 / (1.0 - m ** 2 * S_J ** 2)
    t = 1.0 / (1.0 - m * S_J)
    D = np.empty(N)
    D[0] = float(K) ** 2
    D[1:] = np.sin(np.pi * j[1:] * K / N) ** 2 / SIN2
    num = float(np.sum(w * D))
    return num / N, float(np.sum(w * t * D)) / num


def slopes(m, K):
    v1, t1 = V_tau(m, K)
    v2, t2 = V_tau(m, 2 * K)
    return np.log2(v2 / v1), np.log2(t2 / t1)


LM = [1e-2, 1e-4, 1e-6, 1e-8]          # perfect squares: K = x/sqrt(1-m) stays integer
XS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]

print("=" * 98)
print("COLLAPSE  local alpha as a function of x = K sqrt(1-m)   [ring k=4, n = 2^20]")
print("=" * 98)
print(f"{'1-m':<10}{'K range':<20}" + "".join(f"{'x=' + str(x):>11}" for x in XS))
for lm in LM:
    m, r = 1 - lm, np.sqrt(lm)
    Ks = [int(round(x / r)) for x in XS]
    vals = [slopes(m, K)[0] for K in Ks]
    print(f"{lm:<10.0e}{str(Ks[0]) + '-' + str(Ks[-1]):<20}"
          + "".join(f"{v:>11.5f}" for v in vals))

print()
print(f"{'1-m':<10}{'K range':<20}" + "".join(f"{'x=' + str(x):>11}" for x in XS))
for lm in LM:
    m, r = 1 - lm, np.sqrt(lm)
    Ks = [int(round(x / r)) for x in XS]
    vals = [slopes(m, K)[1] for K in Ks]
    print(f"{lm:<10.0e}{str(Ks[0]) + '-' + str(Ks[-1]):<20}"
          + "".join(f"{v:>11.5f}" for v in vals))
print("  (local beta)")

print()
print("=" * 98)
print("THE SCALING CURVE  (1-m = 1e-6, so x is resolved finely)   alpha(x), beta(x)")
print("=" * 98)
lm = 1e-6
m, r = 1 - lm, np.sqrt(lm)
print(f"{'x':<9}{'K':<10}{'alpha':<11}{'beta':<11}")
rows = []
for K in sorted(set(int(round(v)) for v in np.exp(np.linspace(np.log(50), np.log(60000), 34)))):
    a, b = slopes(m, K)
    rows.append((K * r, K, a, b))
for x, K, a, b in rows:
    mark = ""
    if abs(a - 1.5) < 0.06:
        mark += "   <- alpha = 1.5"
    print(f"{x:<9.4f}{K:<10}{a:<11.5f}{b:<11.5f}{mark}")

a_arr = np.array([r_[2] for r_ in rows])
b_arr = np.array([r_[3] for r_ in rows])
x_arr = np.array([r_[0] for r_ in rows])
i = int(np.nanargmax(b_arr))
print()
print(f"  beta is maximal at x = {x_arr[i]:.4f} :  beta_max = {b_arr[i]:.5f}, "
      f"alpha there = {a_arr[i]:.5f}")
k15 = np.interp(1.5, a_arr[::-1], np.arange(len(a_arr))[::-1])
x15 = np.interp(k15, np.arange(len(x_arr)), x_arr)
b15 = np.interp(k15, np.arange(len(b_arr)), b_arr)
print(f"  alpha = 1.5 is reached at x = {x15:.4f} :  beta there = {b15:.5f}")
print(f"  distance of the closest point on the whole curve to (1.5, 0.2), "
      f"metric sqrt(da^2 + 4 db^2):")
dist = np.hypot(a_arr - 1.5, 2 * (b_arr - 0.2))
jm = int(np.nanargmin(dist))
print(f"      x = {x_arr[jm]:.4f}  ->  (alpha, beta) = ({a_arr[jm]:.4f}, "
      f"{b_arr[jm]:.4f}),  distance {dist[jm]:.4f}")

print()
print("=" * 98)
print("CONSEQUENCE  m and K are not independent knobs: only x = K sqrt(1-m) matters.")
print("  So a FITTED exponent over a window [K1, K2] depends on m only through")
print("  where that window sits on the single curve above.")
print("=" * 98)
print(f"{'1-m':<11}{'fit window K':<18}{'x window':<22}{'alpha fit':<12}{'beta fit'}")
for lm in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    m, r = 1 - lm, np.sqrt(lm)
    Ks = [2 ** p for p in range(1, 7)]           # K = 2..64, as fit_scaling uses
    lv = [V_tau(m, K) for K in Ks]
    lx = np.log(Ks)
    af = np.polyfit(lx, np.log([v for v, _ in lv]), 1)[0]
    bf = np.polyfit(lx, np.log([t for _, t in lv]), 1)[0]
    print(f"{lm:<11.0e}{'2 - 64':<18}"
          f"{f'{2*r:.4f} - {64*r:.4f}':<22}{af:<12.5f}{bf:.5f}")
