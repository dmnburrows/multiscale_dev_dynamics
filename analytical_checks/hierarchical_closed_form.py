"""The DETERMINISTIC hierarchical network is exactly solvable too.

The ring closes because its adjacency commutes with a transitive symmetry group
(cyclic shifts). Any graph with a transitive automorphism group closes the same
way. The binary tree's automorphism group acts transitively on leaves, so if the
hierarchical wiring is DETERMINISTIC WEIGHTS rather than random edges,

    A[i,j] = Phi^(-level(i,j)),   level = highest differing bit of i xor j

then everything is known in closed form. Writing B_l for the indicator of
"i and j agree above bit l" (nested block matrices, all commuting),

    A = sum_{b=0}^{L-1} c_b (B_{b+1} - B_b),   c_b = Phi^(-b),   n = 2^L

The common eigenbasis is the Haar wavelet basis on the tree. Since
B_l v = 2^l v for l <= s-1 and 0 for l >= s for a wavelet at tree level s:

    uniform mode:      lambda_P = sum_{b=0}^{L-1} (2/Phi)^b
    wavelet level s:   lambda_s  = sum_{b=0}^{s-2} (2/Phi)^b - (2/Phi)^(s-1)
                       multiplicity 2^(L-s),  s = 1..L

Greedy ICG is FORCED onto the tree's own blocks (correlation depends only on
level and decreases with it), and a wavelet's squared overlap with a block of
size K, averaged over blocks, is K^2/n when its support exceeds K and 0 when it
does not. So with K = 2^t:

    V(K)   = (K^2/n) [ w_P + sum_{s=t+1}^{L} 2^(L-s) w_s ]
    tau(K) = [ w_P t_P + sum_{s=t+1}^{L} 2^(L-s) w_s t_s ] / [ w_P + sum ... ]

Elementary functions of m, Phi, n, K. Tested against linear_theory.py below.
"""
import sys
sys.path.insert(0, "/Users/k2585057/Library/CloudStorage/Dropbox/PhD/Analysis/"
                   "my_scripts/GitHub/multiscale_dev_dynamics")
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import linear_theory as lt

np.seterr(all="ignore")


def A_deterministic(n, phi):
    """A[i,j] = phi^(-level(i,j)); level as in lt.make_network's 'hm' branch."""
    labels = np.arange(n)
    xor = labels[:, None] ^ labels[None, :]
    level = np.floor(np.log2(np.maximum(xor, 1))).astype(int)
    A = phi ** (-level.astype(float))
    np.fill_diagonal(A, 0.0)
    return A


def spectrum_closed_form(n, phi):
    """Return (eigenvalues, multiplicities) from the formula, no linear algebra."""
    L = int(round(np.log2(n)))
    r = 2.0 / phi
    lam_P = sum(r ** b for b in range(L))
    out = [(lam_P, 1)]
    for s in range(1, L + 1):
        lam_s = sum(r ** b for b in range(s - 1)) - r ** (s - 1)
        out.append((lam_s, 2 ** (L - s)))
    return out


def levels_closed_form(n, phi, m, Ks):
    """V(K), tau(K) from the formula: no matrix, no eigendecomposition, no ICG."""
    L = int(round(np.log2(n)))
    spec = spectrum_closed_form(n, phi)
    rho = max(abs(v) for v, _ in spec)
    lam_P = spec[0][0]
    s_P = lam_P / rho
    wP, tP = 1 / (1 - m ** 2 * s_P ** 2), 1 / (1 - m * s_P)
    rows = []
    for K in Ks:
        t = int(round(np.log2(K)))
        num_w, num_wt = wP, wP * tP
        for s in range(t + 1, L + 1):
            lam_s = spec[s][0]
            ss = lam_s / rho
            w = 1 / (1 - m ** 2 * ss ** 2)
            mult = 2 ** (L - s)
            num_w += mult * w
            num_wt += mult * w / (1 - m * ss)
        V = (K ** 2 / n) * num_w
        rows.append({"K": K, "variance": V, "tau": num_wt / num_w})
    return pd.DataFrame(rows)


N = 512
PHIS = [1.5, 2.0, 2.83, 3.5, 5.0]

print("=" * 100)
print("STEP 1  closed-form spectrum vs eigh, on the deterministic hierarchical matrix")
print("=" * 100)
print(f"{'Phi':<8}{'n':<7}{'distinct eigs predicted':<27}{'max abs error vs eigh':<25}{'rho formula':<14}{'rho eigh'}")
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev = eigh(A, eigvals_only=True, check_finite=False)
    spec = spectrum_closed_form(N, phi)
    predicted = np.concatenate([np.full(mult, val) for val, mult in spec])
    err = np.abs(np.sort(predicted) - np.sort(ev)).max()
    print(f"{phi:<8}{N:<7}{len(spec):<27}{err:<25.3e}"
          f"{max(abs(v) for v,_ in spec):<14.6f}{np.abs(ev).max():.6f}")

print()
print("=" * 100)
print("STEP 2  is greedy ICG forced onto the tree blocks?  (cluster = contiguous")
print("        run of 2^t consecutive labels, which is what the tree's blocks are)")
print("=" * 100)
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev, V = eigh(A, check_finite=False)
    ok_all = []
    for m in (0.5, 0.99, 0.9999):
        S, L_ = lt.stationary_covariances(ev, V, m)
        memb, K, ok = np.arange(N), 1, True
        while len(S) >= 4:
            for c in range(memb.max() + 1):
                idx = np.flatnonzero(memb == c)
                if not (len(idx) == K and idx[0] % K == 0 and np.all(np.diff(idx) == 1)):
                    ok = False
            pairs = lt.greedy_pairs(S)
            a, b = pairs.T
            rl = np.empty(len(S), int); rl[a] = np.arange(len(pairs)); rl[b] = np.arange(len(pairs))
            memb = rl[memb]
            S, L_ = lt.pool(S, pairs), lt.pool(L_, pairs)
            K *= 2
        ok_all.append(ok)
    print(f"  Phi={phi:<6} tree-block partition at every level, m=0.5/0.99/0.9999 -> {ok_all}")

print()
print("=" * 100)
print("STEP 3  closed form vs linear_theory.py:  alpha and beta")
print("=" * 100)
print(f"{'Phi':<8}{'m':<10}{'alpha module':<15}{'alpha formula':<16}{'beta module':<14}"
      f"{'beta formula':<15}{'|d alpha|':<12}{'|d beta|'}")
worst = 0.0
for phi in PHIS:
    A = A_deterministic(N, phi)
    ev, V = eigh(A, check_finite=False)
    for m in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999):
        S, L_ = lt.stationary_covariances(ev, V, m)
        lv = lt.coarse_grain(S, L_, 4)
        cf = levels_closed_form(N, phi, m, lv.K.tolist())
        fm, fc = lt.fit_scaling(lv), lt.fit_scaling(cf)
        da, db = abs(fm["alpha"] - fc["alpha"]), abs(fm["beta"] - fc["beta"])
        worst = max(worst, da, db)
        print(f"{phi:<8}{m:<10}{fm['alpha']:<15.6f}{fc['alpha']:<16.6f}"
              f"{fm['beta']:<14.6f}{fc['beta']:<15.6f}{da:<12.2e}{db:.2e}")
print(f"\n  worst discrepancy anywhere: {worst:.3e}")

print()
print("=" * 100)
print("STEP 4  what the formula gives that the 512x512 matrix cannot: alpha, beta over")
print("        Phi x m, at n = 2^20, fitted over K = 2..2^16")
print("=" * 100)
n_big = 2 ** 20
Ks = [2 ** p for p in range(1, 17)]
print(f"{'Phi':<8}" + "".join(f"{'1-m=' + f'{v:.0e}':>22}" for v in (1e-2, 1e-4, 1e-6)))
print(f"{'':<8}" + "".join(f"{'(alpha, beta)':>22}" for _ in range(3)))
for phi in [1.5, 1.8, 2.0, 2.25, 2.5, 2.83, 3.2, 3.5, 4.0, 5.0]:
    cells = []
    for lm in (1e-2, 1e-4, 1e-6):
        f = lt.fit_scaling(levels_closed_form(n_big, phi, 1 - lm, Ks))
        cells.append("({:.3f}, {:.3f})".format(f["alpha"], f["beta"]))
    print(f"{phi:<8}" + "".join(f"{c:>22}" for c in cells))
