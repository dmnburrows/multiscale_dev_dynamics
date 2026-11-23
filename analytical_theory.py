"""Analytical layer on top of linear_theory.py: where alpha and beta come from.

linear_theory.py is the canonical pipeline and is imported UNMODIFIED. Nothing
here replaces its numerics; everything here is checked against them.

The model is  x(t+1) = W x(t) + xi(t),  W = (m/rho(A)) A,  Q = I,  A symmetric.
With eigenpairs (lambda_k, v_k) of W, lambda_k = m s_k and s_k = eig_k/rho(A):

    Sigma = sum_k w_k v_k v_k^T,          w_k   = 1/(1 - lambda_k^2)
    Lambda = sum_k w_k t_k v_k v_k^T,     t_k   = 1/(1 - lambda_k)

(Lambda is the resolvent sum whose trace ratio gives the ICG timescale.)


PART 1 -- THE MODAL IDENTITY (exact, machine precision)
-------------------------------------------------------
Let abar_k(K) = < (v_k . 1_S)^2 > be mode k's squared overlap with a cluster,
averaged over the clusters at ICG level K. Then exactly

    V(K)   = sum_k w_k abar_k(K)
    tau(K) = sum_k w_k t_k abar_k(K) / sum_k w_k abar_k(K)

and alpha, beta are the log-log slopes of these. Verified to 8.9e-16.

Topology sets {s_k} AND {v_k}; m supplies only the scalar in lambda_k = m s_k.
abar_k is weakly m-dependent too, because the greedy partition is chosen from
Sigma(m) -- freezing partitions across m shifts alpha by <=0.038, beta <=0.041.


PART 2 -- THE MECHANISM (exact local-slope identities)
------------------------------------------------------
With pi_k(K) = w_k abar_k(K) / sum_j w_j abar_j(K)   (mode k's share of V(K))
and gamma_k  = d log abar_k / d log K                (topology, m-free):

    alpha_loc = log2 sum_k pi_k 2^gamma_k                            EXACT
    beta_loc  = log2 [<t 2^gamma>_pi / <t>_pi] - log2 <2^gamma>_pi   EXACT

Linearised (INTUITION ONLY -- off by up to 0.22 in alpha, and badly wrong on
the ring): alpha ~ <gamma>_pi, beta ~ Cov_pi(gamma, t)/<t>_pi.

Reading: alpha wants pi concentrated on high-gamma modes, which is monotone in
m and saturates at gamma_max ~ 2. beta wants gamma and t to DISAGREE across
modes, so beta dies identically when one mode owns V(K). The same dial is asked
for concentration and for diversity, which is why the joint fit to (1.5, 0.2)
has an interior optimum -- an argument made wholly inside m < 1.


PART 3 -- THE RING CLOSES IN CLOSED FORM
----------------------------------------
Circulant adjacency: s_j = (2/k) sum_{d=1..k/2} cos(2 pi j d/n); eigenvectors
are Fourier modes; greedy ICG gives contiguous blocks, whose squared overlap
with Fourier mode j is the Dirichlet kernel, independent of the block's start:

    abar_j(K) = D_K(j)/n,  D_K(j) = sin^2(pi j K/n)/sin^2(pi j/n),  D_K(0) = K^2

Exact to 1e-16 -- 3e-14 at k = 4 and 6 (k = 2 fails: ties break contiguity).
Continuum limit gives xi ~ (1-m)^(-1/2) and a one-variable collapse: alpha and
beta are functions of x = K sqrt(1-m) ALONE (verified across six decades).
beta has a single maximum, beta_max = 0.21324 at x* = 1.556 where alpha = 1.527;
the closest approach of the entire curve to (1.5, 0.2) is 0.0377 and lands on
that maximum. Fixed-window fitting as m -> 1 drives (alpha, beta) -> (2, 0).


PART 4 -- THE DETERMINISTIC HIERARCHY CLOSES TOO
------------------------------------------------
Closed forms need three things: the spectrum, the eigenvectors, and -- the
binding one -- the PARTITION. A transitive automorphism group supplies all
three (every node equivalent => greedy's choice forced by symmetry; every
K-block equivalent => the cluster average is one term). Random graphs fail on
all three independently. So vertex-transitive graphs are solvable: rings,
lattices, tori, hypercubes, complete graphs, circulants, Cayley graphs.

The binary tree's automorphisms act transitively on leaves, so swapping Munn's
stochastic wiring P_l = Phi^(-l) for deterministic WEIGHTS A_ij = Phi^(-l(i,j))
puts the hierarchy in that class. Haar wavelets on the tree are the eigenbasis;
with r = 2/Phi, n = 2^L:

    lambda_P = sum_{b=0}^{L-1} r^b                                  (mult 1)
    lambda_s = sum_{b=0}^{s-2} r^b - r^(s-1)                        (mult 2^(L-s))
    abar_s(K) = K^2/n  if 2^s > K  else  0

Exact to 1e-14 -- 1e-16, under two genuine conditions: Phi >= 2.83, and K = 2
must be dropped. K = 2 fails because the level-1 wavelet has eigenvalue exactly
-1 for every Phi, so the most strongly wired pair is the LEAST correlated close
pair -- ICG's first merge crosses the first module boundary -- and all level-1
partners are exactly tied, so the K = 2 pairing flips under a 1e-14 perturbation
(6/6 trials). The first coarse-graining level is decided by rounding noise.

Best in the Phi x m grid at n = 2^20: Phi = 4.0, 1-m = 1e-3 -> (1.4595, 0.2153),
distance 0.0507, interior in both Phi and m.


Run the verifications:

    python analytical_theory.py all
    python analytical_theory.py modal mechanism ring collapse hier hier-diag grid

(Harmless spurious BLAS FPE flags on this Mac are silenced with np.seterr.)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt


# =============================================================================
# PART 1  the modal identity
# =============================================================================

def cascade(S, L, min_clusters=4):
    """lt.coarse_grain, but also returning the unit->cluster membership per level.

    Bit-for-bit the cascade run_sweep uses: lt.greedy_pairs and lt.pool do the
    work, membership tracking is only carried alongside.
    """
    memb = np.arange(len(S))
    levels, K = [], 1
    while len(S) >= min_clusters:
        levels.append(dict(K=K, clusters=len(S),
                           variance=np.trace(S) / len(S),
                           tau=np.trace(L) / np.trace(S),
                           memb=memb.copy()))
        if len(S) == 1:
            break
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs))
        relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return levels


def overlaps(memb, eigenvectors):
    """abar_k = < (v_k . 1_S)^2 > averaged over the clusters at this level."""
    n_clusters = memb.max() + 1
    indicator = np.zeros((len(memb), n_clusters))
    indicator[np.arange(len(memb)), memb] = 1.0
    projected = eigenvectors.T @ indicator          # (modes, clusters)
    return np.mean(projected ** 2, axis=1)


def modal_weights(eigenvalues, m):
    """w_k = 1/(1 - lambda_k^2) and t_k = 1/(1 - lambda_k), lambda_k = m s_k."""
    lam = m * eigenvalues / np.max(np.abs(eigenvalues))
    w = 1.0 / ((1.0 - lam) * (1.0 + lam))
    return w, 1.0 / (1.0 - lam)


def modal_levels(eigenvalues, eigenvectors, m, min_clusters=4):
    """V(K), tau(K) rebuilt from the modal identity instead of from traces."""
    S, L = lt.stationary_covariances(eigenvalues, eigenvectors, m)
    levels = cascade(S, L, min_clusters)
    w, t = modal_weights(eigenvalues, m)
    rows = []
    for d in levels:
        ab = overlaps(d["memb"], eigenvectors)
        num = float(np.sum(w * ab))
        rows.append({"K": d["K"], "variance": num,
                     "tau": float(np.sum(w * t * ab)) / num})
    return pd.DataFrame(rows), levels


def slope(K, y, drop_first=1, drop_last=1):
    """Log-log slope with the same window convention as lt.fit_scaling."""
    end = len(K) - drop_last if drop_last else len(K)
    x = np.log(np.asarray(K[drop_first:end], float))
    yy = np.log(np.asarray(y[drop_first:end], float))
    return float(np.polyfit(x, yy, 1)[0])


# =============================================================================
# PART 2  the mechanism: pi, gamma, and the exact local slopes
# =============================================================================

def mode_statistics(eigenvalues, eigenvectors, m, min_clusters=4):
    """Per-level pi, gamma, exact and linearised local slopes, N_eff.

    Returns one row per K -> 2K doubling:
      alpha_true/beta_true   from the cascade's own traces
      alpha_exact/beta_exact from the log-sum-exp identities (should match)
      alpha_lin/beta_lin     the linearised forms (intuition only)
      perron                 pi of the top mode
      n_eff                  1/sum pi^2, effective number of amplified modes
      cov_over_tau           Cov_pi(gamma, t)/<t>_pi
      corr_gamma_tau         pi-weighted correlation of gamma and t
    """
    S, L = lt.stationary_covariances(eigenvalues, eigenvectors, m)
    levels = cascade(S, L, min_clusters)
    w, t = modal_weights(eigenvalues, m)
    ab = [overlaps(d["memb"], eigenvectors) for d in levels]
    rows = []
    for i in range(len(levels) - 1):
        g = np.log2(ab[i + 1] / ab[i])
        pi = w * ab[i] / np.sum(w * ab[i])
        mt = float(np.sum(pi * t))
        mg = float(np.sum(pi * g))
        sd_t = np.sqrt(max(float(np.sum(pi * t ** 2)) - mt ** 2, 0.0))
        sd_g = np.sqrt(max(float(np.sum(pi * g ** 2)) - mg ** 2, 0.0))
        cov = float(np.sum(pi * g * t)) - mg * mt
        rows.append({
            "K": levels[i]["K"],
            "alpha_true": np.log2(levels[i + 1]["variance"] / levels[i]["variance"]),
            "alpha_exact": float(np.log2(np.sum(pi * 2.0 ** g))),
            "alpha_lin": mg,
            "beta_true": np.log2(levels[i + 1]["tau"] / levels[i]["tau"]),
            "beta_exact": float(np.log2(np.sum(pi * t * 2.0 ** g) / np.sum(pi * t))
                                - np.log2(np.sum(pi * 2.0 ** g))),
            "beta_lin": cov / mt,
            "perron": float(pi[-1]),
            "n_eff": float(1.0 / np.sum(pi ** 2)),
            "sd_tau_rel": sd_t / mt,
            "cov_over_tau": cov / mt,
            "corr_gamma_tau": cov / (sd_g * sd_t) if sd_g * sd_t > 0 else np.nan,
            "gamma_max": float(g.max()),
            "gamma_perron": float(g[-1]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# PART 3  the ring closed form, and its scaling collapse
# =============================================================================

def ring_spectrum(n, k):
    """s_j = lambda_j / rho for the ring; rho = k, so no eigendecomposition."""
    j = np.arange(n)
    d = np.arange(1, k // 2 + 1)[:, None]
    lam = 2.0 * np.cos(2.0 * np.pi * j[None, :] * d / n).sum(axis=0)
    return lam / k


def dirichlet(n, K):
    """D_K(j) = sin^2(pi j K/n)/sin^2(pi j/n), D_K(0) = K^2.

    This is the squared overlap of Fourier mode j with a contiguous block of K
    sites ONLY for integer K, so fractional K is rejected -- evaluating it off
    the integers is meaningless and silently produces nonsense slopes.
    """
    if not float(K).is_integer():
        raise ValueError("The Dirichlet kernel is a block overlap only at integer K.")
    K = int(K)
    j = np.arange(n)
    out = np.empty(n)
    out[0] = float(K) ** 2
    out[1:] = np.sin(np.pi * j[1:] * K / n) ** 2 / np.sin(np.pi * j[1:] / n) ** 2
    return out


def ring_levels(n, k, m, Ks, s=None):
    """V(K), tau(K) from the formula: no matrix, no eigendecomposition, no ICG."""
    if s is None:
        s = ring_spectrum(n, k)
    w = 1.0 / (1.0 - m ** 2 * s ** 2)
    t = 1.0 / (1.0 - m * s)
    rows = []
    for K in Ks:
        D = dirichlet(n, K)
        V = float(np.sum(w * D)) / n
        rows.append({"K": K, "variance": V,
                     "tau": float(np.sum(w * t * D)) / n / V})
    return pd.DataFrame(rows)


def ring_local_slopes(n, k, m, K, s=None):
    """(alpha_loc, beta_loc) at K by the K -> 2K doubling, both integers."""
    lv = ring_levels(n, k, m, [int(K), 2 * int(K)], s)
    return (float(np.log2(lv.variance[1] / lv.variance[0])),
            float(np.log2(lv.tau[1] / lv.tau[0])))


def ring_collapse_curve(n, k, one_minus_m, Ks):
    """alpha, beta against the scaling variable x = K sqrt(1-m)."""
    m, r = 1.0 - one_minus_m, np.sqrt(one_minus_m)
    s = ring_spectrum(n, k)
    rows = []
    for K in Ks:
        a, b = ring_local_slopes(n, k, m, K, s)
        rows.append({"x": K * r, "K": int(K), "alpha": a, "beta": b})
    return pd.DataFrame(rows)


def partition_is_contiguous(n, k, m=0.99, seed=7):
    """Does lt.greedy_pairs give contiguous blocks (so the Dirichlet kernel applies)?"""
    A = lt.make_network(n, "ring", k, seed)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, m)
    memb, K, report = np.arange(n), 1, []
    while len(S) >= 4:
        ok = True
        for c in range(memb.max() + 1):
            idx = np.flatnonzero(memb == c)
            d = np.diff(idx)                      # contiguous, allowing wraparound
            if not (np.all(d == 1) or (np.sum(d != 1) == 1 and idx[0] == 0)):
                ok = False
        report.append((K, memb.max() + 1, ok))
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs))
        relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return report


# =============================================================================
# PART 4  the deterministic hierarchy
# =============================================================================

def hier_adjacency(n, phi):
    """A[i,j] = phi^(-level(i,j)); level as in lt.make_network's 'hm' branch."""
    labels = np.arange(n)
    xor = labels[:, None] ^ labels[None, :]
    level = np.floor(np.log2(np.maximum(xor, 1))).astype(int)
    A = phi ** (-level.astype(float))
    np.fill_diagonal(A, 0.0)
    return A


def hier_spectrum(n, phi):
    """[(eigenvalue, multiplicity)]: index 0 is the uniform mode, s is wavelet level s."""
    L = int(round(np.log2(n)))
    r = 2.0 / phi
    lam_P = sum(r ** b for b in range(L))
    out = [(lam_P, 1)]
    for s in range(1, L + 1):
        lam_s = sum(r ** b for b in range(s - 1)) - r ** (s - 1)
        out.append((lam_s, 2 ** (L - s)))
    return out


def hier_levels(n, phi, m, Ks):
    """V(K), tau(K) from the formula: no matrix, no eigendecomposition, no ICG.

    Valid for phi >= 2.83 and K >= 4 (see the module docstring on K = 2).
    """
    L = int(round(np.log2(n)))
    spec = hier_spectrum(n, phi)
    rho = max(abs(v) for v, _ in spec)
    s_P = spec[0][0] / rho
    wP, tP = 1 / (1 - m ** 2 * s_P ** 2), 1 / (1 - m * s_P)
    rows = []
    for K in Ks:
        t = int(round(np.log2(K)))
        num_w, num_wt = wP, wP * tP
        for s in range(t + 1, L + 1):             # only modes whose support exceeds K
            ss = spec[s][0] / rho
            w = 1 / (1 - m ** 2 * ss ** 2)
            mult = 2 ** (L - s)
            num_w += mult * w
            num_wt += mult * w / (1 - m * ss)
        rows.append({"K": K, "variance": (K ** 2 / n) * num_w,
                     "tau": num_wt / num_w})
    return pd.DataFrame(rows)


def hier_tree_blocks(n, phi, m, from_K=4):
    """Are the ICG clusters the tree's own blocks (runs of 2^t aligned labels)?"""
    A = hier_adjacency(n, phi)
    ev, V = eigh(A, check_finite=False)
    S, L = lt.stationary_covariances(ev, V, m)
    memb, K, ok = np.arange(n), 1, True
    while len(S) >= 4:
        if K >= from_K:
            for c in range(memb.max() + 1):
                idx = np.flatnonzero(memb == c)
                if not (len(idx) == K and idx[0] % K == 0
                        and np.all(np.diff(idx) == 1)):
                    ok = False
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs))
        relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2
    return ok


def hier_tie_instability(n, phi, m=0.99, eps=1e-14, trials=6):
    """Is the K = 2 pairing stable? Perturb Sigma by eps and see if it flips."""
    A = hier_adjacency(n, phi)
    ev, V = eigh(A, check_finite=False)
    S, _ = lt.stationary_covariances(ev, V, m)
    base = set(map(tuple, map(sorted, lt.greedy_pairs(S))))
    same = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        P = S * (1.0 + eps * rng.standard_normal(S.shape))
        P = (P + P.T) / 2
        same.append(base == set(map(tuple, map(sorted, lt.greedy_pairs(P)))))
    return same, lt.greedy_pairs(S)


def hier_grid(n, phis, one_minus_m, Ks, target=(1.5, 0.2), beta_weight=2):
    """alpha, beta and distance to the Munn exponents over Phi x m, formula only."""
    rows = []
    for phi in phis:
        for lm in one_minus_m:
            f = lt.fit_scaling(hier_levels(n, phi, 1 - lm, Ks))
            d = np.hypot(f["alpha"] - target[0], beta_weight * (f["beta"] - target[1]))
            rows.append({"phi": phi, "one_minus_m": lm, "alpha": f["alpha"],
                         "beta": f["beta"], "distance": d})
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def distance_to_target(alpha, beta, target=(1.5, 0.2), beta_weight=2):
    """The metric used throughout: sqrt(d_alpha^2 + (beta_weight d_beta)^2)."""
    return np.hypot(np.asarray(alpha) - target[0],
                    beta_weight * (np.asarray(beta) - target[1]))


# =============================================================================
# VERIFICATIONS  (python analytical_theory.py all)
# =============================================================================

N = 512
NETWORKS = [("hm2.83", "hm", 2.83), ("er6.2", "er", 6.2), ("ring4", "ring", 4)]
PHIS = [1.5, 2.0, 2.83, 3.5, 5.0]


def _spectra(n=N, seed=7):
    out = {}
    for name, fam, par in NETWORKS:
        A = lt.make_network(n, fam, par, seed)
        out[name] = eigh(A, check_finite=False)
    return out


def check_modal():
    print("=" * 96)
    print("1  MODAL IDENTITY   V(K) = sum_k w_k abar_k(K),  tau(K) = <w t abar>/<w abar>")
    print("=" * 96)
    print(f"{'network':<9}{'m':<9}{'max rel err V':<16}{'max rel err tau':<18}"
          f"{'|d alpha|':<12}{'|d beta|'}")
    spec = _spectra()
    for name, _, _ in NETWORKS:
        ev, V = spec[name]
        for m in (0.5, 0.9, 0.99, 0.999):
            pred, levels = modal_levels(ev, V, m)
            true = pd.DataFrame({"K": [d["K"] for d in levels],
                                 "variance": [d["variance"] for d in levels],
                                 "tau": [d["tau"] for d in levels]})
            ref, got = lt.fit_scaling(true), lt.fit_scaling(pred)
            print(f"{name:<9}{m:<9}"
                  f"{np.abs(pred.variance / true.variance - 1).max():<16.3e}"
                  f"{np.abs(pred.tau / true.tau - 1).max():<18.3e}"
                  f"{abs(got['alpha'] - ref['alpha']):<12.3e}"
                  f"{abs(got['beta'] - ref['beta']):.3e}")

    print()
    print("  How m-dependent is abar itself? (the partition is chosen from Sigma(m))")
    print(f"{'network':<9}{'m_ref':<8}{'m':<9}{'alpha true':<12}{'alpha frozen':<14}"
          f"{'beta true':<11}{'beta frozen':<13}{'K=64 agreement'}")
    for name, _, _ in NETWORKS:
        ev, V = spec[name]
        parts = {}
        for m in (0.5, 0.9, 0.99, 0.999):
            S, L = lt.stationary_covariances(ev, V, m)
            parts[m] = cascade(S, L)
        for m_ref in (0.5, 0.999):
            for m in (0.5, 0.9, 0.99, 0.999):
                w, t = modal_weights(ev, m)
                Ks = [d["K"] for d in parts[m_ref]]
                v_f, t_f = [], []
                for d in parts[m_ref]:
                    ab = overlaps(d["memb"], V)
                    nv = float(np.sum(w * ab))
                    v_f.append(nv)
                    t_f.append(float(np.sum(w * t * ab)) / nv)
                ref = lt.fit_scaling(pd.DataFrame(
                    {"K": [d["K"] for d in parts[m]],
                     "variance": [d["variance"] for d in parts[m]],
                     "tau": [d["tau"] for d in parts[m]]}))
                lv = {d["K"]: d["memb"] for d in parts[m]}
                lr = {d["K"]: d["memb"] for d in parts[m_ref]}
                agree = float(((lv[64][:, None] == lv[64][None, :])
                               == (lr[64][:, None] == lr[64][None, :])).mean())
                print(f"{name:<9}{m_ref:<8}{m:<9}{ref['alpha']:<12.4f}"
                      f"{slope(Ks, v_f):<14.4f}{ref['beta']:<11.4f}"
                      f"{slope(Ks, t_f):<13.4f}{agree:.4f}")


def check_mechanism():
    print("=" * 104)
    print("2  EXACT LOCAL-SLOPE IDENTITIES vs the linearised forms   (HM Phi=2.83)")
    print("=" * 104)
    A = lt.make_network(N, "hm", 2.83, 7)
    ev, V = eigh(A, check_finite=False)
    print(f"{'m':<9}{'K':<6}{'alpha true':<13}{'alpha exact':<14}{'alpha ~<g>':<13}"
          f"{'beta true':<12}{'beta exact':<13}{'beta ~Cov/<t>'}")
    for m in (0.9, 0.99, 0.999):
        df = mode_statistics(ev, V, m)
        for _, r in df[df.K.isin([2, 8, 32])].iterrows():
            print(f"{m:<9}{int(r.K):<6}{r.alpha_true:<13.6f}{r.alpha_exact:<14.6f}"
                  f"{r.alpha_lin:<13.6f}{r.beta_true:<12.6f}{r.beta_exact:<13.6f}"
                  f"{r.beta_lin:.6f}")
    print("  (the exact columns match to 6 dp; the linearised ones do not)")

    print()
    print("=" * 104)
    print("  MODE CONCENTRATION vs beta: does beta die when one mode owns V(K)?  [K=32]")
    print("=" * 104)
    print(f"{'network':<9}{'m':<9}{'alpha':<9}{'beta':<9}{'perron pi':<11}{'N_eff':<9}"
          f"{'sd_tau/<tau>':<14}{'Cov/<tau>':<12}{'corr(g,tau)'}")
    for name, fam, par in NETWORKS:
        A = lt.make_network(N, fam, par, 7)
        ev, V = eigh(A, check_finite=False)
        for m in (0.5, 0.9, 0.99, 0.995, 0.999, 0.9999):
            S, L = lt.stationary_covariances(ev, V, m)
            lv = cascade(S, L)
            fit = lt.fit_scaling(pd.DataFrame(
                {"K": [d["K"] for d in lv],
                 "variance": [d["variance"] for d in lv],
                 "tau": [d["tau"] for d in lv]}))
            r = mode_statistics(ev, V, m).set_index("K").loc[32]
            print(f"{name:<9}{m:<9}{fit['alpha']:<9.4f}{fit['beta']:<9.4f}"
                  f"{r.perron:<11.4f}{r.n_eff:<9.2f}{r.sd_tau_rel:<14.4f}"
                  f"{r.cov_over_tau:<12.4f}{r.corr_gamma_tau:.4f}")

    print()
    print("=" * 104)
    print("  gamma IS m-FREE: per-mode overlap growth at K=32->64 across m")
    print("=" * 104)
    for name, fam, par in NETWORKS:
        A = lt.make_network(N, fam, par, 7)
        ev, V = eigh(A, check_finite=False)
        print(f"\n  {name}:   s_2 = {ev[-2]/np.abs(ev).max():.4f}, "
              f"s_min = {ev[0]/np.abs(ev).max():.4f}")
        print(f"  {'m':<9}{'gamma_perron':<16}{'gamma_max':<12}")
        for m in (0.5, 0.9, 0.99, 0.999):
            r = mode_statistics(ev, V, m).set_index("K").loc[32]
            print(f"  {m:<9}{r.gamma_perron:<16.4f}{r.gamma_max:<12.4f}")


def check_ring():
    print("=" * 96)
    print("3  RING CLOSED FORM")
    print("=" * 96)
    for k in (2, 4, 6):
        rep = partition_is_contiguous(N, k)
        print(f"  ring k={k}: contiguous blocks at every level -> "
              f"{all(ok for _, _, ok in rep)}   (K = 1..{rep[-1][0]})")
    print("  (k=2 fails: exact correlation ties break contiguity, so the kernel")
    print("   does not apply there. k=4 and k=6 are the valid cases.)")

    print()
    print(f"{'k':<5}{'m':<11}{'alpha module':<15}{'alpha closed':<15}{'beta module':<14}"
          f"{'beta closed':<14}{'|d alpha|':<12}{'|d beta|'}")
    for k in (2, 4, 6):
        for m in (0.5, 0.9, 0.99, 0.995, 0.999, 0.9999, 0.99999):
            A = lt.make_network(N, "ring", k, 7)
            ev, V = eigh(A, check_finite=False)
            S, L = lt.stationary_covariances(ev, V, m)
            lv = lt.coarse_grain(S, L, 4)
            cf = ring_levels(N, k, m, lv.K.tolist())
            fm, fc = lt.fit_scaling(lv), lt.fit_scaling(cf)
            print(f"{k:<5}{m:<11}{fm['alpha']:<15.6f}{fc['alpha']:<15.6f}"
                  f"{fm['beta']:<14.6f}{fc['beta']:<14.6f}"
                  f"{abs(fm['alpha']-fc['alpha']):<12.2e}"
                  f"{abs(fm['beta']-fc['beta']):.2e}")

    print()
    print("  Where the 512x512 matrix cannot go: n to 2^20, 1-m to 1e-9")
    print(f"{'n':<10}{'k':<4}{'1-m':<11}{'alpha':<11}{'beta'}")
    for n in (512, 4096, 65536, 1048576):
        for lm in (1e-2, 1e-6, 1e-9):
            f = lt.fit_scaling(ring_levels(n, 4, 1 - lm, [2 ** i for i in range(12)]))
            print(f"{n:<10}{4:<4}{lm:<11.1e}{f['alpha']:<11.4f}{f['beta']:.4f}")


def check_collapse():
    print("=" * 96)
    print("4  SCALING COLLAPSE   alpha and beta are functions of x = K sqrt(1-m) alone")
    print("=" * 96)
    n, k = 2 ** 20, 4
    xs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
    for which, lab in ((0, "local alpha"), (1, "local beta")):
        print(f"\n  {lab}")
        print(f"  {'1-m':<10}{'K range':<18}" + "".join(f"{'x='+str(x):>11}" for x in xs))
        for lm in (1e-2, 1e-4, 1e-6, 1e-8):       # perfect squares keep K integral
            m, r = 1 - lm, np.sqrt(lm)
            s = ring_spectrum(n, k)
            Ks = [int(round(x / r)) for x in xs]
            vals = [ring_local_slopes(n, k, m, K, s)[which] for K in Ks]
            print(f"  {lm:<10.0e}{f'{Ks[0]}-{Ks[-1]}':<18}"
                  + "".join(f"{v:>11.5f}" for v in vals))

    lm = 1e-6
    Ks = sorted(set(int(round(v)) for v in
                    np.exp(np.linspace(np.log(50), np.log(60000), 34))))
    curve = ring_collapse_curve(n, k, lm, Ks)
    i = int(np.nanargmax(curve.beta.values))
    d = distance_to_target(curve.alpha.values, curve.beta.values)
    j = int(np.nanargmin(d))
    print()
    print(f"  beta is maximal at x = {curve.x[i]:.4f}:  beta_max = {curve.beta[i]:.5f}, "
          f"alpha there = {curve.alpha[i]:.5f}")
    print(f"  closest approach of the whole curve to (1.5, 0.2): x = {curve.x[j]:.4f} -> "
          f"({curve.alpha[j]:.4f}, {curve.beta[j]:.4f}), distance {d[j]:.4f}")
    print("  => the best fit to the Munn exponents SITS ON the maximum of beta.")

    print()
    print("  Consequence: a fitted exponent over a FIXED window depends on m only")
    print("  through where that window sits on the single curve above.")
    print(f"  {'1-m':<11}{'x window':<24}{'alpha fit':<12}{'beta fit'}")
    for lm in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        m, r = 1 - lm, np.sqrt(lm)
        f = lt.fit_scaling(ring_levels(n, k, m, [2 ** p for p in range(1, 7)]))
        print(f"  {lm:<11.0e}{f'{2*r:.4f} - {64*r:.4f}':<24}{f['alpha']:<12.5f}"
              f"{f['beta']:.5f}")
    print("  => fixed-window fitting as m -> 1 drives (alpha, beta) -> (2, 0),")
    print("     NOT to (1.5, 0.2).")


def check_hier():
    print("=" * 100)
    print("5  DETERMINISTIC HIERARCHY: closed-form spectrum vs eigh")
    print("=" * 100)
    print(f"{'Phi':<8}{'distinct eigs':<16}{'max abs err':<16}{'rho formula':<14}{'rho eigh'}")
    for phi in PHIS:
        ev = eigh(hier_adjacency(N, phi), eigvals_only=True, check_finite=False)
        spec = hier_spectrum(N, phi)
        predicted = np.concatenate([np.full(mult, val) for val, mult in spec])
        print(f"{phi:<8}{len(spec):<16}"
              f"{np.abs(np.sort(predicted) - np.sort(ev)).max():<16.3e}"
              f"{max(abs(v) for v, _ in spec):<14.6f}{np.abs(ev).max():.6f}")

    print()
    print("  Are the ICG clusters the tree's own blocks, for K >= 4?")
    for phi in PHIS:
        res = [hier_tree_blocks(N, phi, m, from_K=4)
               for m in (0.5, 0.9, 0.99, 0.999, 0.9999)]
        print(f"  Phi={phi:<6} m=0.5..0.9999 -> {res}")
    print("  (Phi=1.5 and 2.0 leave the tree blocks even at K>=4: the closed form")
    print("   is scoped to Phi >= 2.83.)")

    print()
    print("=" * 100)
    print("  closed form vs linear_theory.py, fitting K = 4..64  (drop_first=2)")
    print("=" * 100)
    print(f"{'Phi':<8}{'m':<11}{'alpha module':<15}{'alpha formula':<16}"
          f"{'beta module':<14}{'beta formula':<15}{'|d alpha|':<12}{'|d beta|'}")
    worst, worst_in_scope = 0.0, 0.0
    for phi in PHIS:
        ev, V = eigh(hier_adjacency(N, phi), check_finite=False)
        for m in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999):
            S, L = lt.stationary_covariances(ev, V, m)
            lv = lt.coarse_grain(S, L, 4)
            cf = hier_levels(N, phi, m, lv.K.tolist())
            fm = lt.fit_scaling(lv, drop_first=2, drop_last=1)
            fc = lt.fit_scaling(cf, drop_first=2, drop_last=1)
            da, db = abs(fm["alpha"] - fc["alpha"]), abs(fm["beta"] - fc["beta"])
            worst = max(worst, da, db)
            if phi >= 2.83:
                worst_in_scope = max(worst_in_scope, da, db)
            print(f"{phi:<8}{m:<11}{fm['alpha']:<15.8f}{fc['alpha']:<16.8f}"
                  f"{fm['beta']:<14.8f}{fc['beta']:<15.8f}{da:<12.2e}{db:.2e}")
    print(f"\n  worst discrepancy, K=2 excluded, all Phi:        {worst:.3e}")
    print(f"  worst discrepancy, K=2 excluded, Phi >= 2.83:    {worst_in_scope:.3e}")
    print("  (the first number is carried entirely by the out-of-scope Phi=1.5, 2.0)")


def check_hier_diag():
    print("=" * 100)
    print("6  WHY K = 2 HAS TO GO: the level-1 wavelet eigenvalue is exactly -1")
    print("=" * 100)
    print(f"{'Phi':<8}{'lambda_1 formula':<20}{'nearest eigh':<20}"
          f"{'corr(0,1) sibling':<20}{'corr(0,2) one out'}")
    for phi in PHIS:
        ev, V = eigh(hier_adjacency(N, phi), check_finite=False)
        lam1 = hier_spectrum(N, phi)[1][0]
        S, _ = lt.stationary_covariances(ev, V, 0.99)
        sd = np.sqrt(np.diag(S))
        C = S / np.outer(sd, sd)
        print(f"{phi:<8}{lam1:<20.12f}{ev[np.argmin(np.abs(ev - lam1))]:<20.12f}"
              f"{C[0,1]:<20.8f}{C[0,2]:.8f}")
    print("  => the nearest neighbour is the LEAST correlated close partner, so")
    print("     ICG's first merge CROSSES the first module boundary.")

    print()
    print("  And all level-1 partners are exactly tied, so K=2 is noise-decided:")
    same, pairs = hier_tie_instability(N, 2.83)
    print(f"    pairing unchanged under a 1e-14 relative perturbation: {same}")
    print(f"    first 6 pairs unperturbed: {pairs[:6].tolist()}")

    print()
    print("=" * 100)
    print("  overlap rule  abar_s(K) = K^2/n if 2^s > K else 0,  checked at K >= 4")
    print("=" * 100)
    phi = 2.83
    ev, V = eigh(hier_adjacency(N, phi), check_finite=False)
    spec = hier_spectrum(N, phi)
    lev = np.array([int(np.argmin([abs(e - v) for v, _ in spec])) for e in ev])
    S, L = lt.stationary_covariances(ev, V, 0.99)
    memb, K = np.arange(N), 1
    while len(S) >= 4:
        if K in (4, 16, 64):
            ab = overlaps(memb, V)
            above = np.array([(l == 0) or (2 ** l > K) for l in lev])
            print(f"  K={K:<4} predicted K^2/n = {K**2/N:.6f}   "
                  f"measured mean over 'support > K' = {ab[above].mean():.6f}   "
                  f"max dev = {np.abs(ab[above]-K**2/N).max():.2e}")
            print(f"         measured max over 'support <= K' = {ab[~above].max():.2e}"
                  f"  (predicted 0)")
        pairs = lt.greedy_pairs(S)
        a, b = pairs.T
        relabel = np.empty(len(S), int)
        relabel[a] = np.arange(len(pairs))
        relabel[b] = np.arange(len(pairs))
        memb = relabel[memb]
        S, L = lt.pool(S, pairs), lt.pool(L, pairs)
        K *= 2


def check_grid():
    print("=" * 100)
    print("7  WHAT THE FORMULA BUYS: Phi x m at n = 2^20, fitted over K = 4..2^15")
    print("=" * 100)
    n = 2 ** 20
    Ks = [2 ** p for p in range(1, 17)]
    phis = [1.5, 1.8, 2.0, 2.25, 2.5, 2.83, 3.2, 3.5, 4.0, 5.0]
    lms = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    grid = hier_grid(n, phis, lms, Ks)
    wide = grid.pivot(index="phi", columns="one_minus_m", values="alpha")
    wideb = grid.pivot(index="phi", columns="one_minus_m", values="beta")
    print(f"{'Phi':<8}" + "".join(f"{'1-m='+f'{v:.0e}':>22}" for v in lms))
    for phi in phis:
        cells = ["({:.3f}, {:.3f})".format(wide.loc[phi, v], wideb.loc[phi, v])
                 for v in lms]
        print(f"{phi:<8}" + "".join(f"{c:>22}" for c in cells))
    print()
    print("  closest to (1.5, 0.2), metric sqrt(d_alpha^2 + 4 d_beta^2):")
    print(grid.head(8).to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
    print("  => every Phi row has its own interior optimum in m; the m -> 1 column")
    print("     is the worst in every row.")


CHECKS = {"modal": check_modal, "mechanism": check_mechanism, "ring": check_ring,
          "collapse": check_collapse, "hier": check_hier,
          "hier-diag": check_hier_diag, "grid": check_grid}


if __name__ == "__main__":
    np.seterr(all="ignore")                       # spurious BLAS FPE flags on this Mac
    args = sys.argv[1:] or ["all"]
    names = list(CHECKS) if "all" in args else args
    for i, name in enumerate(names):
        if name not in CHECKS:
            raise SystemExit(f"unknown check {name!r}; choose from {list(CHECKS)} or 'all'")
        if i:
            print("\n")
        CHECKS[name]()
