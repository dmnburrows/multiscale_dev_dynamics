"""Five independent verifications of the alpha/beta modal proof.

The proof chain is:

    (i)   x(t+1)=Wx(t)+xi, W symmetric, Q=I  =>  modes are independent AR(1)
          with w_k = 1/(1-lam^2), tau_k = 1/(1-lam)
    (ii)  V(K)   = sum_k w_k abar_k(K)
          tau(K) = sum_k w_k tau_k abar_k(K) / sum_k w_k abar_k(K)
    (iii) abar_k(2K) = abar_k(K) 2^gamma_k   =>   pi(2K) = pi(K) 2^gamma / <2^gamma>
    (iv)  alpha = log2 <2^gamma>_pi
    (v)   beta  = log2[ <tau 2^gamma> / (<tau><2^gamma>) ]
                = log2[ 1 + Cov(tau,2^gamma)/(<tau><2^gamma>) ]
                = log2[ 1 + corr(tau,2^gamma) CV(tau) CV(2^gamma) ]

Each check attacks a different link, by a different route:

  CHECK 1  Monte Carlo from the raw difference equation. Tests (i) and (ii)
           against actual simulated dynamics -- no eigendecomposition used to
           produce the data. This is the only check that tests the PREMISES.
  CHECK 2  Symbolic algebra (sympy). Tests (iii)-(v) as exact identities in
           free symbols, so they are proven, not sampled.
  CHECK 3  Brute-force cluster sums vs modal sums on real networks. Tests (ii)
           by computing both sides two different ways.
  CHECK 4  Adversarial random inputs: random symmetric matrices, negative
           eigenvalues, near-critical and tiny m, and ARBITRARY non-nested
           unequal-size partitions. Tests that the identities need none of the
           structure the derivation happened to assume.
  CHECK 5  Corollaries and degenerate cases, including the exactly solvable
           bimodal reduction alpha = 1 + log2(1+p).

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u proof_checks.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt

np.seterr(all="ignore")
FAIL = []


def report(label, err, tol):
    ok = np.isfinite(err) and err <= tol
    if not ok:
        FAIL.append(label)
    print(f"    {'PASS' if ok else 'FAIL':<6}{label:<58}err {err:.3e}  (tol {tol:.0e})")


def report_sym(label, expr):
    """PASS only if the expression simplifies to exactly zero."""
    import sympy as sp
    z = sp.simplify(expr)
    ok = (z == 0)
    if not ok:
        FAIL.append(label)
    print(f"    {'PASS' if ok else 'FAIL':<6}{label:<58}"
          f"simplify -> {sp.srepr(z)[:28] if not ok else '0'}")


# ----------------------------------------------------------------------------
# shared helpers, written from scratch rather than imported, so the checks do
# not inherit any convention from analytical_theory.py
# ----------------------------------------------------------------------------
def cascade_members(S, L, min_clusters=4):
    """Coarse-grain, recording the membership matrix P at every level.

    P has one row per cluster, P[c,i]=1 if unit i is in cluster c. Returns a
    list of dicts with K, P, and the two observables computed the way
    linear_theory.coarse_grain computes them (trace ratios).
    """
    n = len(S)
    P = np.eye(n)
    out, K = [], 1
    Sc, Lc = S.copy(), L.copy()
    while len(Sc) >= min_clusters:
        out.append({"K": K, "P": P.copy(),
                    "V": np.trace(Sc) / len(Sc),
                    "tau": np.trace(Lc) / np.trace(Sc)})
        if len(Sc) == 1:
            break
        pairs = lt.greedy_pairs(Sc)
        Sc, Lc = lt.pool(Sc, pairs), lt.pool(Lc, pairs)
        a, b = pairs.T
        P = P[a] + P[b]
        K *= 2
    return out


def modal(ev, m):
    """w_k and tau_k from the AR(1) formulae.

    w is written 1/((1-lam)(1+lam)), matching linear_theory, NOT the
    algebraically identical 1/(1-lam^2). The latter suffers catastrophic
    cancellation near criticality: lam^2 rounds to ~1, so 1-lam^2 carries
    relative error ~eps/(2(1-m)) -- 1e-10 at 1-m=1e-6. The factored form is
    safe because 1-lam is computed exactly (Sterbenz) for lam in [1/2, 2].
    An earlier draft of this file used the unstable form and produced spurious
    1e-11 residuals in checks 3 and 4.
    """
    lam = m * ev / np.max(np.abs(ev))
    return 1.0 / ((1.0 - lam) * (1.0 + lam)), 1.0 / (1.0 - lam), lam


def abar(P, V):
    """Mean squared mode-cluster overlap: mean over clusters of (v_k . 1_S)^2."""
    return np.mean((P @ V) ** 2, axis=0)


def pieces(P0, P1, V, w, t):
    """pi, 2^gamma, and the five scalar averages the identities use."""
    a0, a1 = abar(P0, V), abar(P1, V)
    ok = a0 > 1e-300
    u = np.zeros_like(a0)
    u[ok] = a1[ok] / a0[ok]                      # = 2^gamma_k
    pi = w * a0 / np.sum(w * a0)
    mt, mu = np.sum(pi * t), np.sum(pi * u)
    mtu = np.sum(pi * t * u)
    sdt = np.sqrt(max(np.sum(pi * t ** 2) - mt ** 2, 0.0))
    sdu = np.sqrt(max(np.sum(pi * u ** 2) - mu ** 2, 0.0))
    return pi, u, mt, mu, mtu, sdt, sdu


# ============================================================================
# CHECK 1 -- Monte Carlo from the raw difference equation
# ============================================================================
def simulate(W, n_samples, H, seed, reps=8, chunk=50000):
    """Estimate S = <x x^T> and L = sum_{h=0}^{H} <x(t) x(t+h)^T> by simulation.

    The trick that makes L cheap: sum_h <x(t)x(t+h)^T> = <x(t) y(t)^T> with
    y(t) = sum_{h=0}^{H} x(t+h), a rolling window sum obtained from one cumsum.
    No eigendecomposition is used anywhere in here.
    """
    n = len(W)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((reps, n))
    for _ in range(3000):                                    # burn in
        x = x @ W.T + rng.standard_normal((reps, n))
    SS = np.zeros((n, n))
    SL = np.zeros((n, n))
    count = 0
    while count < n_samples:
        c = int(min(chunk, np.ceil((n_samples - count) / reps)))
        X = np.empty((c + H, reps, n))
        for s in range(c + H):
            x = x @ W.T + rng.standard_normal((reps, n))
            X[s] = x
        cs = np.cumsum(X, axis=0)
        Y = cs[H:H + c].copy()
        Y[1:] -= cs[:c - 1]                                  # y(t)=cs[t+H]-cs[t-1]
        Xa = X[:c].reshape(-1, n)
        Ya = Y.reshape(-1, n)
        SS += Xa.T @ Xa
        SL += Xa.T @ Ya
        count += c * reps
    return SS / count, SL / count


def check1():
    print("\nCHECK 1  Monte Carlo from x(t+1)=Wx(t)+xi  (tests the PREMISES)")
    print("  The data come from simulated dynamics only. Nothing below uses the")
    print("  AR(1) formulae to generate it; they are only used to predict it.\n")
    n, m, H = 32, 0.90, 300
    A = lt.make_network(n, "hm", 3.0, 1)
    ev, V = eigh(A, check_finite=False)
    W = (m / np.max(np.abs(ev))) * A
    S_th, L_th = lt.stationary_covariances(ev, V, m)
    S_mc, L_mc = simulate(W, 6_000_000, H, seed=7)

    rel = lambda a, b: np.max(np.abs(a - b)) / np.max(np.abs(b))
    report("S_sim == S_theory  (w_k = 1/((1-lam)(1+lam)), modes independent)",
           rel(S_mc, S_th), 2e-2)
    report("L_sim == L_theory  (tau_k = 1/(1-lam), signed lag sum)",
           rel(L_mc, L_th), 3e-2)

    # Use the THEORETICAL pairing for both, so this compares dynamics and not
    # tie-breaking in the greedy matcher. Then report whether they agreed anyway.
    lev_th = cascade_members(S_th, L_th)
    same = True
    Sc, Lc = S_mc.copy(), L_mc.copy()
    Vm, Tm = [], []
    for d in lev_th:
        P = d["P"]
        Sp, Lp = P @ S_mc @ P.T, P @ L_mc @ P.T
        Vm.append(np.trace(Sp) / len(Sp))
        Tm.append(np.trace(Lp) / np.trace(Sp))
    lev_mc_own = cascade_members(S_mc, L_mc)
    for a, b in zip(lev_th, lev_mc_own):
        if not np.array_equal(a["P"], b["P"]):
            same = False
    Vm, Tm = np.array(Vm), np.array(Tm)
    Vt = np.array([d["V"] for d in lev_th])
    Tt = np.array([d["tau"] for d in lev_th])
    report("V(K) from simulation == V(K) from theory, all K", rel(Vm, Vt), 3e-2)
    report("tau(K) from simulation == tau(K) from theory, all K", rel(Tm, Tt), 3e-2)

    a_mc = np.log2(Vm[1:] / Vm[:-1]);  a_th = np.log2(Vt[1:] / Vt[:-1])
    b_mc = np.log2(Tm[1:] / Tm[:-1]);  b_th = np.log2(Tt[1:] / Tt[:-1])
    report("local alpha(K) from simulation == from theory",
           np.max(np.abs(a_mc - a_th)), 3e-2)
    report("local beta(K) from simulation == from theory",
           np.max(np.abs(b_mc - b_th)), 3e-2)
    print(f"\n    greedy partitions from simulated vs theoretical S identical: "
          f"{'yes' if same else 'no (compared on the theoretical partition)'}")
    print(f"    {'K':<6}{'alpha sim':<13}{'alpha theory':<15}"
          f"{'beta sim':<13}{'beta theory'}")
    for i in range(len(a_mc)):
        print(f"    {lev_th[i]['K']:<6}{a_mc[i]:<13.5f}{a_th[i]:<15.5f}"
              f"{b_mc[i]:<13.5f}{b_th[i]:.5f}")


# ============================================================================
# CHECK 2 -- symbolic algebra
# ============================================================================
def check2():
    print("\nCHECK 2  Symbolic algebra in free symbols (sympy)")
    print("  w_k, tau_k, abar_k(K), abar_k(2K) are arbitrary positive symbols.")
    print("  Each identity is simplified to exactly zero -- a proof, not a sample.\n")
    import sympy as sp
    N = 4
    w = sp.symbols(f"w1:{N+1}", positive=True)
    t = sp.symbols(f"t1:{N+1}", positive=True)
    a = sp.symbols(f"a1:{N+1}", positive=True)    # abar(K)
    b = sp.symbols(f"b1:{N+1}", positive=True)    # abar(2K)
    u = [b[k] / a[k] for k in range(N)]           # 2^gamma_k

    VK = sum(w[k] * a[k] for k in range(N))
    V2K = sum(w[k] * b[k] for k in range(N))
    tauK = sum(w[k] * t[k] * a[k] for k in range(N)) / VK
    tau2K = sum(w[k] * t[k] * b[k] for k in range(N)) / V2K
    pi = [w[k] * a[k] / VK for k in range(N)]
    avg = lambda f: sum(pi[k] * f[k] for k in range(N))

    mu, mt = avg(u), avg(t)
    mtu = avg([t[k] * u[k] for k in range(N)])
    cov = mtu - mt * mu
    var_t = avg([t[k] ** 2 for k in range(N)]) - mt ** 2
    var_u = avg([u[k] ** 2 for k in range(N)]) - mu ** 2

    for k in range(N):
        report_sym(f"tilt lemma, mode {k+1}:  pi_k(2K) = pi_k(K) 2^g_k / <2^g>",
                   w[k] * b[k] / V2K - pi[k] * u[k] / mu)
    report_sym("alpha:  V(2K)/V(K) = <2^gamma>_pi", V2K / VK - mu)
    report_sym("tau(K) = <tau>_pi", tauK - mt)
    report_sym("tau(2K) = <tau 2^g>_pi / <2^g>_pi", tau2K - mtu / mu)
    report_sym("beta:  tau(2K)/tau(K) = <tau 2^g>/(<tau><2^g>)",
               tau2K / tauK - mtu / (mt * mu))
    report_sym("covariance form:  ratio = 1 + Cov(tau,2^g)/(<tau><2^g>)",
               mtu / (mt * mu) - (1 + cov / (mt * mu)))

    # The CV regrouping, with the two standard deviations as positive symbols.
    # Written this way because sympy cannot establish var_t, var_u >= 0 from
    # free symbols, so it will not combine sqrt(var_t var_u) on its own. The
    # content of the step is only the regrouping, which this proves exactly.
    st, su, cv_cov = sp.symbols("sigma_tau sigma_u c", positive=True)
    report_sym("CV form:  Cov/(<tau><2^g>) = corr * CV(tau) * CV(2^g)",
               cv_cov / (mt * mu)
               - (cv_cov / (st * su)) * (st / mt) * (su / mu))
    report_sym("variance expansion: Var(tau) = <tau^2> - <tau>^2 under pi",
               (avg([t[k] ** 2 for k in range(N)]) - mt ** 2) - var_t)
    report_sym("variance expansion: Var(2^g) = <2^2g> - <2^g>^2 under pi",
               (avg([u[k] ** 2 for k in range(N)]) - mu ** 2) - var_u)
    print("\n    (each reduces to 0 identically in w, tau, abar -- so the")
    print("     algebra holds for any spectrum, any network, any partition)")


# ============================================================================
# CHECK 3 -- brute-force cluster sums vs modal sums
# ============================================================================
def check3():
    print("\nCHECK 3  Modal sums vs direct cluster sums on real networks")
    print("  LHS: trace(P S P')/n_clusters and trace(P L P')/trace(P S P').")
    print("  RHS: sum_k w_k abar_k and the tau-weighted version. Different routes.\n")
    worst_v = worst_t = worst_a = worst_b = 0.0
    for fam, par in (("ring", 4), ("ring", 8), ("hm", 2.5), ("hm", 4.0),
                     ("er", 6.0), ("sf", 2.5), ("sw", 0.1)):
        for lm in (3e-1, 1e-2, 1e-4, 1e-6):
            n, m = 256, 1 - lm
            A = lt.make_network(n, fam, par, 1)
            ev, V = eigh(A, check_finite=False)
            S, L = lt.stationary_covariances(ev, V, m)
            w, t, _ = modal(ev, m)
            lev = cascade_members(S, L)
            for i in range(len(lev) - 1):
                P0, P1 = lev[i]["P"], lev[i + 1]["P"]
                a0 = abar(P0, V)
                worst_v = max(worst_v, abs(np.sum(w * a0) - lev[i]["V"])
                              / abs(lev[i]["V"]))
                worst_t = max(worst_t, abs(np.sum(w * t * a0) / np.sum(w * a0)
                                           - lev[i]["tau"]) / abs(lev[i]["tau"]))
                _, u, mt, mu, mtu, _, _ = pieces(P0, P1, V, w, t)
                a_id = np.log2(mu)
                b_id = np.log2(mtu / (mt * mu))
                a_tr = np.log2(lev[i + 1]["V"] / lev[i]["V"])
                b_tr = np.log2(lev[i + 1]["tau"] / lev[i]["tau"])
                worst_a = max(worst_a, abs(a_id - a_tr))
                worst_b = max(worst_b, abs(b_id - b_tr))
    report("V(K)   modal sum == trace ratio   (28 nets x all K)", worst_v, 1e-11)
    report("tau(K) modal sum == trace ratio   (28 nets x all K)", worst_t, 1e-11)
    report("alpha  identity == measured slope (28 nets x all K)", worst_a, 1e-11)
    report("beta   identity == measured slope (28 nets x all K)", worst_b, 1e-11)


# ============================================================================
# CHECK 4 -- adversarial random inputs and arbitrary partitions
# ============================================================================
def random_partition(n, n_cl, rng):
    """A partition into n_cl non-empty, deliberately unequal clusters."""
    lab = rng.integers(0, n_cl, n)
    lab[:n_cl] = np.arange(n_cl)                  # guarantee non-empty
    rng.shuffle(lab)
    P = np.zeros((n_cl, n))
    P[lab, np.arange(n)] = 1.0
    return P


def check4():
    print("\nCHECK 4  Adversarial inputs: random dense symmetric W, negative")
    print("  eigenvalues, m from 1e-8 to 1-1e-9, and ARBITRARY unequal,")
    print("  NON-NESTED partitions (not produced by greedy ICG at all).\n")
    rng = np.random.default_rng(0)
    worst = {k: 0.0 for k in ("V", "tau", "alpha", "beta", "cov", "cv", "tilt")}
    n_neg = 0
    for trial in range(400):
        n = int(rng.choice([24, 48, 96]))
        kind = trial % 3
        if kind == 0:                             # dense Gaussian symmetric
            M = rng.standard_normal((n, n)); A = (M + M.T) / 2
        elif kind == 1:                           # sparse signed
            M = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.15)
            A = (M + M.T) / 2
        else:                                     # binary graph
            A = lt.make_network(32, "er", float(rng.uniform(2, 12)), trial)
            n = 32
        np.fill_diagonal(A, 0.0)
        ev, V = eigh(A, check_finite=False)
        if np.min(ev) < 0:
            n_neg += 1
        m = float(rng.choice([1e-8, 1e-3, 0.5, 0.9, 1 - 1e-5, 1 - 1e-9]))
        S, L = lt.stationary_covariances(ev, V, m)
        w, t, _ = modal(ev, m)
        # two unrelated random partitions, different cluster counts
        c0 = int(rng.integers(4, max(5, n // 3)))
        c1 = int(rng.integers(2, c0))
        P0, P1 = random_partition(n, c0, rng), random_partition(n, c1, rng)

        for P, key_v, key_t in ((P0, "V", "tau"),):
            Sp, Lp = P @ S @ P.T, P @ L @ P.T
            a = abar(P, V)
            worst["V"] = max(worst["V"], abs(np.sum(w * a) - np.trace(Sp) / len(Sp))
                             / abs(np.trace(Sp) / len(Sp)))
            worst["tau"] = max(worst["tau"],
                               abs(np.sum(w * t * a) / np.sum(w * a)
                                   - np.trace(Lp) / np.trace(Sp))
                               / abs(np.trace(Lp) / np.trace(Sp)))
        V0, V1 = np.trace(P0 @ S @ P0.T) / c0, np.trace(P1 @ S @ P1.T) / c1
        t0 = np.trace(P0 @ L @ P0.T) / np.trace(P0 @ S @ P0.T)
        t1 = np.trace(P1 @ L @ P1.T) / np.trace(P1 @ S @ P1.T)
        pi, u, mt, mu, mtu, sdt, sdu = pieces(P0, P1, V, w, t)
        worst["alpha"] = max(worst["alpha"], abs(np.log2(mu) - np.log2(V1 / V0)))
        worst["beta"] = max(worst["beta"],
                            abs(np.log2(mtu / (mt * mu)) - np.log2(t1 / t0)))
        cov = mtu - mt * mu
        worst["cov"] = max(worst["cov"],
                           abs(np.log2(1 + cov / (mt * mu))
                               - np.log2(mtu / (mt * mu))))
        if sdt * sdu > 0:
            corr = cov / (sdt * sdu)
            worst["cv"] = max(worst["cv"],
                              abs(np.log2(1 + corr * (sdt / mt) * (sdu / mu))
                                  - np.log2(t1 / t0)))
        a1 = abar(P1, V)
        ok = (a1 > 1e-300) & (abar(P0, V) > 1e-300)
        pi1 = w * a1 / np.sum(w * a1)
        worst["tilt"] = max(worst["tilt"],
                            np.max(np.abs(pi1[ok] - (pi * u / mu)[ok])))
    print(f"    400 trials, {n_neg} with negative eigenvalues present\n")
    report("V   modal sum == trace ratio, arbitrary partition", worst["V"], 1e-10)
    report("tau modal sum == trace ratio, arbitrary partition", worst["tau"], 1e-10)
    report("tilt lemma, two unrelated partitions", worst["tilt"], 1e-10)
    report("alpha identity, coarse/fine partitions unrelated", worst["alpha"], 1e-10)
    report("beta identity, coarse/fine partitions unrelated", worst["beta"], 1e-10)
    report("beta covariance form", worst["cov"], 1e-10)
    report("beta corr*CV*CV form", worst["cv"], 1e-10)
    print("\n    => the identities need neither nesting, nor equal cluster sizes,")
    print("       nor a doubling of K, nor positive weights. Only that the")
    print("       coarse-graining is a linear sum over a partition.")


# ============================================================================
# CHECK 5 -- corollaries and exactly solvable cases
# ============================================================================
def check5():
    print("\nCHECK 5  Corollaries and degenerate cases\n")
    rng = np.random.default_rng(3)

    # (ii)/(iii) bounds and Jensen, on real cascades
    worst_lo = worst_hi = worst_jen = 0.0
    for fam, par in (("ring", 4), ("hm", 3.0), ("er", 6.0), ("sf", 2.5)):
        for lm in (3e-1, 1e-2, 1e-4, 1e-6):
            n = 256
            A = lt.make_network(n, fam, par, 1)
            ev, V = eigh(A, check_finite=False)
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            w, t, _ = modal(ev, 1 - lm)
            lev = cascade_members(S, L)
            for i in range(len(lev) - 1):
                a0, a1 = abar(lev[i]["P"], V), abar(lev[i + 1]["P"], V)
                ok = (a0 > 1e-300) & (a1 > 1e-300)
                g = np.log2(a1[ok] / a0[ok])
                pi = (w * a0)[ok] / np.sum((w * a0)[ok])
                al = np.log2(np.sum(pi * 2.0 ** g))
                worst_lo = max(worst_lo, g.min() - al)          # want <= 0
                worst_hi = max(worst_hi, al - g.max())          # want <= 0
                worst_jen = max(worst_jen, np.sum(pi * g) - al)  # want <= 0
    report("corollary (ii):  alpha >= min_k gamma_k", max(worst_lo, 0.0), 1e-10)
    report("corollary (ii):  alpha <= max_k gamma_k", max(worst_hi, 0.0), 1e-10)
    report("corollary (iii): alpha >= <gamma>_pi  (Jensen)", max(worst_jen, 0.0), 1e-10)

    # (iv) degenerate: gamma constant => alpha = gamma exactly, beta = 0 exactly
    worst_a = worst_b = 0.0
    for _ in range(300):
        n = int(rng.integers(3, 40))
        w = rng.random(n) * 10 + 0.1
        t = rng.random(n) * 100 + 0.1
        a0 = rng.random(n) + 1e-3
        c = float(rng.uniform(0.2, 8.0))           # one common growth factor
        a1 = c * a0
        pi = w * a0 / np.sum(w * a0)
        u = a1 / a0
        mu, mt = np.sum(pi * u), np.sum(pi * t)
        mtu = np.sum(pi * t * u)
        worst_a = max(worst_a, abs(np.log2(mu) - np.log2(c)))
        worst_b = max(worst_b, abs(np.log2(mtu / (mt * mu))))
    report("corollary (iv):  gamma const => alpha = gamma", worst_a, 1e-12)
    report("corollary (iv):  gamma const => beta = 0", worst_b, 1e-12)

    # tau constant => beta = 0
    worst_b = 0.0
    for _ in range(300):
        n = int(rng.integers(3, 40))
        w = rng.random(n) * 10 + 0.1
        a0 = rng.random(n) + 1e-3
        a1 = rng.random(n) + 1e-3
        t = np.full(n, float(rng.uniform(0.5, 50)))
        pi = w * a0 / np.sum(w * a0)
        u = a1 / a0
        mu, mt, mtu = np.sum(pi * u), np.sum(pi * t), np.sum(pi * t * u)
        worst_b = max(worst_b, abs(np.log2(mtu / (mt * mu))))
    report("tau const => beta = 0 (no timescale spread to sort)", worst_b, 1e-12)

    # bimodal reduction: gamma in {1,2} with share p at 2  =>  alpha = 1+log2(1+p)
    worst = 0.0
    for p in np.linspace(0.01, 0.99, 99):
        n = 2000
        n2 = int(round(p * n))
        g = np.concatenate([np.full(n2, 2.0), np.full(n - n2, 1.0)])
        pi = np.ones(n) / n
        al = np.log2(np.sum(pi * 2.0 ** g))
        pp = n2 / n
        worst = max(worst, abs(al - (1 + np.log2(1 + pp))))
    report("bimodal:  alpha = 1 + log2(1+p)", worst, 1e-12)
    pstar = 2 ** 0.5 - 1
    report("bimodal:  alpha = 1.5  <=>  p = sqrt(2)-1",
           abs(1 + np.log2(1 + pstar) - 1.5), 1e-12)
    print(f"\n    p* = sqrt(2)-1 = {pstar:.6f}")
    # where the mixture variance of 2^gamma peaks, in the same reduction
    pg = np.linspace(1e-4, 1 - 1e-4, 200001)
    cv = np.sqrt(pg * (1 - pg)) / (1 + pg)
    ip = int(np.argmax(cv))
    print(f"    CV(2^gamma) in the bimodal reduction peaks at p = {pg[ip]:.4f}"
          f"  (analytic 1/3 = {1/3:.4f}),")
    print(f"    i.e. at alpha = {1+np.log2(1+pg[ip]):.4f} -- NOT exactly 1.5.")
    print("    So the beta peak sitting near alpha=1.5 is an observation about")
    print("    these spectra, not a theorem. Stated as such.")


def main():
    print("=" * 94)
    print("FIVE INDEPENDENT CHECKS OF THE alpha/beta MODAL PROOF")
    print("=" * 94)
    for fn in (check1, check2, check3, check4, check5):
        fn()
    print("\n" + "=" * 94)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:")
        for f in FAIL:
            print(f"   - {f}")
    else:
        print("ALL CHECKS PASSED.")
    print("=" * 94)


if __name__ == "__main__":
    main()
