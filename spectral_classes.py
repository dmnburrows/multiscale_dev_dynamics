"""Gapped vs gapless, made precise, and swept over every family's parameter range.

DEFINITION. Let s_1 >= s_2 >= ... be the adjacency spectrum normalised by its
radius, s_k = lambda_k / rho(A). The object that matters is not the gap alone
but the COUNTING FUNCTION at the top edge,

    N(delta) = #{ k : 1 - s_k <= delta }

because the modal weight w_k = 1/(1 - m^2 s_k^2) is large exactly for modes
with 1 - s_k <~ (1 - m). Three classes, by how N behaves as n grows:

  (G)  GAPPED.            N(delta) = 1 for all delta < delta_0, with delta_0 > 0
                          independent of n. One isolated top mode. Equivalently
                          liminf_n (s_1 - s_2) > 0.

  (B)  GAPLESS, EXTENSIVE EDGE ("band").   N(delta) ~ n delta^nu, nu > 0.
                          An extensive number of modes accumulates at the edge.
                          The amplified modes number M ~ n (1-m)^nu, so there is
                          a correlation volume n/M ~ (1-m)^(-nu), and alpha and
                          beta depend on K only through x = K/(n/M). This is the
                          class where the single-variable law is exact.

  (S)  GAPLESS, SUB-EXTENSIVE EDGE.   N(delta) ~ n^theta g(delta), 0 < theta < 1.
                          Modes do accumulate, but a vanishing fraction of them.
                          There is no correlation volume proportional to n, and
                          no single variable: the crossover scale and the number
                          of competing modes grow with different powers of n.

ANALYTIC RESULTS, one per family (all checked numerically below).

  Ring / d-torus, k-nearest per axis. Circulant, so s(q) = 1 - c q^2 + O(q^4)
      exactly, with c = (1/k) sum_{j<=k/2} 2 j^2 (2 pi / n)^2 ... in reduced
      units s(q) = 1 - c q^2. Counting modes with c q^2 <= delta in d dimensions
      gives
          N(delta) = C_d n (delta/c)^{d/2},     so  nu = d/2        -> class B
      and the gap is the smallest nonzero mode:
          s_1 - s_2 = c (2 pi/n)^2  ~  n^{-2}                       (d = 1)

  Erdos-Renyi, mean degree z. The Perron eigenvalue is lambda_1 ~ z while the
      bulk edge is the Kesten-McKay value 2 sqrt(z - 1), so
          gap  g(z) = 1 - 2 sqrt(z - 1) / z                         -> class G
      which is positive for z > 2 and CLOSES as z -> 2. So sparse random graphs
      approach the band class at the percolation point and nowhere else.

  Deterministic hierarchy, weights Phi^{-level}, r = 2/Phi. From the exact
      wavelet spectrum, 1 - s_s = (2 - r) r^{s-1} with multiplicity 2^{L-s}, so
          N(delta) = sum_{s > s_delta} 2^{L-s} ~ n (delta/(2-r))^{nu},
          nu = 1 / log2(Phi/2)                                      -> class B
      and the gap closes as n^{-log2(Phi/2)}. Matching nu = d/2 gives the
      effective dimension d_eff = 2/log2(Phi/2) used elsewhere.

  Scale-free (Chung-Lu, exponent tau). The top eigenvalues localise on hubs and
      separate from the bulk, so the gap stays open and grows with hub
      dominance                                                     -> class G

  Stochastic hierarchical modular. No closed form. Measured by the HARD count
      N(delta) at delta = 3e-3, it stays at 1 for every n up to 4096, so at
      accessible sizes it is class G, with a gap of 0.03 to 0.04 that drifts
      down slowly (slope about -0.5 in n, noisy across seeds) and whose limit
      is NOT resolved here.
      (An earlier pass called this class S on the strength of the SOFT count
      M(m) growing as n^0.66. That was wrong: M is a participation ratio over
      the whole spectrum, so its growth tracks the bulk, not an edge band. The
      hard count is the one that defines the class.)

This module sweeps every family across its whole parameter range, records the
spectral class diagnostics together with the exponents and the leading-mode
statistics, and writes one tidy table for the figures.

    python spectral_classes.py analytic   # the closed-form checks
    python spectral_classes.py sweep      # the full parameter sweep -> CSV
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import analytical_theory as at

np.seterr(all="ignore")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_data")

# Range-weighted distance: alpha spans about [1, 2] and beta about [0, 0.5],
# so beta is weighted by the ratio of those ranges.
R_ALPHA, R_BETA = 1.0, 0.5
TARGET = (1.5, 0.2)

PARAM_GRID = {
    "ring": [2, 4, 6, 8, 12, 16],
    "hm": [1.5, 2.0, 2.5, 2.83, 3.2, 3.5, 4.0, 5.0, 6.0],
    "er": [2.5, 3.0, 4.0, 6.2, 9.0, 14.0, 22.0, 40.0],
    "sw": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "sf": [2.1, 2.3, 2.5, 2.8, 3.1, 3.5],
}


def distance(alpha, beta, target=TARGET):
    """Range-weighted distance to the empirical pair."""
    return np.hypot((np.asarray(alpha) - target[0]) / R_ALPHA,
                    (np.asarray(beta) - target[1]) / R_BETA)


def counting(s, deltas):
    """N(delta) = #{k : 1 - s_k <= delta}."""
    g = 1.0 - s
    return np.array([int(np.sum(g <= d)) for d in deltas])


def leading_stats(ev, V, m, memb):
    """Leading mode's share of V(K), and how much slower it is than the rest."""
    w, t = at.modal_weights(ev, m)
    ab = at.overlaps(memb, V)
    contrib = w * ab
    tot = float(np.sum(contrib))
    top = int(np.argmax(contrib))
    mask = np.ones(len(w), bool)
    mask[top] = False
    tau_rest = float(np.sum(contrib[mask] * t[mask]) / np.sum(contrib[mask]))
    n_eff = float(tot ** 2 / np.sum(contrib ** 2))
    return float(contrib[top] / tot), float(t[top] / tau_rest), n_eff


# =============================================================================

def analytic():
    print("=" * 100)
    print("A  RING: gap ~ n^-2 and N(delta) ~ n sqrt(delta)   (class B, nu = 1/2)")
    print("=" * 100)
    print(f"{'k':<5}{'n':<8}{'gap measured':<16}{'gap ~ c(2pi/n)^2':<20}"
          f"{'N(1e-4)/n':<13}{'pred n*sqrt(d)/n':<18}")
    for k in (4, 8):
        for n in (1024, 4096, 16384):
            s = at.ring_spectrum(n, k)
            srt = np.sort(s)[::-1]
            d = np.arange(1, k // 2 + 1)
            # 2cos(qd) = 2 - q^2 d^2 + ..., so s(q) = 1 - (1/k) sum d^2 q^2
            c = float(np.sum(d ** 2) / k)              # k=4 -> c = 5/4
            gpred = c * (2 * np.pi / n) ** 2
            nd = counting(s, [1e-4])[0]
            print(f"{k:<5}{n:<8}{srt[0]-srt[1]:<16.3e}{gpred:<20.3e}"
                  f"{nd/n:<13.5f}{np.sqrt(1e-4/c)/np.pi:<18.5f}")

    print()
    print("=" * 100)
    print("B  ERDOS-RENYI: gap g(z) = 1 - 2 sqrt(z-1)/z   (class G, closes at z -> 2)")
    print("=" * 100)
    print(f"{'mean degree z':<16}{'gap predicted':<17}{'gap measured (n=4096)':<24}{'ratio'}")
    for z in PARAM_GRID["er"]:
        gp = 1.0 - 2.0 * np.sqrt(max(z - 1.0, 0.0)) / z
        gs = []
        for seed in (1, 2):
            A = lt.make_network(4096, "er", z, seed)
            ev = eigh(A, eigvals_only=True, check_finite=False)
            s = np.sort(ev / np.max(np.abs(ev)))[::-1]
            gs.append(s[0] - s[1])
        gm = float(np.mean(gs))
        print(f"{z:<16}{gp:<17.5f}{gm:<24.5f}{gm/gp if gp > 0 else np.nan:.4f}")

    print()
    print("=" * 100)
    print("C  DETERMINISTIC HIERARCHY: gap ~ n^-log2(Phi/2), N(delta) ~ n delta^nu")
    print("=" * 100)
    print(f"{'Phi':<7}{'nu = 1/log2(Phi/2)':<21}{'gap slope in n':<18}"
          f"{'pred -log2(Phi/2)':<20}{'N(delta) slope in delta':<24}{'pred nu'}")
    for phi in (2.83, 3.2, 3.5, 4.0, 5.0, 6.0):
        nu = 1.0 / np.log2(phi / 2.0)
        ns = [2 ** p for p in (10, 12, 14, 16)]
        gaps = []
        for n in ns:
            spec = at.hier_spectrum(n, phi)
            rho = max(abs(v) for v, _ in spec)
            vals = np.sort(np.concatenate(
                [np.full(mult, val / rho) for val, mult in spec]))[::-1]
            gaps.append(vals[0] - vals[1])
        gslope = np.polyfit(np.log(ns), np.log(gaps), 1)[0]
        n = 2 ** 16
        spec = at.hier_spectrum(n, phi)
        rho = max(abs(v) for v, _ in spec)
        s = np.concatenate([np.full(mult, val / rho) for val, mult in spec])
        ds = np.array([1e-5, 1e-4, 1e-3, 1e-2])
        Nd = counting(s, ds)
        ok = Nd > 1
        nslope = np.polyfit(np.log(ds[ok]), np.log(Nd[ok]), 1)[0] if ok.sum() > 1 else np.nan
        print(f"{phi:<7}{nu:<21.5f}{gslope:<18.4f}{-np.log2(phi/2):<20.4f}"
              f"{nslope:<24.4f}{nu:.4f}")

    print()
    print("=" * 100)
    print("D  WHICH CLASS IS EACH STOCHASTIC FAMILY? N(delta) vs n at fixed delta")
    print("   slope 1 = extensive band (B); slope 0 = gapped (G); between = (S)")
    print("=" * 100)
    delta = 3e-3
    ns = [512, 1024, 2048, 4096]
    print(f"{'family':<9}{'param':<9}" + "".join(f"{f'n={v}':>9}" for v in ns)
          + f"{'slope theta':>14}{'class'}")
    for fam, params in (("ring", [4]), ("hm", [2.83, 4.0]), ("er", [3.0, 6.2]),
                        ("sw", [0.01, 0.1]), ("sf", [2.5])):
        for par in params:
            counts = []
            for n in ns:
                cc = []
                for seed in (1, 2):
                    A = lt.make_network(n, fam, par, seed)
                    ev = eigh(A, eigvals_only=True, check_finite=False)
                    s = ev / np.max(np.abs(ev))
                    cc.append(counting(s, [delta])[0])
                counts.append(float(np.mean(cc)))
            th = np.polyfit(np.log(ns), np.log(np.maximum(counts, 1e-9)), 1)[0]
            cls = "B" if th > 0.8 else ("G" if th < 0.2 else "S")
            print(f"{fam:<9}{par:<9}" + "".join(f"{v:>9.1f}" for v in counts)
                  + f"{th:>14.3f}  {cls}")


# =============================================================================

def sweep():
    os.makedirs(OUT, exist_ok=True)
    n = 1024
    seeds = [1, 2]
    lms = np.logspace(-1, -5, 21)
    deltas = [1e-4, 1e-3, 1e-2]
    rows = []
    for fam, params in PARAM_GRID.items():
        for par in params:
            for seed in seeds:
                A = lt.make_network(n, fam, par, seed)
                ev, V = eigh(A, check_finite=False)
                s = ev / np.max(np.abs(ev))
                srt = np.sort(s)[::-1]
                gap = float(srt[0] - srt[1])
                Nd = counting(s, deltas)
                for lm in lms:
                    m = 1.0 - lm
                    w = 1.0 / (1.0 - m ** 2 * s ** 2)
                    M = float(np.sum(w) ** 2 / np.sum(w ** 2))
                    P = float(np.max(w) / np.sum(w))
                    S, L = lt.stationary_covariances(ev, V, m)
                    levels = at.cascade(S, L, 4)
                    Vv = np.array([d["variance"] for d in levels])
                    Tt = np.array([d["tau"] for d in levels])
                    Ks = np.array([d["K"] for d in levels], float)
                    a = np.log2(Vv[1:] / Vv[:-1])
                    b = np.log2(Tt[1:] / Tt[:-1])
                    fit = lt.fit_scaling(pd.DataFrame(
                        {"K": Ks, "variance": Vv, "tau": Tt}))
                    for i in range(len(a)):
                        p_lead, rho, n_eff = leading_stats(
                            ev, V, m, levels[i]["memb"])
                        rows.append({
                            "family": fam, "param": par, "seed": seed, "n": n,
                            "one_minus_m": lm, "gap": gap,
                            "N_1e4": Nd[0], "N_1e3": Nd[1], "N_1e2": Nd[2],
                            "M": M, "P": P, "K": Ks[i],
                            "alpha": a[i], "beta": b[i],
                            "alpha_fit": fit["alpha"], "beta_fit": fit["beta"],
                            "p_lead": p_lead, "rho": rho, "n_eff": n_eff,
                            "dist": float(distance(a[i], b[i])),
                        })
            print(f"  done {fam} {par}", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "full_sweep.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {len(df)} rows to {path}")
    print(df.groupby("family").dist.min())


SECTIONS = {"analytic": analytic, "sweep": sweep}

if __name__ == "__main__":
    args = sys.argv[1:] or ["analytic"]
    for i, name in enumerate(args):
        if name not in SECTIONS:
            raise SystemExit(f"unknown section {name!r}")
        if i:
            print("\n")
        SECTIONS[name]()
