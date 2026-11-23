"""When does m act through a single variable, and when does it not?

The collapse alpha = A(x), beta = B(x) with x = K M(m)/n is exact for the ring
and the deterministic hierarchy and fails for the stochastic families. This
module tests the two candidate reasons, both measurable:

  GAP.  A gapless spectrum (s_2 -> s_1 as n grows) has a band of modes just
        below the top, so raising m widens a continuum and there is a genuine
        correlation length. A GAPPED spectrum has one isolated top mode, so
        raising m just transfers variance to that single mode. There is no
        diverging length, and M(m) saturates at 1 instead of tracking a scale.

  PARTITION DRIFT.  The collapse assumes the overlaps abar_k(K) are fixed while
        m moves. Greedy ICG chooses the partition FROM Sigma(m), so if the
        clusters reorganise as m changes, abar moves too and no single variable
        can absorb m.

Prediction: collapse quality tracks (small gap) AND (stable partition), and the
stochastic families fail on both.

    python why_collapse_fails.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linear_theory as lt
import analytical_theory as at
import universal_variable as uv

np.seterr(all="ignore")

FAMILIES = uv.FAMILIES
SEEDS = [1, 2, 3]
LMS = [10.0 ** -e for e in np.arange(1.0, 5.01, 0.25)]


def gap(s):
    """s_1 - s_2 on the normalised spectrum (0 = gapless)."""
    srt = np.sort(s)[::-1]
    return float(srt[0] - srt[1])


def partition_stability(ev, V, n, K_probe=32):
    """Mean pairwise cluster agreement at K_probe across the m range."""
    membs = []
    for lm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        for d in at.cascade(S, L, 4):
            if d["K"] == K_probe:
                membs.append(d["memb"])
                break
    if len(membs) < 2:
        return np.nan
    scores = []
    for i in range(len(membs)):
        for j in range(i + 1, len(membs)):
            a = membs[i][:, None] == membs[i][None, :]
            b = membs[j][:, None] == membs[j][None, :]
            scores.append(float((a == b).mean()))
    return float(np.mean(scores))


def main():
    n = 2048
    print("=" * 104)
    print("WHY THE COLLAPSE HOLDS FOR SOME FAMILIES AND NOT OTHERS   n = 2048")
    print("=" * 104)
    print(f"{'family':<10}{'gap s1-s2':<13}{'M(1e-1)/n':<13}{'M(1e-5)/n':<13}"
          f"{'M ratio':<11}{'partition stab':<17}{'collapse alpha':<16}{'beta'}")
    rows = []
    for label, fam, par in FAMILIES:
        gaps, Ms, stabs = [], [], []
        for seed in SEEDS:
            A = lt.make_network(n, fam, par, seed)
            ev, V = eigh(A, check_finite=False)
            s = ev / np.max(np.abs(ev))
            gaps.append(gap(s))
            Ms.append((uv.participation(s, 0.9) / n,
                       uv.participation(s, 1 - 1e-5) / n))
            stabs.append(partition_stability(ev, V, n))
        df = uv.sweep(n, fam, par, SEEDS, LMS)
        qa, qb, _ = uv.collapse_quality(df, "x")
        m_lo = float(np.mean([x[0] for x in Ms]))
        m_hi = float(np.mean([x[1] for x in Ms]))
        rows.append({"family": label, "gap": float(np.mean(gaps)),
                     "stab": float(np.mean(stabs)), "qa": qa, "qb": qb,
                     "m_ratio": m_lo / m_hi})
        print(f"{label:<10}{np.mean(gaps):<13.5f}{m_lo:<13.5f}{m_hi:<13.6f}"
              f"{m_lo/m_hi:<11.1f}{np.mean(stabs):<17.4f}{qa:<16.4f}{qb:.4f}")

    d = pd.DataFrame(rows)
    print()
    print("  Correlations across families (n = 6, so indicative only):")
    for col, name in (("gap", "spectral gap"), ("stab", "partition stability"),
                      ("m_ratio", "dynamic range of M")):
        ca = np.corrcoef(d[col], d.qa)[0, 1]
        cb = np.corrcoef(d[col], d.qb)[0, 1]
        print(f"    {name:<24} vs collapse alpha {ca:+.3f}   vs beta {cb:+.3f}")

    print()
    print("=" * 104)
    print("DOES M(m) SATURATE? the number of amplified modes as m -> 1")
    print("=" * 104)
    print(f"{'family':<10}" + "".join(f"{f'1-m={v:.0e}':>14}"
                                      for v in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)))
    for label, fam, par in FAMILIES:
        A = lt.make_network(n, fam, par, 1)
        ev = eigh(A, eigvals_only=True, check_finite=False)
        s = ev / np.max(np.abs(ev))
        vals = [uv.participation(s, 1 - lm) for lm in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)]
        print(f"{label:<10}" + "".join(f"{v:>14.3f}" for v in vals))
    print()
    print("  A family whose M falls to ~1 has put all its variance in one mode:")
    print("  there is no length scale left for m to set, so no single variable")
    print("  can carry the m-dependence.")


if __name__ == "__main__":
    main()
