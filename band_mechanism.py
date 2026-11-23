"""WHERE does beta live -- between range bands, or inside them?

range_composition.py established the exact decomposition of both exponents over
dyadic distance bands, and found ring and HM have opposite band profiles:
beta_b ~ 0 in every ring band, beta_b ~ 0.2 in every HM band. This script shows
the mechanism behind that, by printing the two quantities the band form is built
from, level by level:

    V_b(K)        the band's share of the cluster variance
    tau_b(K)      = T_b(K)/V_b(K), the band's own mean timescale

beta_b is by definition the slope of log2 tau_b against log2 K. So:

    tau_b flat in K            -> beta_b = 0, the band contributes no scaling
    tau_b rising with b        -> mixing bands raises the mean timescale
    tau_b rising in K          -> the band itself slows down as clusters grow

THE CLAIM UNDER TEST. The ring is circulant, so distance -> wavenumber ->
eigenvalue is a monotone map: a pair's separation very nearly determines its
timescale. A band is therefore also a timescale band, with little spread inside
it, so tau_b cannot grow within a band and all of beta must come from the
changing MIX of bands. HM has no such map -- its XOR distance says which
hierarchical level a pair's connection sits at, not how fast that pair is -- so
each band holds a wide spread of timescales and ICG can select the slow end of
it inside a fixed band.

Prediction if that is right:
    ring  tau_b rises steeply ACROSS bands, flat WITHIN each band
    HM    tau_b similar across bands, rising WITHIN each band

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u band_mechanism.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spectral_classes as sc
from range_composition import N, band_cascade, band_index, band_label, best_m, net

np.seterr(all="ignore")

CONFIGS = [
    ("ring k=4",    "ring", 4,   "ring"),
    ("ring k=8",    "ring", 8,   "ring"),
    ("hm phi=3.0",  "hm",   3.0, "module"),
    ("hm phi=2.6",  "hm",   2.6, "module"),
    ("geo sigma=4", "geo",  4.0, "ring"),
    ("geo sigma=2", "geo",  2.0, "ring"),
]
SEED = 1
NB = int(np.log2(N))  # bands 0 .. log2(N)-1 cover every dyadic distance


def table(label, kind, par, metric):
    A = net(kind, par, SEED)
    ev, V = eigh(A, check_finite=False)
    dist, lm, f = best_m(ev, V)
    S, L = lt.stationary_covariances(ev, V, 1 - lm)
    B = band_index(N, metric)
    nb = int(B.max()) + 1
    rows = band_cascade(S, L, B, nb)

    print("\n" + "=" * 112)
    print(f"{label}   metric={metric}   1-m={lm:.2e}   alpha={f['alpha']:.3f}  "
          f"beta={f['beta']:.3f}   distance to (1.5,0.2) = {dist:.3f}")
    print("=" * 112)

    Ks = [r["K"] for r in rows]
    # ---- share of variance per band per level
    print("\n  VARIANCE SHARE  V_b(K)/V(K).  '.' = band carries no variance yet")
    hdr = "  " + "band".ljust(9) + "".join(f"K={k:<9}" for k in Ks)
    print(hdr)
    for b in range(nb):
        cells = []
        for r in rows:
            s = r["V_b"][b] / r["V"]
            cells.append("   .     " if abs(s) < 1e-12 else f"{s:<9.3f}")
        print(f"  {band_label(b, metric):<9}" + "".join(cells))

    nlive = [int(np.sum(np.abs(r["V_b"]) > 1e-12)) for r in rows]
    print(f"  {'live':<9}" + "".join(f"{v:<9d}" for v in nlive))
    new = [nlive[i] - nlive[i - 1] for i in range(1, len(nlive))]
    print(f"  new bands activated per doubling: {new}   "
          f"mean {np.mean(new):+.2f}")

    # ---- band's own timescale per level
    print("\n  BAND TIMESCALE  tau_b(K) = T_b(K)/V_b(K).   beta_b is its slope in log2 K")
    print(hdr)
    tb = np.full((nb, len(rows)), np.nan)
    for b in range(nb):
        cells = []
        for j, r in enumerate(rows):
            v, t = r["V_b"][b], r["T_b"][b]
            if abs(v) < 1e-12 * abs(r["V"]):
                cells.append("   .     ")
                continue
            tb[b, j] = t / v
            cells.append(f"{t / v:<9.2f}")
        print(f"  {band_label(b, metric):<9}" + "".join(cells))
    print(f"  {'TOTAL':<9}" + "".join(f"{r['tau']:<9.2f}" for r in rows))

    # ---- the two directions, quantified
    lk = np.log2(np.array(Ks, dtype=float))
    print("\n  THE TWO DIRECTIONS")
    print(f"  {'band':<9}{'share@K=64':<13}{'tau_b@K=64':<13}"
          f"{'beta_b = d log2 tau_b / d log2 K':<34}{'levels'}")
    j64 = Ks.index(64) if 64 in Ks else len(Ks) - 1
    for b in range(nb):
        ok = np.isfinite(tb[b]) & (tb[b] > 0)
        if ok.sum() < 2:
            continue
        sl = np.polyfit(lk[ok], np.log2(tb[b][ok]), 1)[0]
        sh = rows[j64]["V_b"][b] / rows[j64]["V"]
        t64 = tb[b, j64]
        print(f"  {band_label(b, metric):<9}{sh:<13.3f}"
              f"{t64 if np.isfinite(t64) else np.nan:<13.2f}{sl:<34.3f}{int(ok.sum())}")

    # across-band spread of timescales at a fixed level, variance weighted
    for jj in (j64,):
        v = np.array([rows[jj]["V_b"][b] for b in range(nb)])
        t = tb[:, jj]
        ok = np.isfinite(t) & (v > 0)
        w = v[ok] / v[ok].sum()
        mu = float((w * t[ok]).sum())
        sd = float(np.sqrt((w * (t[ok] - mu) ** 2).sum()))
        print(f"\n  across bands at K={Ks[jj]}: tau_b = {mu:.2f} +- {sd:.2f}  "
              f"(CV {sd / mu:.3f}) over {int(ok.sum())} bands carrying variance")
        print(f"  ratio longest-range tau_b / self tau_b = "
              f"{t[ok][-1] / t[ok][0]:.2f}")


def main():
    print("Does the pair's RANGE determine its TIMESCALE?")
    print("If yes, bands are timescale bands: beta_b must be ~0 and all of beta")
    print("comes from re-mixing bands. If no, beta can be generated inside a band.")
    for label, kind, par, metric in CONFIGS:
        table(label, kind, par, metric)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
