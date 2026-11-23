"""Is the "beta lives between bands vs inside bands" story BINNING-INVARIANT?

THE WORRY. range_composition.py split pairs into DYADIC distance bands and found
beta_b ~ 0 for the ring (so beta is inter-band reweighting) and beta_b ~ 0.2 for
HM (so beta is intra-band). But that attribution may be an artifact of the bin
width I chose:

    one band holding every pair     ->  beta_b = beta,  0% between-band
    one band per distance           ->  for a circulant net tau_b = l(d)/s(d)
                                        is a single fixed number, so beta_b = 0
                                        EXACTLY and 100% is between-band

Same network, same beta, opposite story. So the dyadic result on its own says
nothing. What could still be real is the FINE-BIN LIMIT: refine the bins until
each holds one distance and ask whether any within-band scaling survives. For a
circulant network it cannot, by construction. For HM it can, because pairs at
equal XOR distance have different S_ij and L_ij. That asymmetry, if it holds, is
binning-invariant and is the only defensible version of the claim.

THE DECOMPOSITION. Freeze one factor at a reference level and let the other move:

    tau_weights_frozen(K) = sum_b w_b(ref) tau_b(K)     within-band growth only
    tau_taus_frozen(K)    = sum_b w_b(K) tau_b(ref)     reweighting only

with w_b = V_b/V and ref the LAST level (where every band is populated, so no
band is frozen at zero weight). beta_within and beta_reweight are their log-log
slopes, and beta_within + beta_reweight should recover beta approximately.

Run at two binnings. If beta_within/beta is stable the story is about the
network; if it collapses to zero under refinement the story was about my bins.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u band_binning_invariance.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
from range_composition import N, band_cascade, band_index, best_m, net

np.seterr(all="ignore")

KLO, KHI = 2, 64


def fine_index(n, metric):
    """One band per distance value: the finest possible binning."""
    i = np.arange(n)
    if metric == "ring":
        return np.minimum(np.abs(i[:, None] - i[None, :]),
                          n - np.abs(i[:, None] - i[None, :]))
    return i[:, None] ^ i[None, :]


def split(rows, nb):
    """beta, beta_within, beta_reweight over the fitted window."""
    sel = [r for r in rows if KLO <= r["K"] <= KHI]
    lk = np.log2([r["K"] for r in sel], dtype=float)
    ref = sel[-1]
    wref = ref["V_b"] / ref["V"]

    # bands usable at every level in the window: strictly positive variance
    live = np.ones(nb, dtype=bool)
    for r in sel:
        live &= r["V_b"] > 0
    tref = np.where(live, ref["T_b"] / np.where(live, ref["V_b"], 1.0), 0.0)
    cover = float(wref[live].sum())

    tw, tt = [], []
    for r in sel:
        tb = np.where(live, r["T_b"] / np.where(live, r["V_b"], 1.0), 0.0)
        wb = r["V_b"] / r["V"]
        tw.append(float((wref * live) @ tb))
        tt.append(float((wb * live) @ tref))
    b = np.polyfit(lk, np.log2([r["tau"] for r in sel]), 1)[0]
    bw = np.polyfit(lk, np.log2(tw), 1)[0]
    br = np.polyfit(lk, np.log2(tt), 1)[0]
    return b, bw, br, int(live.sum()), cover


def run(label, kind, par, metric):
    A = net(kind, par, 1)
    ev, V = eigh(A, check_finite=False)
    dist, lm, f = best_m(ev, V)
    S, L = lt.stationary_covariances(ev, V, 1 - lm)
    print(f"\n{label}   1-m={lm:.2e}   alpha={f['alpha']:.3f}  beta={f['beta']:.3f}")
    print(f"  {'binning':<22}{'bands':<8}{'beta':<9}{'beta_within':<14}"
          f"{'beta_reweight':<16}{'within/beta':<14}{'live':<8}{'cover'}")
    for name, B in (("dyadic (as published)", band_index(N, metric)),
                    ("one band per distance", fine_index(N, metric))):
        nb = int(B.max()) + 1
        rows = band_cascade(S, L, B, nb)
        b, bw, br, nl, cov = split(rows, nb)
        print(f"  {name:<22}{nb:<8}{b:<9.3f}{bw:<14.3f}{br:<16.3f}"
              f"{bw / b:<14.3f}{nl:<8}{cov:.3f}")


def main():
    print("beta_within is the scaling that survives when the band WEIGHTS are held")
    print("fixed; beta_reweight is what survives when the band TIMESCALES are held")
    print("fixed. 'cover' is the variance share of bands usable at every level.")
    for a in (("ring k=4", "ring", 4, "ring"),
              ("ring k=8", "ring", 8, "ring"),
              ("hm phi=3.0", "hm", 3.0, "module"),
              ("hm phi=2.6", "hm", 2.6, "module"),
              ("geo sigma=4", "geo", 4.0, "ring"),
              ("geo sigma=2", "geo", 2.0, "ring")):
        run(*a)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
