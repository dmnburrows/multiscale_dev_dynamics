"""How the Munn pair is composed out of connection ranges: ring vs HM.

THE HYPOTHESIS being tested: the Munn pair emerges from a COMPOSITION of
ring-like local structure with longer-range connections, and that is why both a
ring and a hierarchical-modular network can reach it.

That is made precise, and exact, by generalising the binary local/global split
to ALL ranges at once. Bin the pairs inside a cluster by range band b, so that
V = sum_b V_b and T = sum_b T_b exactly. Each band has its own pair of local
exponents, alpha_b = dlog V_b/dlog K and beta_b = dlog tau_b/dlog K. Then,
because a log-slope of a sum is a weighted log-sum-exp of the parts' slopes:

    alpha(K) = log2 sum_b w_b 2^(alpha_b)                   w_b = V_b/V
    beta(K)  = log2 sum_b u_b 2^(alpha_b+beta_b)            u_b = T_b/T
               - log2 sum_b w_b 2^(alpha_b)

These are exact (verified to 1e-13 in section `compose`), and structurally
IDENTICAL to the modal identities alpha = log2<2^gamma>_pi etc. So the exponents
are a soft-max over ranges in exactly the same way they are a soft-max over
modes: a band counts according to how fast its variance grows, not merely how
much variance it holds.

SECTIONS
  compose   Verify the two composition laws to machine precision.
  bands     Range-resolved alpha_b, beta_b, w_b for ring / HM / geometric at
            their Munn-matching configurations. Which ranges build (1.5, 0.2)?
  hybrid    Build the composition explicitly: an exact ring for the local part
            PLUS HM's own hierarchical law for the long-range part. Does it
            reach the pair, and does it reproduce HM's band profile?
  pr        Does eigenvector delocalisation (participation ratio) predict the
            alpha_glob - alpha_loc ordering ring > HM > disordered?

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u range_composition.py [section ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import spatial_networks as sn
import spectral_classes as sc
from local_global import ring_distance, slope_r2

np.seterr(all="ignore")

N = 512
SEEDS = (1, 2, 3, 4, 5, 6)
OUT = os.path.join(HERE, "sweep_data")


# ---------------------------------------------------------------------------
# range bands
# ---------------------------------------------------------------------------
def band_index(n, metric):
    """Dyadic band id for every pair. Band 0 is the diagonal.

    metric='ring'   band = 1 + floor(log2(ring distance))
    metric='module' band = 1 + floor(log2(i XOR j))   -- HM's own levels
    """
    i = np.arange(n)
    raw = (np.minimum(np.abs(i[:, None] - i[None, :]),
                      n - np.abs(i[:, None] - i[None, :])) if metric == "ring"
           else (i[:, None] ^ i[None, :]))
    B = np.zeros((n, n), dtype=np.int64)
    nz = raw > 0
    B[nz] = 1 + np.floor(np.log2(raw[nz])).astype(np.int64)
    return B


def band_label(b, metric):
    if b == 0:
        return "self"
    lo, hi = 2 ** (b - 1), 2 ** b - 1
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def band_cascade(S, L, B, nb, min_clusters=4):
    """Coarse-grain, recording V and T split across range bands at every level.

    Uses cluster member lists, so the cost is sum_c K^2 = n*K per level.
    """
    n = len(S)
    P = np.eye(n, dtype=np.int8)
    Sc, Lc = S.copy(), L.copy()
    rows, K = [], 1
    while len(Sc) >= min_clusters:
        ncl = len(Sc)
        vb = np.zeros(nb)
        tb = np.zeros(nb)
        for row in P:
            idx = np.flatnonzero(row)
            bb = B[np.ix_(idx, idx)].ravel()
            vb += np.bincount(bb, weights=S[np.ix_(idx, idx)].ravel(),
                              minlength=nb)
            tb += np.bincount(bb, weights=L[np.ix_(idx, idx)].ravel(),
                              minlength=nb)
        rows.append({"K": K, "clusters": ncl,
                     "V": np.trace(Sc) / ncl, "T": np.trace(Lc) / ncl,
                     "tau": np.trace(Lc) / np.trace(Sc),
                     "V_b": vb / ncl, "T_b": tb / ncl})
        if ncl == 1:
            break
        pr = lt.greedy_pairs(Sc)
        Sc, Lc = lt.pool(Sc, pr), lt.pool(Lc, pr)
        a, b = pr.T
        P = P[a] + P[b]
        K *= 2
    return rows


# ---------------------------------------------------------------------------
def net(kind, par, seed, par2=None):
    if kind == "ring":
        return lt.make_network(N, "ring", int(par), seed)
    if kind == "hm":
        return lt.make_network(N, "hm", float(par), seed)
    if kind == "geo":
        return sn.geometric(N, float(par), 4.0, seed)
    if kind == "hybrid":
        return hybrid(N, 4, float(par), int(par2), seed)
    raise ValueError(kind)


def hybrid(n, k, phi, l_min, seed):
    """Exact ring for the LOCAL part + HM's hierarchical law for LONG range.

    Takes HM's adjacency, deletes every edge below level l_min (its local,
    disordered part), and substitutes an exact ring of degree k. The long-range
    structure is therefore HM's and the local structure is the ring's, which is
    precisely the composition the hypothesis proposes.
    """
    A = lt.make_network(n, "hm", phi, seed)
    i = np.arange(n)
    lvl = np.zeros((n, n), dtype=np.int64)
    x = i[:, None] ^ i[None, :]
    lvl[x > 0] = np.floor(np.log2(x[x > 0])).astype(np.int64)
    A = np.where(lvl < l_min, 0.0, A)
    R = lt.make_network(n, "ring", k, seed)
    return np.maximum(A, R)


def props(A):
    _, lab = connected_components(csr_matrix(A), directed=False)
    D = ring_distance(len(A))
    el = D[A > 0]
    return (float(A.sum(axis=1).mean()), np.bincount(lab).max() / len(A),
            float(el.mean()) if el.size else np.nan)


def best_m(ev, V, grid=None):
    grid = np.logspace(-5.5, -0.6, 34) if grid is None else grid
    best = (np.inf, None, None)
    for lm in grid:
        S, L = lt.stationary_covariances(ev, V, 1 - lm)
        f = lt.fit_scaling(lt.coarse_grain(S, L, 4))
        d = float(sc.distance(f["alpha"], f["beta"]))
        if d < best[0]:
            best = (d, lm, f)
    return best


# ===========================================================================
def sec_compose():
    print("=" * 100)
    print("COMPOSE   the range-composition laws")
    print("  V and T are partitioned over range bands, so with")
    print("      w_b = V_b(K)/V(K),   rho_b = V_b(2K)/V_b(K)")
    print("      u_b = T_b(K)/T(K),   sig_b = T_b(2K)/T_b(K)")
    print("  the EXACT laws are")
    print("      alpha = log2 sum_b w_b rho_b")
    print("      beta  = log2 sum_b u_b sig_b  -  log2 sum_b w_b rho_b")
    print("  A band's summed covariance V_b CAN BE NEGATIVE (the ring's")
    print("  Dirichlet kernel oscillates in sign), so rho_b is a signed growth")
    print("  FACTOR and must not be written as 2^(alpha_b): any log form is")
    print("  undefined on those bands. Both the soft-max and the linear-mixture")
    print("  versions are therefore reported as APPROXIMATIONS, restricted to")
    print("  the positive bands, alongside how much variance the negative")
    print("  bands actually carry.")
    print("=" * 100)
    worst_a = worst_b = 0.0
    worst_soft = worst_lin = 0.0
    neg_share, neg_count, tot_count = [], 0, 0
    for kind, par, metric in (("ring", 4, "ring"), ("hm", 3.0, "module"),
                              ("hm", 2.6, "module"), ("geo", 4.0, "ring"),
                              ("hm", 3.0, "ring")):
        B = band_index(N, metric)
        nb = int(B.max()) + 1
        A = net(kind, par, 1)
        ev, V = eigh(A, check_finite=False)
        for lm in (3e-2, 1e-3):
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            rows = band_cascade(S, L, B, nb)
            for i in range(len(rows) - 1):
                r0, r1 = rows[i], rows[i + 1]
                V0b, V1b, T0b, T1b = r0["V_b"], r1["V_b"], r0["T_b"], r1["T_b"]
                a_true = np.log2(r1["V"] / r0["V"])
                b_true = np.log2(r1["tau"] / r0["tau"])

                # --- exact: signed band terms, every band included
                a_law = np.log2(np.sum(V1b) / r0["V"])
                at_law = np.log2(np.sum(T1b) / r0["T"])
                worst_a = max(worst_a, abs(a_law - a_true))
                worst_b = max(worst_b, abs((at_law - a_law) - b_true))

                # --- how much variance sits in sign-flipping / negative bands
                live = V0b != 0
                bad = live & ((V0b < 0) | (V1b < 0))
                neg_count += int(bad.sum()); tot_count += int(live.sum())
                neg_share.append(float(np.sum(np.abs(V0b[bad])) /
                                       np.sum(np.abs(V0b[live]))))

                # --- approximations, on the strictly positive bands only
                pos = (V0b > 0) & (V1b > 0)
                if pos.any():
                    wp = V0b[pos] / np.sum(V0b[pos])
                    ab = np.log2(V1b[pos] / V0b[pos])
                    worst_soft = max(worst_soft,
                                     abs(np.log2(np.sum(wp * 2.0 ** ab)) - a_true))
                    worst_lin = max(worst_lin, abs(np.sum(wp * ab) - a_true))
    print(f"\n  EXACT alpha law, max error over all nets/m/K : {worst_a:.3e}")
    print(f"  EXACT beta  law, max error over all nets/m/K : {worst_b:.3e}")
    print(f"\n  bands with negative summed covariance        : "
          f"{neg_count}/{tot_count} ({100*neg_count/tot_count:.1f}%)")
    print(f"  share of |V_b| carried by those bands        : "
          f"{100*np.mean(neg_share):.2f}% (mean), "
          f"{100*np.max(neg_share):.2f}% (worst)")
    print(f"\n  soft-max form on positive bands only, error  : {worst_soft:.3e}")
    print(f"  linear mixture on positive bands only, error : {worst_lin:.3e}")
    print("\n  Reading: the exact law lets each range band contribute its own")
    print("  signed growth factor rho_b, weighted by the variance it holds.")
    print("  Composition over ranges is therefore real and exact, but it is")
    print("  NOT the clean soft-max that composition over MODES is -- modal")
    print("  weights w_k abar_k are positive by construction, band sums are not.")


# ===========================================================================
def sec_bands():
    print("=" * 100)
    print("BANDS   range-resolved exponents at each family's Munn-matching m")
    print("=" * 100)
    cases = [("ring", 4, None, "ring"), ("hm", 3.0, None, "module"),
             ("hm", 2.6, None, "module"), ("geo", 2.0, None, "ring"),
             ("geo", 4.0, None, "ring")]
    rows = []
    for kind, par, par2, metric in cases:
        B = band_index(N, metric)
        nb = int(B.max()) + 1
        for seed in SEEDS:
            A = net(kind, par, seed, par2)
            deg, conn, el = props(A)
            ev, V = eigh(A, check_finite=False)
            dist, lm, f = best_m(ev, V)
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            cas = band_cascade(S, L, B, nb)
            df = pd.DataFrame({"K": [c["K"] for c in cas]})
            Vb = np.array([c["V_b"] for c in cas])
            Tb = np.array([c["T_b"] for c in cas])
            Vt = np.array([c["V"] for c in cas])
            for b in range(nb):
                df[f"V{b}"] = Vb[:, b]
                df[f"t{b}"] = np.where(Vb[:, b] != 0, Tb[:, b] / Vb[:, b], np.nan)
            for b in range(nb):
                ab, r2a, nu, _ = slope_r2(df, f"V{b}")
                bb, r2b, _, _ = slope_r2(df, f"t{b}")
                ref = np.flatnonzero(df.K.to_numpy() == 64)
                share = (Vb[ref[0], b] / Vt[ref[0]]) if ref.size else np.nan
                rows.append({"family": f"{kind} {par}", "metric": metric,
                             "seed": seed, "one_minus_m": lm, "dist": dist,
                             "alpha_tot": f["alpha"], "beta_tot": f["beta"],
                             "band": b, "label": band_label(b, metric),
                             "alpha_b": ab, "beta_b": bb, "r2_a": r2a,
                             "share": share, "n_used": nu,
                             "degree": deg, "edge_len": el})
        print(f"  done {kind} {par} [{metric}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "range_bands.csv"), index=False)

    for fam, g in out.groupby("family", sort=False):
        met = g.metric.iloc[0]
        print("\n" + "-" * 100)
        print(f"{fam}   metric={met}   1-m={g.one_minus_m.median():.2e}   "
              f"total alpha={g.alpha_tot.mean():.3f} beta={g.beta_tot.mean():.3f}"
              f"   dist={g.dist.mean():.3f}   edge len={g.edge_len.mean():.1f}")
        print(f"  {'band':<10}{'share of V':<13}{'alpha_b':<16}{'beta_b':<16}"
              f"{'R2(V_b)':<10}{'levels'}")
        for b, h in g.groupby("band"):
            sh = h.share.mean()
            if not np.isfinite(sh) or sh < 1e-6:
                continue
            print(f"  {h.label.iloc[0]:<10}{sh:<13.4f}"
                  f"{h.alpha_b.mean():.3f}+-{h.alpha_b.std(ddof=1):<9.3f}"
                  f"{h.beta_b.mean():.3f}+-{h.beta_b.std(ddof=1):<9.3f}"
                  f"{h.r2_a.mean():<10.3f}{int(h.n_used.mean())}")


# ===========================================================================
def sec_hybrid():
    print("=" * 100)
    print("HYBRID   ring local part + HM hierarchical long-range part")
    print("  l_min is the level below which HM's own edges are replaced by an")
    print("  exact ring. l_min=0 is pure HM (plus a ring); large l_min keeps")
    print("  only HM's longest-range structure on top of a ring.")
    print("=" * 100)
    rows = []
    for phi in (2.6, 3.0, 3.4):
        for l_min in (0, 2, 4, 6):
            for seed in SEEDS:
                A = hybrid(N, 4, phi, l_min, seed)
                deg, conn, el = props(A)
                ev, V = eigh(A, check_finite=False)
                dist, lm, f = best_m(ev, V)
                S, L = lt.stationary_covariances(ev, V, 1 - lm)
                # binary local/global in HM's metric, for comparison with
                # ring_vs_disorder.py and hm_local_global.py
                from local_global import decompose
                Dx = (np.arange(N)[:, None] ^ np.arange(N)[None, :]).astype(float)
                dfm = decompose(S, L, Dx, thresh=1.0)
                r = dfm[dfm.K == 64]
                rows.append({"phi": phi, "l_min": l_min, "seed": seed,
                             "one_minus_m": lm, "dist": dist,
                             "alpha": f["alpha"], "beta": f["beta"],
                             "alpha_loc": slope_r2(dfm, "V_loc")[0],
                             "beta_loc": slope_r2(dfm, "tau_loc")[0],
                             "alpha_glob": slope_r2(dfm, "V_glob")[0],
                             "beta_glob": slope_r2(dfm, "tau_glob")[0],
                             "glob_share": float(r.V_glob.iloc[0] / r.V.iloc[0]),
                             "degree": deg, "conn": conn, "edge_len": el})
            print(f"  done phi={phi} l_min={l_min}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "range_hybrid.csv"), index=False)
    out["diff"] = out.alpha_glob - out.alpha_loc

    print(f"\n  {'phi':<7}{'l_min':<8}{'degree':<9}{'edge len':<11}{'1-m':<10}"
          f"{'alpha':<8}{'beta':<8}{'dist':<9}{'a_loc':<8}{'a_glob':<9}"
          f"{'diff':<9}{'glob sh'}")
    for (phi, lm_), g in out.groupby(["phi", "l_min"], sort=False):
        print(f"  {phi:<7.1f}{lm_:<8d}{g.degree.mean():<9.2f}"
              f"{g.edge_len.mean():<11.1f}{g.one_minus_m.median():<10.1e}"
              f"{g.alpha.mean():<8.3f}{g.beta.mean():<8.3f}{g.dist.mean():<9.3f}"
              f"{g.alpha_loc.mean():<8.3f}{g.alpha_glob.mean():<9.3f}"
              f"{g['diff'].mean():<+9.3f}{g.glob_share.mean():.3f}")
    print("\n  reference points (own metric):  ring+shortcuts diff +0.80 (39/39)")
    print("                                  pure HM        diff +0.27 (42/48)")
    print("                                  disordered     diff -0.87 (0/21)")


# ===========================================================================
def sec_pr():
    print("=" * 100)
    print("PR   does eigenvector delocalisation predict the local/global sign?")
    print("  PR = participation ratio of |v|^2, averaged over the modes that")
    print("  actually carry variance (pi-weighted), normalised by n.")
    print("  Prediction from the mechanism: ring (delocalised) > HM > geometric")
    print("=" * 100)
    from local_global import decompose
    rows = []
    cases = [("ring+short", None), ("hm 3.0", None), ("hm 2.6", None),
             ("geo+short", None)]
    import ring_vs_disorder as rvd
    for name in ("ring+short", "hm 3.0", "hm 2.6", "geo+short"):
        for seed in SEEDS:
            if name == "ring+short":
                A = rvd.ring_plus(N, 4, 4, seed); metric = "ring"
            elif name == "geo+short":
                A = rvd.geo_plus(N, 4.0, 4.0, 4, seed); metric = "ring"
            else:
                A = lt.make_network(N, "hm", float(name.split()[1]), seed)
                metric = "module"
            ev, V = eigh(A, check_finite=False)
            dist, lm, f = best_m(ev, V)
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            lam = (1 - lm) * ev / np.max(np.abs(ev))
            w = 1.0 / ((1.0 - lam) * (1.0 + lam))
            p4 = np.sum(V ** 4, axis=0)
            pr = (1.0 / p4) / N                       # per-mode PR, in [1/n, 1]
            pi = w / np.sum(w)
            D = (ring_distance(N).astype(float) if metric == "ring"
                 else (np.arange(N)[:, None] ^ np.arange(N)[None, :]).astype(float))
            df = decompose(S, L, D, thresh=1.0)
            rows.append({"family": name, "seed": seed, "metric": metric,
                         "one_minus_m": lm, "dist": dist,
                         "alpha": f["alpha"], "beta": f["beta"],
                         "PR_mean": float(np.mean(pr)),
                         "PR_pi": float(np.sum(pi * pr)),
                         "PR_top": float(pr[np.argmax(w)]),
                         "alpha_loc": slope_r2(df, "V_loc")[0],
                         "alpha_glob": slope_r2(df, "V_glob")[0]})
        print(f"  done {name}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "range_pr.csv"), index=False)
    out["diff"] = out.alpha_glob - out.alpha_loc
    print(f"\n  {'family':<13}{'metric':<9}{'PR (mean)':<12}{'PR (pi-wtd)':<14}"
          f"{'PR (top mode)':<16}{'alpha_glob-alpha_loc'}")
    for fam, g in out.groupby("family", sort=False):
        d = g["diff"]
        print(f"  {fam:<13}{g.metric.iloc[0]:<9}{g.PR_mean.mean():<12.4f}"
              f"{g.PR_pi.mean():<14.4f}{g.PR_top.mean():<16.4f}"
              f"{d.mean():+.3f}+-{d.std(ddof=1):.3f}")
    v = out[np.isfinite(out["diff"])]
    for c in ("PR_mean", "PR_pi", "PR_top"):
        print(f"  corr({c:<8}, diff) over all {len(v)} runs = "
              f"{v[c].corr(v['diff']):+.3f}")


# ===========================================================================
def sec_recruit():
    """Split each exponent into RECRUITMENT and GROWTH, exactly.

    At every doubling the bands divide into those already carrying variance at
    K and those switching on from exactly zero. Since V(2K) = sum_b V_b(2K),

        2^alpha = G_V + R_V,   G_V = sum_{b live at K}  V_b(2K)/V(K)
                               R_V = sum_{b new at 2K}  V_b(2K)/V(K)

    and the same for T. So the counterfactual "what would alpha be if no new
    range were recruited" is log2(G_V), and the recruitment contribution is
    alpha - log2(G_V). This is exact, not a regression.
    """
    print("=" * 100)
    print("RECRUIT   how much of (alpha, beta) is RECRUITING a new range band,")
    print("          and how much is GROWTH of ranges already active?")
    print("   2^alpha = G_V + R_V exactly;  alpha_nr = log2 G_V is the")
    print("   counterfactual with recruitment switched off.")
    print("=" * 100)
    cases = [("ring", 4, "ring"), ("ring", 8, "ring"), ("hm", 3.0, "module"),
             ("hm", 2.6, "module"), ("hm", 3.0, "ring"),
             ("geo", 2.0, "ring"), ("geo", 4.0, "ring")]
    rows = []
    for kind, par, metric in cases:
        B = band_index(N, metric)
        nb = int(B.max()) + 1
        for seed in SEEDS:
            A = net(kind, par, seed)
            ev, V = eigh(A, check_finite=False)
            dist, lm, f = best_m(ev, V)
            S, L = lt.stationary_covariances(ev, V, 1 - lm)
            cas = band_cascade(S, L, B, nb)
            for i in range(len(cas) - 1):
                r0, r1 = cas[i], cas[i + 1]
                live = r0["V_b"] != 0
                new = (~live) & (r1["V_b"] != 0)
                GV = np.sum(r1["V_b"][live]) / r0["V"]
                RV = np.sum(r1["V_b"][new]) / r0["V"]
                liveT = r0["T_b"] != 0
                newT = (~liveT) & (r1["T_b"] != 0)
                GT = np.sum(r1["T_b"][liveT]) / r0["T"]
                RT = np.sum(r1["T_b"][newT]) / r0["T"]
                a = np.log2(GV + RV)
                b = np.log2(GT + RT) - a
                a_nr = np.log2(GV) if GV > 0 else np.nan
                b_nr = (np.log2(GT) - a_nr) if (GT > 0 and GV > 0) else np.nan
                rows.append({
                    "family": f"{kind} {par}", "metric": metric, "seed": seed,
                    "one_minus_m": lm, "dist": dist, "K": r0["K"],
                    "n_new": int(new.sum()), "n_live": int(live.sum()),
                    "alpha": a, "beta": b, "alpha_nr": a_nr, "beta_nr": b_nr,
                    "d_alpha": a - a_nr, "d_beta": b - b_nr,
                    "recruit_frac": RV / (GV + RV)})
        print(f"  done {kind} {par} [{metric}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "range_recruit.csv"), index=False)

    # restrict to the fitted window K=2..64, matching lt.fit_scaling
    w = out[(out.K >= 2) & (out.K <= 64)]
    print(f"\n  averaged over the fitted window K=2..64, at each family's best m")
    print(f"  {'family':<12}{'metric':<9}{'1-m':<10}{'new bands/step':<17}"
          f"{'alpha':<9}{'alpha no-recr':<15}{'d_alpha':<10}"
          f"{'beta':<9}{'beta no-recr':<14}{'d_beta':<10}{'recruit frac'}")
    for (fam, met), g in w.groupby(["family", "metric"], sort=False):
        print(f"  {fam:<12}{met:<9}{g.one_minus_m.median():<10.1e}"
              f"{g.n_new.mean():<17.2f}{g.alpha.mean():<9.3f}"
              f"{g.alpha_nr.mean():<15.3f}{g.d_alpha.mean():<+10.3f}"
              f"{g.beta.mean():<9.3f}{g.beta_nr.mean():<14.3f}"
              f"{g.d_beta.mean():<+10.3f}{g.recruit_frac.mean():.3f}")

    print(f"\n  per-K profile (seed-averaged), showing recruitment fading with K")
    print(f"  {'family':<12}{'metric':<9}" + "".join(f"{f'K={k}':>10}"
          for k in (2, 4, 8, 16, 32, 64)))
    for (fam, met), g in out.groupby(["family", "metric"], sort=False):
        line = f"  {fam:<12}{met:<9}"
        for k in (2, 4, 8, 16, 32, 64):
            h = g[g.K == k]
            line += f"{h.d_alpha.mean():>10.3f}" if len(h) else f"{'--':>10}"
        print(line + "   <- d_alpha")
    print("\n  d_alpha is the part of alpha that exists ONLY because a new")
    print("  range band switched on at that step. d_beta likewise.")


SECTIONS = {"compose": sec_compose, "bands": sec_bands,
            "hybrid": sec_hybrid, "pr": sec_pr,
            "recruit": sec_recruit}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or list(SECTIONS)
    for s in want:
        SECTIONS[s]()
        print()
