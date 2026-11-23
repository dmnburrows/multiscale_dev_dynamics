"""Gapped vs gapless, done properly: the gap is not a property of the network alone.

THE PROBLEM WITH THE SIMPLE STORY. Classifying by the spectral gap g = s_1 - s_2
predicts the distance to (1.5, 0.2) with Spearman rho = 0.75 across 36
architecture settings, but scale-free breaks it badly: the LARGEST gaps in the
whole study (0.30 to 0.61) and among the best fits (0.062 to 0.149). So "gapped
implies cannot match" is false as stated, and the classification needs fixing
before it can carry any argument.

TWO REPAIRS, both tested here.

(1) THE GAP IS ONLY MEANINGFUL RELATIVE TO 1-m.
    The modal weight is w_k = 1/(1 - m^2 s_k^2), which is large for every mode
    with 1 - s_k <~ 1 - m. So a gap g is INVISIBLE when 1 - m >> g, and
    decisive only when 1 - m << g. The operative quantity is therefore not g
    but the number of modes actually amplified at the working m,

        N(1-m) = #{k : 1 - s_k <= 1 - m}

    which makes "gapped" a property of the PAIR (network, m), not of the
    network. Every gapped network has a band-like regime, at m far enough from
    1. Prediction: the best fits of gapped families sit at larger 1-m than
    those of gapless families, and all families match at comparable N(1-m).

(2) A LOCALISED TOP MODE IS NOT A DOMINANT MODE.
    The one-mode argument (alpha -> 2 as one mode takes over) assumes the top
    eigenvector is EXTENDED, so that its overlap with a cluster of K units
    grows like K^2/n. If instead it is localised on a hub, its overlap stops
    growing once the cluster contains the hub, so gamma_top ~ 0 rather than 2
    and the mode never drives alpha to 2 however much weight it carries.
    Scale-free networks have exactly this: the top eigenvector localises on the
    largest hub. Measured by the participation ratio of the eigenvector,

        PR_1 / n = 1 / (n * sum_i v_1(i)^4)      1 = extended, ~1/n = localised

    Prediction: scale-free has a small PR_1/n and a small gamma_top, and that
    is why it escapes the penalty its gap would otherwise impose.

So the classification that matters has TWO axes, not one: how many modes are
amplified, and whether the leading one is extended. This module measures both.

    python gap_theory.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import analytical_theory as at
import spectral_classes as sc

np.seterr(all="ignore")

ORDER = ["ring", "hm", "er", "sw", "sf"]


def participation_vector(v):
    """PR/n of an eigenvector: 1 = spread over everything, ~1/n = on one node."""
    return float(1.0 / (len(v) * np.sum(v ** 4)))


def gamma_of_mode(ev, V, m, levels, idx, mode):
    """d log abar_mode / d log K between level idx and idx+1."""
    a0 = at.overlaps(levels[idx]["memb"], V)[mode]
    a1 = at.overlaps(levels[idx + 1]["memb"], V)[mode]
    return float(np.log2(a1 / a0))


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "full_sweep.csv"))
    df = df[df.K >= 4]
    best = (df.loc[df.groupby(["family", "param"]).dist.idxmin()]
              .reset_index(drop=True))

    n = 1024
    rows = []
    for _, b in best.iterrows():
        A = lt.make_network(n, b.family, b.param, int(b.seed))
        ev, V = eigh(A, check_finite=False)
        s = ev / np.max(np.abs(ev))
        order = np.argsort(s)[::-1]
        pr1 = participation_vector(V[:, order[0]])
        pr2 = participation_vector(V[:, order[1]])
        gap = float(s[order[0]] - s[order[1]])
        N_at_m = int(np.sum((1.0 - s) <= b.one_minus_m))
        # gamma of the leading-variance mode at the best-fit level
        S, L = lt.stationary_covariances(ev, V, 1 - b.one_minus_m)
        levels = at.cascade(S, L, 4)
        idx = next(i for i, d in enumerate(levels) if d["K"] == b.K)
        w, t = at.modal_weights(ev, 1 - b.one_minus_m)
        contrib = w * at.overlaps(levels[idx]["memb"], V)
        lead = int(np.argmax(contrib))
        g_lead = gamma_of_mode(ev, V, 1 - b.one_minus_m, levels, idx, lead)
        rows.append({
            "family": b.family, "param": b.param, "dist": b.dist,
            "gap": gap, "one_minus_m": b.one_minus_m,
            "ratio": b.one_minus_m / gap if gap > 0 else np.inf,
            "N_at_m": N_at_m, "M": b.M, "p_lead": b.p_lead, "rho": b.rho,
            "pr1": pr1, "pr2": pr2, "gamma_lead": g_lead,
            "lead_is_top": int(lead == order[0]),
        })
    r = pd.DataFrame(rows)

    print("=" * 110)
    print("1  AT EACH FAMILY'S BEST FIT: is the gap resolved, and is the top mode extended?")
    print("=" * 110)
    print(f"{'family':<7}{'param':<8}{'dist':<8}{'gap':<9}{'1-m*':<10}"
          f"{'(1-m)/gap':<12}{'N(1-m)':<9}{'p_lead':<9}{'gamma_lead':<12}"
          f"{'PR_1/n':<10}{'lead=top?'}")
    for fam in ORDER:
        for _, x in r[r.family == fam].sort_values("dist").iterrows():
            print(f"{x.family:<7}{x.param:<8g}{x.dist:<8.3f}{x.gap:<9.4f}"
                  f"{x.one_minus_m:<10.1e}{x.ratio:<12.3f}{x.N_at_m:<9d}"
                  f"{x.p_lead:<9.3f}{x.gamma_lead:<12.3f}{x.pr1:<10.4f}"
                  f"{x.lead_is_top}")

    print()
    print("=" * 110)
    print("2  TEST OF REPAIR (1): do good fits need the gap UNRESOLVED, (1-m) >~ gap?")
    print("=" * 110)
    good = r[r.dist < 0.08]
    bad = r[r.dist > 0.15]
    for name, d in (("best fits (d < 0.08)", good), ("worst fits (d > 0.15)", bad)):
        fin = d[np.isfinite(d.ratio)]
        print(f"  {name:<24} n={len(d):<3} "
              f"median (1-m)/gap = {fin.ratio.median():<9.3f} "
              f"median N(1-m) = {d.N_at_m.median():<7.1f} "
              f"median 1-m = {d.one_minus_m.median():.1e}")
    from scipy.stats import spearmanr
    fin = r[np.isfinite(r.ratio)]
    for col, lab in (("ratio", "(1-m)/gap"), ("N_at_m", "N(1-m)"),
                     ("gap", "gap"), ("pr1", "PR_1/n"),
                     ("gamma_lead", "gamma_lead"), ("p_lead", "p_lead")):
        d = fin if col == "ratio" else r
        rho, p = spearmanr(d[col], d.dist)
        print(f"    Spearman  dist vs {lab:<12} rho = {rho:+.3f}   p = {p:.2e}")

    print()
    print("=" * 110)
    print("3  TEST OF REPAIR (2): is the scale-free top mode localised?")
    print("=" * 110)
    print(f"{'family':<8}{'param':<8}{'PR_1/n (top mode)':<20}{'PR_2/n':<12}"
          f"{'gap':<10}{'interpretation'}")
    for fam in ORDER:
        for _, x in r[r.family == fam].iterrows():
            tag = ("localised" if x.pr1 < 0.05 else
                   "extended" if x.pr1 > 0.3 else "partly localised")
            print(f"{x.family:<8}{x.param:<8g}{x.pr1:<20.4f}{x.pr2:<12.4f}"
                  f"{x.gap:<10.4f}{tag}")

    print()
    print("=" * 110)
    print("4  THE TWO-AXIS CLASSIFICATION")
    print("=" * 110)
    print(f"  {'family':<8}{'median gap':<13}{'median PR_1/n':<16}"
          f"{'median gamma_lead':<20}{'best dist':<12}{'class'}")
    for fam in ORDER:
        d = r[r.family == fam]
        g, pr, gl, bd = (d.gap.median(), d.pr1.median(),
                         d.gamma_lead.median(), d.dist.min())
        if g < 1e-3:
            cls = "gapless band"
        elif pr < 0.05:
            cls = "gapped, localised top"
        else:
            cls = "gapped, extended top"
        print(f"  {fam:<8}{g:<13.4f}{pr:<16.4f}{gl:<20.3f}{bd:<12.4f}{cls}")
    print()
    print("  A gapped network penalises the exponents only when its top mode is")
    print("  EXTENDED, because only then does one mode drag alpha towards 2.")


if __name__ == "__main__":
    main()
