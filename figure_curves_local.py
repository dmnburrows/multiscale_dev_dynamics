"""The same curve check, for the best LOCAL-slope fits.

figure_curves.py selects the network whose FITTED (whole-cascade) exponents sit
closest to (1.5, 0.2). This one selects on the best LOCAL slope instead -- the
closest single K -> 2K step anywhere in the cascade -- and marks which step it
was.

Plotted for comparison on the same axes: the local tangent that won (heavy
segment through two points) and the OLS fit over the usual window (thin line).
Where those disagree, the local winner is a tangent to a bend rather than a
description of the curve.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import extended_networks as xn

np.seterr(all="ignore")
FIGDIR = os.path.join(HERE, "figures")
N = 512
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
HUE = {"ring": "#2a78d6", "hm": "#eb6834", "er": "#1baf7a",
       "sw": "#eda100", "sf": "#e87ba4"}
ORDER = ["ring", "hm", "er", "sw", "sf"]
LABEL = {"ring": "Ring", "hm": "Hierarchical modular",
         "er": "Erdős–Rényi", "sw": "Small-world",
         "sf": "Scale-free"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def build(fam, p1, p2):
    if fam == "ring":
        return lt.make_network(N, "ring", int(p1), 1)
    if fam == "hm":
        return lt.make_network(N, "hm", float(p1), 1)
    if fam == "er":
        return lt.make_network(N, "er", float(p1), 1)
    if fam == "sw":
        return xn.small_world(N, int(p1), float(p2), 1)
    return xn.scale_free(N, float(p1), float(p2), 1)


def best_connected(df, fam, limit=120):
    d = df[df.family == fam].sort_values("dist").head(limit)
    for _, r in d.iterrows():
        A = build(fam, r.param, r.param2)
        _, lab = connected_components(csr_matrix(A), directed=False)
        if np.bincount(lab).max() / N > 0.98:
            return r
    return None


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "matched_local.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.0), sharex=True)

    print(f"{'family':<8}{'config':<24}{'1-m':<10}{'K*':<6}"
          f"{'alpha_loc':<11}{'beta_loc':<11}{'d_loc':<9}"
          f"{'alpha_fit':<11}{'beta_fit':<11}{'d_fit'}")
    import spectral_classes as sc
    for col, fam in enumerate(ORDER):
        b = best_connected(df, fam)
        A = build(fam, b.param, b.param2)
        ev, V = eigh(A, check_finite=False)
        S, L = lt.stationary_covariances(ev, V, 1 - b.one_minus_m)
        lv = lt.coarse_grain(S, L, 4)
        f = lt.fit_scaling(lv)
        fitted = lv.iloc[1:len(lv) - 1]
        x = np.log(fitted.K.to_numpy())
        i = int(np.flatnonzero(lv.K.to_numpy() == b.K)[0])

        for row, (cname, loc, gl, lab) in enumerate(
                (("variance", "alpha_loc", "alpha", r"$V(K)$"),
                 ("tau", "beta_loc", "beta", r"$\tau(K)$"))):
            ax = axes[row, col]
            ax.plot(lv.K, lv[cname], "o", color=HUE[fam], ms=5,
                    mec="#fcfcfb", mew=0.6, zorder=3)
            # the OLS fit over the usual window
            y = np.log(fitted[cname].to_numpy())
            icpt = y.mean() - f[gl] * x.mean()
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(np.exp(xs), np.exp(icpt + f[gl] * xs), "-", color=INK2,
                    lw=1.0, zorder=4)
            # the winning local step
            kk = lv[cname].to_numpy()
            ax.plot(lv.K.to_numpy()[i:i + 2], kk[i:i + 2], "-", color=INK,
                    lw=2.6, zorder=6, solid_capstyle="round")
            ax.plot(lv.K.to_numpy()[i:i + 2], kk[i:i + 2], "o", color=INK,
                    ms=7, zorder=7)
            sym = r"\alpha" if gl == "alpha" else r"\beta"
            ax.text(0.04, 0.95,
                    f"local ${sym}$ = {b[loc]:.3f}  (K={int(b.K)})\n"
                    f"fitted ${sym}$ = {f[gl]:.3f}   $R^2$={f[gl+'_r2']:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                              boxstyle="round,pad=0.3"))
            ax.set(xscale="log", yscale="log",
                   ylabel=lab if col == 0 else "",
                   xlabel=r"cluster size $K$" if row == 1 else "")
            if row == 0:
                p2 = "" if not np.isfinite(b.param2) else f", {b.param2:g}"
                ax.set_title(f"{LABEL[fam]}\n{b.param:g}{p2}   "
                             f"1−m={b.one_minus_m:.1e}", fontsize=9)

        dfit = float(sc.distance(f["alpha"], f["beta"]))
        p2 = "" if not np.isfinite(b.param2) else f", {b.param2:g}"
        print(f"{fam:<8}{f'{b.param:g}{p2}':<24}{b.one_minus_m:<10.1e}"
              f"{int(b.K):<6}{b.alpha_loc:<11.4f}{b.beta_loc:<11.4f}"
              f"{b.dist:<9.4f}{f['alpha']:<11.4f}{f['beta']:<11.4f}{dfit:.4f}")

    fig.suptitle("Best LOCAL-slope fits (connected only).  Heavy segment = the "
                 "single K→ 2K step that won; thin line = the OLS fit over "
                 "K = 2..64 for the same network", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_curves_local.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("\n  wrote figures/fig_curves_local.png / .pdf")


if __name__ == "__main__":
    main()
