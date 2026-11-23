"""Sanity check: the actual scaling curves behind each family's best fit.

A slope is only meaningful if the level curve is straight in log-log. This
plots V(K) and tau(K) for the best CONNECTED configuration of each family,
with the fitted line and its R^2 over the fitting window.

On kurtosis: the pipeline is second-moment only -- stationary_covariances
returns Sigma and Lambda and never specifies the noise distribution beyond its
covariance. For Gaussian noise the process is exactly Gaussian, so every
cluster sum is a linear combination of Gaussians and the excess kurtosis is
identically 0 at every K. There is no kurtosis scaling in this model; the third
panel records that rather than pretending to measure it.
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


def connected(fam, p1, p2):
    A = build(fam, p1, p2)
    _, lab = connected_components(csr_matrix(A), directed=False)
    return np.bincount(lab).max() / N > 0.98


def best_connected(df, fam, limit=80):
    d = df[df.family == fam]
    cfg = (d.groupby(["param", "param2"], dropna=False).dist.min().reset_index()
             .sort_values("dist").head(limit))
    for _, r in cfg.iterrows():
        if connected(fam, r.param, r.param2):
            sub = d[(d.param == r.param) &
                    (d.param2.isna() if not np.isfinite(r.param2)
                     else d.param2 == r.param2)]
            return sub.loc[sub.dist.idxmin()]
    return None


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "matched_fitted.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.0), sharex=True)

    print(f"{'family':<8}{'config':<26}{'1-m':<10}{'alpha':<9}{'R2':<9}"
          f"{'beta':<9}{'R2':<9}{'straight?'}")
    for col, fam in enumerate(ORDER):
        b = best_connected(df, fam)
        A = build(fam, b.param, b.param2)
        ev, V = eigh(A, check_finite=False)
        S, L = lt.stationary_covariances(ev, V, 1 - b.one_minus_m)
        lv = lt.coarse_grain(S, L, 4)
        f = lt.fit_scaling(lv)
        fitted = lv.iloc[1:len(lv) - 1]          # the default fitting window
        x = np.log(fitted.K.to_numpy())

        for row, (colname, exp, lab) in enumerate(
                (("variance", "alpha", r"$V(K)$"), ("tau", "beta", r"$\tau(K)$"))):
            ax = axes[row, col]
            ax.plot(lv.K, lv[colname], "o", color=HUE[fam], ms=5,
                    mec="#fcfcfb", mew=0.6, zorder=3)
            ax.plot(fitted.K, fitted[colname], "o", color=HUE[fam], ms=6.5,
                    mec=INK, mew=0.9, zorder=4)
            y = np.log(fitted[colname].to_numpy())
            icpt = y.mean() - f[exp] * x.mean()
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(np.exp(xs), np.exp(icpt + f[exp] * xs), "-", color=INK,
                    lw=1.3, zorder=5)
            sym = r"\alpha" if exp == "alpha" else r"\beta"
            ax.text(0.04, 0.95,
                    f"${sym}$ = {f[exp]:.3f}\n$R^2$ = {f[exp + '_r2']:.4f}",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                              boxstyle="round,pad=0.3"))
            ax.set(xscale="log", yscale="log", ylabel=lab if col == 0 else "",
                   xlabel=r"cluster size $K$" if row == 1 else "")
            if row == 0:
                p2 = "" if not np.isfinite(b.param2) else f", {b.param2:g}"
                ax.set_title(f"{LABEL[fam]}\n{b.param:g}{p2}   "
                             f"1−m={b.one_minus_m:.1e}", fontsize=9)

        straight = "yes" if min(f["alpha_r2"], f["beta_r2"]) > 0.98 else "NO"
        p2 = "" if not np.isfinite(b.param2) else f", {b.param2:g}"
        print(f"{fam:<8}{f'{b.param:g}{p2}':<26}{b.one_minus_m:<10.1e}"
              f"{f['alpha']:<9.4f}{f['alpha_r2']:<9.4f}{f['beta']:<9.4f}"
              f"{f['beta_r2']:<9.4f}{straight}")

    fig.suptitle("Scaling curves behind each family's best CONNECTED fit.  "
                 "Filled-black points are the fitting window (K = 2..64); "
                 "line is the OLS fit", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_curves.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("\n  wrote figures/fig_curves.png / .pdf")


if __name__ == "__main__":
    main()
