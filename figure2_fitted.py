"""Figure 2 on FITTED exponents: one slope per network, across all cluster sizes.

The earlier version plotted the two-point local slope between consecutive ICG
levels, which made cluster size a free axis and let it be cherry-picked. That
answers "is there some K where the local slope passes near (1.5, 0.2)", not
"are this network's exponents (1.5, 0.2)".

Here alpha and beta are what linear_theory.fit_scaling returns on its default
window -- a single log-log slope over the whole cascade, which is how the
empirical exponents are measured. So each (architecture parameter, m, seed)
contributes ONE point instead of one per level.

Colour is the architecture parameter, on a spectral map, as before. No iso-m
lines.

    python figure2_fitted.py             # reuse the cached HM fitted sweep
    python figure2_fitted.py --resweep   # recompute it
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import analytical_theory as at
import spectral_classes as sc
from figure2_spectral import PHI_GRID, ORDER, LABEL, PNAME, LOGSCALE, CMAP

np.seterr(all="ignore")
FIGDIR, DATA = os.path.join(HERE, "figures"), os.path.join(HERE, "sweep_data")
os.makedirs(FIGDIR, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
TARGET = (1.5, 0.2)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def sweep_hm_fitted(n=1024, seeds=(1, 2), lms=None):
    """One fitted (alpha, beta) per (Phi, seed, m), lt.fit_scaling defaults."""
    if lms is None:
        lms = np.logspace(-1, -5, 21)
    rows = []
    for phi in PHI_GRID:
        for seed in seeds:
            A = lt.make_network(n, "hm", phi, seed)
            ev, V = eigh(A, check_finite=False)
            for lm in lms:
                S, L = lt.stationary_covariances(ev, V, 1.0 - lm)
                levels = lt.coarse_grain(S, L, 4)
                f = lt.fit_scaling(levels)
                rows.append({"family": "hm", "param": phi, "seed": seed,
                             "one_minus_m": lm,
                             "alpha_fit": f["alpha"], "beta_fit": f["beta"]})
        print(f"  hm phi={phi} done", flush=True)
    return pd.DataFrame(rows)


def load(resweep=False):
    base = pd.read_csv(os.path.join(DATA, "full_sweep.csv"))
    cols = ["family", "param", "seed", "one_minus_m", "alpha_fit", "beta_fit"]
    # full_sweep repeats the fitted value on every level row: keep one per run
    other = (base[base.family != "hm"][cols]
             .drop_duplicates(["family", "param", "seed", "one_minus_m"]))
    cache = os.path.join(DATA, "hm_phi_1_to_5_fitted.csv")
    if resweep or not os.path.exists(cache):
        hm = sweep_hm_fitted()
        hm.to_csv(cache, index=False)
    else:
        hm = pd.read_csv(cache)
    df = pd.concat([other, hm[cols]], ignore_index=True)
    df = df[np.isfinite(df.alpha_fit) & np.isfinite(df.beta_fit)]
    df["dist"] = sc.distance(df.alpha_fit.values, df.beta_fit.values)
    return df


def main(resweep=False):
    df = load(resweep)
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)

    for ax, fam in zip(axes, ORDER):
        d = df[df.family == fam]
        params = np.array(sorted(d.param.unique()), float)
        norm = (LogNorm(vmin=params.min(), vmax=params.max()) if fam in LOGSCALE
                else Normalize(vmin=params.min(), vmax=params.max()))
        ax.scatter(d.alpha_fit, d.beta_fit, c=d.param.values, cmap=CMAP,
                   norm=norm, s=16, alpha=0.85, linewidths=0.2,
                   edgecolors="#fcfcfb")

        best = d.loc[d.dist.idxmin()]
        ax.plot([best.alpha_fit], [best.beta_fit], "o", mfc="none", mec=INK,
                mew=1.7, ms=12, zorder=6)
        ax.plot([TARGET[0]], [TARGET[1]], "*", color=INK, ms=14, zorder=7)
        for r in (0.1, 0.2):
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(TARGET[0] + r * sc.R_ALPHA * np.cos(th),
                    TARGET[1] + r * sc.R_BETA * np.sin(th),
                    "-", color=INK2, lw=0.7, alpha=0.6, zorder=5)
        ax.text(0.975, 0.975,
                f"best d = {best.dist:.3f}\n{PNAME[fam]} = {best.param:g},  "
                f"1−m = {best.one_minus_m:.1e}",
                transform=ax.transAxes, fontsize=6.8, color=INK, zorder=8,
                ha="right", va="top",
                bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                          boxstyle="round,pad=0.3"))
        ax.set(xlabel=r"$\alpha$   (variance scaling)",
               xlim=(0.95, 2.05), ylim=(-0.05, 0.55))
        ax.set_title(LABEL[fam], pad=8)

        cax = ax.inset_axes([0.06, -0.33, 0.88, 0.055])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                          orientation="horizontal")
        cb.set_label(PNAME[fam], fontsize=7.5, labelpad=2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2)
        if fam in LOGSCALE:
            cb.ax.set_xscale("log")
        step = max(1, len(params) // 6)
        cb.set_ticks(params[::step])
        cb.ax.set_xticklabels([f"{v:g}" for v in params[::step]])

    axes[0].set_ylabel(r"$\beta$   (timescale scaling)")
    fig.suptitle("Exponents fitted across ALL cluster sizes (one point per "
                 "network and coupling).  Star = (1.5, 0.2); rings = "
                 "range-weighted distance 0.1 and 0.2", y=1.04, fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_fitted.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig2_fitted.png / .pdf")

    print()
    print("FITTED EXPONENTS: best per parameter (minimised over m and seed)")
    for fam in ORDER:
        d = df[df.family == fam]
        print(f"\n  {LABEL[fam]}  ({PNAME[fam]})")
        print(f"  {'param':<9}{'min dist':<11}{'alpha':<9}{'beta':<9}{'1-m'}")
        for par in sorted(d.param.unique()):
            dd = d[d.param == par]
            b = dd.loc[dd.dist.idxmin()]
            print(f"  {par:<9g}{b.dist:<11.4f}{b.alpha_fit:<9.4f}"
                  f"{b.beta_fit:<9.4f}{b.one_minus_m:.1e}")
    print()
    print(f"  {'family':<8}{'best fitted distance':<24}{'at'}")
    for fam in ORDER:
        d = df[df.family == fam]
        b = d.loc[d.dist.idxmin()]
        print(f"  {fam:<8}{b.dist:<24.4f}{PNAME[fam]}={b.param:g}, "
              f"1-m={b.one_minus_m:.1e}")


if __name__ == "__main__":
    main("--resweep" in sys.argv)
