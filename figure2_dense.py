"""Figure 2, dense parameter grid, fitted exponents, small markers.

Reads sweep_data/dense_fitted.csv (sweep_dense.py). alpha and beta are a single
log-log slope per network across all cluster sizes -- no local slopes, no
cluster size as a free axis.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectral_classes as sc
from figure2_spectral import ORDER, LABEL, PNAME, LOGSCALE

np.seterr(all="ignore")
FIGDIR = os.path.join(HERE, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
TARGET = (1.5, 0.2)
CMAP = plt.get_cmap("coolwarm")
A_MIN, A_MAX = 0.12, 1.0                 # opacity at m = 0.8 and at m = 0.9999


def opacity(one_minus_m, lo, hi):
    """Opacity from closeness to instability: darker as m approaches 1.

    Linear in log(1 - m), so each decade closer to criticality is an equal
    step in opacity.
    """
    t = (np.log10(one_minus_m) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return A_MAX + (A_MIN - A_MAX) * np.clip(t, 0, 1)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "dense_fitted.csv"))
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)

    for ax, fam in zip(axes, ORDER):
        d = df[df.family == fam]
        params = np.array(sorted(d.param.unique()), float)
        norm = (LogNorm(vmin=params.min(), vmax=params.max()) if fam in LOGSCALE
                else Normalize(vmin=params.min(), vmax=params.max()))
        # colour = architecture parameter, opacity = closeness to criticality.
        # Built as explicit RGBA so the alpha channel can vary per point; the
        # most critical points are drawn last so they sit on top.
        lo, hi = df.one_minus_m.min(), df.one_minus_m.max()
        o = d.sort_values("one_minus_m", ascending=False)
        rgba = CMAP(norm(o.param.values))
        rgba[:, 3] = opacity(o.one_minus_m.values, lo, hi)
        ax.scatter(o.alpha_fit, o.beta_fit, c=rgba, s=2.6, linewidths=0,
                   rasterized=True)

        best = d.loc[d.dist.idxmin()]
        ax.plot([best.alpha_fit], [best.beta_fit], "o", mfc="none", mec=INK,
                mew=1.5, ms=11, zorder=6)
        ax.plot([TARGET[0]], [TARGET[1]], "*", color=INK, ms=13, zorder=7)
        for r in (0.1, 0.2):
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(TARGET[0] + r * sc.R_ALPHA * np.cos(th),
                    TARGET[1] + r * sc.R_BETA * np.sin(th),
                    "-", color=INK2, lw=0.7, alpha=0.6, zorder=5)
        ax.text(0.975, 0.975,
                f"best d = {best.dist:.3f}\n{PNAME[fam]} = {best.param:g},  "
                f"1−m = {best.one_minus_m:.1e}\n{len(params)} values, "
                f"{len(d)} points",
                transform=ax.transAxes, fontsize=6.5, color=INK, zorder=8,
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

    axes[0].set_ylabel(r"$\beta$   (timescale scaling)")

    # key for the opacity encoding
    lo, hi = df.one_minus_m.min(), df.one_minus_m.max()
    for i, mv in enumerate([0.8, 0.99, 0.9999]):
        a = float(opacity(np.array([1 - mv]), lo, hi)[0])
        axes[0].plot([0.0], [0.0], "o", color=INK, alpha=a, ms=5,
                     label=f"m = {mv:g}")
    axes[0].legend(loc="lower left", fontsize=6.3, handletextpad=0.3,
                   borderpad=0.4, labelspacing=0.3,
                   title="opacity", title_fontsize=6.3)

    fig.suptitle("Exponents fitted across ALL cluster sizes.  Colour = "
                 "architecture parameter; opacity = closeness to criticality "
                 "(darker is nearer).  Star = (1.5, 0.2)", y=1.04, fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_dense.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig2_dense.png / .pdf")

    print()
    print(f"  {'family':<8}{'param values':<15}{'points':<10}"
          f"{'best distance':<16}{'at'}")
    for fam in ORDER:
        d = df[df.family == fam]
        b = d.loc[d.dist.idxmin()]
        print(f"  {fam:<8}{d.param.nunique():<15}{len(d):<10}{b.dist:<16.4f}"
              f"{PNAME[fam]}={b.param:g}, 1-m={b.one_minus_m:.1e}")

    print()
    print("  parameter values within distance 0.05 of the target:")
    for fam in ORDER:
        d = df[df.family == fam]
        good = sorted(d[d.dist < 0.05].param.unique())
        if good:
            print(f"    {fam:<6} {len(good):>3} of {d.param.nunique():<4} "
                  f"({min(good):g} to {max(good):g})")
        else:
            print(f"    {fam:<6} none")


if __name__ == "__main__":
    main()
