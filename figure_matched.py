"""Degree-matched exponent figure.

Five panels, all on the same mean-degree range (1.67 to 511, taken from the
hierarchical network at Phi = 1 to 5). Fitted exponents only -- one log-log
slope per network across all cluster sizes.

Encodings
  colour    the family's own topology parameter (coolwarm)
  opacity   closeness to criticality, darker as m approaches 1
  size      mean degree, for small-world and scale-free only, whose topology
            parameter and density are now separate axes

Two markers per panel: the closest point overall (open circle) and the closest
point whose network is actually CONNECTED (square). For ER and scale-free those
differ a lot, because their best scores come from fragmented graphs.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import extended_networks as xn
import spectral_classes as sc

np.seterr(all="ignore")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
N = 512
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
TARGET = (1.5, 0.2)
CMAP = plt.get_cmap("coolwarm")
A_MIN, A_MAX = 0.12, 1.0
S_MIN, S_MAX = 1.5, 26.0

ORDER = ["ring", "hm", "er", "sw", "sf"]
LABEL = {"ring": "Ring", "hm": "Hierarchical modular",
         "er": "Erdős–Rényi", "sw": "Small-world",
         "sf": "Scale-free"}
# which column carries the colour, and how it is scaled
COLOUR = {"ring": ("param", "k", False), "hm": ("param", "Φ", False),
          "er": ("param", "mean degree", True),
          "sw": ("param2", "rewiring p", True),
          "sf": ("param2", "exponent γ", False)}
SIZED = {"sw", "sf"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def opacity(one_minus_m, lo, hi):
    t = (np.log10(one_minus_m) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return A_MAX + (A_MIN - A_MAX) * np.clip(t, 0, 1)


def sizes(degree, lo, hi):
    t = (np.log10(degree) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return S_MIN + (S_MAX - S_MIN) * np.clip(t, 0, 1)


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


def giant_fraction(fam, p1, p2, cache={}):
    key = (fam, p1, p2)
    if key not in cache:
        A = build(fam, p1, p2)
        _, lab = connected_components(csr_matrix(A), directed=False)
        cache[key] = np.bincount(lab).max() / N
    return cache[key]


def best_connected(d, fam, limit=60):
    """Closest point whose graph keeps >98% of nodes in one component."""
    cfg = (d.groupby(["param", "param2"], dropna=False).dist.min()
             .reset_index().sort_values("dist").head(limit))
    for _, r in cfg.iterrows():
        if giant_fraction(fam, r.param, r.param2) > 0.98:
            sub = d[(d.param == r.param) &
                    (d.param2.isna() if not np.isfinite(r.param2)
                     else d.param2 == r.param2)]
            return sub.loc[sub.dist.idxmin()]
    return None


def main():
    df = pd.read_csv(os.path.join(HERE, "sweep_data", "matched_fitted.csv"))
    lo_m, hi_m = df.one_minus_m.min(), df.one_minus_m.max()
    lo_d, hi_d = df.degree.min(), df.degree.max()

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.32, wspace=0.08)

    for ax, fam in zip(axes, ORDER):
        d = df[df.family == fam]
        col, cname, clog = COLOUR[fam]
        vals = d[col].values.astype(float)
        norm = (LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)) if clog
                else Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))

        o = d.sort_values("one_minus_m", ascending=False)
        rgba = CMAP(norm(o[col].values.astype(float)))
        rgba[:, 3] = opacity(o.one_minus_m.values, lo_m, hi_m)
        s = (sizes(o.degree.values, lo_d, hi_d) if fam in SIZED else 3.2)
        ax.scatter(o.alpha_fit, o.beta_fit, c=rgba, s=s, linewidths=0,
                   rasterized=True)

        ax.plot([TARGET[0]], [TARGET[1]], "*", color=INK, ms=14, zorder=8)
        for r in (0.1, 0.2):
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(TARGET[0] + r * sc.R_ALPHA * np.cos(th),
                    TARGET[1] + r * sc.R_BETA * np.sin(th),
                    "-", color=INK2, lw=0.7, alpha=0.6, zorder=5)

        b = d.loc[d.dist.idxmin()]
        ax.plot([b.alpha_fit], [b.beta_fit], "o", mfc="none", mec=INK,
                mew=1.5, ms=12, zorder=9)
        bc = best_connected(d, fam)
        note = f"best  d = {b.dist:.3f}  (deg {b.degree:.1f})"
        if bc is not None:
            ax.plot([bc.alpha_fit], [bc.beta_fit], "s", mfc="none", mec=INK,
                    mew=1.5, ms=11, zorder=9)
            note += f"\nconnected  d = {bc.dist:.3f}  (deg {bc.degree:.1f})"
        ax.text(0.975, 0.975, note, transform=ax.transAxes, fontsize=6.5,
                color=INK, zorder=10, ha="right", va="top",
                bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                          boxstyle="round,pad=0.3"))

        ax.set(xlabel=r"$\alpha$   (variance scaling)",
               xlim=(0.95, 2.05), ylim=(-0.05, 0.55))
        ax.set_title(LABEL[fam], pad=8)

        cax = ax.inset_axes([0.06, -0.36, 0.88, 0.05])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
                          cax=cax, orientation="horizontal")
        cb.set_label(cname, fontsize=7.5, labelpad=2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2)
        if clog:
            cb.ax.set_xscale("log")

    axes[0].set_ylabel(r"$\beta$   (timescale scaling)")

    # keys: opacity on the first panel, marker size on the small-world panel
    for mv in (0.8, 0.99, 0.9999):
        a = float(opacity(np.array([1 - mv]), lo_m, hi_m)[0])
        axes[0].plot([], [], "o", color=INK, alpha=a, ms=5, label=f"m = {mv:g}")
    axes[0].legend(loc="lower left", fontsize=6.2, handletextpad=0.3,
                   borderpad=0.4, labelspacing=0.3, title="opacity",
                   title_fontsize=6.2)
    for dv in (2, 16, 128, 500):
        axes[3].plot([], [], "o", color=INK2,
                     ms=np.sqrt(sizes(np.array([dv]), lo_d, hi_d)[0]) * 1.6,
                     label=f"{dv}")
    axes[3].legend(loc="lower left", fontsize=6.2, handletextpad=0.5,
                   borderpad=0.4, labelspacing=0.5, title="mean degree",
                   title_fontsize=6.2)

    fig.suptitle("Degree-matched comparison, exponents fitted across all cluster "
                 "sizes.  Colour = topology parameter, opacity = closeness to "
                 "criticality, size = mean degree (small-world, scale-free).  "
                 "Circle = closest overall, square = closest connected",
                 y=1.045, fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_matched.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig_matched.png / .pdf")

    print()
    print(f"  {'family':<8}{'best d':<10}{'deg':<8}{'connected d':<14}{'deg':<8}"
          f"{'parameter at connected best'}")
    for fam in ORDER:
        d = df[df.family == fam]
        b = d.loc[d.dist.idxmin()]
        bc = best_connected(d, fam)
        if bc is None:
            print(f"  {fam:<8}{b.dist:<10.4f}{b.degree:<8.2f}{'none':<14}")
            continue
        p2 = "" if not np.isfinite(bc.param2) else f", {bc.param2:g}"
        print(f"  {fam:<8}{b.dist:<10.4f}{b.degree:<8.2f}{bc.dist:<14.4f}"
              f"{bc.degree:<8.2f}{bc.param:g}{p2}")


if __name__ == "__main__":
    main()
