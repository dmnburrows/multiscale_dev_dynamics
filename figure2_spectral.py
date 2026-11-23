"""Figure 2, recoloured: each panel spectral, by that family's own parameter.

Changes from the first version:
  - colour encodes the architecture PARAMETER on a spectral map, with a colour
    bar per panel, instead of a per-family hue
  - the hierarchical modular panel is swept over Phi = 1 to 5 (probabilistic,
    i.e. linear_theory.make_network, the same network the other panels use)

Everything else is as before: every m, every cluster size with K >= 4, the
range-weighted distance to (1.5, 0.2), and the closest point marked.

Note on reading the HM panel: above Phi ~ 3.2 the probabilistic network breaks
into disconnected pieces (at Phi = 3.5, n = 1024, the largest component holds
45% of the nodes). Those points are plotted as asked, but they describe a set
of fragments rather than one network.

    python figure2_spectral.py            # reuses the cached HM sweep if present
    python figure2_spectral.py --resweep  # recompute the HM rows
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import analytical_theory as at
import spectral_classes as sc

np.seterr(all="ignore")

FIGDIR = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "sweep_data")
os.makedirs(FIGDIR, exist_ok=True)

PHI_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.83, 3.0, 3.25, 3.5, 4.0,
            4.5, 5.0]
ORDER = ["ring", "hm", "er", "sw", "sf"]
LABEL = {"ring": "Ring", "hm": "Hierarchical modular",
         "er": "Erdős–Rényi", "sw": "Small-world",
         "sf": "Scale-free"}
PNAME = {"ring": "k", "hm": "Φ", "er": "mean degree",
         "sw": "rewiring p", "sf": "exponent τ"}
LOGSCALE = {"er", "sw"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
TARGET = (1.5, 0.2)
CMAP = plt.get_cmap("Spectral_r")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def sweep_hm(n=1024, seeds=(1, 2), lms=None):
    """The HM rows over the new Phi grid, same pipeline as spectral_classes."""
    if lms is None:
        lms = np.logspace(-1, -5, 21)
    rows = []
    for phi in PHI_GRID:
        for seed in seeds:
            A = lt.make_network(n, "hm", phi, seed)
            ev, V = eigh(A, check_finite=False)
            s = ev / np.max(np.abs(ev))
            for lm in lms:
                m = 1.0 - lm
                S, L = lt.stationary_covariances(ev, V, m)
                levels = at.cascade(S, L, 4)
                Vv = np.array([d["variance"] for d in levels])
                Tt = np.array([d["tau"] for d in levels])
                Ks = np.array([d["K"] for d in levels], float)
                a = np.log2(Vv[1:] / Vv[:-1])
                b = np.log2(Tt[1:] / Tt[:-1])
                for i in range(len(a)):
                    rows.append({"family": "hm", "param": phi, "seed": seed,
                                 "one_minus_m": lm, "K": Ks[i],
                                 "alpha": a[i], "beta": b[i],
                                 "dist": float(sc.distance(a[i], b[i]))})
        print(f"  hm phi={phi} done", flush=True)
    return pd.DataFrame(rows)


M_LINES = [0.8, 0.9, 0.99, 0.999, 0.9999]
# coolwarm for the m lines. It shares the blue-to-red axis with Spectral_r, so
# hue alone will not separate the two encodings: the lines also carry a white
# halo, which is what actually makes them read as a layer above the scatter.
M_CMAP = plt.get_cmap("coolwarm")
M_COLORS = [M_CMAP(t) for t in np.linspace(0.0, 1.0, len(M_LINES))]


def iso_m_curve(fam, param, m, n=1024, seeds=(1, 2, 3, 4, 5)):
    """(alpha, beta) traced over cluster size at fixed architecture and m.

    A constant-m slice of the (parameter, m, K) family is still two-dimensional,
    so it fills a region rather than drawing a line. Fixing the architecture
    parameter as well leaves cluster size as the only free variable, which is
    the curve worth drawing: it is the scaling trajectory at that coupling.

    Single-seed trajectories are jagged because consecutive local slopes on a
    random graph are noisy, so this averages over seeds at matched cluster size.
    """
    A_, B_, K_ = [], [], None
    for seed in seeds:
        A = lt.make_network(n, fam, param, seed)
        ev, V = eigh(A, check_finite=False)
        S, L = lt.stationary_covariances(ev, V, m)
        levels = at.cascade(S, L, 4)
        Vv = np.array([d["variance"] for d in levels])
        Tt = np.array([d["tau"] for d in levels])
        Ks = np.array([d["K"] for d in levels], float)
        a = np.log2(Vv[1:] / Vv[:-1])
        b = np.log2(Tt[1:] / Tt[:-1])
        keep = Ks[:-1] >= 4
        A_.append(a[keep]); B_.append(b[keep]); K_ = Ks[:-1][keep]
    L0 = min(len(x) for x in A_)
    return (np.mean([x[:L0] for x in A_], axis=0),
            np.mean([x[:L0] for x in B_], axis=0), K_[:L0])


def load(resweep=False):
    base = pd.read_csv(os.path.join(DATA, "full_sweep.csv"))
    cache = os.path.join(DATA, "hm_phi_1_to_5.csv")
    if resweep or not os.path.exists(cache):
        hm = sweep_hm()
        hm.to_csv(cache, index=False)
    else:
        hm = pd.read_csv(cache)
    cols = ["family", "param", "seed", "one_minus_m", "K", "alpha", "beta", "dist"]
    out = pd.concat([base[base.family != "hm"][cols], hm[cols]],
                    ignore_index=True)
    return out[out.K >= 4]


def main(resweep=False):
    df = load(resweep)
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.30, wspace=0.08)

    for ax, fam in zip(axes, ORDER):
        d = df[df.family == fam]
        params = np.array(sorted(d.param.unique()), float)
        if fam in LOGSCALE:
            norm = LogNorm(vmin=params.min(), vmax=params.max())
        else:
            norm = Normalize(vmin=params.min(), vmax=params.max())
        ax.scatter(d.alpha, d.beta, c=d.param.values, cmap=CMAP, norm=norm,
                   s=3.0, alpha=0.30, linewidths=0, rasterized=True)

        best = d.loc[d.dist.idxmin()]

        # iso-m lines, drawn at this panel's best-fitting parameter
        for mi, mv in enumerate(M_LINES):
            ca, cb, cK = iso_m_curve(fam, best.param, mv,
                                     seeds=(1, 2, 3, 4, 5, 6, 7, 8))
            if len(ca) < 2:
                continue
            col = M_COLORS[mi]
            ax.plot(ca, cb, "-", color=col, lw=1.7, zorder=4 + mi,
                    solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=3.4,
                                                foreground="#fcfcfb")])
            ax.plot(ca[-1], cb[-1], "o", color=col, ms=3.6, zorder=4 + mi,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground="#fcfcfb")])
        ax.plot([best.alpha], [best.beta], "o", mfc="none", mec=INK, mew=1.7,
                ms=11, zorder=6)
        ax.plot([TARGET[0]], [TARGET[1]], "*", color=INK, ms=14, zorder=7)
        for r in (0.1, 0.2):
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(TARGET[0] + r * sc.R_ALPHA * np.cos(th),
                    TARGET[1] + r * sc.R_BETA * np.sin(th),
                    "-", color=INK2, lw=0.7, alpha=0.6, zorder=5)
        ax.text(0.975, 0.975,
                f"best d = {best.dist:.3f}\n{PNAME[fam]} = {best.param:g},  "
                f"1−m = {best.one_minus_m:.1e}\nK = {best.K:g}",
                transform=ax.transAxes, fontsize=6.6, color=INK, zorder=8,
                ha="right", va="top",
                bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                          boxstyle="round,pad=0.3"))
        ax.set(xlabel=r"$\alpha$   (variance scaling)",
               xlim=(1.0, 2.05), ylim=(-0.02, 0.72))
        ax.set_title(LABEL[fam], pad=8)

        cax = ax.inset_axes([0.06, -0.33, 0.88, 0.055])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
                          cax=cax, orientation="horizontal")
        cb.set_label(PNAME[fam], fontsize=7.5, labelpad=2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=2)
        if fam in LOGSCALE:
            cb.ax.set_xscale("log")
        cb.set_ticks(params[::max(1, len(params) // 6)])
        cb.ax.set_xticklabels([f"{v:g}" for v in params[::max(1, len(params) // 6)]])

    axes[0].set_ylabel(r"$\beta$   (timescale scaling)")
    handles = [Line2D([], [], color=c, lw=2.2, label=f"m = {v:g}")
               for c, v in zip(M_COLORS, M_LINES)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=len(M_LINES), fontsize=8, handlelength=1.8,
               columnspacing=1.6, handletextpad=0.5)
    fig.suptitle("Reachable exponent pairs, coloured by each architecture's own "
                 "parameter, with iso-m trajectories at each panel's best parameter.  "
                 "Star = (1.5, 0.2); rings = range-weighted distance "
                 "0.1 and 0.2", y=1.10, fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_spectral.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig2_spectral.png / .pdf")

    print()
    print("BEST PER PARAMETER VALUE (K >= 4, minimised over m and cluster size)")
    for fam in ORDER:
        d = df[df.family == fam]
        print(f"\n  {LABEL[fam]}  ({PNAME[fam]})")
        print(f"  {'param':<9}{'min dist':<11}{'alpha':<9}{'beta':<9}{'1-m':<10}{'K'}")
        for par in sorted(d.param.unique()):
            dd = d[d.param == par]
            b = dd.loc[dd.dist.idxmin()]
            print(f"  {par:<9g}{b.dist:<11.4f}{b.alpha:<9.4f}{b.beta:<9.4f}"
                  f"{b.one_minus_m:<10.1e}{b.K:g}")


if __name__ == "__main__":
    main("--resweep" in sys.argv)
