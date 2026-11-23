"""Figures for the Fig 3 analysis: spectral class, reachable exponents, mechanism.

Reads sweep_data/full_sweep.csv (written by spectral_classes.py sweep) and the
closed forms, and writes figures/ as PNG + PDF.

Colour rules followed: family identity is a categorical hue, in fixed order,
from a palette validated for colour-vision deficiency; a parameter's MAGNITUDE
within a family is lightness on that family's hue (sequential), never a second
categorical set; distance is a single-hue sequential ramp, dark = worse. No
dual axes anywhere.

    python make_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linear_theory as lt
import analytical_theory as at
import spectral_classes as sc

np.seterr(all="ignore")

FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# validated categorical palette (node scripts/validate_palette.js: all checks pass)
HUE = {"ring": "#2a78d6", "hm": "#eb6834", "er": "#1baf7a",
       "sw": "#eda100", "sf": "#e87ba4"}
LABEL = {"ring": "Ring (k)", "hm": "Hierarchical modular (Φ)",
         "er": "Erdős–Rényi (mean degree)",
         "sw": "Small-world (rewiring p)", "sf": "Scale-free (exponent)"}
ORDER = ["ring", "hm", "er", "sw", "sf"]
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


def shades(hue, k):
    """Sequential ramp on one hue: light -> dark, for an ordered parameter."""
    base = np.array(matplotlib.colors.to_rgb(hue))
    return [tuple(np.clip(base * (0.45 + 0.75 * t) + (1 - (0.45 + 0.75 * t)) * 0.93, 0, 1))
            for t in np.linspace(0.12, 1.0, k)]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .pdf")


# =============================================================================
# Figure 1 - spectral classification
# =============================================================================

def figure1():
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))

    # (a) gap vs n
    ax = axes[0]
    ns = [256, 512, 1024, 2048, 4096]
    probe = {"ring": 4, "hm": 2.83, "er": 6.2, "sw": 0.1, "sf": 2.5}
    for fam in ORDER:
        gs = []
        for n in ns:
            vals = []
            for seed in (1, 2):
                A = lt.make_network(n, fam, probe[fam], seed)
                ev = eigh(A, eigvals_only=True, check_finite=False)
                s = np.sort(ev / np.max(np.abs(ev)))[::-1]
                vals.append(s[0] - s[1])
            gs.append(np.mean(vals))
        ax.plot(ns, gs, "o-", color=HUE[fam], lw=2, ms=5, label=fam)
        ax.annotate(fam, (ns[-1], gs[-1]), xytext=(4, 0),
                    textcoords="offset points", color=HUE[fam], fontsize=8,
                    va="center")
    ax.plot(ns, 0.05 * (np.array(ns) / 256.0) ** -2.0, "--", color=INK2, lw=1)
    ax.annotate("$n^{-2}$", (ns[2], 0.05 * (ns[2] / 256.) ** -2), color=INK2,
                fontsize=8, xytext=(-24, -2), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="network size $n$",
           ylabel="spectral gap  $s_1 - s_2$",
           title="a  Only the ring's gap closes")
    ax.set_xlim(220, 9000)

    # (b) ER gap: closed form vs measured
    ax = axes[1]
    zs = sc.PARAM_GRID["er"]
    pred = [1 - 2 * np.sqrt(z - 1) / z for z in zs]
    meas = []
    for z in zs:
        vals = []
        for seed in (1, 2):
            A = lt.make_network(4096, "er", z, seed)
            ev = eigh(A, eigvals_only=True, check_finite=False)
            s = np.sort(ev / np.max(np.abs(ev)))[::-1]
            vals.append(s[0] - s[1])
        meas.append(np.mean(vals))
    ax.plot(zs, pred, "-", color=INK2, lw=2, label="$1-2\\sqrt{z-1}/z$")
    ax.plot(zs, meas, "o", color=HUE["er"], ms=6, label="measured, $n$=4096")
    ax.axhline(0, color=GRID, lw=1)
    ax.set(xscale="log", xlabel="mean degree $z$", ylabel="spectral gap",
           title="b  Random graphs: gap in closed form")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.annotate("gap closes\nas $z\\to2$", (2.5, 0.02), xytext=(10, 28),
                textcoords="offset points", fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))

    # (c) hierarchy: gap exponent vs prediction
    ax = axes[2]
    phis = [2.83, 3.2, 3.5, 4.0, 5.0, 6.0]
    xs, ys = [], []
    for phi in phis:
        nn = [2 ** p for p in (10, 12, 14, 16)]
        gaps = []
        for n in nn:
            spec = at.hier_spectrum(n, phi)
            rho = max(abs(v) for v, _ in spec)
            vals = np.sort(np.concatenate(
                [np.full(mult, val / rho) for val, mult in spec]))[::-1]
            gaps.append(vals[0] - vals[1])
        ys.append(np.polyfit(np.log(nn), np.log(gaps), 1)[0])
        xs.append(-np.log2(phi / 2))
    lim = [min(xs) - 0.15, max(xs) + 0.15]
    ax.plot(lim, lim, "--", color=INK2, lw=1)
    ax.plot(xs, ys, "o", color=HUE["hm"], ms=7)
    for x, y, phi in zip(xs, ys, phis):
        ax.annotate(f"$\\Phi$={phi}", (x, y), xytext=(5, -3),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set(xlabel="predicted  $-\\log_2(\\Phi/2)$",
           ylabel="measured  $d\\log(\\mathrm{gap})/d\\log n$",
           title="c  Deterministic hierarchy: exact")
    ax.set_xlim(*lim)

    # (d) counting function
    ax = axes[3]
    ds = np.logspace(-5, -1, 40)
    for fam in ORDER:
        A = lt.make_network(4096, fam, probe[fam], 1)
        ev = eigh(A, eigvals_only=True, check_finite=False)
        s = ev / np.max(np.abs(ev))
        N = sc.counting(s, ds) / 4096.0
        ax.plot(ds, N, "-", color=HUE[fam], lw=2)
        ax.annotate(fam, (ds[-1], N[-1]), xytext=(4, 0),
                    textcoords="offset points", color=HUE[fam], fontsize=8,
                    va="center")
    ax.set(xscale="log", yscale="log", xlabel="$\\delta$",
           ylabel="$N(\\delta)\\,/\\,n$",
           title="d  Modes within $\\delta$ of the edge")
    ax.set_xlim(1e-5, 3e-1)

    fig.suptitle("Figure 1  Spectral classification: one family is gapless, the rest are gapped",
                 y=1.04, fontsize=10, x=0.5)
    save(fig, "fig1_spectral_class")


# =============================================================================
# Figure 2 - the reachable (alpha, beta) region
# =============================================================================

def figure2(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), sharex=True, sharey=True)
    for ax, fam in zip(axes, ORDER):
        d = df[df.family == fam]
        params = sorted(d.param.unique())
        cols = shades(HUE[fam], len(params))
        for c, par in zip(cols, params):
            dd = d[d.param == par]
            ax.plot(dd.alpha, dd.beta, ".", color=c, ms=2.0, alpha=0.55,
                    rasterized=True)
        best = d.loc[d.dist.idxmin()]
        ax.plot([best.alpha], [best.beta], "o", mfc="none", mec=INK, mew=1.6,
                ms=10, zorder=5)
        ax.text(0.975, 0.03,
                f"best d = {best.dist:.3f}\n{_pname(fam)} = {best.param:g},  "
                f"1−m = {best.one_minus_m:.1e}\nK = {best.K:g}",
                transform=ax.transAxes, fontsize=6.6, color=INK, zorder=8,
                ha="right", va="bottom",
                bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                          boxstyle="round,pad=0.3"))
        ax.plot([TARGET[0]], [TARGET[1]], "*", color=INK, ms=13, zorder=7)
        for r in (0.1, 0.2):
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(TARGET[0] + r * sc.R_ALPHA * np.cos(th),
                    TARGET[1] + r * sc.R_BETA * np.sin(th),
                    "-", color=INK2, lw=0.7, alpha=0.6)
        # colour key for the parameter ramp, on its own strip above the axes
        for i, (c, par) in enumerate(zip(cols, params)):
            x = 0.035 + i * (0.93 / max(len(params) - 1, 1))
            ax.plot([x], [1.045], "s", color=c, ms=5, clip_on=False,
                    transform=ax.transAxes)
            ax.text(x, 1.085, f"{par:g}", fontsize=6.0, color=INK2,
                    ha="center", va="bottom", transform=ax.transAxes)
        ax.text(-0.01, 1.045, _pname(fam), fontsize=7, color=INK,
                ha="right", va="center", transform=ax.transAxes)
        ax.set(xlabel=r"$\alpha$  (variance scaling)",
               xlim=(1.0, 2.05), ylim=(-0.02, 0.72))
        ax.set_title(LABEL[fam], pad=32)
    axes[0].set_ylabel(r"$\beta$  (timescale scaling)")
    fig.suptitle("Figure 2  Reachable exponent pairs over the full parameter range "
                 "(every $m$, every cluster size). Star = (1.5, 0.2); rings = "
                 "range-weighted distance 0.1 and 0.2",
                 y=1.22, fontsize=10)
    save(fig, "fig2_reachable_exponents")


def _pname(fam):
    return {"ring": "k", "hm": "Φ", "er": "z", "sw": "p", "sf": "τ"}[fam]


# =============================================================================
# Figure 3 - distance landscape over (parameter, m)
# =============================================================================

def figure3(df):
    ramp = LinearSegmentedColormap.from_list(
        "dist", ["#eef4fc", "#9cc2ec", "#2a78d6", "#143c6b"])
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    g = (df.groupby(["family", "param", "one_minus_m"]).dist.min().reset_index())
    vmin, vmax = g.dist.min(), np.percentile(g.dist, 97)
    for ax, fam in zip(axes, ORDER):
        d = g[g.family == fam]
        params = sorted(d.param.unique())
        lms = sorted(d.one_minus_m.unique())
        Z = np.full((len(params), len(lms)), np.nan)
        for i, p in enumerate(params):
            for j, l in enumerate(lms):
                v = d[(d.param == p) & (d.one_minus_m == l)].dist
                if len(v):
                    Z[i, j] = v.iloc[0]
        im = ax.pcolormesh(np.array(lms), np.arange(len(params)), Z,
                           cmap=ramp, vmin=vmin, vmax=vmax, shading="nearest")
        i, j = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot([lms[j]], [i], "o", mfc="none", mec="#fcfcfb", mew=2.0, ms=11)
        ax.plot([lms[j]], [i], "o", mfc="none", mec=INK, mew=1.1, ms=11)
        ax.set(xscale="log", yticks=np.arange(len(params)),
               yticklabels=[f"{p:g}" for p in params],
               xlabel="$1-m$", ylabel=_pname(fam), title=LABEL[fam])
        ax.grid(False)
        ax.invert_xaxis()
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.set_label("minimum range-weighted distance to (1.5, 0.2)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Figure 3  Distance landscape: minimised over cluster size, "
                 "for every architecture parameter and every distance from instability",
                 y=1.05, fontsize=10)
    save(fig, "fig3_distance_landscape")


# =============================================================================
# Figure 4 - mechanism
# =============================================================================

def figure4(df):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    near = df[np.abs(df.alpha - 1.5) < 0.05]

    # (a) beta vs rho at alpha ~ 1.5
    ax = axes[0]
    for fam in ORDER:
        d = near[near.family == fam]
        ax.plot(d.rho, d.beta, ".", color=HUE[fam], ms=3.5, alpha=0.6,
                rasterized=True)
    ax.axhline(TARGET[1], color=INK, lw=1.1, ls="--")
    ax.annotate(r"$\beta=0.2$", (1.3, 0.21), fontsize=7.5, color=INK)
    ax.set(xscale="log", xlabel=r"$\rho=\tau_{\rm lead}/\tau_{\rm rest}$",
           ylabel=r"$\beta$", title=r"a  Where $\alpha\approx1.5$")

    # (b) distance vs rho
    ax = axes[1]
    for fam in ORDER:
        d = df[df.family == fam]
        ax.plot(d.rho, d.dist, ".", color=HUE[fam], ms=2.0, alpha=0.35,
                rasterized=True)
        b = d.loc[d.dist.idxmin()]
        ax.plot([b.rho], [b.dist], "o", color=HUE[fam], mec=INK, mew=0.9, ms=7)
    ax.set(xscale="log", yscale="log", xlabel=r"$\rho$",
           ylabel="distance to (1.5, 0.2)",
           title="b  Best fits need modest $\\rho$")

    # (c) alpha vs leading-mode share, with the one-mode law
    ax = axes[2]
    for fam in ORDER:
        d = df[df.family == fam]
        ax.plot(d.p_lead, d.alpha, ".", color=HUE[fam], ms=2.0, alpha=0.35,
                rasterized=True)
    p = np.linspace(0, 1, 200)
    ax.plot(p, 1 + np.log2(1 + p), "-", color=INK, lw=1.8)
    ax.annotate(r"$\alpha=1+\log_2(1+p)$", (0.45, 1.46), fontsize=7.5,
                color=INK, rotation=22)
    ax.set(xlabel="leading mode's share of variance $p$", ylabel=r"$\alpha$",
           title="c  One-mode law holds only when\n   one mode dominates")

    # (d) n_eff vs beta
    ax = axes[3]
    for fam in ORDER:
        d = df[df.family == fam]
        ax.plot(d.n_eff, d.beta, ".", color=HUE[fam], ms=2.0, alpha=0.35,
                rasterized=True)
        ax.plot([], [], "o", color=HUE[fam], ms=5, label=LABEL[fam])
    ax.axhline(TARGET[1], color=INK, lw=1.1, ls="--")
    ax.set(xscale="log", xlabel=r"$N_{\rm eff}$ (modes sharing the variance)",
           ylabel=r"$\beta$", title=r"d  $\beta$ needs several modes")
    ax.legend(fontsize=6.6, loc="upper right")

    fig.suptitle("Figure 4  Mechanism: the leading mode's share and its relative "
                 "slowness decide where an architecture lands",
                 y=1.05, fontsize=10)
    save(fig, "fig4_mechanism")


def figure5(df):
    from scipy.stats import spearmanr
    g = (df.groupby(["family", "param"])
           .agg(dist=("dist", "min"), gap=("gap", "mean")).reset_index())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    fig.subplots_adjust(wspace=0.34, bottom=0.30)

    # (a) distance vs gap
    ax = axes[0]
    for fam in ORDER:
        d = g[g.family == fam]
        ax.plot(np.maximum(d.gap, 2e-5), d.dist, "o", color=HUE[fam], ms=7,
                mec="#fcfcfb", mew=0.6, label=LABEL[fam].split(" (")[0])
    rho, p = spearmanr(g.gap, g.dist)
    lg, ld = np.log10(np.maximum(g.gap, 2e-5)), np.log10(g.dist)
    xs = np.linspace(lg.min(), lg.max(), 50)
    sl, ic = np.polyfit(lg, ld, 1)
    ax.plot(10 ** xs, 10 ** (sl * xs + ic), "-", color=INK2, lw=1.4)
    ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.2f}\n"
            f"$p$ = {p:.1e},  $N$ = {len(g)}", transform=ax.transAxes,
            fontsize=7.5, color=INK, va="top",
            bbox=dict(facecolor="#fcfcfb", alpha=0.9, edgecolor=GRID,
                      boxstyle="round,pad=0.3"))
    sfd = g[g.family == "sf"]
    ax.annotate("scale-free breaks the trend:\nlargest gaps, good fits",
                (sfd.gap.max(), sfd.dist.min()), xytext=(-118, 30),
                textcoords="offset points", fontsize=7, color=HUE["sf"],
                arrowprops=dict(arrowstyle="->", color=HUE["sf"], lw=0.9))
    ax.set(xscale="log", yscale="log", xlabel="spectral gap  $s_1-s_2$",
           ylabel="best distance to (1.5, 0.2)",
           title="a  Gaplessness predicts the exponents")
    ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=3, handletextpad=0.3, columnspacing=1.0)

    # (b) within-family optimum
    ax = axes[1]
    for fam in ORDER:
        d = g[g.family == fam].sort_values("param")
        x = np.arange(len(d)) / max(len(d) - 1, 1)
        ax.plot(x, d.dist, "o-", color=HUE[fam], lw=2, ms=5)
        i = int(np.argmin(d.dist.values))
        ax.plot([x[i]], [d.dist.values[i]], "o", mfc="none", mec=INK, mew=1.5,
                ms=11)
        ax.annotate(f"{_pname(fam)}={d.param.values[i]:g}",
                    (x[i], d.dist.values[i]), xytext=(0, -14),
                    textcoords="offset points", fontsize=6.8,
                    color=HUE[fam], ha="center")
    ax.set(yscale="log", xlabel="parameter, low → high (per family)",
           ylabel="best distance", title="b  Every family has an interior optimum")

    # (c) the shape of the edge: one power law, four steps
    ax = axes[2]
    ds = np.logspace(-4, 0, 60)
    ylab = {"ring": 0.80, "hm": 0.62, "er": 0.46, "sf": 0.30}
    for fam, par in (("ring", 4), ("hm", 3.5), ("er", 6.2), ("sf", 2.1)):
        A = lt.make_network(4096, fam, par, 1)
        ev = eigh(A, eigvals_only=True, check_finite=False)
        s = ev / np.max(np.abs(ev))
        N = sc.counting(s, ds)
        ax.plot(ds, N, "-", color=HUE[fam], lw=2)
        ax.text(0.04, ylab[fam], f"{fam} {par:g}", transform=ax.transAxes,
                color=HUE[fam], fontsize=7.5, va="center")
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(1.3e-4, 1.25, "one mode only", fontsize=6.8, color=INK2)
    ax.set(xscale="log", yscale="log", xlabel="$\\delta$",
           ylabel="$N(\\delta)$  (modes within $\\delta$ of the edge)",
           title="c  One power law, four steps")
    ax.set_xlim(1e-4, 1.2)

    fig.suptitle("Figure 5  The spectral edge, not the distance from instability, "
                 "decides which architectures match", y=1.04, fontsize=10)
    save(fig, "fig5_gap_predicts")


def summary(df, df_all):
    print()
    print("=" * 92)
    print("MINIMUM RANGE-WEIGHTED DISTANCE TO (1.5, 0.2), by family and parameter")
    print("  d = sqrt( ((alpha-1.5)/1.0)^2 + ((beta-0.2)/0.5)^2 ),  minimised over m and K")
    print("  K >= 4 only: the K=1 and K=2 levels are the ones greedy ICG gets wrong")
    print("  (it crosses the first module boundary there), so they are excluded.")
    print("=" * 92)
    print(f"{'family':<8}{'param':<9}{'min dist':<11}{'alpha':<9}{'beta':<9}"
          f"{'1-m':<10}{'K':<7}{'p_lead':<9}{'rho':<11}{'N_eff'}")
    for fam in ORDER:
        d = df[df.family == fam]
        for par in sorted(d.param.unique()):
            dd = d[d.param == par]
            b = dd.loc[dd.dist.idxmin()]
            print(f"{fam:<8}{par:<9g}{b.dist:<11.4f}{b.alpha:<9.4f}{b.beta:<9.4f}"
                  f"{b.one_minus_m:<10.1e}{b.K:<7g}{b.p_lead:<9.4f}"
                  f"{b.rho:<11.2f}{b.n_eff:.2f}")

    print()
    print("  How much the first two levels were flattering each family:")
    print(f"  {'family':<8}{'best d, K>=4':<16}{'best d, all K':<16}{'K of the all-K best'}")
    for fam in ORDER:
        a = df[df.family == fam].dist.min()
        bb = df_all[df_all.family == fam]
        b = bb.loc[bb.dist.idxmin()]
        print(f"  {fam:<8}{a:<16.4f}{b.dist:<16.4f}{b.K:g}")


if __name__ == "__main__":
    df_all = pd.read_csv(os.path.join(HERE, "sweep_data", "full_sweep.csv"))
    df = df_all[df_all.K >= 4].copy()
    print(f"loaded {len(df_all)} rows; {len(df)} with K >= 4")
    figure1()
    figure2(df)
    figure3(df)
    figure4(df)
    figure5(df)
    summary(df, df_all)
