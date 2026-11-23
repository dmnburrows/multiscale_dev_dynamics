"""Figure: alpha and beta vs m, and the (alpha, beta) trajectory, under six
different ICG pairing rules.  Reads sweep_data/structural_icg_curves.csv.

    /Users/k2585057/anaconda3/envs/py39/bin/python -W ignore -u figure_structural_icg.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIGDIR = os.path.join(HERE, "figures")
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

# Okabe-Ito: colourblind-safe by construction. Linestyle is a second channel,
# so the two warm hues never have to be told apart by colour alone.
STYLE = {
    "corr":   ("#0072B2", "-",  2.0, "correlation (baseline)"),
    "shared": ("#D55E00", "-",  1.3, "shared connections"),
    "nbr":    ("#009E73", "-",  1.3, "shared partners only"),
    "adj":    ("#CC79A7", "-",  1.3, "strongest direct edge"),
    "dyadic": ("#E69F00", "--", 1.3, "true module tree"),
    "random": ("#555555", ":",  1.3, "random matching"),
}
ORDER = ["corr", "shared", "nbr", "adj", "dyadic", "random"]
NETS = ["ring k=4", "ring k=6", "ring k=8", "sw p=0.004",
        "hm phi=3.4", "hm phi=3.0", "hm phi=2.6"]
PRETTY = {"ring k=4": "Ring $k=4$", "ring k=6": "Ring $k=6$",
          "ring k=8": "Ring $k=8$", "sw p=0.004": "Small-world $p=0.004$",
          "hm phi=3.4": r"HM $\phi=3.4$", "hm phi=3.0": r"HM $\phi=3.0$",
          "hm phi=2.6": r"HM $\phi=2.6$"}


def main():
    d = pd.read_csv(os.path.join(HERE, "sweep_data", "structural_icg_curves.csv"))
    agg = (d.groupby(["network", "rule", "one_minus_m"], sort=False)
             .agg(alpha=("alpha", "mean"), beta=("beta", "mean")).reset_index())

    fig, axes = plt.subplots(len(NETS), 3, figsize=(9.6, 2.35 * len(NETS)))
    for r, net in enumerate(NETS):
        g = agg[agg.network == net]
        ax0, ax1, ax2 = axes[r]
        for rule in ORDER:
            h = g[g.rule == rule].sort_values("one_minus_m")
            c, ls, lw, _ = STYLE[rule]
            ax0.plot(h.one_minus_m, h.alpha, ls, color=c, lw=lw)
            ax1.plot(h.one_minus_m, h.beta, ls, color=c, lw=lw)
            ax2.plot(h.alpha, h.beta, ls, color=c, lw=lw)
        for ax, y in ((ax0, TARGET[0]), (ax1, TARGET[1])):
            ax.set_xscale("log")
            ax.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)))
            ax.set_xlabel(r"$1-m$")
        ax0.set_ylabel(r"$\alpha$")
        ax1.set_ylabel(r"$\beta$")
        ax2.plot(*TARGET, "*", ms=11, mfc="#fcfcfb", mec=INK, mew=1.1, zorder=5)
        if net.startswith("ring"):
            # The five non-random rules are superimposed, not missing.
            for ax in (ax0, ax1, ax2):
                ax.text(0.03, 0.04, "5 rules coincide", transform=ax.transAxes,
                        fontsize=7.5, color=INK2, ha="left", va="bottom")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"$\beta$")
        ax2.set_xlim(1.0, 2.0)
        ax0.text(-0.30, 0.5, PRETTY[net], transform=ax0.transAxes,
                 rotation=90, va="center", ha="center", fontsize=9.5)
        if r == 0:
            ax0.set_title(r"$\alpha$ vs distance to criticality", fontsize=9)
            ax1.set_title(r"$\beta$ vs distance to criticality", fontsize=9)
            ax2.set_title(r"trajectory in the $(\alpha,\beta)$ plane", fontsize=9)

    handles = [plt.Line2D([], [], color=c, ls=ls, lw=lw, label=lab)
               for c, ls, lw, lab in (STYLE[k] for k in ORDER)]
    handles.append(plt.Line2D([], [], ls="none", marker="*", ms=10,
                              mfc="#fcfcfb", mec=INK, label="Munn point (1.5, 0.2)"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.012), fontsize=8.5)
    fig.suptitle("ICG pairing rule: correlation vs wiring", y=0.998, fontsize=11)
    fig.tight_layout(rect=(0.015, 0.022, 1, 0.992))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig_structural_icg.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig_structural_icg.png / .pdf")


if __name__ == "__main__":
    main()
