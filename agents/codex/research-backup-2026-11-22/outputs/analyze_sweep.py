"""Summarize the sweep and verify an exact non-identifiability example."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from griffiths_sweep import curves


def counterexample():
    i = np.eye(4)
    cross = np.ones((4, 4)) / 4
    graphs = [
        np.block([[p, cross], [cross, p]]) / 2
        for p in (i[[1, 0, 3, 2]], i[[2, 3, 0, 1]])
    ]
    np.testing.assert_array_equal(graphs[0] @ graphs[0], graphs[1] @ graphs[1])
    m = .9
    groups = [np.eye(8), np.eye(8).reshape(4, 2, 8).sum(axis=1)]
    for b in graphs:
        lam, modes = eigh(b)
        values = [(g @ modes)**2 for g in groups]
        v, w, _ = curves(values, lam, m)
        # Independently check the stationary covariance and lag-sum formula.
        c = np.linalg.inv(np.eye(8) - m*m*(b @ b))
        s = np.linalg.solve(np.eye(8) - m*b, c)
        np.testing.assert_allclose(v, [np.diag(g @ c @ g.T).mean() for g in groups])
        np.testing.assert_allclose(w, [np.diag(g @ s @ g.T).mean() for g in groups])
        # Check the response formula with finite differences at fixed groups.
        def log_var(local_m, r):
            return np.log([q.mean(axis=0) @ (1 / (1 - (local_m * lam + r)**2))
                           for q in values])
        h = 1e-6
        dm = (log_var(m+h, 0) - log_var(m-h, 0))/(2*h)
        dr = (log_var(m, h) - log_var(m, -h))/(2*h)
        np.testing.assert_allclose(1 + .5*(m*dm + dr), w/v, rtol=1e-8)
        print('counterexample', 'alpha,beta', np.log2(v[1]/v[0]),
              np.log2((w/v)[1]/(w/v)[0]), 'variance', v, 'timescale', w/v)


def main():
    path = Path(__file__).parent
    df = pd.concat([pd.read_csv(f) for f in sorted(path.glob('sweep_n*.csv'))])
    assert np.max(np.abs(df.alpha + df.beta - df.gamma)) < 1e-12
    grouped = df.groupby(['n', 'family', 'seed', 'policy', 'k_min', 'k_max'])
    nearest = df.reset_index(drop=True)
    keys = ['n', 'family', 'seed', 'policy', 'k_min', 'k_max']
    nearest = nearest.loc[nearest.groupby(keys).target_error.idxmin()].copy()
    peak = grouped.beta.max().rename('peak_beta').reset_index()
    nearest = nearest.merge(peak, on=keys, validate='one_to_one')
    nearest.to_csv(path / 'nearest_to_target.csv', index=False)
    print('rows', len(df))
    columns = ['m', 'alpha', 'beta', 'beta_mean_time', 'peak_beta', 'target_error']
    sel = nearest.query('policy == "greedy" and k_min == 2')
    print(sel.groupby(['n','family'])[columns].median().round(5).to_string())
    print('matched target options')
    for n, sub in df.query('policy == "greedy"').groupby('n'):
        near = sub.loc[(sub.alpha-1.5).abs().le(.05) & (sub.beta-.2).abs().le(.05)]
        print(n, near.groupby('family').m.agg(['min','max','size']).to_string())
        print('sensitivity to averaging', sub.eval('abs(beta-beta_mean_time)').quantile([.5,.9,.99,1]).to_dict())
    print('ring, target by indexed window')
    print(nearest.query('family == "ring" and policy == "indexed"')[['n','k_min','k_max',*columns]].to_string(index=False))
    print('m ambiguity conditional on alpha and beta (same window within each size)')
    near = df.query('policy == "greedy" and k_min == 2 and k_max == 32').copy()
    print(near.loc[(near.alpha-1.5).abs().le(.05) & (near.beta-.2).abs().le(.05)]
          .groupby(['n','family']).m.agg(['min','max','size']).to_string())
    counterexample()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, size in zip(axs[0], [256, 1024]):
        q = df.query('n == @size and policy == "greedy" and seed == 0 and k_min == 2')
        for family, sub in q.groupby('family'):
            ax.plot(sub.alpha, sub.beta, label=family, lw=1.5)
        ax.scatter(1.5,.2,marker='x',s=72,color='black',zorder=5,label='Target')
        ax.set(title=f'{size:,} nodes · greedy clusters · K = 2–32',
               xlabel=r'Variance exponent $\alpha$', ylabel=r'Timescale exponent $\beta$')
        ax.legend(ncol=2, frameon=False, fontsize=8)
    for ax, column, label in zip(axs[1], ['alpha','beta'], [r'$\alpha$',r'$\beta$']):
        q = df.query('n == 256 and family == "ring" and policy == "indexed"')
        for (lo, hi), sub in q.groupby(['k_min','k_max']):
            ax.plot(-np.log10(1-sub.m), sub[column], label=f'{lo}–{hi}', lw=1.5)
        ax.set(title='Ring · varying the fitted cluster sizes',
               xlabel=r'Proximity to instability $-\log_{10}(1-m)$', ylabel=label)
        ax.legend(title='K window', frameon=False, fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.savefig(path / 'exponent_sweep.png', dpi=180)


if __name__ == '__main__':
    main()
