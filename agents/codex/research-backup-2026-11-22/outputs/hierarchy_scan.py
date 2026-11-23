"""Brute-force a weighted dyadic hierarchy using its exact Haar spectrum."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import root

from griffiths_sweep import graph, groups, slope


def spectrum(levels, phi):
    j = np.arange(1, levels + 1)
    n = 2**levels
    mass = (2/phi)**(j-1)
    lam = np.r_[np.cumsum(mass) - 2*mass, mass.sum()] / mass.sum()
    k = 2.0**np.arange(levels)
    detail = np.where(k[:,None] < 2.0**j, k[:,None]**2 / 2.0**j, 0)
    return lam, np.column_stack([detail, k**2/n])


def validate():
    n = 256
    lam, modes = eigh(graph('hierarchy',n,np.random.default_rng(0)))
    lam /= lam.max()
    detail, weight = spectrum(8, 4)
    q = np.array([((g @ modes)**2).mean(axis=0) for g in groups(None,'indexed',n,np.arange(n))])
    for m in (.1,.9,.999):
        np.testing.assert_allclose(q @ (1/(1-(m*lam)**2)),
                                   weight @ (1/(1-(m*detail)**2)), rtol=1e-10)
        np.testing.assert_allclose(q @ (1/((1-(m*lam)**2)*(1-m*lam))),
                                   weight @ (1/((1-(m*detail)**2)*(1-m*detail))), rtol=1e-10)


def main():
    validate()
    path = Path(__file__).parent
    ms = np.unique(np.r_[np.linspace(0,.99,501), 1-np.geomspace(.1,1e-9,2001)])
    phis = np.unique(np.r_[np.geomspace(1.1,32,501),4])
    windows = [(1,5),(2,6),(3,7),(4,8),(5,9),(6,10)]
    result = []
    fit = np.arange(-2,3) / 10
    for levels in (8,10,12,15):
        for phi in phis:
            lam, q = spectrum(levels, phi)
            a = ms[:, None] * lam
            v = (1/(1-a*a)) @ q.T
            w = (1/((1-a*a)*(1-a))) @ q.T
            lv, lt = np.log2(v), np.log2(w/v)
            for lo, hi in windows:
                if hi >= levels:
                    continue
                alpha, beta = lv[:,lo:hi+1] @ fit, lt[:,lo:hi+1] @ fit
                error = np.hypot(alpha-1.5,beta-.2)
                idx, peak = np.argmin(error), np.argmax(beta)
                result.append((2**levels,phi,2**lo,2**hi,ms[idx],alpha[idx],beta[idx],
                               error[idx],ms[peak],alpha[peak],beta[peak]))
    with (path/'hierarchy_scan.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n','phi','k_min','k_max','best_m','alpha','beta','error',
                         'peak_m','alpha_at_peak','peak_beta'])
        writer.writerows(result)
    data = np.asarray(result)
    fig, axs = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for lo in (2,4,8,16,32,64):
        sub = data[(data[:,0] == 32768) & (data[:,2] == lo)]
        for ax, y in zip(axs, [sub[:,7],-np.log10(1-sub[:,4]),sub[:,10]]):
            ax.semilogx(sub[:,1],y,label=f'{lo}–{16*lo}')
    axs[0].legend(title='K window',frameon=False,fontsize=8)
    for ax,label in zip(axs,['Closest distance to target','Proximity at closest point','Peak timescale exponent']):
        ax.set(xlabel=r'Weight decay parameter $\Phi$',ylabel=label)
        ax.axvline(4,color='black',ls=':',lw=1)
        ax.grid(alpha=.2)
    fig.savefig(path/'hierarchy_scan.png',dpi=180)
    for n in (256,1024,4096,32768):
        sub = data[data[:,0] == n]
        print(n,'best', sub[np.argmin(sub[:,7])])
        print(n,'Phi=4',sub[sub[:,1]==4])
    print('grid fits',len(result)*len(ms))
    solutions = []
    for lo, hi in windows:
        sub = data[(data[:,0] == 32768) & (data[:,2] == 2**lo)]
        _, phi0, _, _, m0, *_ = sub[np.argmin(sub[:,7])]
        win = slice(lo, hi+1)

        def residual(parameters):
            phi, m = parameters
            lam, q = spectrum(15, phi)
            a = m*lam
            v = q @ (1/(1-a*a))
            w = q @ (1/((1-a*a)*(1-a)))
            return np.array([slope(v[win]) - 1.5, slope((w/v)[win]) - .2])

        sol = root(residual, [phi0, m0], tol=1e-11)
        if not sol.success or np.linalg.norm(sol.fun) > 1e-9:
            raise RuntimeError(f'Failed to refine target fit at K = {2**lo}–{2**hi}')
        phi, m = sol.x
        solutions.append((32768, 2**lo, 2**hi, phi, m, np.linalg.norm(sol.fun)))
    with (path/'hierarchy_target_solutions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n','k_min','k_max','phi','m','target_error'])
        writer.writerows(solutions)


if __name__ == '__main__':
    main()
