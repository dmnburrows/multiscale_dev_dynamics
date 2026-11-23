"""Compare the continuum diffusion prediction with exact discrete ring sums."""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from griffiths_sweep import slope


def continuum(x):
    shape = x + np.expm1(-x)
    alpha = -x * np.expm1(-x) / shape
    return shape, (3 - alpha) / 2


def run():
    n = 32768
    q = 2*np.pi*np.fft.rfftfreq(n)
    lam = (np.cos(q) + np.cos(2*q)) / 2
    multiplicity = np.full(len(q), 2)
    multiplicity[[0, -1]] = 1
    x_values = np.geomspace(.04, 20, 260)
    band_scales = 2. ** np.arange(-2, 3)
    path = Path(__file__).parent
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    rows = []
    for center in (16, 32, 64, 128, 256, 512):
        sizes = (center * band_scales).astype(int)
        filt = np.array([(k*np.sinc(k*q/(2*np.pi))/np.sinc(q/(2*np.pi)))**2
                         for k in sizes]) * multiplicity / n
        for x in x_values:
            m = 1 / (1 + 1.25*(x/center)**2)
            a = m*lam
            v = filt @ (1 / (1-a*a))
            t = (filt @ (1 / ((1-a*a)*(1-a)))) / v
            cv, ct = continuum(x*band_scales)
            rows.append((center, x, m, slope(v), slope(t), slope(cv), slope(ct)))
        band = np.array(rows[-len(x_values):])
        for ax, col in zip(axs[:2], [3, 4]):
            ax.semilogx(band[:,1],band[:,col],label=f'{sizes[0]}–{sizes[-1]}')
        axs[2].plot(band[:,3],band[:,4])
        mask = (band[:,1] >= .3) & (band[:,1] <= 5)
        print('ring', center, 'max errors x=.3..5', np.max(np.abs(band[mask,3:5]-band[mask,5:7]), axis=0))
    for ax, col, label in zip(axs[:2], [5, 6], [r'$\alpha$',r'$\beta$']):
        ax.semilogx(band[:,1],band[:,col],label='Continuum',color='black',ls='--')
        ax.set(xlabel=r'$K_{\rm mid}/\xi$', ylabel=label)
    axs[0].legend(title='K window',fontsize=8,frameon=False)
    axs[2].plot(band[:,5],band[:,6],color='black',ls='--',label='Continuum')
    axs[2].scatter(1.5,.2,marker='x',color='black',label='Target')
    axs[2].set(xlabel=r'$\alpha$',ylabel=r'$\beta$')
    axs[2].legend(frameon=False)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(path/'ring_collapse.png',dpi=180)
    with (path/'ring_collapse.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k_mid','x','m','alpha','beta','alpha_cont','beta_cont'])
        writer.writerows(rows)
    for scales in (band_scales, 2.**np.array([-.001, .001])):
        def exponents(x):
            var,time=continuum(x*scales)
            # slope() assumes unit log2 steps.
            step=np.log2(scales[1]/scales[0])
            return slope(var)/step, slope(time)/step
        peak=minimize_scalar(lambda x:-exponents(x)[1],bounds=(.05,10),method='bounded')
        target=minimize_scalar(lambda x:sum((np.array(exponents(x))-[1.5,.2])**2),
                               bounds=(.05,10),method='bounded')
        print('continuum peak', peak.x, exponents(peak.x),
              'closest to target',target.x,exponents(target.x))


if __name__ == '__main__':
    run()
