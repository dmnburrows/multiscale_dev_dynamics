"""Continuum lattice predictions, integer d=1..7; no finite-N claim."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from continuum_dimension_peaks import pair

ROOT=Path(__file__).parent
SIDE=32.
DIMENSIONS=range(1,8)
DIMS=np.array(list(DIMENSIONS))
XS=np.geomspace(1e-5,60,1000)
# Precompute universal alpha-beta curves once per dimension.
AB={d:np.array([pair(x,d) for x in XS]) for d in DIMENSIONS}
colors=plt.colormaps['viridis'](np.linspace(.08,.92,7))
fig,axes=plt.subplots(5,3,figsize=(15,19),layout='constrained')
rows=[]
for row,neighbours in enumerate(range(2,7)):
    axs=axes[row]
    for d,color in zip(DIMENSIONS,colors):
        ab=AB[d]
        # lambda(q)=1-c|q|^2+... for equal-strength axis-aligned
        # connections at offsets +-1,...,+-neighbours on each axis.
        c=np.sum(np.arange(1,neighbours+1,dtype=float)**2)/(2*neighbours*d)
        m=1/(1+c*(XS/SIDE)**2)
        axs[0].plot(ab[:,0],ab[:,1],color=color,label=f'd = {d}',lw=1.6)
        axs[1].plot(m,ab[:,0],color=color,lw=1.6)
        axs[2].plot(m,ab[:,1],color=color,lw=1.6)
        peak=int(np.argmax(ab[:,1]))
        # A dot denotes an observed interior maximum, not an endpoint.
        if 0<peak<len(XS)-1:
            axs[0].scatter(ab[peak,0],ab[peak,1],s=18,color=color)
            axs[1].scatter(m[peak],ab[peak,0],s=18,color=color)
            axs[2].scatter(m[peak],ab[peak,1],s=18,color=color)
        rows.extend(zip([neighbours]*len(XS),[d]*len(XS),[SIDE]*len(XS),XS,m,ab[:,0],ab[:,1]))
    axs[0].scatter(1.5,.2,marker='x',color='black',s=40)
    axs[0].axvline(1.5,color='grey',ls=':',lw=.7)
    axs[0].axhline(.2,color='grey',ls=':',lw=.7)
    axs[1].axhline(1.5,color='grey',ls=':',lw=.7)
    axs[2].axhline(.2,color='grey',ls=':',lw=.7)
    axs[0].set(xlabel='Local variance exponent alpha',ylabel='Local timescale exponent beta',
               xlim=(.98,2.02),ylim=(-.005,.53))
    axs[1].set(xlabel='Coupling m',ylabel='Local variance exponent alpha',ylim=(.98,2.02))
    axs[2].set(xlabel='Coupling m',ylabel='Local timescale exponent beta',ylim=(-.005,.53))
    for ax in axs[1:]:
        ax.set_xscale('function',
            functions=(lambda m:-np.log10(np.maximum(1-np.asarray(m),1e-12)),
                       lambda x:1-10.**(-np.asarray(x))))
        ax.set_xlim(.9,.9999999)
        ax.set_xticks([.9,.99,.999,.9999,.99999,.999999,.9999999])
        ax.set_xticklabels(['.9','.99','.999','.9999','.99999','.999999','.9999999'],
                           rotation=35,ha='right',fontsize=8)
    for col,title in enumerate(('alpha versus beta','alpha versus m','beta versus m')):
        axs[col].set_title(f'{neighbours} neighbours per side per axis · {title}',fontsize=10)
        axs[col].grid(alpha=.15)
    axs[0].legend(frameon=False,fontsize=8,ncol=2)
fig.suptitle('Continuum lattice predictions · colour = dimension d · cluster side = 32 lattice spacings\n'
             'Rows vary connection range; total degree = 2 × range × d. '
             'Dots mark interior beta peaks.',fontsize=12)
fig.savefig(ROOT/'ring-range-dimensions-1-to-7.png',dpi=160)
pd.DataFrame(rows,columns=['neighbours_per_side_per_axis','dimension','cluster_side','x',
                          'm','alpha','beta']).to_csv(ROOT/'ring-range-dimensions-1-to-7.csv',index=False)
