"""Exact finite periodic lattices, using permutation-compressed Fourier sums.

Axis length L=16, d=1..7, r=2..6 neighbours on EACH side of EACH axis.
Total neurons L**d; degree 2*r*d. No continuum approximation or simulated noise.
Cube clusters of sides 1,2,4,8. All finite stationary moments are computed exactly
up to floating-point arithmetic, without constructing the full adjacency.
"""
from itertools import combinations_with_replacement
from math import factorial
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
L=16
DIMS=range(1,8)
RANGES=range(2,7)
SIDES=np.array([1,2,4,8])
DELTAS=np.geomspace(1,1e-8,601)

def compressed_modes(length,d,sides):
    frequencies=np.arange(length//2+1)
    multiplicity=np.full(len(frequencies),2.)
    multiplicity[0]=1
    if length%2==0:multiplicity[-1]=1
    counts=np.array([np.bincount(items,minlength=len(frequencies))
                     for items in combinations_with_replacement(frequencies,d)])
    permutations=np.array([factorial(d)/np.prod([factorial(int(v)) for v in row])
                            for row in counts])
    # A cube cluster has separable squared Fourier overlaps; permutations
    # of coordinate frequencies have identical eigenvalues and overlaps.
    weights=[]
    for side in sides:
        wave=2*np.pi*frequencies/length
        one=(side*np.sinc(side*wave/(2*np.pi))/np.sinc(wave/(2*np.pi)))**2
        one[(frequencies>0)&((frequencies*side)%length==0)]=0
        base=one*multiplicity/length
        q=permutations*np.prod(base[None,:]**counts,axis=1)
        np.testing.assert_allclose(q.sum(),float(side)**d,rtol=1e-12)
        weights.append(q)
    return frequencies,counts,np.array(weights)

def evaluate(length,d,r,sides,delta):
    if not 1<=r<length/2:raise ValueError('Offsets must be distinct on the periodic axis')
    freq,counts,q=compressed_modes(length,d,sides)
    wave=2*np.pi*freq/length
    axisgap=np.mean([2*np.sin(offset*wave/2)**2 for offset in range(1,r+1)],axis=0)
    gap=counts@axisgap/d
    delta=np.atleast_1d(delta)
    h=delta[:,None]+(1-delta[:,None])*gap
    gain=1/(h*(2-h))
    v=gain@q.T
    w=(gain/h)@q.T
    return v,w/v,len(gap)

def regress(values,x):
    xc=x-x.mean()
    return np.log(values)@xc/(xc@xc)

def check_dense():
    results=[]
    for length,d,r in [(16,1,6),(8,2,2),(4,3,1)]:
        N=length**d
        coords=np.array(np.unravel_index(np.arange(N),(length,)*d)).T
        adjacency=np.zeros((N,N))
        for i,coordinate in enumerate(coords):
            for axis in range(d):
                for step in range(1,r+1):
                    for sign in (-1,1):
                        other=coordinate.copy()
                        other[axis]=(other[axis]+sign*step)%length
                        j=np.ravel_multi_index(tuple(other),(length,)*d)
                        adjacency[i,j]=1
        np.testing.assert_array_equal(adjacency.sum(axis=1),np.full(N,2*r*d))
        b=adjacency/(2*r*d)
        sides=np.array([1,2])
        for m in (.4,.95,.999):
            cov=solve_discrete_lyapunov(m*b,np.eye(N))
            integrated=np.linalg.solve(np.eye(N)-m*b,cov)
            vv=[];tt=[]
            for side in sides:
                membership=np.all(coords<side,axis=1).astype(float)
                v=membership@cov@membership
                vv.append(v);tt.append((membership@integrated@membership)/v)
            v,t,_=evaluate(length,d,r,sides,1-m)
            error=max(np.max(abs(v[0]/np.array(vv)-1)),np.max(abs(t[0]/np.array(tt)-1)))
            np.testing.assert_allclose(v[0],vv,rtol=1e-8)
            np.testing.assert_allclose(t[0],tt,rtol=1e-8)
            results.append(dict(L=length,d=d,r=r,m=m,max_relative_error=float(error)))
    return results

def draw(data,convention):
    fig,axs=plt.subplots(5,3,figsize=(15,19),layout='constrained')
    colors=plt.colormaps['viridis'](np.linspace(.08,.92,7))
    for row,r in enumerate(RANGES):
        for d,color in zip(DIMS,colors):
            sub=data[(data.r==r)&(data.d==d)&(data.convention==convention)].sort_values('m')
            a=sub.alpha.to_numpy();b=sub.beta.to_numpy();m=sub.m.to_numpy()
            index=np.argmax(b)
            axs[row,0].plot(a,b,color=color,label=f'd={d} · degree={2*r*d}',lw=1.5)
            axs[row,1].plot(m,a,color=color,lw=1.5)
            axs[row,2].plot(m,b,color=color,lw=1.5)
            axs[row,0].scatter(a[index],b[index],s=16,color=color)
            axs[row,1].scatter(m[index],a[index],s=16,color=color)
            axs[row,2].scatter(m[index],b[index],s=16,color=color)
        axs[row,0].scatter(1.5,.2,marker='x',color='black',s=35)
        axs[row,0].axvline(1.5,color='grey',ls=':',lw=.7)
        axs[row,0].axhline(.2,color='grey',ls=':',lw=.7)
        axs[row,0].set(xlabel='Variance exponent alpha',ylabel='Timescale exponent beta')
        axs[row,1].set(xlabel='Coupling m',ylabel='Variance exponent alpha')
        axs[row,2].set(xlabel='Coupling m',ylabel='Timescale exponent beta')
        for col,ax in enumerate(axs[row]):
            ax.set_title(f'{r} neighbours per side per axis',fontsize=10)
            ax.grid(alpha=.16)
            if col>0:
                ax.set_xscale('function',
                    functions=(lambda m:-np.log10(np.maximum(1-np.asarray(m),1e-12)),
                               lambda x:1-10.**(-np.asarray(x))))
                ax.set_xlim(0,1-1e-8)
                ticks=[0,.9,.99,.999,.9999,.99999,.999999,.9999999,.99999999]
                ax.set_xticks(ticks)
                ax.set_xticklabels(['0','.9','.99','.999','.9999','.99999','.999999',
                                    '.9999999','.99999999'],rotation=40,ha='right',fontsize=7)
        axs[row,0].legend(frameon=False,fontsize=7,ncol=2)
    description=('Cube side 4 to 8; K = 4^d to 8^d; interval slopes divided by d'
                 if convention=='interval' else 'Cube sides 1, 2, 4, 8; equal-weight OLS versus log K')
    fig.suptitle(f'Finite periodic lattices · axis length L=16 · N=16^d · degree=2rd\n'
                 f'{description} · exact stationary Fourier sums · dots mark sampled beta maxima',
                 fontsize=12)
    fig.savefig(ROOT/f'finite-lattice-dimensions-{convention}.png',dpi=160)
    # Export individual rows for legibility.
    for row,r in enumerate(RANGES):
        bbox=matplotlib.transforms.Bbox.union([ax.get_tightbbox(fig.canvas.get_renderer())
                                               for ax in axs[row]])
        bbox=bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
        fig.savefig(ROOT/f'finite-lattice-{convention}-r{r}.png',dpi=160,bbox_inches=bbox)
    plt.close(fig)

if __name__=='__main__':
    checks=check_dense()
    rows=[]
    configurations=[]
    for r in RANGES:
        for d in DIMS:
            v,t,count=evaluate(L,d,r,SIDES,DELTAS)
            np.testing.assert_allclose(v[0],SIDES.astype(float)**d,rtol=1e-12)
            np.testing.assert_allclose(t[0],np.ones(len(SIDES)),rtol=1e-12)
            logk=d*np.log(SIDES)
            for convention in ('interval','global'):
                sel=np.array([2,3]) if convention=='interval' else np.arange(4)
                a=regress(v[:,sel],logk[sel])
                b=regress(t[:,sel],logk[sel])
                for delta,alpha,beta in zip(DELTAS,a,b):
                    rows.append((r,d,L,L**d,2*r*d,convention,1-delta,delta,alpha,beta))
            configurations.append(dict(r=r,d=d,N=L**d,degree=2*r*d,
                                       compressed_modes=count))
        print('Neighbour range completed',r,flush=True)
    df=pd.DataFrame(rows,columns=['r','d','L','N','degree','convention','m','delta','alpha','beta'])
    df.to_csv(ROOT/'finite-lattice-dimension-sweep.csv',index=False)
    peaks=df.loc[df.groupby(['r','d','convention']).beta.idxmax()]
    peaks.to_csv(ROOT/'finite-lattice-dimension-peaks.csv',index=False)
    (ROOT/'finite-lattice-dimension-checks.json').write_text(
        json.dumps(dict(dense_checks=checks,configurations=configurations,
                        note='Exact finite moments; peaks selected on 601-point log-delta grid'),indent=2)+'\n')
    draw(df,'interval')
    draw(df,'global')
    print('All calculations and figures complete',flush=True)
