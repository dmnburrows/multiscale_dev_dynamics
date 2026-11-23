"""Reproducible peak audit: exact spatial sums and finite greedy cascades.

Run with numpy, scipy, pandas, matplotlib. Numerical peak searches use delta=1-m.
Stationary moments are exact. "Local" means a consecutive K-to-2K secant.
The global estimator is OLS with intercept, equal weight per dyadic size.
"""
import argparse
import csv
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve_discrete_lyapunov
from scipy.optimize import brentq, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from griffiths_sweep import groups
from hierarchy_scan import spectrum as haar_spectrum

ROOT=Path(__file__).parent
CONFIGS=[('local',1,2),('local',3,4),('local',5,6),('local',7,8),
         ('global',1,6),('global',2,7),('global',3,8)]
PHIS=(3.,4.,5.,7.765787162650601)

def measure(K,v,t,config):
    convention,lo,hi=config
    keep=(K>=2**lo)&(K<=2**hi)
    x=np.log(K[keep]);xc=x-x.mean()
    def regression(y):
        logy=np.log(y[keep])
        slope=xc@logy/(xc@xc)
        if len(x)==2 or np.ptp(logy) < 1e-12:
            r2=np.nan
        else:
            r2=np.corrcoef(x,logy)[0,1]**2
        return slope,r2
    alpha,ar2=regression(v);beta,br2=regression(t)
    return np.array([alpha,beta,ar2,br2])

def factory(family,n,phi=None):
    L=int(np.log2(n))
    K=2.**np.arange(L)
    if family=='Ring':
        k=2*np.pi*np.fft.rfftfreq(n)
        gap=np.sin(k/2)**2+np.sin(k)**2
        mult=np.full(len(k),2.);mult[[0,-1]]=1
        overlaps=np.array([(size*np.sinc(size*k/(2*np.pi))/np.sinc(k/(2*np.pi)))**2
                           for size in K])*mult/n
    else:
        # Haar spectrum is independently checked against the exact dense matrix.
        eig,overlaps=haar_spectrum(L,phi)
        gap=1-eig
    def evaluate(delta):
        h=delta+(1-delta)*gap
        gain=1/(h*(2-h))
        v=overlaps@gain
        t=(overlaps@(gain/h))/v
        return K,v,t
    return evaluate

def search_peak(evaluate,config,points):
    x=np.linspace(np.log(1e-10),0,points)
    vals=np.array([measure(*evaluate(np.exp(z)),config) for z in x])
    # Refine every sampled local maximum, then choose the highest.
    candidates=[(x[0],vals[0,1]),(x[-1],vals[-1,1])]
    for j in range(1,len(x)-1):
        if vals[j,1]>=vals[j-1,1] and vals[j,1]>=vals[j+1,1]:
            opt=minimize_scalar(lambda z:-measure(*evaluate(np.exp(z)),config)[1],
                bounds=(x[j-1],x[j+1]),method='bounded',options={'xatol':1e-11})
            candidates.append((opt.x,-opt.fun))
    xp,_=max(candidates,key=lambda z:z[1])
    peak=measure(*evaluate(np.exp(xp)),config)
    crossings=[]
    for j in range(len(x)-1):
        if (vals[j,0]-1.5)*(vals[j+1,0]-1.5)<0:
            z=brentq(lambda z:measure(*evaluate(np.exp(z)),config)[0]-1.5,x[j],x[j+1])
            crossings.append(dict(delta=float(np.exp(z)),
                                  beta=float(measure(*evaluate(np.exp(z)),config)[1])))
    closest=min(crossings,key=lambda z:abs(np.log(z['delta'])-xp)) if crossings else None
    row=dict(delta_peak=float(np.exp(xp)),m_peak=float(1-np.exp(xp)),
             alpha_at_peak=float(peak[0]),beta_peak=float(peak[1]),
             alpha_r2_at_peak=float(peak[2]),beta_r2_at_peak=float(peak[3]),
             delta_at_alpha_1_5=closest['delta'] if closest else np.nan,
             beta_at_alpha_1_5=closest['beta'] if closest else np.nan,
             delta_ratio_peak_to_crossing=float(np.exp(xp)/closest['delta']) if closest else np.nan,
             alpha_crossing_count=len(crossings),
             peak_at_search_boundary=bool(xp in (x[0],x[-1])))
    return row,x,vals

def dense_checks():
    errors={}
    for family,phi in [('Ring',None),('HM',3.),('HM',5.)]:
        n=64;ids=np.arange(n)
        if family=='Ring':
            distance=np.minimum(abs(ids[:,None]-ids),n-abs(ids[:,None]-ids))
            adj=((distance>0)&(distance<=2)).astype(float)
        else:
            xor=ids[:,None]^ids
            lev=np.floor(np.log2(np.maximum(1,xor)))+1
            adj=np.where(xor>0,phi**(-lev),0)
        adj/=max(abs(np.linalg.eigvalsh(adj)))
        for m in (.3,.95,.999):
            s=solve_discrete_lyapunov(m*adj,np.eye(n))
            w=np.linalg.solve(np.eye(n)-m*adj,s)
            K,v,t=factory(family,n,phi)(1-m)
            vv=[];tt=[]
            for size in K.astype(int):
                g=np.eye(n).reshape(n//size,size,n).sum(axis=1)
                vv.append(np.trace(g@s@g.T)/(n//size))
                tt.append(np.trace(g@w@g.T)/np.trace(g@s@g.T))
            np.testing.assert_allclose(v,vv,rtol=1e-9)
            np.testing.assert_allclose(t,tt,rtol=1e-9)
            errors[f'{family}_{phi}_{m}']=float(max(np.max(abs(v/np.array(vv)-1)),
                                                     np.max(abs(t/np.array(tt)-1))))
    return errors

def analytical():
    checks=dense_checks()
    rows=[];traces=[];convergence=[]
    for n in (256,512,4096):
        for family,phi in [('Ring',None),*[('HM',p) for p in PHIS]]:
            evaluate=factory(family,n,phi)
            for config in CONFIGS:
                if config[2] >= int(np.log2(n)):
                    continue
                row,x,vals=search_peak(evaluate,config,601)
                repeat,_,_=search_peak(evaluate,config,1201)
                error=max(abs(row['beta_peak']-repeat['beta_peak']),
                          abs(row['alpha_at_peak']-repeat['alpha_at_peak']))
                if error>1e-5:
                    raise RuntimeError(f'Peak search failed resolution check: {family} {phi} {config}')
                convergence.append(error)
                row.update(family=family,phi=phi,n=n,clustering='contiguous' if family=='Ring' else 'aligned tree',
                           convention=config[0],K_min=2**config[1],K_max=2**config[2])
                rows.append(row)
                for z,value in zip(x,vals):
                    traces.append([family,phi,n,row['clustering'],config[0],row['K_min'],row['K_max'],
                                   1-np.exp(z),np.exp(z),*value])
            print('Exact sums done',family,phi,n,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'beta-peak-exact.csv',index=False)
    pd.DataFrame(traces,columns=['family','phi','n','clustering','convention','K_min','K_max',
                                'm','delta','alpha','beta','alpha_r2','beta_r2']).to_csv(
                                    ROOT/'beta-peak-trajectories.csv',index=False)
    (ROOT/'beta-peak-checks.json').write_text(json.dumps(
        dict(dense_relative_errors=checks,maximum_search_resolution_difference=max(convergence),
             convention='one-sided infinite-lag integrated autocorrelation; interval or OLS slopes',
             networks='degree-4 ring; deterministic weighted dyadic hierarchy'),indent=2)+'\n')
    return df

def canonical_greedy_groups(covariance,n):
    membership=np.eye(n)
    yield membership
    while len(membership)>2:
        c=membership@covariance@membership.T
        sd=np.sqrt(np.diag(c))
        ix,jx=np.triu_indices(len(c),1)
        correlations=c[ix,jx]/(sd[ix]*sd[jx])
        # Treat correlations equal to 12 decimal places as ties.
        # Stable enumeration then resolves ties by cluster-index order.
        ranking=np.argsort(-np.round(correlations,12),kind='stable')
        used=set();pairs=[]
        for index in ranking:
            i,j=int(ix[index]),int(jx[index])
            if i not in used and j not in used:
                used.update((i,j));pairs.append((i,j))
                if len(used)==len(c):break
        membership=membership[np.array(pairs)].sum(axis=1)
        yield membership

def greedy(points=181):
    # Every m has an independently recomputed exact-covariance greedy cascade.
    # Symmetric networks retain stochastic white drive; no simulated trajectories.
    n=256;ids=np.arange(n);allrows=[];curves=[]
    grid=np.linspace(np.log(1e-7),0,points)
    for family,phi in [('Ring',None),('HM',3.),('HM',4.),('HM',5.)]:
        if family=='Ring':
            distance=np.minimum(abs(ids[:,None]-ids),n-abs(ids[:,None]-ids))
            adj=((distance>0)&(distance<=2)).astype(float)
        else:
            xor=ids[:,None]^ids
            level=np.floor(np.log2(np.maximum(1,xor)))+1
            adj=np.where(xor>0,phi**(-level),0)
        ev,U=eigh(adj);ev/=max(abs(ev))
        @lru_cache(maxsize=None)
        def evaluate(logdelta):
            delta=np.exp(logdelta);a=(1-delta)*ev
            gain=1/(1-a*a);tau=1/(1-a)
            cov=(U*gain)@U.T
            gs=list(canonical_greedy_groups(cov,n))
            q=[(g@U)**2 for g in gs]
            v=np.array([z.mean(axis=0)@gain for z in q])
            w=np.array([z.mean(axis=0)@(gain*tau) for z in q])
            return 2.**np.arange(len(v)),v,w/v
        configs=[c for c in CONFIGS if c[2]<=7]
        tables=[evaluate(z) for z in grid]
        for config in configs:
            vals=np.array([measure(*tab,config) for tab in tables])
            j=vals[:,1].argmax()
            # Refine near the five highest sampled points. A jump can make
            # smooth scalar optimization inappropriate, so use finer grids.
            top=np.argsort(vals[:,1])[-5:]
            refined=[]
            for idx in top:
                local=np.linspace(grid[max(0,idx-1)],grid[min(len(grid)-1,idx+1)],17)
                for z in local:
                    value=measure(*evaluate(z),config)
                    refined.append((z,value))
            xp,val=max(refined,key=lambda item:item[1][1])
            row=dict(family=family,phi=phi,n=n,clustering='greedy',convention=config[0],
                     K_min=2**config[1],K_max=2**config[2],delta_peak=np.exp(xp),
                     m_peak=1-np.exp(xp),alpha_at_peak=val[0],beta_peak=val[1],
                     alpha_r2_at_peak=val[2],beta_r2_at_peak=val[3],
                     peak_at_search_boundary=bool(xp in (grid[0],grid[-1])))
            allrows.append(row)
            for z,value in zip(grid,vals):
                curves.append([family,phi,n,config[0],2**config[1],2**config[2],np.exp(z),*value])
        pd.DataFrame(allrows).to_csv(ROOT/'beta-peak-greedy.csv',index=False)
        pd.DataFrame(curves,columns=['family','phi','n','convention','K_min','K_max','delta',
                                    'alpha','beta','alpha_r2','beta_r2']).to_csv(
                                        ROOT/'beta-peak-greedy-trajectories.csv',index=False)
        print('Greedy done',family,phi,flush=True)

def plots():
    df=pd.read_csv(ROOT/'beta-peak-exact.csv')
    trace=pd.read_csv(ROOT/'beta-peak-trajectories.csv')
    colors={3.:'#4477aa',4.:'#228833',5.:'#cc6677',PHIS[-1]:'#aa3377'}
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,(family,phi) in enumerate([('Ring',None),('HM',4.)]):
        for row,convention in enumerate(('local','global')):
            ax=axs[row,column]
            use=(trace.family==family)&(trace.n==512)&(trace.convention==convention)
            if phi is not None:use &= np.isclose(trace.phi,phi)
            for (lo,hi),sub in trace[use].groupby(['K_min','K_max']):
                line,=ax.plot(sub.alpha,sub.beta,label=f'K {lo}–{hi}',lw=1.4)
                pp=df[(df.family==family)&(df.n==512)&(df.convention==convention)&
                      (df.K_min==lo)&(df.K_max==hi)]
                if phi is not None:pp=pp[np.isclose(pp.phi,phi)]
                ax.scatter(pp.alpha_at_peak,pp.beta_peak,color=line.get_color(),s=35,zorder=4)
            ax.axvline(1.5,color='black',lw=.7,ls=':')
            ax.axhline(.2,color='black',lw=.7,ls=':')
            ax.scatter([1.5],[.2],marker='x',color='black',s=55)
            ax.set(title=f'{family if phi is None else "Deterministic HM, Phi = 4"} · {convention} · N = 512',
                   xlabel='Variance exponent alpha',ylabel='Timescale exponent beta')
            ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(alpha=.15)
    fig.savefig(ROOT/'beta-peak-trajectories.png',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,convention in enumerate(('local','global')):
        for row,metric in enumerate(('alpha_at_peak','beta_peak')):
            ax=axs[row,col]
            for family,phi in [('Ring',None),*[('HM',p) for p in PHIS]]:
                for n,ls,marker in [(512,'-','o'),(4096,'--','s')]:
                    use=(df.family==family)&(df.n==n)&(df.convention==convention)
                    if phi is not None:use &= np.isclose(df.phi,phi)
                    sub=df[use].sort_values('K_min')
                    color='black' if family=='Ring' else colors[phi]
                    label=f'Ring, N={n}' if phi is None else f'HM Phi={phi:.3g}, N={n}'
                    ax.plot(sub.K_min,sub[metric],ls=ls,marker=marker,ms=4,color=color,label=label)
            ax.axhline(1.5 if row==0 else .2,color='grey',ls=':',lw=1)
            ax.set_xscale('log',base=2)
            ax.set(title=f'{convention.capitalize()} slopes',xlabel='Smallest fitted cluster size',
                   ylabel='Alpha at beta maximum' if row==0 else 'Maximum beta')
            ax.grid(alpha=.15)
    axs[0,1].legend(frameon=False,fontsize=7,ncol=2)
    fig.savefig(ROOT/'beta-peak-locations.png',dpi=180)
    plt.close(fig)
    # Coupling-coordinate plot for the standard fixed global window.
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for family,phi in [('Ring',None),*[('HM',p) for p in PHIS]]:
        use=(trace.family==family)&(trace.n==512)&(trace.convention=='global')&(trace.K_min==2)
        if phi is not None:use &= np.isclose(trace.phi,phi)
        sub=trace[use]
        label='Ring' if phi is None else f'HM Phi={phi:.3g}'
        for ax,metric in zip(axs,('alpha','beta')):
            ax.plot(-np.log10(sub.delta),sub[metric],label=label)
            ax.set(xlabel='Proximity to instability: -log10(1-m)',ylabel=metric)
            ax.grid(alpha=.15)
    axs[0].axhline(1.5,color='grey',ls=':');axs[1].axhline(.2,color='grey',ls=':')
    axs[1].legend(frameon=False,fontsize=8)
    fig.savefig(ROOT/'beta-peak-vs-coupling.png',dpi=180)
    plt.close(fig)
    greedy_path=ROOT/'beta-peak-greedy.csv'
    if greedy_path.exists():
        gd=pd.read_csv(greedy_path)
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,convention in zip(axs,('local','global')):
            for family,phi in [('Ring',None),('HM',3.),('HM',4.),('HM',5.)]:
                sub=gd[(gd.family==family)&(gd.convention==convention)].sort_values('K_min')
                label='Ring, greedy' if phi is None else f'HM Phi={phi:g}, greedy'
                line,=ax.plot(sub.alpha_at_peak,sub.beta_peak,'o-',label=label,ms=5)
                exact=df[(df.family==family)&(df.n==256)&(df.convention==convention)]
                if phi is not None:exact=exact[np.isclose(exact.phi,phi)]
                ax.scatter(exact.alpha_at_peak,exact.beta_peak,marker='x',color=line.get_color(),s=40)
            ax.axvline(1.5,color='grey',ls=':');ax.axhline(.2,color='grey',ls=':')
            ax.set(title=f'{convention.capitalize()} peaks: greedy N=256; crosses aligned N=256',
                   xlabel='Alpha at beta maximum',ylabel='Maximum beta')
            ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
        fig.savefig(ROOT/'beta-peak-greedy-comparison.png',dpi=180)
        plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--greedy',action='store_true')
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--greedy-points',type=int,default=361)
    args=parser.parse_args()
    if not args.plots_only:
        if args.greedy:greedy(args.greedy_points)
        else:analytical()
    plots()
