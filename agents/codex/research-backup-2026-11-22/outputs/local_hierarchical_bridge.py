"""Explore local + hierarchical coupling at fixed dynamics and observation windows.

All matrices are symmetric and nonnegative. Exact stationary moments are used.
The target is the rounded pair (1.5,.2), with equal exponent weighting.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
TARGET=np.array([1.5,.2])
WINDOWS=[('K2–64',1,6),('K2–16',1,4),('K4–32',2,5),('K8–64',3,6)]

def ring(n):
    i=np.arange(n)
    distance=np.minimum(abs(i[:,None]-i),n-abs(i[:,None]-i))
    return ((distance>0)&(distance<=2)).astype(float)/4

def hierarchy(n,phi):
    i=np.arange(n)
    xor=i[:,None]^i
    level=np.floor(np.log2(np.maximum(1,xor)))
    a=np.where(xor>0,phi**(-level),0)
    return a/a.sum(axis=1)[0]

def balanced(a):
    scale=np.ones(len(a))
    for _ in range(3000):
        rows=scale*(a@scale)
        change=np.max(abs(rows-1))
        if change<1e-11:break
        scale*=1/np.sqrt(np.maximum(rows,1e-300))
    b=scale[:,None]*a*scale[None,:]
    if np.max(abs(b.sum(axis=1)-1))>1e-8:
        raise RuntimeError('Symmetric row balancing did not converge')
    return b

def make(spec,n):
    family=spec['family'];g=spec.get('g',0);phi=spec.get('phi',4.)
    rng=np.random.default_rng(spec.get('seed',0))
    if family=='ring':return ring(n)
    if family=='hierarchy':return hierarchy(n,phi)
    if family=='ring+hierarchy':
        return (1-g)*ring(n)+g*hierarchy(n,phi)
    if family=='module rings+hierarchy':
        block=spec['block'];modules=n//block
        return ((1-g)*np.kron(np.eye(modules),ring(block))+
                g*np.kron(hierarchy(modules,phi),np.ones((block,block))/block))
    if family=='ring+uniform':
        return (1-g)*ring(n)+g*(np.ones((n,n))-np.eye(n))/(n-1)
    if family=='ring+random cycles':
        random=np.zeros((n,n))
        for _ in range(2):
            order=rng.permutation(n)
            for i,j in zip(order,np.roll(order,-1)):
                random[i,j]+=1;random[j,i]+=1
        return (1-g)*ring(n)+g*random/4
    if family=='powerlaw ring':
        i=np.arange(n)
        distance=np.minimum(abs(i[:,None]-i),n-abs(i[:,None]-i))
        a=np.where(distance>0,np.maximum(1,distance)**(-spec['power']),0)
        return a/a.sum(axis=1)[0]
    if family=='ring+stochastic hierarchy':
        i=np.arange(n)
        xor=i[:,None]^i
        level=np.floor(np.log2(np.maximum(1,xor)))
        probability=np.where(xor>0,phi**(-level),0)
        upper=np.triu(rng.random((n,n))<probability,1)
        h=(upper|upper.T).astype(float)
        h/=h.sum()/n
        a=(1-g)*ring(n)+g*h
        return balanced(a) if spec.get('balance',False) else a
    raise ValueError(family)

def specifications():
    yield dict(family='ring')
    for phi in (2.5,2.83,3.,4.,5.,8.):
        yield dict(family='hierarchy',phi=phi)
        for g in (.001,.003,.01,.03,.1,.3,.6,.9):
            yield dict(family='ring+hierarchy',phi=phi,g=g)
        for block in (16,32,64):
            for g in (.003,.01,.03,.1,.3,.6,.9):
                yield dict(family='module rings+hierarchy',phi=phi,g=g,block=block)
    for g in (.001,.003,.01,.03,.1,.3,.6,.9):
        yield dict(family='ring+uniform',g=g)
        for seed in range(3):
            yield dict(family='ring+random cycles',g=g,seed=seed)
    for power in (.75,1.2,1.5,1.8,2.2,3.):
        yield dict(family='powerlaw ring',power=power)
    for phi in (2.5,2.83,3.2):
        for g in (.01,.1,.3,.6,.9):
            for seed in range(3):
                for balance in (False,True):
                    yield dict(family='ring+stochastic hierarchy',phi=phi,g=g,
                               seed=seed,balance=balance)

def fixed_weights(modes):
    n=len(modes)
    projection=modes.copy()
    qs=[]
    while len(projection)>=4:
        qs.append((projection**2).mean(axis=0))
        projection=projection.reshape(len(projection)//2,2,n).sum(axis=1)
    return np.array(qs)

def fits(v,t):
    result=[]
    for _,lo,hi in WINDOWS:
        x=np.arange(lo,hi+1,dtype=float)*np.log(2)
        xc=x-x.mean()
        result.append(np.column_stack((np.log(v[:,lo:hi+1])@xc/(xc@xc),
                                       np.log(t[:,lo:hi+1])@xc/(xc@xc))))
    return np.stack(result,axis=1)

def summarize(spec_id,spec,n,deltas,v,t,partition):
    ab=fits(v,t)
    primary=np.linalg.norm(ab[:,0,:]-TARGET,axis=1)
    moving=np.sqrt(np.mean(np.sum((ab[:,1:,:]-TARGET)**2,axis=2),axis=1))
    local_a=np.diff(np.log2(v[:,3:7]),axis=1)
    local_b=np.diff(np.log2(t[:,3:7]),axis=1)
    interval=np.sqrt(np.mean((local_a-1.5)**2+(local_b-.2)**2,axis=1))
    # Joint robustness score rewards both moving global windows and interval slopes.
    score=np.sqrt((moving*moving+interval*interval)/2)
    out=[]
    for objective,values in [('primary',primary),('robust',score)]:
        j=np.argmin(values)
        out.append(dict(id=spec_id,n=n,partition=partition,objective=objective,
                        delta=float(deltas[j]),m=float(1-deltas[j]),
                        alpha=float(ab[j,0,0]),beta=float(ab[j,0,1]),
                        primary_distance=float(primary[j]),moving_window_error=float(moving[j]),
                        interval_error=float(interval[j]),robust_score=float(score[j]),
                        alpha_interval_min=float(local_a[j].min()),
                        alpha_interval_max=float(local_a[j].max()),
                        beta_interval_min=float(local_b[j].min()),
                        beta_interval_max=float(local_b[j].max()),
                        **spec))
    return out

def spectral_diagnostics(b,eigen,modes):
    lead=modes[:,-1]
    return dict(Perron_participation_fraction=float(1/(len(b)*np.sum(lead**4))),
                spectral_gap=float(1-eigen[-2]),
                components=int(connected_components(b>0,directed=False,return_labels=False)),
                row_strength_cv=float(np.std(b.sum(axis=1))/np.mean(b.sum(axis=1))),
                positive_mode_count=int(np.sum(eigen>0)))

def screen():
    n=256
    delta=np.geomspace(1.,1e-8,801)
    specs=list(specifications())
    results=[];diagnostics=[]
    for index,spec in enumerate(specs):
        b=make(spec,n)
        ev,U=eigh(b);ev/=max(abs(ev))
        q=fixed_weights(U)
        h=delta[:,None]+(1-delta[:,None])*(1-ev)
        gain=1/(h*(2-h))
        v=gain@q.T
        t=((gain/h)@q.T)/v
        np.testing.assert_allclose(v[0],2.**np.arange(len(q)),rtol=1e-11)
        results.extend(summarize(index,spec,n,delta,v,t,'fixed spatial'))
        diagnostics.append(dict(id=index,n=n,**spec,**spectral_diagnostics(b,ev,U)))
        if index%25==0:
            print('Screen',index,'of',len(specs),flush=True)
    pd.DataFrame(results).to_csv(ROOT/'bridge-screen.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT/'bridge-spectral-diagnostics.csv',index=False)
    (ROOT/'bridge-specifications.json').write_text(json.dumps(specs,indent=2)+'\n')
    return pd.DataFrame(results)

def greedy_pairs(cov):
    sd=np.sqrt(np.diag(cov))
    i,j=np.triu_indices(len(cov),1)
    corr=cov[i,j]/(sd[i]*sd[j])
    order=np.argsort(-np.round(corr,12),kind='stable')
    used=np.zeros(len(cov),bool)
    left=[];right=[]
    for k in order:
        a,b=i[k],j[k]
        if not used[a] and not used[b]:
            used[a]=used[b]=True
            left.append(a);right.append(b)
            if len(left)==len(cov)//2:break
    return np.array(left),np.array(right)

def cascade(b,eigen,modes,delta,estimator='weighted',lag=None):
    h=delta+(1-delta)*(1-eigen)
    a=1-h
    gain=1/(h*(2-h))
    if lag is None:
        tau=1/h
    else:
        # Trapezoidal integral over lags 0,...,lag, in simulation time steps.
        tau=(1-a**(lag+1))/h-.5*(1+a**lag)
    cov=(modes*gain)@modes.T
    projection=modes.copy()
    variance=[];times=[];mean_times=[]
    while len(cov)>=4:
        vd=np.diag(cov)
        wd=projection**2@(gain*tau)
        variance.append(vd.mean())
        times.append(wd.sum()/vd.sum())
        mean_times.append(np.mean(wd/vd))
        if len(cov)==4:break
        i,j=greedy_pairs(cov)
        cov=cov[np.ix_(i,i)]+cov[np.ix_(i,j)]+cov[np.ix_(j,i)]+cov[np.ix_(j,j)]
        projection=projection[i]+projection[j]
    return np.array(variance),np.array(times),np.array(mean_times)

def select_ids(df):
    chosen=set()
    for family,g in df.groupby('family'):
        chosen.update(g[g.objective=='robust'].nsmallest(2,'robust_score').id.astype(int))
        chosen.update(g[g.objective=='primary'].nsmallest(1,'primary_distance').id.astype(int))
    # Include explicit reference hierarchies, weak/strong bridges, and paired
    # balanced/unbalanced stochastic examples when selected.
    chosen.update(df[(df.family=='hierarchy')&df.phi.isin([2.83,4.,8.])].id.astype(int))
    specs=json.loads((ROOT/'bridge-specifications.json').read_text())
    for k in list(chosen):
        s=specs[k]
        if s['family']=='ring+stochastic hierarchy':
            partner={**s,'balance':not s['balance']}
            for j,z in enumerate(specs):
                if z==partner:chosen.add(j)
    return sorted(chosen)

def followup():
    df=pd.read_csv(ROOT/'bridge-screen.csv')
    specs=json.loads((ROOT/'bridge-specifications.json').read_text())
    chosen=select_ids(df)
    rows=[]
    delta=np.geomspace(1.,1e-8,101)
    for n in (256,512):
        for k in chosen:
            spec=specs[k]
            b=make(spec,n);ev,U=eigh(b);ev/=max(abs(ev))
            v=[];t=[];mt=[]
            for d in delta:
                vv,tt,mm=cascade(b,ev,U,d)
                v.append(vv);t.append(tt);mt.append(mm)
            rows.extend(summarize(k,spec,n,delta,np.array(v),np.array(t),'greedy weighted'))
            rows.extend(summarize(k,spec,n,delta,np.array(v),np.array(mt),'greedy equal weights'))
            print('Greedy',n,k,spec,flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'bridge-greedy-followup.csv',index=False)
    return pd.DataFrame(rows)

def plot():
    data=pd.read_csv(ROOT/'bridge-screen.csv')
    family_order=['ring','hierarchy','ring+hierarchy','module rings+hierarchy',
                  'ring+uniform','ring+random cycles','powerlaw ring','ring+stochastic hierarchy']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    colors=plt.colormaps['tab10'](np.arange(8))
    for color,family in zip(colors,family_order):
        s=data[(data.family==family)&(data.objective=='primary')]
        axes[0].scatter(s.alpha,s.beta,s=18,alpha=.6,color=color,label=family)
        r=data[(data.family==family)&(data.objective=='robust')]
        axes[1].scatter(r.primary_distance,r.robust_score,s=18,alpha=.6,color=color,label=family)
    axes[0].scatter(1.5,.2,marker='x',color='black',s=70,zorder=5)
    axes[0].set(xlabel='Alpha at closest global fit',ylabel='Beta at closest global fit')
    axes[1].set(xlabel='Distance to target in K=2–64 fit',
                ylabel='Joint error across moving windows and interval slopes')
    axes[1].legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('Local–hierarchical bridge screen · N=256 · fixed spatial clusters\n'
                 'A close single-window fit and stable exponents are scored separately')
    fig.savefig(ROOT/'bridge-screen.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['screen','followup','plot'],default='screen',nargs='?')
    args=parser.parse_args()
    if args.stage=='screen':screen();plot()
    elif args.stage=='followup':followup()
    else:plot()
