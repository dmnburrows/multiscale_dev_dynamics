"""Held-coupling size validation and finite-lag estimator sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import local_hierarchical_bridge as bridge

ROOT=Path(__file__).parent
LAGS=[16,24,32,64,128,256,1024,"infinite trapezoid",None]
SELECTED=[0,36,91,101,169,307,308]

def tau_integral(a,h,lag):
    if lag is None:return 1/h
    # Stable for positive a close to one.
    power=np.power(a,lag+1)
    ratio=np.where(a>0,-np.expm1((lag+1)*np.log(np.maximum(a,1e-300)))/h,
                   (1-power)/h)
    return ratio-.5*(1+a**lag)

def verify():
    a=np.array([-.9,-.2,0.,.2,.9,.999])
    for lag in (1,16,64):
        direct=.5+.5*a**lag
        for k in range(1,lag):direct+=a**k
        np.testing.assert_allclose(tau_integral(a,1-a,lag),direct,rtol=1e-11,atol=1e-11)
    # Exact decomposition of module-local and intermodule modes.
    n=128;block=16;g=.1;phi=4.
    matrix=bridge.make(dict(family='module rings+hierarchy',block=block,g=g,phi=phi),n)
    ring=bridge.make(dict(family='ring'),block)
    local=np.linalg.eigvalsh(ring)
    hier=bridge.make(dict(family='hierarchy',phi=phi),n//block)
    collective=np.linalg.eigvalsh(hier)
    expected=np.r_[np.tile((1-g)*local[:-1],n//block),(1-g)+g*collective]
    np.testing.assert_allclose(np.sort(expected),np.linalg.eigvalsh(matrix),atol=1e-12)

def all_estimators(b,ev,U,delta,greedy):
    h=delta+(1-delta)*(1-ev);a=1-h
    gain=1/(h*(2-h))
    times=np.column_stack([1/h-.5 if lag=="infinite trapezoid" else tau_integral(a,h,lag)
                           for lag in LAGS])
    cov=(U*gain)@U.T
    projection=U.copy()
    variance=[];weighted=[];equal=[]
    while len(cov)>=4:
        vd=np.diag(cov)
        wd=projection**2@(gain[:,None]*times)
        variance.append(vd.mean())
        weighted.append(wd.sum(axis=0)/vd.sum())
        equal.append((wd/vd[:,None]).mean(axis=0))
        if len(cov)==4:break
        if greedy:
            i,j=bridge.greedy_pairs(cov)
        else:
            i=np.arange(0,len(cov),2);j=i+1
        cov=cov[np.ix_(i,i)]+cov[np.ix_(i,j)]+cov[np.ix_(j,i)]+cov[np.ix_(j,j)]

        projection=projection[i]+projection[j]
    return np.array(variance),np.array(weighted),np.array(equal)

def run():
    verify()
    screen=pd.read_csv(ROOT/'bridge-screen.csv')
    specs=json.loads((ROOT/'bridge-specifications.json').read_text())
    rows=[]
    for k in SELECTED:
        spec=specs[k]
        for n in (256,512,1024):
            b=bridge.make(spec,n)
            ev,U=eigh(b);ev/=max(abs(ev))
            for objective in ('primary','robust'):
                delta=float(screen[(screen.id==k)&(screen.objective==objective)].delta.iloc[0])
                for partition,greedy in [('fixed spatial',False),('greedy',True)]:
                    v,wt,et=all_estimators(b,ev,U,delta,greedy)
                    for estimator,values in [('weighted',wt),('equal weights',et)]:
                        for j,lag in enumerate(LAGS):
                            result=bridge.summarize(k,spec,n,np.array([delta]),v[None,:],
                                                    values[:,j][None,:],partition)[0]
                            result['trained_objective']=objective
                            result['estimator']=estimator
                            result['lag_steps']='infinite inclusive' if lag is None else str(lag)
                            rows.append(result)
            print('Held-state validation',k,n,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'bridge-held-state-validation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    colors=plt.colormaps['tab10'](np.arange(len(SELECTED)))
    for k,color in zip(SELECTED,colors):
        s=df[(df.id==k)&(df.n==512)&(df.trained_objective=='robust')&
             (df.partition=='greedy')&(df.estimator=='equal weights')].copy()
        x=np.array([4096 if z=='infinite trapezoid' else 8192 if z=='infinite inclusive'
                    else int(z) for z in s.lag_steps])
        order=np.argsort(x);s=s.iloc[order];x=x[order]
        label=f'{k}: {specs[k]["family"]}'
        axes[0].semilogx(x,s.beta,'o-',color=color,label=label)
        axes[1].semilogx(x,s.robust_score,'o-',color=color,label=label)
    axes[0].axhline(.2,color='black',ls=':',lw=1)
    axes[0].set(xlabel='Lag horizon (final points: infinite trapezoid, infinite inclusive)',
                ylabel='Global beta, K=2–64')
    axes[1].set(xlabel='Lag horizon (final points: infinite trapezoid, infinite inclusive)',
                ylabel='Error across windows and interval slopes')
    for ax in axes:ax.grid(alpha=.2)
    axes[1].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle('Finite-lag sensitivity at fixed coupling trained on N=256\n'
                 'Shown at N=512; no refit of coupling or topology parameters')
    fig.savefig(ROOT/'bridge-finite-lag-sensitivity.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    run()
