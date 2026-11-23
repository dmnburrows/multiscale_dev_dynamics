"""Locate rounded Munn target using exact hierarchy sums and global regression."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import solve_discrete_lyapunov

ROOT=Path(__file__).parent

def spectrum(L,phi):
    r=2/phi
    # Exact normalized gaps avoid subtraction of nearly equal eigenvalues.
    s=np.arange(1,L+1)
    if abs(r-1)<1e-10:
        gaps=(L-s+2)/L
    else:
        gaps=((2-r)*r**(s-1)-r**L)/(1-r**L)
    return np.r_[0.,gaps],np.r_[1.,2.**(L-s)]

def curves(L,phi,delta,ts):
    gap,mult=spectrum(L,phi)
    h=delta+(1-delta)*gap  # 1 - m lambda, exactly.
    gain=1/(h*(2-h))
    q=np.column_stack([np.ones(len(ts)),np.arange(1,L+1)[None,:]>ts[:,None]])
    q=q*mult
    s=q@gain
    w=q@(gain/h)
    K=2.**ts
    return K*K/2**L*s,w/s,q*gain,q*(gain/h),gap

def slope(x,y):
    xc=x-x.mean()
    return xc@np.log(y)/(xc@xc)

def pair(L,phi,delta,ts):
    v,t,*_=curves(L,phi,delta,ts)
    x=ts*np.log(2)
    return np.array([slope(x,v),slope(x,t)])

def diagnostics(L,phi,delta,ts):
    v,t,modal,weighted,gap=curves(L,phi,delta,ts)
    p=modal/modal.sum(axis=1,keepdims=True)
    pt=weighted/weighted.sum(axis=1,keepdims=True)
    h=delta+(1-delta)*gap
    times=1/h
    rows=[]
    for j,level in enumerate(ts):
        for i in range(L+1):
            if modal[j,i]==0: continue
            rows.append([2**int(level),'Perron' if i==0 else str(i),p[j,i],pt[j,i],times[i]])
    def r2(y):
        x=ts*np.log(2)
        z=np.log(y)
        return float(np.corrcoef(x,z)[0,1]**2)
    # Remove just the Perron contribution and refit.
    vp=(2.**ts)**2/2**L*modal[:,1:].sum(axis=1)
    tp=weighted[:,1:].sum(axis=1)/modal[:,1:].sum(axis=1)
    return dict(alpha_r2=r2(v),beta_r2=r2(t),
                without_Perron_pair=[float(slope(ts*np.log(2),vp)),
                                     float(slope(ts*np.log(2),tp))],
                alpha=float(pair(L,phi,delta,ts)[0]),
                beta=float(pair(L,phi,delta,ts)[1]),
                alpha_intervals=(np.diff(np.log2(v))/np.diff(ts)).tolist(),
                beta_intervals=(np.diff(np.log2(t))/np.diff(ts)).tolist(),
                K=(2.**ts).tolist(),variance=v.tolist(),timescale=t.tolist(),
                Perron_variance_fraction=p[:,0].tolist(),
                Perron_integrated_fraction=pt[:,0].tolist()),rows

# Independence check against dense matrix covariance with aligned clusters.
L=7;phi=3.03;delta=.001
n=2**L;ids=np.arange(n)
levels=np.floor(np.log2(np.maximum(1,ids[:,None]^ids))).astype(int)+1
adj=np.where(ids[:,None]!=ids,float(phi)**(-levels),0)
adj/=np.max(abs(np.linalg.eigvalsh(adj)))
cov=solve_discrete_lyapunov((1-delta)*adj,np.eye(n))
summed=np.linalg.solve(np.eye(n)-(1-delta)*adj,cov)
ts=np.arange(1,6)
vv=[];tt=[]
for t in ts:
    k=2**int(t);g=np.eye(n).reshape(n//k,k,n).sum(axis=1)
    diag=np.diag(g@cov@g.T)
    vv.append(diag.mean());tt.append(np.diag(g@summed@g.T).mean()/diag.mean())
v,t,*_=curves(L,phi,delta,ts)
np.testing.assert_allclose(v,vv,rtol=1e-9)
np.testing.assert_allclose(t,tt,rtol=1e-9)

target=np.array([1.5,.2])
solutions=[]
modalrows=[]
for L,lo,hi in ((9,1,6),(9,2,7),(12,1,6),(12,3,8),(16,1,6),(16,4,9)):
    ts=np.arange(lo,hi+1)
    def residual(theta):
        phi=np.exp(theta[0]);delta=np.exp(theta[1])
        return pair(L,phi,delta,ts)-target
    candidates=[]
    for phi0 in (3,5,9,15):
        for d0 in (.01,.0001):
            sol=least_squares(residual,np.log([phi0,d0]),
                              bounds=(np.log([2.01,1e-10]),np.log([40,.3])),
                              xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
            if np.linalg.norm(sol.fun)<1e-7:
                phi,d=np.exp(sol.x)
                if all(np.linalg.norm(sol.x-old[0])>1e-4 for old in candidates):
                    candidates.append((sol.x,phi,d))
    for _,phi,d in candidates:
        diag,rows=diagnostics(L,phi,d,ts)
        r=2/phi
        sstar=np.log((2-r)/(r*d))/np.log(1/r)
        result=dict(n=2**L,K_min=2**lo,K_max=2**hi,phi=float(phi),
                    m=float(1-d),delta=float(d),Kstar_approx=float(2**sstar),**diag)
        solutions.append(result)
        if L==9 and lo==1:
            modalrows.extend(rows)
fixed=[]
L=9;ts=np.arange(1,7)
for phi in (2.83,3.03,3.5,4,5,8,10):
    grid=np.linspace(np.log(1e-9),np.log(.3),800)
    objective=lambda logd:np.sum((pair(L,phi,np.exp(logd),ts)-target)**2)
    gvals=np.array([objective(x) for x in grid])
    idx=np.argmin(gvals)
    opt=minimize_scalar(objective,bounds=(grid[max(0,idx-1)],grid[min(len(grid)-1,idx+1)]),
                        method='bounded')
    delta=np.exp(opt.x)
    diag,_=diagnostics(L,phi,delta,ts)
    fixed.append(dict(phi=phi,m=float(1-delta),delta=float(delta),
                      distance=float(np.sqrt(opt.fun)),**diag))
with (ROOT/'hierarchy-global-modal-shares.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['K','level','variance_share','integrated_covariance_share','mode_time'])
    w.writerows(modalrows)
output=dict(target=target.tolist(),estimator='variance-weighted one-sided lag-zero-inclusive infinite sum',
            exact_target_solutions=solutions,fixed_phi_nearest=fixed,
            dense_check_max_relative_error=float(max(np.max(np.abs(v/np.array(vv)-1)),
                                                    np.max(np.abs(t/np.array(tt)-1)))))
(ROOT/'hierarchy-global-target.json').write_text(json.dumps(output,indent=2)+'\n')
for row in solutions:
    print('SOLUTION', {k:row[k] for k in ('n','K_min','K_max','phi','m','Kstar_approx','alpha','beta')})
    print('interval alpha',np.round(row['alpha_intervals'],4),'beta',np.round(row['beta_intervals'],4))
    print('Perron shares',np.round(row['Perron_variance_fraction'],4),np.round(row['Perron_integrated_fraction'],4))
for row in fixed:
    print('FIXED',{k:row[k] for k in ('phi','m','alpha','beta','distance')})

# Figures use exact finite sums only.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    row=next(z for z in solutions if z['n']==512 and z['K_min']==2)
    phi=row['phi'];delta=row['delta'];ts=np.arange(1,7)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,values,exponent,label in zip(axs[:2],
        [row['variance'],row['timescale']],[1.5,.2],['Mean variance','Integrated timescale']):
        k=np.array(row['K']);y=np.array(values);x=np.log(k)
        intercept=np.mean(np.log(y)-exponent*x)
        ax.loglog(k,y,'o-',label='Exact hierarchy')
        ax.loglog(k,np.exp(intercept)*k**exponent,'--',color='black',label='Global power-law fit')
        ax.set(xlabel='Cluster size K',ylabel=label)
        ax.legend(fontsize=8,frameon=False)
        ax.grid(alpha=.2)
    ds=np.geomspace(1e-6,.3,600)
    pairs=np.array([pair(9,phi,d,ts) for d in ds])
    axs[2].plot(pairs[:,0],pairs[:,1])
    axs[2].scatter([1.5],[.2],marker='x',color='black',s=65,label='Rounded target')
    axs[2].set(xlabel='Global variance exponent alpha',ylabel='Global timescale exponent beta')
    axs[2].legend(frameon=False,fontsize=8)
    axs[2].grid(alpha=.2)
    fig.savefig(ROOT/'hierarchy-global-target.png',dpi=180)
    print('TARGET FIT DIAGNOSTICS',row['alpha_r2'],row['beta_r2'],
          'without Perron',row['without_Perron_pair'])
