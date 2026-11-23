"""Test rounded empirical exponents in spatially low-dimensional networks.

Exact stationary moments for symmetric linear dynamics; no continuum approximation.
Fixed and correlation-greedy dyadic partitions; interval and global slopes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
NETWORKS=['Ring','Open chain','Narrow cylinder','Weighted ring, mild',
          'Weighted ring, strong','1D geometric']
CONFIGS=[('Interval K=8–16',3,4),('Interval K=32–64',5,6),('Global K=2–64',1,6)]
COLORS=dict(zip(NETWORKS,['#4477aa','#66ccee','#228833','#ccbb44','#ee6677','#aa3377']))

def adjacency(name,n,seed):
    rng=np.random.default_rng(seed)
    a=np.zeros((n,n))
    if name=='Narrow cylinder':
        width=4;length=n//width
        for x in range(length):
            for y in range(width):
                i=width*x+y
                for dx,dy in [(1,0),(2,0),(0,1)]:
                    j=width*((x+dx)%length)+(y+dy)%width
                    a[i,j]=a[j,i]=1
    elif name=='1D geometric':
        # Variable positions; local probabilistic edges plus a connected backbone.
        positions=np.sort(rng.uniform(0,n,n))
        distance=abs(positions[:,None]-positions)
        distance=np.minimum(distance,n-distance)
        backbone=np.zeros((n,n),bool)
        for i in range(n):
            backbone[i,(i+1)%n]=backbone[(i+1)%n,i]=True
        kernel=np.exp(-distance/2)
        np.fill_diagonal(kernel,0)
        kernel[backbone]=0
        lo,hi=0.,20.
        for _ in range(60):
            mid=(lo+hi)/2
            degree=np.minimum(mid*kernel,1).sum()/n+2
            if degree>4:hi=mid
            else:lo=mid
        probability=np.minimum((lo+hi)/2*kernel,1)
        upper=np.triu(rng.random((n,n))<probability,1)
        a=(upper|upper.T|backbone).astype(float)
    else:
        sigma=.2 if name=='Weighted ring, mild' else .8
        for i in range(n):
            for step in (1,2):
                if name=='Open chain' and i+step>=n:continue
                j=(i+step)%n
                value=np.exp(sigma*rng.normal()-.5*sigma*sigma) if name.startswith('Weighted') else 1.
                a[i,j]=a[j,i]=value
    return a

def choose_pairs(cov,greedy):
    size=len(cov)
    if not greedy:
        return np.arange(0,size,2),np.arange(1,size,2)
    sd=np.sqrt(np.diag(cov))

    ii,jj=np.triu_indices(size,1)
    corr=cov[ii,jj]/(sd[ii]*sd[jj])
    rank=np.argsort(-np.round(corr,12),kind='stable')
    used=np.zeros(size,bool)
    left=[];right=[]
    for idx in rank:
        i,j=ii[idx],jj[idx]
        if not used[i] and not used[j]:
            left.append(i);right.append(j);used[i]=used[j]=True
            if len(left)==size//2:break
    return np.array(left),np.array(right)

def cascade(cov,modes,gain_time,greedy):
    projection=modes.copy()
    result=[]
    K=1
    while len(cov)>=4:
        vd=np.diag(cov)
        wd=projection**2@gain_time
        assert np.all(vd>0)
        result.append((K,vd.mean(),wd.sum()/vd.sum(),np.mean(wd/vd)))
        if len(cov)==4:break
        a,b=choose_pairs(cov,greedy)
        cov=cov[np.ix_(a,a)]+cov[np.ix_(a,b)]+cov[np.ix_(b,a)]+cov[np.ix_(b,b)]
        projection=projection[a]+projection[b]
        K*=2
    return np.array(result)

def slopes(table,timescale):
    output=[]
    logK=np.log(table[:,0])
    for label,lo,hi in CONFIGS:
        sel=slice(lo,hi+1)
        x=logK[sel];xc=x-x.mean()
        alpha=xc@np.log(table[sel,1])/(xc@xc)
        beta=xc@np.log(table[sel,timescale])/(xc@xc)
        output.append((label,float(alpha),float(beta)))
    return output

def run():
    records=[];checks=[];metadata=[]
    initial=np.linspace(np.log(1e-7),0,121)
    for n in (256,512):
        for name in NETWORKS:
            seeds=range(3) if name.startswith('Weighted') or name=='1D geometric' else range(1)
            for seed in seeds:
                adj=adjacency(name,n,seed)
                eigen,modes=eigh(adj)
                rho=max(abs(eigen));eigen/=rho
                normalized=adj/rho
                metadata.append(dict(network=name,n=n,seed=seed,
                                     mean_degree=float((adj>0).sum()/n),
                                     mean_strength=float(adj.sum()/n)))
                cache={}
                def evaluate(logdelta):
                    key=float(logdelta)
                    if key in cache:return cache[key]
                    delta=np.exp(logdelta)
                    h=delta+(1-delta)*(1-eigen)
                    gain=1/(h*(2-h))
                    cov=(modes*gain)@modes.T
                    if not checks or np.isclose(delta,.01,rtol=.04):
                        residual=cov-(1-delta)**2*normalized@cov@normalized-np.eye(n)
                        checks.append(float(np.linalg.norm(residual)/np.linalg.norm(cov)))
                    output=[]
                    for partition,greedy in [('Fixed spatial blocks',False),('Greedy correlation',True)]:
                        table=cascade(cov,modes,gain/h,greedy)
                        for estimator,column in [('Variance weighted',2),('Equal cluster weights',3)]:
                            for label,alpha,beta in slopes(table,column):
                                output.append((partition,estimator,label,alpha,beta))
                    cache[key]=output
                    return output
                base=[evaluate(x) for x in initial]
                # Refine around both beta peaks and nearest target matches for
                # every estimator/window/partition, reusing each evaluation.
                focus=set()
                for k in range(len(base[0])):
                    a=np.array([r[k][3] for r in base])
                    b=np.array([r[k][4] for r in base])
                    candidates=[int(np.argmax(b)),int(np.argmin((a-1.5)**2+(b-.2)**2))]
                    for j in candidates:
                        for x in np.linspace(initial[max(0,j-1)],initial[min(len(initial)-1,j+1)],9):
                            focus.add(float(x))
                for x in sorted(focus):evaluate(x)
                for logdelta,items in cache.items():
                    delta=np.exp(logdelta)
                    for partition,estimator,label,a,b in items:
                        records.append((name,n,seed,partition,estimator,label,delta,1-delta,a,b,
                                        np.hypot(a-1.5,b-.2)))
                print('Completed',name,n,seed,'couplings',len(cache),flush=True)
        # Checkpoint after each size.
        df=pd.DataFrame(records,columns=['network','n','seed','partition','estimator','window',
                                        'delta','m','alpha','beta','distance'])
        df.to_csv(ROOT/'low-dimensional-trajectories.csv',index=False)
    keys=['network','n','seed','partition','estimator','window']
    peak=df.loc[df.groupby(keys).beta.idxmax()].copy()
    peak=peak.rename(columns={'alpha':'alpha_at_beta_peak','beta':'beta_peak',
                              'm':'m_at_beta_peak','delta':'delta_at_beta_peak'})
    closest=df.loc[df.groupby(keys).distance.idxmin()].copy()
    closest=closest.rename(columns={'alpha':'alpha_at_closest','beta':'beta_at_closest',
                                    'm':'m_at_closest','delta':'delta_at_closest',
                                    'distance':'closest_distance'})
    summary=peak[keys+['alpha_at_beta_peak','beta_peak','m_at_beta_peak','delta_at_beta_peak']].merge(
        closest[keys+['alpha_at_closest','beta_at_closest','m_at_closest','delta_at_closest',
                      'closest_distance']],on=keys,validate='one_to_one')
    summary['joint_tolerance_hit']=(abs(summary.alpha_at_closest-1.5)<=.05)&(
        abs(summary.beta_at_closest-.2)<=.05)
    summary.to_csv(ROOT/'low-dimensional-summary.csv',index=False)
    pd.DataFrame(metadata).to_csv(ROOT/'low-dimensional-networks.csv',index=False)
    (ROOT/'low-dimensional-checks.json').write_text(json.dumps(
        dict(max_stationary_covariance_relative_residual=max(checks),
             target=[1.5,.2],distance='equal-weight Euclidean distance',
             illustrative_joint_tolerance=[.05,.05],
             greedy_tie_rounding_decimals=12,
             seed_count_random_families=3,
             note='Refined sampled extrema; original empirical finite-lag estimator is not reproduced.'),
        indent=2)+'\n')

def plot():
    df=pd.read_csv(ROOT/'low-dimensional-trajectories.csv')
    summary=pd.read_csv(ROOT/'low-dimensional-summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for row,partition in enumerate(['Fixed spatial blocks','Greedy correlation']):
        for col,(label,_,_) in enumerate(CONFIGS):
            ax=axes[row,col]
            sub=df[(df.n==512)&(df.partition==partition)&
                   (df.estimator=='Variance weighted')&(df.window==label)]
            for name in NETWORKS:
                for seed,g in sub[sub.network==name].groupby('seed'):
                    # Refined grids differ by configuration; display in delta order.
                    g=g.sort_values('delta')
                    ax.plot(g.alpha,g.beta,color=COLORS[name],lw=1.3,
                            alpha=.9 if seed==0 else .4,label=name if seed==0 else None)
                pp=summary[(summary.n==512)&(summary.partition==partition)&
                           (summary.estimator=='Variance weighted')&(summary.window==label)&
                           (summary.network==name)]
                ax.scatter(pp.alpha_at_beta_peak,pp.beta_peak,color=COLORS[name],s=20,zorder=4)
                ax.scatter(pp.alpha_at_closest,pp.beta_at_closest,edgecolors=COLORS[name],
                           facecolors='none',s=42,zorder=5)
            ax.scatter(1.5,.2,color='black',marker='x',s=55,zorder=6)
            ax.axvline(1.5,color='grey',ls=':',lw=.7)
            ax.axhline(.2,color='grey',ls=':',lw=.7)
            ax.set(title=f'{partition}\n{label}',xlabel='Variance exponent alpha',
                   ylabel='Timescale exponent beta')
            ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.suptitle('Other spatially low-dimensional networks · N=512 · exact linear stationary moments\n'
                 'Filled points: beta peaks · hollow points: closest to rounded target · faint curves: extra seeds',
                 fontsize=12)
    fig.savefig(ROOT/'low-dimensional-network-curves.png',dpi=180)
    plt.close(fig)

    # Peak and target robustness across sizes, using common global window.
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,partition in zip(axes,['Fixed spatial blocks','Greedy correlation']):
        for name in NETWORKS:
            for n,marker in [(256,'o'),(512,'s')]:
                s=summary[(summary.partition==partition)&(summary.network==name)&
                          (summary.n==n)&(summary.window=='Global K=2–64')&
                          (summary.estimator=='Variance weighted')]
                ax.scatter(s.alpha_at_beta_peak,s.beta_peak,color=COLORS[name],
                           marker=marker,s=45,label=f'{name}, N={n}')
        ax.scatter(1.5,.2,color='black',marker='x',s=65)
        ax.axvline(1.5,color='grey',ls=':',lw=.7)
        ax.axhline(.2,color='grey',ls=':',lw=.7)
        ax.set(title=partition,xlabel='Alpha at beta peak',ylabel='Maximum beta')
        ax.grid(alpha=.18)
    axes[1].legend(frameon=False,fontsize=7,ncol=2,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle('Global peak locations · K=2–64 · circle N=256, square N=512')
    fig.savefig(ROOT/'low-dimensional-global-peaks.png',dpi=180)
    plt.close(fig)


def mode_diagnostics():
    rows=[]
    for name in NETWORKS:
        seeds=range(3) if name.startswith('Weighted') or name=='1D geometric' else range(1)
        for seed in seeds:
            adj=adjacency(name,512,seed)
            ev,vec=eigh(adj)
            perron=vec[:,-1]
            rows.append((name,seed,1/(512*np.sum(perron**4)),1-ev[-2]/ev[-1]))
    pd.DataFrame(rows,columns=['network','seed','Perron_participation_fraction',
                               'normalized_spectral_gap']).to_csv(
                                   ROOT/'low-dimensional-mode-diagnostics.csv',index=False)

if __name__=='__main__':
    run()
    plot()
    mode_diagnostics()
