"""Plots for the exact local/collective mode decomposition and held-state results."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_hierarchical_bridge import make,greedy_pairs

ROOT=Path(__file__).parent
training=pd.read_csv(ROOT/'bridge-finite-horizon-screen.csv')
chosen=training[(training.objective=='robust')&(training.phi==8)&(training.g==.03)&
                (training.block==32)&(training.lag_steps=='24')].iloc[0]
delta=float(chosen.delta)
SPEC=dict(family='module rings+hierarchy',phi=8.,g=.03,block=32)
HORIZON=24

def measurements(spec,n,greedy=True):
    b=make(spec,n);ev,U=eigh(b);ev/=max(abs(ev))
    h=delta+(1-delta)*(1-ev)
    a=1-h;gain=1/(h*(2-h))
    tau=(1-a**(HORIZON+1))/h-.5*(1+a**HORIZON)
    cov=(U*gain)@U.T;projection=U.copy()
    K=1;rows=[];shares=[]
    if spec['family']=='module rings+hierarchy':
        bsize=spec['block']
        collective=(U.reshape(n//bsize,bsize,n).sum(axis=1)**2).sum(axis=0)/bsize
        assert np.all(np.minimum(abs(collective),abs(collective-1))<1e-8)
        iscollective=collective>.5
        perron=np.argmax(ev)
    while len(cov)>=4:
        q=projection**2

        vd=np.diag(cov);wd=q@(gain*tau)
        rows.append((K,vd.mean(),np.mean(wd/vd)))
        if spec['family']=='module rings+hierarchy':
            mass=q.mean(axis=0)*gain
            total=mass.sum()
            cp=mass[perron]/total
            shares.append((K,mass[~iscollective].sum()/total,
                           mass[iscollective].sum()/total-cp,cp))
        if len(cov)==4:break
        if greedy:i,j=greedy_pairs(cov)
        else:i=np.arange(0,len(cov),2);j=i+1
        cov=cov[np.ix_(i,i)]+cov[np.ix_(i,j)]+cov[np.ix_(j,i)]+cov[np.ix_(j,j)]
        projection=projection[i]+projection[j];K*=2
    table=np.array(rows)
    slopes=np.column_stack((table[:-1,0],np.diff(np.log2(table[:,1])),
                             np.diff(np.log2(table[:,2]))))
    return table,slopes,np.array(shares)

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
records=[]
for n,color in zip((256,512,1024),['#4477aa','#228833','#cc6677']):
    _,s,_=measurements(SPEC,n)
    axes[0,0].semilogx(s[:,0],s[:,1],'o-',base=2,color=color,label=f'N={n}')
    axes[0,1].semilogx(s[:,0],s[:,2],'o-',base=2,color=color,label=f'N={n}')
    records.extend((n,*row) for row in s)
for label,spec,color in [('Local ring only',dict(family='ring'),'#4477aa'),
                          ('Hierarchy only, Phi=8',dict(family='hierarchy',phi=8.),'#cc6677'),
                          ('Local module rings + hierarchy',SPEC,'#228833')]:
    _,s,_=measurements(spec,512)
    axes[1,0].semilogx(s[:,0],s[:,1],'o-',base=2,color=color,label=label)
    axes[1,1].semilogx(s[:,0],s[:,2],'o-',base=2,color=color,label=label)
for row in range(2):
    axes[row,0].axhline(1.5,color='black',ls=':',lw=1)
    axes[row,1].axhline(.2,color='black',ls=':',lw=1)
    for ax in axes[row]:
        ax.axvspan(8,32,color='grey',alpha=.08)
        ax.set_xlabel('Interval start K · slope measured from K to 2K')
        ax.grid(alpha=.2)
    axes[row,0].set_ylabel('Variance interval exponent alpha')
    axes[row,1].set_ylabel('Timescale interval exponent beta')
axes[0,0].set_title('Same parameters and coupling across network sizes')
axes[0,1].set_title('Finite integration horizon held at 24 updates')
axes[1,0].set_title('Same coupling and estimator across component controls')
axes[1,1].set_title('Greedy correlation clustering')
axes[0,0].legend(frameon=False,fontsize=8)
axes[1,0].legend(frameon=False,fontsize=8)
fig.suptitle(f'Exploratory local–hierarchical candidate · Phi=8, bridge fraction 0.03, module size 32\n'
             f'm={1-delta:.6f} fixed; equal average of cluster-normalized autocorrelations')
fig.savefig(ROOT/'bridge-held-coupling-intervals.png',dpi=180)
plt.close(fig)
pd.DataFrame(records,columns=['n','K','alpha_interval','beta_interval']).to_csv(
    ROOT/'bridge-candidate-intervals.csv',index=False)

table,slopes,shares=measurements(SPEC,512,greedy=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
axes[0].stackplot(shares[:,0],shares[:,1:].T,
                  labels=['Within-module ring modes','Between-module hierarchy modes','Perron mode'],
                  colors=['#4477aa','#228833','#cc6677'])
axes[0].set_xscale('log',base=2)
axes[0].set(xlabel='Cluster size K',ylabel='Fraction of cluster variance',ylim=(0,1),
            title='Aligned clusters: how pooling selects the two mode families')
axes[0].legend(frameon=False,fontsize=8)
for ax,values,label in [(axes[1],table[:,1],'Variance')]:
    ax.loglog(table[:,0],values,'o-',color='#4477aa',label=label)
    ax.set(xlabel='Cluster size K',ylabel='Mean variance')
    ax2=ax.twinx()
    ax2.semilogx(table[:,0],table[:,2],'s-',base=2,color='#228833',label='Timescale')
    ax2.set_ylabel('Finite-horizon timescale (updates)',color='#228833')
    ax.set_title('Same fixed network and coupling')
for ax in axes:ax.grid(alpha=.18)
fig.suptitle('Composition of local and hierarchical activity · N=512')
fig.savefig(ROOT/'bridge-mode-composition.png',dpi=180)
plt.close(fig)
