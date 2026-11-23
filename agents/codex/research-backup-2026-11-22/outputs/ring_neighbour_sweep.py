"""Exact finite-ring moments; vary neighbours per side and coupling explicitly."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).parent
N=512
CONFIGS=[('Local K = 8–16',np.array([8,16])),
         ('Local K = 32–64',np.array([32,64])),
         ('Global K = 2–64',2**np.arange(1,7))]
DELTAS=np.geomspace(1.,1e-7,1201)
M=1-DELTAS
COLORS=['#4477aa','#228833','#cc6677']

def curves(side,config,delta):
    _,sizes=config
    wave=2*np.pi*np.fft.rfftfreq(N)
    # Stable exact expression 1-lambda for a ring with 2*side neighbours.
    gaps=np.mean([2*np.sin(offset*wave/2)**2 for offset in range(1,side+1)],axis=0)
    multiplicity=np.full(len(wave),2.)
    multiplicity[[0,-1]]=1
    overlap=np.array([(size*np.sinc(size*wave/(2*np.pi))/np.sinc(wave/(2*np.pi)))**2
                      for size in sizes])*multiplicity/N
    delta=np.atleast_1d(delta)
    h=delta[:,None]+(1-delta[:,None])*gaps
    gain=1/(h*(2-h))
    variance=gain@overlap.T
    times=((gain/h)@overlap.T)/variance
    x=np.log(sizes.astype(float));xc=x-x.mean()
    alpha=np.log(variance)@xc/(xc@xc)
    beta=np.log(times)@xc/(xc@xc)
    return alpha,beta

def render_row(axs,side,rows,peakrows):
    for config,color in zip(CONFIGS,COLORS):
        label,sizes=config
        alpha,beta=curves(side,config,DELTAS)
        assert np.isfinite(alpha).all() and np.isfinite(beta).all()
        np.testing.assert_allclose([alpha[0],beta[0]],[1,0],atol=1e-12)
        index=np.argmax(beta)
        lo,hi=max(0,index-1),min(len(M)-1,index+1)
        search=minimize_scalar(lambda x:-curves(side,config,np.exp(x))[1][0],
            bounds=sorted(np.log(DELTAS[[lo,hi]])),method='bounded')
        delta=np.exp(search.x)
        ap,bp=curves(side,config,delta)
        coupling=1-delta
        axs[0].plot(alpha,beta,color=color,label=label)
        axs[0].scatter(ap,bp,color=color,s=26,zorder=5)
        for ax,values,yvalue in zip(axs[1:],[alpha,beta],[ap[0],bp[0]]):
            ax.plot(M,values,color=color,label=label)
            ax.scatter([coupling],[yvalue],color=color,s=26,zorder=5)
        rows.extend(zip([side]*len(M),[2*side]*len(M),[label]*len(M),M,DELTAS,alpha,beta))
        peakrows.append([side,2*side,label,coupling,delta,ap[0],bp[0]])
    axs[0].scatter([1.5],[.2],marker='x',color='black',s=45,zorder=6)
    axs[0].axvline(1.5,color='grey',ls=':',lw=.8)
    axs[0].axhline(.2,color='grey',ls=':',lw=.8)
    axs[1].axhline(1.5,color='grey',ls=':',lw=.8)
    axs[2].axhline(.2,color='grey',ls=':',lw=.8)
    axs[0].set(xlabel='Variance exponent alpha',ylabel='Timescale exponent beta',
               xlim=(.98,2.02),ylim=(-.005,.45))
    axs[1].set(xlabel='Coupling m',ylabel='Variance exponent alpha',ylim=(.98,2.02))
    axs[2].set(xlabel='Coupling m',ylabel='Timescale exponent beta',ylim=(-.005,.45))
    for ax in axs[1:]:
        ax.set_xscale('function',functions=(lambda m:-np.log10(np.maximum(1-np.asarray(m),1e-12)),
                                           lambda x:1-10.**(-np.asarray(x))))
        ax.set_xlim(0,1-1e-7)
        ax.set_xticks([0,.9,.99,.999,.9999,.99999,.999999,.9999999])
        ax.set_xticklabels(['0','.9','.99','.999','.9999','.99999','.999999','.9999999'],
                           rotation=35,ha='right',fontsize=8)
    for ax in axs:
        ax.grid(alpha=.18)
    axs[0].legend(title=f'{side} neighbours per side · degree {2*side}',frameon=False,
                  fontsize=8,title_fontsize=9)

fig,axs=plt.subplots(5,3,figsize=(15,19),layout='constrained')
rows=[];peaks=[]
for i,side in enumerate(range(2,7)):
    render_row(axs[i],side,rows,peaks)
    for j,title in enumerate(('alpha versus beta','alpha versus coupling','beta versus coupling')):
        axs[i,j].set_title(f'{side} neighbours per side · {title}',fontsize=10)
fig.suptitle('Finite 1D rings · N = 512 · exact stationary sums · dots mark beta maxima\n'
             'Coupling axes expand the region near m = 1',fontsize=13)
fig.savefig(ROOT/'ring-neighbours-2-to-6.png',dpi=160)
plt.close(fig)
# Individual rows remain readable without enlarging the full figure.
for side in range(2,7):
    fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    render_row(axs,side,[],[])
    fig.suptitle(f'1D ring · N = 512 · {side} neighbours per side ({2*side} total)',fontsize=12)
    fig.savefig(ROOT/f'ring-neighbours-per-side-{side}.png',dpi=160)
    plt.close(fig)
pd.DataFrame(rows,columns=['neighbours_per_side','degree','convention','m','one_minus_m','alpha','beta']).to_csv(
    ROOT/'ring-neighbour-curves.csv',index=False)
df=pd.DataFrame(peaks,columns=['neighbours_per_side','degree','convention','m_peak','one_minus_m_peak',
                             'alpha_at_beta_peak','beta_peak'])
df.to_csv(ROOT/'ring-neighbour-peaks.csv',index=False)
print(df.round(6).to_string(index=False))
