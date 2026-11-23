from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('outputs')
data=pd.read_csv(root/'beta-peak-trajectories.csv')
peaks=pd.read_csv(root/'beta-peak-exact.csv')
phis=sorted(data.loc[data.family=='HM','phi'].unique())
fig,axs=plt.subplots(2,4,figsize=(18,8),layout='constrained')
for col,phi in enumerate(phis):
    for row,convention in enumerate(('local','global')):
        ax=axs[row,col]
        use=data[(data.family=='HM')&(data.n==512)&np.isclose(data.phi,phi)&
                 (data.convention==convention)]
        for (lo,hi),sub in use.groupby(['K_min','K_max']):
            line,=ax.plot(sub.alpha,sub.beta,label=f'{lo}–{hi}',lw=1.6)
            p=peaks[(peaks.family=='HM')&(peaks.n==512)&np.isclose(peaks.phi,phi)&
                    (peaks.convention==convention)&(peaks.K_min==lo)&(peaks.K_max==hi)]
            ax.scatter(p.alpha_at_peak,p.beta_peak,color=line.get_color(),s=28,zorder=4)
        ax.axvline(1.5,color='grey',ls=':',lw=.8)
        ax.axhline(.2,color='grey',ls=':',lw=.8)
        ax.scatter([1.5],[.2],color='black',marker='x',s=45,zorder=5)
        ax.set(title=f'Phi = {phi:.4g} · {convention}',xlabel='Variance exponent alpha',
               xlim=(.97,2.03),ylim=(-.005,.55))
        if col==0:
            ax.set_ylabel('Timescale exponent beta')
        ax.grid(alpha=.15)
        ax.legend(title='Cluster sizes K',frameon=False,fontsize=8,title_fontsize=8)
fig.savefig(root/'beta-trajectories-all-phi.png',dpi=180)
