"""Earlier continuum calculation: d=2, side=32, vary axis neighbour range."""
from pathlib import Path
import ast
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).parent
source=ROOT/'continuum_dimension_peaks.py'
tree=ast.parse(source.read_text())
scope={'np':np,'erf':erf}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                             and n.name in ('g','pair')],type_ignores=[]),str(source),'exec'),scope)
x=np.geomspace(1e-5,60,1200)
ab=np.array([scope['pair'](v,2) for v in x])
fig,axs=plt.subplots(1,3,figsize=(17,5),layout='constrained')
cmap=plt.colormaps['viridis'];norm=Normalize(2,20)
for r in range(2,21):
    c=np.sum(np.arange(1,r+1,dtype=float)**2)/(2*r*2)
    m=1/(1+c*(x/32)**2)
    p=-np.log10(1-m)
    color=cmap(norm(r))
    axs[0].plot(ab[:,0],ab[:,1],color=color,lw=1.6)
    axs[1].plot(p,ab[:,0],color=color,lw=1.6)
    axs[2].plot(p,ab[:,1],color=color,lw=1.6)
    i=np.argmax(ab[:,1])
    for ax,xx,yy in [(axs[0],ab[:,0],ab[:,1]),(axs[1],p,ab[:,0]),(axs[2],p,ab[:,1])]:
        ax.scatter(xx[i],yy[i],color=color,s=20,zorder=4)
axs[0].scatter(1.5,.2,color='black',marker='x',s=55,zorder=5)
axs[0].axvline(1.5,color='grey',ls=':',lw=.8)
axs[0].axhline(.2,color='grey',ls=':',lw=.8)
axs[1].axhline(1.5,color='grey',ls=':',lw=.8)
axs[2].axhline(.2,color='grey',ls=':',lw=.8)
axs[0].set(title='Alpha versus beta · curves overlap',
           xlabel='Local variance exponent alpha',ylabel='Local timescale exponent beta')
axs[1].set(title='Alpha versus coupling',ylabel='Local variance exponent alpha')
axs[2].set(title='Beta versus coupling',ylabel='Local timescale exponent beta')
for ax in axs[1:]:
    ax.set(xlabel='Proximity to instability: -log10(1-m)',xlim=(0,10))
for ax in axs:ax.grid(alpha=.18)
bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs,
                ticks=[2,5,8,11,14,17,20],pad=.015)
bar.set_label('Neighbours per side per axis')
fig.suptitle('Earlier continuum method · d = 2 · instantaneous local slopes · cluster side = 32\n'
             'Neighbour range enters only through diffusion coefficient · dots mark beta maxima',
             fontsize=12)
fig.savefig(ROOT/'continuum-2d-neighbours-2-to20.png',dpi=180)
i=np.argmax(ab[:,1])
print('Peak alpha,beta',ab[i])
