"""Original continuum dimension sweep with an explicit coupling convention."""
import ast
from pathlib import Path
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
source=ROOT/'continuum_dimension_peaks.py'
tree=ast.parse(source.read_text())
ns={'np':np,'erf':erf}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                             and n.name in ('g','pair','boundary')],type_ignores=[]),
             str(source),'exec'),ns)
dims=[1.,1.5,2.,2.5,3.,4.,5.,6.]
xs=np.geomspace(1e-6,100,1600)
fig,axs=plt.subplots(1,3,figsize=(17,5.4),layout='constrained')
for d,color in zip(dims,plt.colormaps['tab10'](np.arange(len(dims)))):
    ab=np.array([ns['pair'](x,d) for x in xs])
    # Same formal nearest/second-nearest axis-normalized coefficient as before.
    c=1.25/d
    delta=c*(xs/32)**2/(1+c*(xs/32)**2)
    coupling_coordinate=-np.log10(delta)
    axs[0].plot(ab[:,0],ab[:,1],color=color,label=f'd = {d:g}',lw=1.7)
    axs[1].plot(coupling_coordinate,ab[:,0],color=color,lw=1.7)
    axs[2].plot(coupling_coordinate,ab[:,1],color=color,lw=1.7)
    j=np.argmax(ab[:,1])
    if 0<j<len(xs)-1:
        axs[0].scatter(ab[j,0],ab[j,1],color=color,s=25,zorder=4)
        axs[1].scatter(coupling_coordinate[j],ab[j,0],color=color,s=25,zorder=4)
        axs[2].scatter(coupling_coordinate[j],ab[j,1],color=color,s=25,zorder=4)
    else:
        a,b=ns['boundary'](d)
        axs[0].scatter(a,b,edgecolors=color,facecolors='white',s=45,zorder=4)
axs[0].scatter(1.5,.2,marker='x',color='black',s=65,zorder=5)
axs[0].axvline(1.5,color='grey',ls=':',lw=.7)
axs[0].axhline(.2,color='grey',ls=':',lw=.7)
axs[1].axhline(1.5,color='grey',ls=':',lw=.7)
axs[2].axhline(.2,color='grey',ls=':',lw=.7)
axs[0].set(title='Alpha versus beta',xlabel='Local variance exponent alpha',
           ylabel='Local timescale exponent beta')
axs[1].set(title='Alpha versus coupling',ylabel='Local variance exponent alpha')
axs[2].set(title='Beta versus coupling',ylabel='Local timescale exponent beta')
for ax in axs[1:]:
    ax.set(xlabel='Proximity to instability: -log10(1-m)',xlim=(0,12))
for ax in axs:ax.grid(alpha=.18)
axs[0].legend(frameon=False,ncol=2,fontsize=9)
fig.suptitle('Original continuum box-model sweep · colour = dimension d, including fractional values\n'
             'Instantaneous local slopes · cluster side 32 · D = 1.25m/d\n'
             'Filled dots: interior maxima · open circles: analytical boundary limits',
             fontsize=12)
fig.savefig(ROOT/'continuum-original-dimension-three-panels.png',dpi=180)
