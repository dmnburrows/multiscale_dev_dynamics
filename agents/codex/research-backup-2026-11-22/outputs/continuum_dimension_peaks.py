"""Continuum diffusion with box clusters: dimension-dependent local exponents.

d is the dimension in the separable box kernel, not a fitted spectral dimension
for an arbitrary graph. x is cluster side divided by correlation length.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
# Stable g: series at large w, expm1 elsewhere.
def g(w):
    b=1/(2*np.sqrt(w))
    direct=2*np.pi*(erf(b)+np.expm1(-b*b)/(b*np.sqrt(np.pi)))
    series=np.sqrt(np.pi/w)*(1-1/(24*w)+1/(480*w*w))
    return np.where(w>1e4,series,direct)

def pair(x,d,step=.008):
    # Integrate in v=log w. Tail cutoff adjusts with x.
    v=np.arange(-32.,max(40.,-2*np.log(x)+9),step)
    w=np.exp(v)
    lf=-w*x*x+d*np.log(g(w))+v
    lf-=lf.max()
    mass=np.exp(lf)
    z=np.trapz(mass,v)
    first=np.trapz(mass*w,v)/z
    second=np.trapz(mass*w*w,v)/z
    return np.array([1+(2-2*x*x*first)/d,
                     (2-2*x*x*(second-first*first)/first)/d])

def boundary(d):
    if d<=2:return min(2,1+2/d),0.
    return 1+2/d, (d-2)/d if d<4 else 2/d

dims=[1.,1.5,2.,2.5,3.,4.,5.,6.]
xs=np.geomspace(1e-5,30,450)
rows=[]
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for d in dims:
    ab=np.array([pair(x,d) for x in xs])
    index=np.argmax(ab[:,1])
    if 0<index<len(xs)-1:
        opt=minimize_scalar(lambda logx:-pair(np.exp(logx),d)[1],
                            bounds=np.log(xs[[index-1,index+1]]),method='bounded')
        xpeak=np.exp(opt.x)
        peak=pair(xpeak,d)
        kind='interior numerical maximum'
        # Compare halved quadrature step.
        np.testing.assert_allclose(peak,pair(xpeak,d,step=.004),atol=1e-7)
    else:
        a,b=boundary(d)
        xpeak=0.
        peak=np.array([a,b])
        kind='boundary supremum as x tends to zero; checked on sampled range'
    row=dict(d=d,x_peak=float(xpeak),alpha_at_peak=float(peak[0]),
             beta_peak=float(peak[1]),location=kind)
    rows.append(row)
    line,=axs[0].plot(ab[:,0],ab[:,1],label=f'd = {d:g}')
    axs[0].scatter(*peak,color=line.get_color(),s=32,zorder=4)
    axs[1].semilogx(xs,ab[:,1],color=line.get_color(),label=f'd = {d:g}')
    if xpeak>0:axs[1].scatter(xpeak,peak[1],color=line.get_color(),s=32,zorder=4)
axs[0].scatter(1.5,.2,marker='x',color='black',s=70,zorder=5,label='Rounded empirical target')
axs[0].set(xlabel='Local variance exponent alpha',ylabel='Local timescale exponent beta',
           title='Continuum box clusters; dots mark maxima or boundary limits')
axs[1].set(xlabel='Cluster side / correlation length',ylabel='Local timescale exponent beta',
           title='How the timescale slope changes with observation scale')
axs[0].legend(frameon=False,fontsize=8,ncol=2)
for ax in axs:ax.grid(alpha=.18)
fig.savefig(ROOT/'continuum-dimension-curves.png',dpi=180)
(ROOT/'continuum-dimension-peaks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
