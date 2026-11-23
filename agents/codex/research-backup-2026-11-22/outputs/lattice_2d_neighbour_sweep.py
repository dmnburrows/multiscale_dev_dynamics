"""Exact 2D finite periodic lattices; vary axis-neighbour range at fixed size."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).parent
spec=spec_from_file_location('finite_lattice',ROOT/'finite_lattice_dimension_sweep.py')
finite=module_from_spec(spec)
spec.loader.exec_module(finite)

L=64
D=2
SIDES=np.array([4,8])
DELTAS=np.geomspace(1.,1e-10,1001)
X=-np.log10(DELTAS)
cmap=plt.colormaps['viridis']
norm=Normalize(2,20)
fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
rows=[]
peaks=[]

def exponents(r,delta):
    variance,time,_=finite.evaluate(L,D,r,SIDES,delta)
    denominator=D*np.log(SIDES[1]/SIDES[0])
    return np.column_stack((np.log(variance[:,1]/variance[:,0])/denominator,
                            np.log(time[:,1]/time[:,0])/denominator))

for r in range(2,21):
    ab=exponents(r,DELTAS)
    np.testing.assert_allclose(ab[0],[1,0],atol=1e-12)
    j=np.argmax(ab[:,1])
    opt=minimize_scalar(lambda x:-exponents(r,np.exp(x))[0,1],
         bounds=sorted(np.log(DELTAS[[max(0,j-1),min(len(DELTAS)-1,j+1)]])),
         method='bounded')
    delta=np.exp(opt.x)
    peak=exponents(r,delta)[0]
    px=-np.log10(delta)
    color=cmap(norm(r))
    axes[0].plot(ab[:,0],ab[:,1],color=color,lw=1.6)
    axes[1].plot(X,ab[:,0],color=color,lw=1.6)
    axes[2].plot(X,ab[:,1],color=color,lw=1.6)
    axes[0].scatter(*peak,s=18,color=color)
    axes[1].scatter(px,peak[0],s=18,color=color)
    axes[2].scatter(px,peak[1],s=18,color=color)
    rows.extend(zip([r]*len(X),[4*r]*len(X),DELTAS,ab[:,0],ab[:,1]))
    peaks.append([r,4*r,1-delta,delta,*peak])
axes[0].scatter(1.5,.2,marker='x',color='black',s=60,zorder=5)
axes[0].axvline(1.5,color='grey',ls=':',lw=.8)
axes[0].axhline(.2,color='grey',ls=':',lw=.8)
axes[1].axhline(1.5,color='grey',ls=':',lw=.8)
axes[2].axhline(.2,color='grey',ls=':',lw=.8)
axes[0].set(title='Alpha versus beta',xlabel='Variance exponent alpha',ylabel='Timescale exponent beta')
axes[1].set(title='Alpha versus coupling',ylabel='Variance exponent alpha')
axes[2].set(title='Beta versus coupling',ylabel='Timescale exponent beta')
for ax in axes[1:]:
    ax.set(xlabel='Proximity to instability: -log10(1-m)',xlim=(0,10))
for ax in axes:ax.grid(alpha=.18)
bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,
                ticks=[2,5,8,11,14,17,20],pad=.015)
bar.set_label('Neighbours per side per axis r · degree = 4r')
fig.suptitle('Fixed 2D periodic lattice · 64 × 64 neurons · N = 4096\n'
             'Neighbours per side per axis = 2–20 · square clusters side 4 to 8 (K = 16 to 64)\n'
             'Exact stationary sums · interval slopes · dots mark beta maxima',fontsize=12)
fig.savefig(ROOT/'lattice-2d-neighbours-2-to20.png',dpi=180)
pd.DataFrame(rows,columns=['neighbours_per_side_per_axis','degree','one_minus_m','alpha','beta']).to_csv(
    ROOT/'lattice-2d-neighbour-sweep.csv',index=False)
pd.DataFrame(peaks,columns=['neighbours_per_side_per_axis','degree','m_peak','one_minus_m_peak',
                           'alpha_at_beta_peak','beta_peak']).to_csv(
    ROOT/'lattice-2d-neighbour-peaks.csv',index=False)
print('Finished 19 neighbour ranges at fixed d, N and cluster sizes.')
