"""Exact finite periodic lattice sums, dimension 1..20, axis length 8."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


from importlib.util import spec_from_file_location, module_from_spec

ROOT=Path(__file__).parent
spec=spec_from_file_location('finite_lattice',ROOT/'finite_lattice_dimension_sweep.py')
module=module_from_spec(spec)
spec.loader.exec_module(module)

length=8
neighbours=2
sides=np.array([1,2,4])
deltas=np.geomspace(1,1e-24,721)
proximity=-np.log10(deltas)
fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
cmap=plt.colormaps['viridis']
norm=Normalize(1,20)
rows=[]
for d in range(1,21):
    variance,timescale,count=module.evaluate(length,d,neighbours,sides,deltas)
    # Interval side 2 to 4. Cluster volumes are K=2^d and 4^d.
    alpha=np.log2(variance[:,2]/variance[:,1])/d
    beta=np.log2(timescale[:,2]/timescale[:,1])/d
    np.testing.assert_allclose([alpha[0],beta[0]],[1,0],atol=1e-12)
    assert np.isfinite(alpha).all() and np.isfinite(beta).all()
    color=cmap(norm(d))
    axes[0].plot(alpha,beta,color=color,lw=1.5)
    axes[1].plot(proximity,alpha,color=color,lw=1.5)
    axes[2].plot(proximity,beta,color=color,lw=1.5)
    peak=np.argmax(beta)
    axes[0].scatter(alpha[peak],beta[peak],color=color,s=16)
    axes[1].scatter(proximity[peak],alpha[peak],color=color,s=16)
    axes[2].scatter(proximity[peak],beta[peak],color=color,s=16)
    rows.extend(zip([d]*len(deltas),[4*d]*len(deltas),[length**d]*len(deltas),
                    deltas,alpha,beta))
    print('Dimension completed',d,'compressed modes',count,flush=True)
axes[0].scatter(1.5,.2,marker='x',s=50,color='black')
axes[0].axvline(1.5,color='grey',ls=':',lw=.8)
axes[0].axhline(.2,color='grey',ls=':',lw=.8)
axes[0].set(xlabel='Variance exponent alpha',ylabel='Timescale exponent beta',
            title='Alpha versus beta')
for ax,ylabel,title in zip(axes[1:],['Variance exponent alpha','Timescale exponent beta'],
                          ['Alpha versus coupling proximity','Beta versus coupling proximity']):
    ax.set(xlabel='Proximity to instability: -log10(1-m)',ylabel=ylabel,title=title)
    ax.set_xlim(0,24)
    ax.set_xticks([0,3,6,9,12,15,18,21,24])
for ax in axes:ax.grid(alpha=.18)
bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,
                ticks=[1,4,8,12,16,20],pad=.015)
bar.set_label('Dimension d · degree = 4d')
fig.suptitle('Exact finite periodic lattices · d = 1–20 · two neighbours per side per axis\n'
             'Axis length 8 · N = 8^d · interval slopes K = 2^d to 4^d · dots mark sampled beta peaks',
             fontsize=12)
fig.savefig(ROOT/'lattice-d1-to20.png',dpi=180)
data=pd.DataFrame(rows,columns=['d','degree','N','one_minus_m','alpha','beta'])
data.to_csv(ROOT/'lattice-d1-to20.csv',index=False)
data.loc[data.groupby('d').beta.idxmax()].to_csv(ROOT/'lattice-d1-to20-peaks.csv',index=False)
