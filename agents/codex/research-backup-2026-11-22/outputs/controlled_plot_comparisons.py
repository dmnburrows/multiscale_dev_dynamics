"""Controlled comparisons with explicit references for continuum and finite models."""
import ast
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import json
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
tree=ast.parse((ROOT/'continuum_dimension_peaks.py').read_text())
ns={'np':np,'erf':erf}
exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name=='g'],
                       type_ignores=[]),'kernel','exec'),ns)
dims=[1.,1.5,2.,2.5,3.,4.,5.,6.]
colors=plt.colormaps['tab10'](np.arange(8))
v=np.arange(-34,55,.012)
w=np.exp(v)
logg=np.log(ns['g'](w))
xx=np.geomspace(1e-7,1e3,2000)
tables={}
for d in dims:
    out=[]
    for x in xx:
        lf=-w*x*x+d*logg+v
        maximum=lf.max()
        mass=np.exp(lf-maximum)
        z=np.trapz(mass,v)
        mean=np.trapz(mass*w,v)/z
        second=np.trapz(mass*w*w,v)/z
        alpha=1+(2-2*x*x*mean)/d
        beta=(2-2*x*x*(second-mean*mean)/mean)/d
        out.append((np.log(z)+maximum,np.log(mean),alpha,beta))
    tables[d]=np.array(out)
    print('Continuum table',d,flush=True)

def continuum(d,ell,p,method):
    delta=10.**(-p);m=1-delta
    D=1.25*m/d
    x=ell*np.sqrt(delta/D)
    if method=='instantaneous':
        return np.column_stack([np.interp(np.log(x),np.log(xx),tables[d][:,i]) for i in (2,3)])
    sides=ell*2.**np.arange(-2,3)
    logside=np.log(sides);center=logside-logside.mean()
    logx=np.log(sides[None,:]*np.sqrt(delta[:,None]/D[:,None]))
    logz=np.interp(logx,np.log(xx),tables[d][:,0])
    logmean=np.interp(logx,np.log(xx),tables[d][:,1])
    logvar=(d+2)*logside+logz
    logtime=2*logside+logmean
    denom=d*(center@center)
    return np.column_stack([logvar@center/denom,logtime@center/denom])

def add_curves(axs,ab,p,color,label):
    axs[0].plot(ab[:,0],ab[:,1],color=color,label=label,lw=1.5)
    axs[1].plot(p,ab[:,0],color=color,lw=1.5)
    axs[2].plot(p,ab[:,1],color=color,lw=1.5)

def finish_axes(axes,titles,suptitle,path):
    for col,title in enumerate(titles):
        axes[0,col].set_title(title,fontsize=11)
        axes[0,col].scatter([1.5],[.2],marker='x',s=40,color='black')
        axes[0,col].set(xlabel='Alpha',ylabel='Beta')
        axes[0,col].axvline(1.5,color='grey',ls=':',lw=.7)
        axes[0,col].axhline(.2,color='grey',ls=':',lw=.7)
        for row,name in ((1,'Alpha'),(2,'Beta')):
            axes[row,col].set(xlabel='Proximity to instability: -log10(1-m)',ylabel=name)
            axes[row,col].axhline(1.5 if row==1 else .2,color='grey',ls=':',lw=.7)
        for ax in axes[:,col]:
            ax.grid(alpha=.17)
    # Use identical limits across each row so visual changes are comparable.
    for row in range(3):
        ylim=(min(ax.get_ylim()[0] for ax in axes[row]),
              max(ax.get_ylim()[1] for ax in axes[row]))
        for ax in axes[row]:
            ax.set_ylim(ylim)
    for ax in axes[0]:
        ax.set_xlim(.98,2.02)
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    axes[0,0].figure.suptitle(suptitle,fontsize=12)
    axes[0,0].figure.savefig(ROOT/path,dpi=180)

p=np.linspace(.05,12,700)
fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
for d,color in zip(dims,colors):
    for col,(ell,method) in enumerate([(32,'instantaneous'),(32,'global'),(16,'instantaneous')]):
        ab=continuum(d,ell,p,method)
        add_curves(axes[:,col],ab,p,color,f'd={d:g}')
finish_axes(axes,
            ['Baseline: instantaneous · side 32',
             'Change slope only: global · sides 8–128',
             'Change cluster side only: side 16'],
            'Same infinite continuum model, dimensions, coupling grid and diffusion coefficient\n'
            'Columns change only the stated slope definition or observation scale',
            'controlled-continuum-comparison.png')
plt.close(fig)

# Second figure: hold box interval and integer dimensions fixed. First change
# the dynamics at fixed finite lattice, then size at fixed continuous dynamics.
spec=spec_from_file_location('finite',ROOT/'finite_lattice_dimension_sweep.py')
finite=module_from_spec(spec);spec.loader.exec_module(finite)
p2=np.linspace(.05,10,301)
delta=10.**(-p2)
sides=np.array([8,16])

def finite_pair(length,d,kind):
    freq,counts,q=finite.compressed_modes(length,d,sides)
    wave=2*np.pi*freq/length
    axisgap=np.mean([2*np.sin(j*wave/2)**2 for j in (1,2)],axis=0)
    gap=counts@axisgap/d
    values=[];times=[]
    # Batch coupling values to bound memory for the 3D lattice.
    for start in range(0,len(delta),16):
        ds=delta[start:start+16,None]
        h=ds+(1-ds)*gap
        if kind=='continuous':
            gain=1/(2*h)
        else:
            gain=1/(h*(2-h))
        variance=gain@q.T
        weighted=(gain/h)@q.T
        values.extend(variance);times.extend(weighted/variance)
    values=np.array(values);times=np.array(times)
    denominator=d*np.log(2)
    return np.column_stack((np.log(values[:,1]/values[:,0])/denominator,
                            np.log(times[:,1]/times[:,0])/denominator))

fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
records=[]
for d,color in zip((1,2,3),colors[[0,2,4]]):
    for col,(length,kind) in enumerate([(64,'continuous'),(64,'discrete'),(128,'continuous')]):
        ab=finite_pair(length,d,kind)
        add_curves(axes[:,col],ab,p2,color,f'd={d}')
        records.extend(zip([d]*len(p2),[length]*len(p2),[kind]*len(p2),p2,ab[:,0],ab[:,1]))
    print('Finite matched comparison',d,flush=True)
finish_axes(axes,
            ['Finite reference: continuous time · L=64',
             'Change update rule only: discrete time · L=64',
             'Change network size only: continuous time · L=128'],
            'Same periodic lattice, two neighbours per side per axis, cube sides 8 to 16\n'
            'Same interval-slope definition; integer d only; N = L^d',
            'controlled-finite-comparison.png')
plt.close(fig)
pd.DataFrame(records,columns=['d','L','update','proximity','alpha','beta']).to_csv(
    ROOT/'controlled-finite-comparison.csv',index=False)

# In the continuum, rescaling cluster side moves the coupling coordinate,
# but it cannot change the full parametric alpha-beta trajectory.
for d in dims:
    for x in (.1,1,3):
        m32=1/(1+(1.25/d)*(x/32)**2)
        m16=1/(1+(1.25/d)*(x/16)**2)
        a=continuum(d,32,np.array([-np.log10(1-m32)]),'instantaneous')
        b=continuum(d,16,np.array([-np.log10(1-m16)]),'instantaneous')
        np.testing.assert_allclose(a,b,atol=1e-7)
print('Matched-scale continuum checks passed.')
