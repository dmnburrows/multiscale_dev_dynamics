"""Check a direct finite-window relation with no local-slope calculation."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar

def fit(x,y):
    xc=x-x.mean()
    return xc@np.log(y)/(xc@xc)

def global_pair(u0, ratios):
    u=u0*ratios
    f=u+np.expm1(-u)
    h=2*u-3+(u+3)*np.exp(-u)
    x=np.log(ratios)
    return np.array([fit(x,f),fit(x,h)-fit(x,f)])

results=[]
for count in (3,5,6,8):
    ratios=2.**(np.arange(count)-(count-1)/2)
    peak=minimize_scalar(lambda logu:-global_pair(np.exp(logu),ratios)[1],
                         bounds=(-3,3),method='bounded')
    close=minimize_scalar(lambda logu:np.sum((global_pair(np.exp(logu),ratios)-[1.5,.2])**2),
                          bounds=(-3,3),method='bounded')
    results.append(dict(levels=count,span=int(2**(count-1)),
                        peak_u_mid=float(np.exp(peak.x)),
                        peak_pair=global_pair(np.exp(peak.x),ratios).tolist(),
                        closest_u_mid=float(np.exp(close.x)),
                        closest_pair=global_pair(np.exp(close.x),ratios).tolist()))

# Independent check: direct finite Fourier sums, then regression.
n=32768
k=2*np.pi*np.fft.rfftfreq(n)
lam=(np.cos(k)+np.cos(2*k))/2
multiplicity=np.full(len(k),2.)
multiplicity[[0,-1]]=1
comparisons=[]
for midpoint in (32,128,512):
    sizes=(midpoint*2.**np.arange(-2,3)).astype(int)
    filt=np.array([(K*np.sinc(K*k/(2*np.pi))/np.sinc(k/(2*np.pi)))**2
                   for K in sizes])*multiplicity/n
    worst=np.zeros(2)
    for umid in np.geomspace(.3,5,50):
        m=1/(1+1.25*(umid/midpoint)**2)
        a=m*lam
        variance=filt@(1/(1-a*a))
        summed=filt@(1/((1-a*a)*(1-a)))
        observed=np.array([fit(np.log(sizes),variance),
                           fit(np.log(sizes),summed/variance)])
        predicted=global_pair(umid,sizes/midpoint)
        worst=np.maximum(worst,np.abs(observed-predicted))
    comparisons.append(dict(K_mid=midpoint,max_alpha_error=float(worst[0]),
                            max_beta_error=float(worst[1])))

# General finite-network, finite-window overlap-ratio identity.
rng=np.random.default_rng(18)
maxerr=0.
for _ in range(100):
    q=rng.uniform(.01,3,(6,15))
    lam=rng.uniform(-1,1,15)
    a=.93*lam
    s=q/(1-a*a)
    v=s.sum(axis=1)
    tau=1/(1-a)
    w=s@tau
    p0=s[0]/v[0]
    ptime0=s[0]*tau/w[0]
    ratios=q/q[0]
    predicted_v=ratios@p0
    predicted_w=ratios@ptime0
    np.testing.assert_allclose(predicted_v,v/v[0],rtol=1e-13)
    np.testing.assert_allclose(predicted_w,w/w[0],rtol=1e-13)
    xx=np.log(np.array([2,4,8,16,32,64]))
    reconstructed=np.array([fit(xx,predicted_v),
                             fit(xx,predicted_w)-fit(xx,predicted_v)])
    direct=np.array([fit(xx,v),fit(xx,w/v)])
    maxerr=max(maxerr,float(np.max(np.abs(reconstructed-direct))))
out=dict(window_results=results,discrete_ring_comparison=comparisons,
         general_identity_max_absolute_error=maxerr)
Path(__file__).with_name('global-ring-checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
