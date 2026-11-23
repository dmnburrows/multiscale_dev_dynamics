"""Independent matrix, quadrature, and derivative checks of the proof."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize_scalar

from griffiths_sweep import curves

out = Path(__file__).parent
rng = np.random.default_rng(713)
errors = {}
def record(name, predicted, independent):
    predicted, independent = np.asarray(predicted), np.asarray(independent)
    err = float(np.max(np.abs(predicted-independent)/np.maximum(1, np.abs(independent))))
    errors[name] = max(errors.get(name, 0), err)
    np.testing.assert_allclose(predicted, independent, rtol=2e-7, atol=2e-8)

# Positive and negative eigenvalues, unequal cluster variances, non-dyadic sizes.
for repeat in range(12):
    n = 12
    raw = rng.normal(size=(n,n))
    b = (raw + raw.T)/2
    b /= np.max(np.abs(np.linalg.eigvalsh(b)))
    lam, modes = np.linalg.eigh(b)
    for m in (0, .35, .9, .99):
        a = m*b
        cov = solve_discrete_lyapunov(a, np.eye(n))
        expected = (modes/(1-(m*lam)**2))@modes.T
        record('stationary_covariance', expected, cov)
        integrated = np.linalg.solve(np.eye(n)-a, cov)
        memberships = [np.eye(n), rng.integers(0,2,size=(5,n))]
        # Ensure nonempty clusters.
        memberships[1][:,0] = 1
        weights = [(g@modes)**2 for g in memberships]
        v,w,t = curves(weights,lam,m)
        record('mean_cluster_variance',v,[np.diag(g@cov@g.T).mean() for g in memberships])
        record('integrated_cluster_covariance',w,
               [np.diag(g@integrated@g.T).mean() for g in memberships])
        record('mean_normalized_timescale',t,
               [np.mean(np.diag(g@integrated@g.T)/np.diag(g@cov@g.T)) for g in memberships])
        for lag in (0,1,2,7):
            record('lag_covariance',
                   (modes*((m*lam)**lag/(1-(m*lam)**2)))@modes.T,
                   np.linalg.matrix_power(a,lag)@cov)
        record('overlap_sum',np.sum(weights[1],axis=1),memberships[1].sum(axis=1))

# Smooth overlaps are a mathematical extension, independently differentiated.
for repeat in range(30):
    lam=rng.uniform(-1,1,15)
    m=.83
    exponents=rng.uniform(-1,3,15)
    coefficients=rng.uniform(.1,2,15)
    x=.4
    def values(x):
        s=coefficients*np.exp(exponents*x)/(1-(m*lam)**2)
        p=s/s.sum()
        tau=1/(1-m*lam)
        return s.sum(),p@tau,p,tau
    v,t,p,tau=values(x)
    alpha=p@exponents
    beta=(p@(exponents*tau)-alpha*(p@tau))/(p@tau)
    h=1e-5
    vp,tp,*_=values(x+h)
    vm,tm,*_=values(x-h)
    record('local_alpha',alpha,(np.log(vp)-np.log(vm))/(2*h))
    record('local_beta',beta,(np.log(tp)-np.log(tm))/(2*h))
    # Arbitrary common weighted regression.
    xx=np.array([0,.3,1.1,2.2,3.8])
    vv,tt=np.array([values(k)[:2] for k in xx]).T
    ww=rng.uniform(.1,2,len(xx))
    xc=xx-np.average(xx,weights=ww)
    def slope(y):
        return np.sum(ww*xc*np.log(y))/np.sum(ww*xc*xc)
    record('fitted_slope_identity',slope(vv*tt)-slope(vv),slope(tt))

# Continuous diffusion model, noise covariance 2Q delta(r-r') delta(t-t').
# Check interval variance by real-space integration and W by Fourier quadrature.
for delta,D,K in ((.7,1.3,.8),(.2,2,3),(.03,.9,12),(.01,2,30)):
    Q=1
    xi=np.sqrt(D/delta)
    u=K/xi
    f=u+np.expm1(-u)
    alpha=-u*np.expm1(-u)/f
    variance=Q*np.sqrt(D)*delta**(-1.5)*f
    t=(3-alpha)/(2*delta)
    real=2*quad(lambda r:(K-r)*Q/(2*np.sqrt(D*delta))*np.exp(-r/xi),0,K)[0]
    record('continuum_variance_real_space',variance,real)
    def overlap(k):
        return (K*np.sinc(k*K/(2*np.pi)))**2
    w=quad(lambda k: Q*overlap(k)/(delta+D*k*k)**2/np.pi,
           0,np.inf,epsabs=1e-8,limit=1500)[0]
    record('continuum_integrated_covariance',variance*t,w)
    def alpha_at(x):
        z=np.exp(x)
        return -z*np.expm1(-z)/(z+np.expm1(-z))
    h=1e-5
    ap=(alpha_at(np.log(u)+h)-alpha_at(np.log(u)-h))/(2*h)
    beta=-ap/(3-alpha)
    def time_at(x):
        return (3-alpha_at(x))/(2*delta)
    record('continuum_beta',beta,
           (np.log(time_at(np.log(u)+h))-np.log(time_at(np.log(u)-h)))/(2*h))

def exact_ab(u):
    f=u+np.expm1(-u)
    fp=-np.expm1(-u)
    alpha=u*fp/f
    dalpha=u*((fp+u*np.exp(-u))*f-u*fp*fp)/(f*f)
    return alpha,-dalpha/(3-alpha)
peak=minimize_scalar(lambda u:-exact_ab(u)[1],bounds=(.1,8),method='bounded',
                     options={'xatol':1e-12})
errors['peak_u']=float(peak.x)
errors['peak_alpha'],errors['peak_beta']=map(float,exact_ab(peak.x))
# Nearest-neighbour bipartite ring invalidates a universal exponential kernel.
n=64
b=np.zeros((n,n))
for j in range(n):
    b[j,(j-1)%n]=b[j,(j+1)%n]=.5
cov=solve_discrete_lyapunov(.9*b,np.eye(n))
assert abs(cov[0,1]) < 1e-12
errors['nearest_neighbour_equal_time_adjacent_covariance']=float(cov[0,1])
(out/'proof-check-results.json').write_text(json.dumps(errors,indent=2)+'\n')
print(json.dumps(errors,indent=2))
