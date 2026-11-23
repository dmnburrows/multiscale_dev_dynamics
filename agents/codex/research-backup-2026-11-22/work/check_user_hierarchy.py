import ast
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
source=Path('/Users/k2585057/.codex/attachments/af5cb045-102c-4b55-b249-04fe8df917e3/Pasted text.txt')
tree=ast.parse(source.read_text())
ns={'np':np}
names={'spectrum','S_T','alpha_beta','alpha_beta_largeL'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
worst=0
for L in (3,5,7):
 n=2**L
 ids=np.arange(n)
 xor=ids[:,None]^ids
 levels=np.floor(np.log2(np.maximum(1,xor))).astype(int)+1
 for phi in (1.5,2,2.83,3.5,5):
  A=np.where(xor>0,float(phi)**(-levels),0)
  vals=np.linalg.eigvalsh(A)
  B=A/np.max(abs(vals))
  sp,ss,mult=ns['spectrum'](L,phi)
  predicted=np.sort(np.r_[sp,np.repeat(ss,mult.astype(int))])
  np.testing.assert_allclose(predicted,np.linalg.eigvalsh(B),atol=1e-12)
  for m in (.2,.95,.999):
   S=solve_discrete_lyapunov(m*B,np.eye(n))
   W=np.linalg.solve(np.eye(n)-m*B,S)
   vs=[];ts=[]
   for t in range(L+1):
    K=2**t
    C=np.eye(n).reshape(n//K,K,n).sum(axis=1)
    v=np.trace(C@S@C.T)/(n//K)
    w=np.trace(C@W@C.T)/(n//K)
    s0,t0=ns['S_T'](t,L,phi,m)
    expected_v=K*K/n*s0
    expected_t=t0/s0
    err=max(abs(expected_v/v-1),abs(expected_t/(w/v)-1))
    worst=max(worst,err)
    np.testing.assert_allclose([expected_v,expected_t],[v,w/v],rtol=1e-9)
    vs.append(v);ts.append(w/v)
   for t in range(L):
    np.testing.assert_allclose(ns['alpha_beta'](t,L,phi,m),
      [np.log2(vs[t+1]/vs[t]),np.log2(ts[t+1]/ts[t])],atol=1e-9)
print('dense exact worst relative error',worst)
print('collapse checks at L=24')
for phi in (3.,4.,5.):
 r=2/phi
 for t in (6,9):
  d=1e-3
  v0=np.array(ns['alpha_beta'](t,24,phi,1-d))
  v1=np.array(ns['alpha_beta'](t+1,24,phi,1-d*r))
  print(phi,t,v0,v1,'difference',v1-v0)
print('gap approximation errors at L=9')
for phi in (2.83,3.5,5.):
 r=2/phi
 _,s,_=ns['spectrum'](9,phi)
 eps=(2-r)*r**np.arange(9)
 exact=((2-r)*r**np.arange(9)-r**9)/(1-r**9)
 np.testing.assert_allclose(1-s,exact,atol=1e-14)
 print(phi,'max abs',max(abs(eps-(1-s))),'top rel',eps[-1]/(1-s[-1])-1)
print('claimed large-L comparison')
for phi in (2.83,3.5,5.):
 for d in (1e-3,1e-5):
  for t in (6,10):
   exact=np.array(ns['alpha_beta'](t,20,phi,1-d))
   approx=np.array(ns['alpha_beta_largeL'](t,20,phi,1-d))
   print(phi,d,t,'error',approx-exact)
