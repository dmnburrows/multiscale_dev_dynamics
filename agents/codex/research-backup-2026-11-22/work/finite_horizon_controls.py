from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from local_hierarchical_bridge import make,fixed_weights,summarize
ROOT=Path('outputs')
delta=np.geomspace(.3,1e-4,241)
specs=[dict(family='ring')]
for phi in (2.5,2.83,3.,4.,5.,8.,12.):
 specs.append(dict(family='hierarchy',phi=phi))
for g in (.001,.003,.01,.03,.1,.3,.6):
 specs.append(dict(family='ring+uniform',g=g))
for power in (.75,1.2,1.5,1.8,2.2,3.):
 specs.append(dict(family='powerlaw ring',power=power))
rows=[]
for k,spec in enumerate(specs):
 b=make(spec,256);ev,U=eigh(b);ev/=max(abs(ev))
 q=fixed_weights(U)
 h=delta[:,None]+(1-delta[:,None])*(1-ev)
 a=1-h;gain=1/(h*(2-h));v=gain@q.T
 for lag in (8,12,16,24,32,48,64,128,None):
  tau=1/h if lag is None else (1-a**(lag+1))/h-.5*(1+a**lag)
  t=((gain*tau)@q.T)/v
  for row in summarize(k,spec,256,delta,v,t,'fixed spatial'):
   row['lag_steps']='infinite inclusive' if lag is None else str(lag)
   rows.append(row)
df=pd.DataFrame(rows)
df.to_csv(ROOT/'bridge-finite-horizon-controls.csv',index=False)
print(df[df.objective=='robust'].groupby('family').apply(
 lambda x:x.nsmallest(1,'robust_score'),include_groups=False).to_string(index=False))
