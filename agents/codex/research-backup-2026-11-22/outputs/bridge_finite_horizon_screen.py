"""Exploratory finite-horizon extension; integration steps are not calibrated seconds."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from local_hierarchical_bridge import make, fixed_weights, summarize

ROOT=Path(__file__).parent
delta=np.geomspace(.3,1e-4,241)
HORIZONS=(8,12,16,24,32,48,64,128,None)
results=[]
counter=0
for block in (16,32,64):
    for phi in (3.,4.,5.,6.,8.,12.):
        for g in (.003,.01,.02,.03,.05,.1,.3):
            spec=dict(family='module rings+hierarchy',phi=phi,g=g,block=block)
            b=make(spec,256);ev,U=eigh(b);ev/=max(abs(ev))
            q=fixed_weights(U)
            h=delta[:,None]+(1-delta[:,None])*(1-ev)

            a=1-h;gain=1/(h*(2-h))
            v=gain@q.T
            for lag in HORIZONS:
                tau=1/h if lag is None else (1-a**(lag+1))/h-.5*(1+a**lag)
                t=((gain*tau)@q.T)/v
                rows=summarize(counter,spec,256,delta,v,t,'fixed spatial')
                for row in rows:
                    row['lag_steps']='infinite inclusive' if lag is None else str(lag)
                    results.append(row)
            counter+=1
    print('Finished block',block,flush=True)
pd.DataFrame(results).to_csv(ROOT/'bridge-finite-horizon-screen.csv',index=False)
df=pd.DataFrame(results)
print(df[df.objective=='robust'].nsmallest(12,'robust_score')[
    ['phi','g','block','lag_steps','delta','alpha','beta','robust_score',
     'alpha_interval_min','alpha_interval_max','beta_interval_min','beta_interval_max']].to_string(index=False))
