import sys
from pathlib import Path
sys.path.insert(0,str(Path('outputs').resolve()))
from low_dimensional_network_tests import adjacency,NETWORKS
from scipy.linalg import eigh
import numpy as np
import pandas as pd
rows=[]
for name in NETWORKS:
    seeds=range(3) if name.startswith('Weighted') or name=='1D geometric' else range(1)
    for seed in seeds:
        A=adjacency(name,512,seed)
        e,u=eigh(A)
        p=u[:,-1]**2
        rows.append([name,seed,1/(512*np.sum(p*p)),1-e[-2]/e[-1]])
df=pd.DataFrame(rows,columns=['network','seed','Perron_participation_fraction','normalized_spectral_gap'])
df.to_csv('outputs/low-dimensional-mode-diagnostics.csv',index=False)
print(df.groupby('network').median(numeric_only=True).to_string())
