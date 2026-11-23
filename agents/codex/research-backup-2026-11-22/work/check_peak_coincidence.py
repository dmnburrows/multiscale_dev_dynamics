import ast,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar,least_squares
tree=ast.parse(Path('outputs/hierarchy_global_target.py').read_text())
ns={'np':np}
exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef)],type_ignores=[]),'functions','exec'),ns)
L=9
phi_global=7.765787162650601

def pair(phi,delta,ts):
 return ns['pair'](L,phi,delta,np.array(ts))
def peak(phi,ts):
 grid=np.linspace(np.log(1e-8),np.log(.5),1000)
 yy=np.array([pair(phi,np.exp(x),ts)[1] for x in grid])
 j=np.argmax(yy)
 sol=minimize_scalar(lambda x:-pair(phi,np.exp(x),ts)[1],
   bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded',
   options={'xatol':1e-12})
 return float(1-np.exp(sol.x)),pair(phi,np.exp(sol.x),ts).tolist()
rows=[]
for name,ts in [('2-4',[1,2]),('4-8',[2,3]),('8-16',[3,4]),('16-32',[4,5]),
 ('global 2-64',list(range(1,7))),('global 4-128',list(range(2,8)))]:
 # Fit the two parameters to the rounded pair, then test peak independently.
 sol=least_squares(lambda th:pair(np.exp(th[0]),np.exp(th[1]),ts)-[1.5,.2],
    np.log([8,.01]),bounds=(np.log([2.01,1e-9]),np.log([30,.5])),
    xtol=1e-13,ftol=1e-13,gtol=1e-13)
 phi,d=np.exp(sol.x)
 mp,ab=peak(phi,ts)
 same_m,same_ab=peak(phi_global,ts)
 row=dict(window=name,target_phi=float(phi),target_m=float(1-d),target_pair=pair(phi,d,ts).tolist(),
          peak_m=mp,pair_at_peak=ab,fixed_phi=phi_global,
          fixed_phi_peak_m=same_m,fixed_phi_pair_at_peak=same_ab)
 rows.append(row)
 print(row)
Path('outputs/hierarchy-peak-coincidence.json').write_text(json.dumps(rows,indent=2)+'\n')
