import ast
from pathlib import Path
import numpy as np
from scipy.special import erf

# Extract just the functions and quadrature arrays, omitting absent dependencies.
p=Path('/Users/k2585057/.codex/attachments/88f6d4a8-3628-401e-a556-f598661f52c9/Pasted text.txt')
tree=ast.parse(p.read_text())
ns={'np':np,'erf':erf}
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('g','moments','alpha_beta'):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
    elif isinstance(node,ast.Assign):
        names=[t.id for t in node.targets if isinstance(t,ast.Name)]
        if any(z in ('_V','_W','_LOGG') for z in names):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
print('Continuum small-x beta')
for d in (2,3,4,5):
 print('d',d,'x=.001',ns['alpha_beta'](.001,d),'x=.0001',ns['alpha_beta'](.0001,d))
print('Two-group fraction at alpha=1.5')
print('overlap ratios(1,4):', (2**1.5-1)/3)
print('overlap ratios(0,4):', 2**1.5/4)
n=2**16;L=16
counts=np.r_[1.,2.**(L-np.arange(1,L+1))]
p=counts/n
print('m=0 grouped-level N_eff',1/(p@p),'actual modal N_eff',n)
print('Two-node bipartite alpha,beta')
for m in (.9,.999,.999999):
 print(m,1,np.log2(1+m))
print('Continuum denominator length ratio at m=.5, .99, .999')
for m in (.5,.99,.999):
 # A/B divided by C/E, omitting common c.
 print(m,(1-m*m)/(2*m*m)/((1-m)/m))
