"""Descriptive arithmetic on saved gradients; does not fit or forward a model."""
from pathlib import Path
import json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
code=r'''
from pathlib import Path
import hashlib,json
import numpy as np
r=Path('/hy-tmp/Q-Attention/control/conditional-readout-geometry-20261123')
s=json.loads((r/'output/summary.json').read_bytes());out={}
for mode,row in s['results'].items():
 p=r/'output'/(mode+'-gradient.private.npz')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['gradient_sha256']
 v=np.load(p,allow_pickle=False);g=v['w'];b=v['bias']
 # A shared change of feature origin mu transforms the fixed-bias gradient
 # to g - b outer mu. The least-squares mu below is descriptive, not a rule.
 mu=(b@g)/float(b@b);rankone=b[:,None]*mu[None,:];residual=g-rankone
 total=float(np.linalg.norm(g));remaining=float(np.linalg.norm(residual))
 orth=float(np.max(np.abs(b@residual)));assert orth<1e-12
 fraction=float(np.sum(rankone**2)/np.sum(g**2));assert 0<=fraction<=1+1e-12
 out[mode]=dict(raw_norm=total,minimum_norm_under_common_feature_translation=remaining,
  remaining_norm_ratio=remaining/total,removed_squared_norm_fraction=fraction,
  optimal_translation_norm=float(np.linalg.norm(mu)),orthogonality_error=orth,
  gradient_sha256=row['gradient_sha256'])
print(json.dumps(dict(results=out,model_forwards=0,optimizer_updates=0,
 scope='Coordinate-sensitivity of saved TRAIN gradient: label-aware descriptive least squares, not a fitted classifier, actual cached-feature mean, convergence gap or gain.')))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=25))
(r/'gradient-decomposition.json').write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps(v))
