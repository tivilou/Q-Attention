from pathlib import Path
import json,numpy as np,hashlib
r=Path('/hy-tmp/Q-Attention/control/observable-value-span-20261126/output')
s=json.loads((r/'summary.json').read_bytes())
with np.load(r/'sufficient.private.npz',allow_pickle=False) as z:
 G,B,ys,y2=[z[k] for k in ('G','B','sum_u','sum_u2')]
with np.load(r/'coefficients.private.npz',allow_pickle=False) as z:
 aq,ad,mu,var=[z[k] for k in ('quantum','diagonal','constant','fit_variance')]
out={}
for kind,g,b,a in [('quantum_span',G,B,aq),('diagonal_span',G[:8,:8],B[:8],ad)]:
 ref,_,rank,_=np.linalg.lstsq(g,b,rcond=1e-10)
 relative=float(np.linalg.norm(ref-a)/np.linalg.norm(a))
 normal=float(np.linalg.norm(g@a-b)/np.linalg.norm(b))
 mse=(y2-2*(a*b).sum(0)+(a*(g@a)).sum(0))/8192
 error=float(np.max(np.abs(mse-np.array(s['metrics']['fit'][kind]['MSE_all40']))))
 assert relative<1e-7 and normal<1e-9 and error<1e-8
 out[kind]={'SVD_same_G_reference_rank':int(rank),'relative_coefficient_difference':relative,'relative_normal_equation_residual':normal,'max_fit_MSE_difference_sufficient_vs_second_pass':error}
assert float(np.max(np.abs(var-(y2/8192-mu**2))))<1e-10
print(json.dumps({'status':'PASS_SAME_SOLUTION_SUFFICIENT_STATISTICS_AUDIT','checks':out,'teacher_FIT_variance_consistent':True,'private_arrays_returned':False,'new_feature_or_checkpoint_reads':0,'new_native_forwards':0,'optimizer_updates':0,'reference_SVD_checks':2,'coefficient_sha256':hashlib.sha256((r/'coefficients.private.npz').read_bytes()).hexdigest()}))
