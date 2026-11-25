"""Synthetic algebra only: no torch, native circuit, task arrays or model import."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
import numpy as np
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
write=lambda n,x:(r/n).write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
s=read('scope.json');hyp=read('hypothesis.json');start=time.process_time()
assert hyp['one_primary_statistic'] if 'one_primary_statistic' in hyp else hyp['formula']['one_primary_statistic']
for p in s['pins']:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
old=json.loads(Path(s['pins'][14]['path']).read_bytes())
v=old['quantum']['psi'];saved=np.asarray(v['real'])+1j*np.asarray(v['imag'])
psi=np.stack([saved[0,0],saved[0,0],saved[1,1],saved[1,1]])
plus=np.zeros(8);plus[[0,2]]=1/math.sqrt(2);minus=plus.copy();minus[2]*=-1
saved_error=float(np.abs(psi-np.stack([plus,plus,minus,minus])).max())
K=np.abs(psi.conj()@psi.T)**2;pop=np.abs(psi)**2;KD=pop@pop.T
vr=np.concatenate([psi.real,psi.imag],axis=1)
J=np.block([[np.zeros((8,8)),-np.eye(8)],[np.eye(8),np.zeros((8,8))]])
K16=(vr@vr.T)**2+(vr@J@vr.T)**2
triu=np.triu_indices(4,1);qoff=K[triu]
a=F(1,10);beta=F(1,2);keycoord=float(a)*np.array([1,1,-1,-1.])
angles=np.zeros((4,2,3,2));angles[:,0,0,0]=math.pi*np.tanh(keycoord*math.atanh(.5)/float(a))
expected_angles=np.zeros_like(angles);expected_angles[:,0,0,0]=[math.pi/2,math.pi/2,-math.pi/2,-math.pi/2]
assert np.max(np.abs(angles-expected_angles))<s['tolerance']
# Q query phi=e0 for all40, zero value/decoder/bias. Every parameter is fixed.
prob=np.abs(psi[:,0])**2;alpha=(prob+1e-6)/(prob+1e-6).sum()
rows=[]
for name,pairs,expect in [('aligned',[(0,1),(2,3)],F(-1,3)),('misaligned',[(0,2),(1,3)],F(1,6))]:
 H=np.eye(4)
 for i,j in pairs:H[i,j]=H[j,i]=float(beta)
 residual=H-np.outer(keycoord,keycoord);L=np.linalg.cholesky(residual)
 h=np.zeros((4,1024));h[:,0]=keycoord;h[:,1:5]=L
 gram=h@h.T;norm=np.sqrt(np.diag(gram));cos=gram/(norm[:,None]*norm[None]);d=1-cos
 off=d[triu]
 def A(kk):
  mass=kk[triu].sum();assert mass>0
  return float(kk[triu]@off/mass-off.mean())
 # Fixed Gaussian admissible projection: choose its first5 scaled coordinates = h[:,:5].
 # This is a constructed shared-input instance, never a trained-baseline reproduction.
 scale=np.sqrt(np.array([3-math.sqrt(2),3+math.sqrt(2)]*6)/24)
 W=np.zeros((12,1028));W[np.arange(5),np.arange(5)]=1/scale[:5]
 kg=h@W[:,:1024].T*scale
 KG=np.exp(-np.square(kg[:,None]-kg[None,:]).sum(-1))
 Aq,Ad,Ag=A(K),A(KD),A(KG)
 assert abs(Aq-float(expect))<s['tolerance']
 assert abs(Ad)<s['tolerance']
 vals=dict(context=name,h=h.tolist(),roles=np.zeros((4,4)).tolist(),mask=[True]*4,
  h_gram=gram.tolist(),target_gram=H.tolist(),residual_min_eigenvalue=float(np.linalg.eigvalsh(residual).min()),
  input_distance=d.tolist(),KQ=K.tolist(),KD_sameQ=KD.tolist(),K16=K16.tolist(),KG=KG.tolist(),
  psi_real=psi.real.tolist(),psi_imag=psi.imag.tolist(),angles=angles.tolist(),
  prepare_assignment={'weight[0,0]':math.atanh(.5)/float(a),'all_other_weights_and_bias':0},
  query_phi='e0 for all40 (all effect coordinates zero)',probability_each_relation=prob.tolist(),alpha_each_relation=alpha.tolist(),
  value_shape=[4,64],value_fill=0,decoder_shape=[40,64],decoder_fill=0,bias_shape=[40],bias_fill=0,logits=[0]*40,
  population=pop.tolist(),density_real=(psi.T@psi.conj()/4).real.tolist(),density_imag=(psi.T@psi.conj()/4).imag.tolist(),
  metrics=dict(AQ=Aq,AD_sameQ=Ad,AG=Ag,S=Aq-Ag,pair_mean_KQ=float(qoff.mean()),purity=float(np.sum(np.abs(psi.T@psi.conj()/4)**2)),
    pair_mean_distance=float(off.mean()),sorted_distances=sorted(off.tolist()),input_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
    quantum_gram_eigenvalues=np.linalg.eigvalsh(K).tolist(),length=4,attention_collision_mass=float(1-alpha@alpha),utility_variance=0),
  exact_AQ=str(expect),gaussian_projection_assignments={f'weight[{j},{j}]':float(W[j,j]) for j in range(5)},gaussian_other_parameters_fill=0,
  max_input_gram_error=float(np.abs(gram-H).max()))
 rows.append(vals)
x,y=[v['metrics'] for v in rows]
assert abs(x['AG']-y['AG'])<s['tolerance']
assert abs((y['AQ']-x['AQ'])-.5)<s['tolerance']
for n in ('purity','length','pair_mean_KQ','attention_collision_mass','utility_variance','pair_mean_distance'):
 assert abs(x[n]-y[n])<s['tolerance'],n
for n in ('sorted_distances','input_gram_eigenvalues','quantum_gram_eigenvalues'):
 assert np.max(np.abs(np.array(x[n])-y[n]))<s['tolerance'],n
assert np.max(np.abs(K-K16))<s['tolerance']
assert saved_error<s['tolerance']
write('algebra-trace.json',dict(schema='synthetic-pair-alignment-algebra.v1',scope_sha256=hashlib.sha256((r/'scope.json').read_bytes()).hexdigest(),
 saved_native_stage_sha256=s['pins'][14]['sha256'],construction='Saved native plus/minus states reused; linear prepare angles checked, no new circuit/model calls.',contexts=rows))
fmean=F(5,6);certificates=[]
for name,weighted in [('aligned',F(1,2)),('misaligned',F(1))]:
 certificates.append(dict(context=name,pair_count=6,kernel_mass=2,unweighted_distance=str(fmean),weighted_distance=str(weighted),AQ=str(weighted-fmean)))
out=dict(status='PASS_SYNTHETIC_DIAGNOSTIC_NONSEQUIVALENCE_AND_INSUFFICIENCY',contexts=2,
 formula='A(K,d)=weighted off-diagonal input distance minus ordinary off-diagonal input distance',
 measurements=[dict(context=v['context'],**v['metrics'],exact_AQ=v['exact_AQ']) for v in rows],
 exact_certificates=certificates,delta_AQ='1/2',max_real16_kernel_error=float(np.abs(K-K16).max()),
 max_h_gram_error=max(v['max_input_gram_error'] for v in rows),saved_state_error=saved_error,
 max_prepare_angle_error=float(np.max(np.abs(angles-expected_angles))),
 conclusion='The association statistic is not a function of purity, kernel spectrum, scalar attention concentration, utility variance, input distance histogram or length alone. It changes by1/2 while original Q and D probabilities, alpha, values and all40 logits are identical; therefore it is not sufficient to establish quantum task error or an intervention.',
 limitation='Artificial fixed native-structure parameter/input construction; no contextual encoder/checkpoint reachability or natural task cause. Exact Gaussian projection is a synthetic instance, not trained strongGcov. Real-state fixture does not certify complex ansatz globally.',
 decision='ONE_EXPLORATORY_DIAGNOSTIC_ONLY_NO_METHOD_OR_TRAINING_ADMISSION',new_theory=False,method_selected=False,training_admitted=False,
 resources=s['resources'],numpy=np.__version__,cpu_seconds=time.process_time()-start,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
write('results.json',out)
print(json.dumps(dict(status=out['status'],AQ=[x['AQ'],y['AQ']],AG=[x['AG'],y['AG']],delta_AQ=.5,unchanged_all40_logits=True,cpu_seconds=out['cpu_seconds'])))
