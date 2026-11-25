"""Reopen complete synthetic inputs; independent scalar/Fraction certificates."""
from pathlib import Path
from fractions import Fraction as F
import cmath,hashlib,json,math
import numpy as np
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read('scope.json');a=read('algebra-trace.json');m=read('results.json');h=read('hypothesis.json')
for p in s['pins']:assert sha(p['path'])==p['sha256']
assert a['scope_sha256']==sha(r/'scope.json') and len(a['contexts'])==2
errors=[];metrics=[];certificate=[]
for context in a['contexts']:
 x=np.asarray(context['h']);gram=np.asarray(context['target_gram']);angles=np.asarray(context['angles'])
 assert x.shape==(4,1024) and context['roles']==[[0.]*4]*4 and context['mask']==[True]*4
 assert np.max(np.abs(x[:,0]-.1*np.array([1,1,-1,-1])))<1e-14
 # Reconstruct every declared linear prepare coordinate. No native model import.
 pre=x[:,0]*context['prepare_assignment']['weight[0,0]']
 angle=[math.pi*math.tanh(t) for t in pre]
 errors.append(max(abs(angle[t]-angles[t,0,0,0]) for t in range(4)))
 saved=json.loads(Path(s['pins'][14]['path']).read_bytes())
 prev=saved['quantum']['angles']['values']
 errors.append(max(abs(angle[t]-prev[0 if t<2 else 1][0 if t<2 else 1][0][0][0]) for t in range(4)))
 # Exact existence: target Gram has eigenvalues1/2,1/2,3/2,3/2;
 # subtracting k*k^T lowers its least eigenvalue by at most ||k||^2=1/25.
 lower=F(1,2)-F(1,25);assert lower==F(23,50) and lower>0
 assert context['residual_min_eigenvalue']>=float(lower)-s['tolerance']
 ps=[[complex(u,v) for u,v in zip(re,im)] for re,im in zip(context['psi_real'],context['psi_imag'])]
 kernel=[];dist=[];kd=[];k16=[]
 for i in range(4):
  for j in range(i+1,4):
   dot=sum(u.conjugate()*v for u,v in zip(ps[i],ps[j]))
   kij=abs(dot)**2
   dij=1-sum(float(u*v) for u,v in zip(x[i],x[j]))/math.sqrt(sum(float(v*v) for v in x[i])*sum(float(v*v) for v in x[j]))
   kernel.append(kij);dist.append(dij)
   kd.append(sum(abs(u)**2*abs(v)**2 for u,v in zip(ps[i],ps[j])))
   vv=[p.real for p in ps[i]]+[p.imag for p in ps[i]]
   ww=[p.real for p in ps[j]]+[p.imag for p in ps[j]]
   jw=[-p.imag for p in ps[j]]+[p.real for p in ps[j]]
   k16.append(sum(u*v for u,v in zip(vv,ww))**2+sum(u*v for u,v in zip(vv,jw))**2)
   errors.extend([abs(kij-context['KQ'][i][j]),abs(dij-context['input_distance'][i][j]),abs(sum(u*v for u,v in zip(vv,ww))**2+sum(u*v for u,v in zip(vv,jw))**2-kij)])
 def assoc(weights):return sum(w*d for w,d in zip(weights,dist))/sum(weights)-sum(dist)/6
 Aq=assoc(kernel);Ad=assoc(kd);assert abs(Ad)<s['tolerance']
 exactdist=[];exactweights=[]
 for i in range(4):
  for j in range(i+1,4):
   exactdist.append(F(1)-F(float(gram[i,j])))
   exactweights.append(F(1 if (i//2)==(j//2) else 0))
 exact=sum(w*d for w,d in zip(exactweights,exactdist))/sum(exactweights)-sum(exactdist)/6
 assert str(exact)==context['exact_AQ'] and abs(Aq-float(exact))<s['tolerance']
 certificate.append(str(exact));metrics.append(context['metrics'])
 assert context['value_fill']==context['decoder_fill']==context['bias_fill']==0
 assert context['logits']==[0]*40 and context['alpha_each_relation']==[.25]*4
 p=[abs(row[0])**2 for row in ps]
 errors.extend(abs(u-v) for u,v in zip(p,context['probability_each_relation']))
 errors.append(float(np.max(np.abs(x@x.T-gram))))
assert certificate==['-1/3','1/6'] and F(certificate[1])-F(certificate[0])==F(1,2)
for key in ('purity','length','pair_mean_KQ','utility_variance','attention_collision_mass','pair_mean_distance','AG'):
 assert abs(metrics[0][key]-metrics[1][key])<s['tolerance'],key
for key in ('sorted_distances','input_gram_eigenvalues','quantum_gram_eigenvalues'):
 assert np.max(np.abs(np.array(metrics[0][key])-metrics[1][key]))<s['tolerance']
assert max(errors)<s['tolerance']
for name in ('prepare','select','audit'):
 e=read(name+'-execution.json');assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/(name+'.py'))
for n in ('select-execution.before-syntax.json','audit-execution.before-scope.json'):assert read(n)['exit_status']==1
assert read('select-execution.before-syntax.json')['code_sha256']==sha(r/'select.before-syntax.py')
assert m['resources']==s['resources'] and all(v==0 for v in m['resources'].values())
assert not m['new_theory'] and not m['method_selected'] and not m['training_admitted']
assert h['diagnostic_gate']['cutoff']=='AUC(S)>=0.55 and AUC(S)-max(AUC(each of the three prespecified scalar/dephased controls))>=0.02.'
assert len(read('source-exclusion.json')['routes'])==6
out=dict(status='PASS_REOPENED_SYNTHETIC_PAIRING_AND_SOURCE_EXCLUSION',
 independent_implementation='Separate scalar dot/distance and exact Fraction Gram/mass certificate; same agent, not external review.',
 contexts=2,exact_AQ=certificate,exact_delta='1/2',residual_positive_definite_lower_bound='23/50',
 maximum_reconstruction_error=max(errors),all40_Q_logits_unchanged=True,
 saved_native_source_and_stage_unchanged=True,source_pins=len(s['pins']),
 scientific_prediction_unchanged_by_syntax_repair=True,failed_exits_preserved=2,
 limitations=m['limitation'],provisional_diagnostic_only=True,task_efficacy_assessed=False,
 resources=m['resources'],newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out))
