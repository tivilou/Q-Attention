from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
from interleaved_math import block,circuit,probabilities,realify,representations,pool,EPS

r=Path(__file__).resolve().parent;root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=read('review.json');packet=read('review-packet.private.json')
assert review['response']['verdict']=='PASS_WITH_RISK' and review['response']['substantive_candidate']=='PROVISIONAL'
assert packet['code']==(r/'interleaved_math.py').read_text('utf8')
for p in packet['source_pins']:assert sha(p['path'])==p['sha256']
assert (r/'primary-adoption.md').is_file()
assert not (r/'math-results.json').exists()
src=Path(packet['source_pins'][0]['path']).parents[2]
sys.path.insert(0,str(src))
import torch
from baselines.retacred.rmea import circuit as native_circuit
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('retained_channel_reference',packet['source_pins'][1]['path'])
channel=importlib.util.module_from_spec(spec);spec.loader.exec_module(channel)
errors={};traces=[];started=time.perf_counter()
def error(name,a,b):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 assert np.isfinite(e),name
 errors[name]=max(errors.get(name,0.),e)
def amp_lifts(w,q):
 w1,w2=block(w[0]),block(w[1]);qu=circuit(q);qd=qu.conj().T
 initial=np.eye(8,dtype=np.complex128)[:,0]
 a=w2.conj().T@initial;b=w1@initial
 amp=(w2@qd@w1)[0,0]
 vec=np.sqrt(8)*np.vdot(qu.reshape(-1,order='F')/np.sqrt(8),np.outer(b,a.conj()).reshape(-1,order='F'))
 rv=realify(w2)@realify(qd)@realify(w1)@np.r_[initial.real,initial.imag]
 error('complex_vec_amplitude',amp,vec);error('complex_real16_amplitude',amp,rv[0]+1j*rv[8])
 rho=np.outer(initial,initial.conj())
 for u in (w1,qd,w2):rho=u@rho@u.conj().T;rho=np.diag(np.diag(rho))
 error('density_dephase_trace',np.trace(rho),1)
 error('density_dephase_score',rho[0,0],probabilities(w,q)['dephased'])
 for u in (w1,w2,qu):error('unitarity',u.conj().T@u,np.eye(8))
 return amp
max_word_last_rz_out_change=0.;max_d_first_rz_change=0.;wrong_adjoint_max_error=0.
j=np.arange(1,13,dtype=np.float64)
for index in range(32):
 w=(1.2*np.sin((index+1)*j*.173)).reshape(2,3,2)
 q=(.9*np.cos((index+2)*j*.137)).reshape(2,3,2)
 p=probabilities(w,q);rep=representations(w,q);amp=amp_lifts(w,q)
 wt=torch.tensor(w,dtype=torch.float64);qt=torch.tensor(q,dtype=torch.float64)
 psi=native_circuit(wt).detach().numpy();phi=native_circuit(qt).detach().numpy()
 error('native_word_vector',psi,circuit(w)[:,0]);error('native_query_vector',phi,circuit(q)[:,0])
 error('native_output_probability',p['out'],abs(np.vdot(phi,psi))**2)
 tm=[channel.layer_matrix(x).detach().numpy() for x in (wt[0],wt[1],qt[0],qt[1])]
 native_mid=(tm[1]@(tm[3]@tm[2]).conj().T@tm[0])[0,0]
 error('native_layer_mid_amplitude',amp,native_mid)
 error('vec_probability',p['mid'],rep['vec']);error('real16_probability',p['mid'],rep['real16'])
 wrong=(block(w[1])@circuit(-q)@block(w[0]))[0,0]
 wrong_adjoint_max_error=max(wrong_adjoint_max_error,float(abs(wrong-amp)))
 for wire in range(3):
  changed=w.copy();changed[1,wire,1]+=.37
  cp=probabilities(changed,q)
  error('last_word_rz_mid_invariant',p['mid'],cp['mid'])
  error('last_word_rz_D_invariant',p['dephased'],cp['dephased'])
  error('last_word_rz_amplitude_phase',(block(changed[1])@circuit(q).conj().T@block(changed[0]))[0,0],np.exp(-.5j*.37)*amp)
  max_word_last_rz_out_change=max(max_word_last_rz_out_change,abs(cp['out']-p['out']))
  changed=w.copy();changed[0,wire,1]+=.37
  max_d_first_rz_change=max(max_d_first_rz_change,abs(probabilities(changed,q)['dephased']-p['dephased']))
 commuting=w.copy();commuting[1]=0;cz=np.zeros((2,3,2))
 cp=probabilities(commuting,cz)
 error('commuting_probability',cp['mid'],cp['out'])
 error('commuting_operators',block(commuting[1])@circuit(cz).conj().T,circuit(cz).conj().T@block(commuting[1]))
 traces.append(dict(index=index,word_angles=w.tolist(),query_angles=q.tolist(),probabilities=p,representations=rep))
errors['first_word_rz_D_invariant']=max_d_first_rz_change

# Reviewer-specified finite native-angle fixture, frozen before this calculation.
wa=np.zeros((2,3,2));wb=wa.copy()
wa[1,0]=[np.pi/2,np.pi/2];wb[1,0]=[-np.pi/2,-np.pi/2]
q0=np.zeros((2,3,2));q0[0,0,0]=-np.pi/2
words=[wa,wb];states=[circuit(w)[:,0] for w in words]
error('completed_density_collision',np.outer(states[0],states[0].conj()),np.outer(states[1],states[1].conj()))
error('fixture_mid_probabilities',[probabilities(w,q0)['mid'] for w in words],[0,1])
error('fixture_D_probabilities',[probabilities(w,q0)['dephased'] for w in words],[.5,.5])
qs=[q0]+[(.7*np.cos((index+1)*j*.219)).reshape(2,3,2) for index in range(1,40)]
allp={name:np.array([[probabilities(w,q)[name] for w in words] for q in qs]) for name in ('out','mid','dephased')}
error('all40_native_equal_token_scores',allp['out'][:,0],allp['out'][:,1])
u=np.zeros((40,2));u[0]=[1,-1]
logits={};alphas={}
for name in allp:
 values=[pool(allp[name][row],u[row],np.ones(2,dtype=bool)) for row in range(40)]
 logits[name]=np.array([x[0] for x in values]);alphas[name]=np.array([x[1] for x in values])
error('native_all40_logits',logits['out'],np.zeros(40))
expected=np.zeros(40);expected[0]=-1/(1+2*EPS)
error('mid_all40_logits',logits['mid'],expected)
error('D_all40_logits',logits['dephased'],np.zeros(40))
for name in ('out','mid','dephased'):
 for token in (0,1):
  value,a=pool(allp[name][0,token:token+1],u[0,token:token+1],np.array([True]))
  error('singleton_logit',value,u[0,token]);error('singleton_alpha',a,[1])
 padded,a=pool(np.r_[allp[name][0],.875],np.r_[u[0],53.],np.array([True,True,False]))
 error('padding_logit',padded,logits[name][0]);error('padding_alpha',a,np.r_[alphas[name][0],0.])
softmax=lambda z:np.exp(z-z.max())/np.exp(z-z.max()).sum()
soft={name:softmax(z) for name,z in logits.items()}
softdiff=float(np.max(np.abs(soft['mid']-soft['out'])))
assert softdiff>1e-9
zero=probabilities(np.zeros((2,3,2)),np.zeros((2,3,2)))
error('zero_angle_endpoint',list(zero.values()),np.ones(3))
assert wrong_adjoint_max_error>1e-3
out=dict(status='PASS' if max(errors.values())<=1e-11 else 'FAIL',threshold=1e-11,max_error=max(errors.values()),errors=errors,
 fixed_tuples=32,native_vector_circuit_calls=64,retained_native_layer_matrix_calls=128,
 fixture=dict(out_row0=float(logits['out'][0]),mid_row0=float(logits['mid'][0]),D_row0=float(logits['dephased'][0]),
  expected_mid=-1/(1+2*EPS),softmax_max_difference=softdiff,native_class0_probability=float(soft['out'][0]),mid_class0_probability=float(soft['mid'][0])),
 capacity=dict(last_word_three_RZ_exactly_invisible_Q_D=True,first_word_three_RZ_invisible_D=True,
  native_out_last_RZ_max_observed_change=max_word_last_rz_out_change,wrong_adjoint_max_amplitude_error=wrong_adjoint_max_error),
 elapsed_seconds=time.perf_counter()-started,resources=dict(GPU=0,task_forward=0,fit=0,new_capacity=0),
 scope='Data-free dense/native-circuit synthetic correspondence; fixed preparer/utility output witness, not task evidence or reparameterized-family separation',
 candidate='PROVISIONAL',training_admitted=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
trace=dict(tuples=traces,fixture=dict(word_angles=[wa.tolist(),wb.tolist()],query_angles=[q.tolist() for q in qs],
 completed_states=[dict(real=x.real.tolist(),imag=x.imag.tolist()) for x in states],probabilities={k:v.tolist() for k,v in allp.items()},
 utilities=u.tolist(),logits={k:v.tolist() for k,v in logits.items()},alphas={k:v.tolist() for k,v in alphas.items()},softmax={k:v.tolist() for k,v in soft.items()}))
for name,value in [('math-results.json',out),('math-trace.json',trace)]:
 (r/name).write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps(out))
assert out['status']=='PASS'
