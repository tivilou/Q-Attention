from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import ast,hashlib,json
import numpy as np
import torch
from pair_math import compute,NAMES
from pair_statistics import auc,summarize
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=read('identity-spec.json');probe=read('identity-probe.json')
assert probe['status']=='PASS_EXISTING_IDENTITY_PROBE' and not probe['errors']
assert sha(spec['prediction']['path'])==spec['prediction']['sha256']
old=Path('D:/CodexHome/task-artifacts/qattn-context-aliasing-source-exclusion-20261126/algebra-trace.json')
j=json.loads(old.read_bytes());errors=[]
for c in j['contexts']:
 h=torch.tensor(c['h'],dtype=torch.float64)[None];mask=torch.ones((1,4),dtype=torch.bool)
 p=torch.tensor(c['psi_real'],dtype=torch.complex128)[None]+1j*torch.tensor(c['psi_imag'],dtype=torch.complex128)[None]
 # Same admissible Gaussian scaled key as the already-certified fixture.
 scores,info=compute(h,mask,p,h[:,:,:5])
 expected=c['metrics'];errors.extend([abs(float(scores[0,4])-float(F(c['exact_AQ']))),abs(float(scores[0,5])-expected['AG']),float(info['real16_error'])])
 assert torch.equal(info['pair_count'],torch.tensor([6]))
assert max(errors)<1e-12
# Exact pair-count oracle, including ties, verifies score orientation and ranking.
scores=np.array([0.,1.,1.,3.,4.]);positive=np.array([False,True,False,True,False])
exact=sum(F(1) if a>b else F(1,2) if a==b else F(0) for a in scores[positive] for b in scores[~positive])/6
assert abs(auc(scores,positive)-float(exact))<1e-15 and exact==F(7,12)
assert auc([1,1],[True,False])==.5 and auc([1],[True]) is None
toy=np.zeros((80,7));labels=np.zeros(80,int);pred={'quantum_distinct':np.r_[np.zeros(40,int),np.ones(40,int)],'gaussian_covariance':np.r_[np.ones(40,int),np.zeros(40,int)]}
assert summarize(toy,labels,pred,True)['decision']=='STOP_THIS_EXACT_ALIASING_PREDICTION'
names=('runner.py','pair_math.py','pair_statistics.py')
for n in names:ast.parse((r/n).read_text('utf8'))
design=dict(profile='qattn-context-pair-diagnostic-20261126-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
 identity_spec=spec,hypothesis_sha256=spec['prediction']['sha256'],
 input_hashes=dict(spec['files'],**spec['inputs']),code_hashes={n:sha(r/n) for n in names},
 batch_size=32,wall_seconds=600,constructor_seed=0,constructor_seed_scope='Initialization overwritten by all four exact checkpoints; no training/seed selection.',
 resources=dict(gpus=1,gpu_mb=4096,cpu=4,ram_mb=12288,disk_mb=128),
 calls=dict(head_forwards=512,native_batched_circuit_calls=512,task_examples=4096,optimizer_updates=0),
 precision='Exact existing FP32 inference from FP16 h cache. New input distances, pair kernels and A statistics use FP64/complex128 on the actual FP32 states/scaledGkeys; no state renormalization. Native TF32 disabled.',
 technical_gates=dict(sealed_logits_scaled_error=2e-4,argmax_mismatches=0,real16_kernel_error=1e-12,independent_numpy_error=1e-10,sealed_macro_error=1e-12,unchanged_states=True),
 scientific_gate=json.loads(Path(spec['prediction']['path']).read_bytes())['diagnostic_gate'],
 selection='One pass/four immutable final checkpoints/all4096 in original order/all40; four fixed first assessment cases for input/kernel/reference trace. No changing logits, labels, checkpoints, classes or thresholds.',
 persistence='One bounded no-update diagnosis, immutable inputs/local-only private population and4case trace; restart_from_inputs capability only, no automatic retry. No COS or new checkpoint/optimizer saves.',
 stop='Failed identity/numerics -> technical nonadmission. Failed frozen AUC/count/purity-dephased-length advantage -> STOP exact prediction. No old fits restarted or subgroup/control/distance/seed rescue.',
 local_math_reference='Reuse previous synthetic inputs and Fraction 7/12 tied-AUC oracle; same agent, not external research review.',
 training_admitted=False,method_selected=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
protocol='''# Frozen no-update pairing diagnostic

Reuse exactly completed gen98 Qdistinct/Ddistinct/Gdistinct and gen91 strongest
declared Gaussian-covariance final checkpoints. Old profiles remain stopped.
Immutable source/input/cohort/sealed prediction identities are in design.json.
Hash the full4.43GB h and34.6MB roles cache inside the new allocation before
reading task tensors; all other48 small/source/checkpoint files already match.

Run one4096-row assessment pass, original32-row batches, all four heads.
Exactly512 head and512 batched circuit calls; one shared GPU/max600seconds,
0updates/0fit/0new capacity. No model API, COS, new heldout, automatic retry.
No training-state or source file is changed; hash before/after and verify all
four saved prediction vectors and all40 confusion matrices.

Original FP32 inference is replayed. New kernel/input association uses FP64 at
the actual stored-precision states and scaled Gaussian keys. No state/query
normalization, epsilon addition, selected pairs or kernel hyperparameter scan.
The one S formula, orientation, AUC>=.55 and advantage>=.02 over best of length,
purity and same-Q-state dephasing were frozen in the prior hypothesis.json.
Every primary undefined case stops this exact prediction. All80 gold-class x
fixed length35 strata are reported, never used to override the pooled gate.

Four fixed first assessment cases retain complete cached inputs, angles,
word states, scaledGkeys, all pair matrices/statistics and sealed/replayed four
head outputs privately on server. Independent NumPy pair algebra verifies all
pairs in those cases; upstream contextual encoder/training trace unavailable.
Same-agent separate implementation is not external review. This diagnosis does
not establish cause, training efficacy, new theory, L3/L4-T, or method admission.
'''
(r/'protocol.md').write_text(protocol,encoding='utf8')
out=dict(status='PASS_LOCAL_PREFLIGHT',synthetic_max_error=max(errors),exact_auc_oracle=str(exact),files={n:sha(r/n) for n in names+('design.json','protocol.md')},
 external_research_review=False,task_calls_before_allocation=0,frozen_prediction_sha256=spec['prediction']['sha256'])
(r/'local-preflight.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],synthetic_max_error=max(errors),exact_auc_oracle=str(exact),calls=design['calls'])))
