from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
r=Path(__file__).resolve().parent;root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
read=lambda n:json.loads((r/n).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read('math-results.json');v=read('independent-verification.json');review=read('review.json');packet=read('review-packet.private.json')
assert m['status']=='PASS' and v['status']=='PASS_EXACT_RATIONAL_COLLISION_AND_DEPHASED_CHAIN'
assert review['model']=='gpt-6.2-astra' and review['effort']=='high' and review['response']['verdict']=='PASS_WITH_RISK'
assert review['packet_sha256']==sha(r/'review-packet.private.json')
assert review['raw_sha256']==sha(r/'response.private.json')
assert packet['code']==(r/'interleaved_math.py').read_text('utf8') and packet['hypothesis']==(r/'hypothesis.md').read_text('utf8')
for p in packet['source_pins']:assert sha(p['path'])==p['sha256']
for name in ('prepare_review','consult','math_preflight','independent_verify'):
 e=read(name+'-execution.json');assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/(name+'.py'))
pre=read('preflight.json')
compact=dict(status=pre['status'],model=pre['model'],effort=pre['effort'],protocol=pre['protocol'],
 metadata_revision=pre['metadata']['revision'],exact_metadata_matches=[x.get('id') for x in pre['metadata']['matches']],
 metadata_attempts=pre['metadata_attempts'],exact_catalog_candidates=pre['exact_catalog_candidates'],
 selection_basis=pre['selection_basis'],capability_ranking='Not inferred from derived metadata or catalog ordering',
 coverage='Main revision and models queried for exact model metadata; not a complete four-source benchmark review.')
(r/'preflight-summary.json').write_bytes((json.dumps(compact,indent=2)+'\n').encode())
pins=[r/n for n in ('hypothesis.md','interleaved_math.py','primary-adoption.md','math-results.json','independent-verification.json','review.json')]
pins += [Path(p['path']) for p in packet['source_pins'][:4]]
next_direction=dict(status='NEXT_IMPLEMENT_FROZEN_PLACEMENT_PLUGIN_AND_COMPACT_SCREEN_CONFIG',
 action='Implement one isolated scientific plugin for the unchanged middle-placement equations and Q/D/real8 controls, plus the compact fixed screen configuration. Reuse stable core/qualification. Resolve exact angle mapping, adjoint order, null directions, shared initial tensor mapping and actual comparable old anchor identities before any task forward or fit.',
 frozen_equations=dict(Q='abs(<0|W2(t) Q_r^dagger W1(t)|0>)^2',D='[M(W2) M(Q_r^dagger) M(W1)]00 with dephasing at these block boundaries only',
 real8='Same placement; twelve word angles as two two-layer-RY blocks and twelve relation angles as four RY layers, fixed order and true transpose to be implemented'),
 invariants=['Original pi*tanh, prepare/effect state schema, h-linear64 utility and EPS1e-6 word normalization retained.',
 'Explicitly drop only analytically invisible phase dependence from the computation; do not hide the zero route gradients or call registered count equal effective capacity.',
 'Blockwise D retains within-Q interference; it is not all-gate dephasing. Real8 changes gate count/geometry and is not a complete complexity match.',
 'One-token logit is utility+bias. Native Q is the placement ablation. Strong native real12/Gaussian/Gcov references remain necessary.',
 'Bind old anchors to exact inputs/cohort/initial mapping/order/optimizer/1024 updates/all40 metric before reusing them; no repeating stopped fits or assuming matching score alone.',
 'After implementation, perform one focused mechanism implementation review and affected finite-gradient/one-update validation before any small efficacy screen. No runner lifecycle matrix or fresh dataset-qualification dossier.',
 'Task hypothesis only: terminal-state aliasing may lose task-relevant preparation-path response. Synthetic collision is not a measured cause or a freely reparameterized-family/risk separation.'],
 provisional_screen=dict(cohort='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0',fit=8192,assessment=4096,exposure='existing TRAIN development',
 seed=13,batch_size=128,updates_per_new_arm=1024,metric='all40 Macro at final update',
 required='Q strictly above every new key control and every declared strongest comparable reference; ties fail; all40 and CE reported.',
 stop='Failure closes this exact placement/ansatz/budget profile; no position/depth/seed/temperature/cohort/budget/control rescue. Pass only permits prospective full-study preparation.'),
 source_pins=[dict(path=str(p),sha256=sha(p)) for p in pins],
 candidate='PROVISIONAL',training_admitted=False,resources_next=dict(GPU=0,new_task_forward=0,fit=0,new_capacity=0),
 retained_stops=['gen98','gen102','gen104','gen105','all other exact stopped profiles'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'next-direction.json').write_bytes((json.dumps(next_direction,ensure_ascii=False,indent=2)+'\n').encode())
verification=dict(status='PASS_INTERLEAVED_PLACEMENT_MATHEMATICAL_CHECKPOINT',review_verdict='PASS_WITH_RISK',candidate='PROVISIONAL',
 reviewed_code_sha256=sha(r/'interleaved_math.py'),source_pin_count=len(packet['source_pins']),review_packet_sha256=review['packet_sha256'],
 max_math_error=m['max_error'],exact_native_logit='0',exact_mid_logit='-500000/500001',
 preserved_blind_spots=['Fixed preparer/utility only; no general task-family separation.','No task data, end-to-end gradient, implementation training or efficacy measurement.','Known process lift and gate identities are not new T; original literature review remains unresolved.'],
 resources=m['resources'],model_API_calls=1,training_admitted=False,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'verification.json').write_bytes((json.dumps(verification,indent=2)+'\n').encode())
names=['hypothesis.md','interleaved_math.py','primary-adoption.md','math_preflight.py','math_preflight-execution.json',
 'math-results.json','math-trace.json','independent_verify.py','independent_verify-execution.json','independent-verification.json',
 'review.json','selection.json','source-pins.json','preflight-summary.json','dispatch.json','prepare_review.py','prepare_review-execution.json',
 'consult.py','consult-execution.json','model_preflight.py','execute.py','analysis.md','next-direction.json','verification.json']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b)
assert sum(map(len,raw.values()))<300000
assert all('.private' not in n and not n.endswith(('.pt','.pdf','.html','.npz','.npy','.png')) for n in names)
package=dict(status='PASS_INTERLEAVED_PLACEMENT_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
 candidate='PROVISIONAL',training_admitted=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(package,indent=2)+'\n').encode())
print(json.dumps(dict(status=verification['status'],files=len(names),public_bytes=sum(map(len,raw.values())),max_math_error=m['max_error'],model_API_calls=1,task_forward=0,fit=0)))
