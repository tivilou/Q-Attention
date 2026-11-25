"""Independent NumPy permutation algebra, source-table arithmetic and record checks."""
from pathlib import Path
import json,hashlib,re,time
import numpy as np
r=Path(__file__).resolve().parent;start=time.monotonic()
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
p=read('synthetic-protocol.json');a=read('assignment-results.json');e=read('execution.json')
assert a['status']=='PASS_ASSIGNMENT_ONLY_REPARAMETRIZATION' and e['exit_status']==0 and e['stderr']==''
assert sha((r/'assignment-audit.py').read_bytes())==a['code_sha256']
assert sha((r/'synthetic-protocol.json').read_bytes())==a['protocol_sha256']==e['inputs']['synthetic-protocol.json']
assert sha((r/'assignment-results.json').read_bytes())==e['output_sha256']
assert json.loads(e['stdout'])['status']==a['status']
for mode in ['quantum','dephased']:
    x=a['results'][mode]
    assert set(x['gradient_errors_in_baseline_coordinates'])=={'decoder','bias','value.weight','value.bias','prepare.weight','prepare.bias','effect','synthetic_h'}
    assert max(x['forward_errors'].values())<p['tolerance']
    assert max(x['gradient_errors_in_baseline_coordinates'].values())<p['tolerance']
    assert np.max(abs(np.array(x['baseline_logits'])-np.array(x['mapped_logits'])))==0
    assert x['parameter_hashes_unchanged'] and x['uncompensated_state_change']>1e-4
P=np.eye(12)[p['permutation']]
assert np.array_equal(P@P.T,np.eye(12))
W=np.sin(np.arange(12*9).reshape(12,9)/31);b=np.arange(12)/17
inputs=np.cos(np.arange(96*9).reshape(96,9)/23)
native=np.pi*np.tanh(inputs@W.T+b)
mapped=(np.pi*np.tanh(inputs@(P.T@W).T+P.T@b))@P.T
algebra_error=float(np.max(abs(native-mapped)))
assert algebra_error<1e-14
assert abs(np.linalg.norm(W)-np.linalg.norm(P.T@W))<1e-14
receipt=read('fulltexts/6c369f2ba8a3bb10/receipt.json');sr=read('source-receipt.json');rr=read('reading-record.json')
assert receipt['identity']=='verified' and receipt['reading']=='read_for_scope'
assert sha(Path(receipt['path']).read_bytes())==sr['pdf_sha256']=='962bf554c8cddc29c9fc59954380fa635316afde13a54b7285746b9bb91f0a6c'
assert sha((r/'fulltexts/6c369f2ba8a3bb10/receipt.json').read_bytes())==sr['final_receipt_sha256']
assert sha((r/'receipt.before-reading.json').read_bytes())==rr['receipt_before_reading_sha256']
assert len(rr['read_pages'])==22 and rr['visual_pages']==[1,15,23,24]
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert sha(Path(receipt['parsed_path']).read_bytes())==receipt['parse_sha256']
# This checks the protocol clauses in context, rather than treating all100 seeds as independent ablation confirmation.
normalize=lambda s:re.sub(r'\s+',' ',s.replace('\u2212','-')).strip()
L=normalize(pages[22]['text']);M=normalize(pages[24]['text'])
assert re.search(r'Hyperparameters.*selected on the test set',L)
assert 'no cross-validation' in L
assert 'optimizing the test R2' in M and '19–20-seed' in M
t=read('source-tables.json');t12=normalize(pages[20]['text']);t15=normalize(pages[23]['text'])
for horizon,tail,rq,rc,rd in t['table12']:
    nums=[str(horizon),str(tail),f'{rq:+.3f}',f'{rc:+.3f}',f'{rd:+.3f}']
    assert ' '.join(nums) in t12,nums
for theta,bi,grp in t['table15']:
    assert f'{bi:.4f}' in t15 and f'{grp:.4f}' in t15
reversals=[dict(horizon=x[0],tail=x[1],r_Q=x[2],r_RBF=x[3]) for x in t['table12'] if x[3]<=x[2]]
assert len(reversals)==6
rows=np.array(t['table15']);bi=rows[:,1];grp=rows[:,2]
table_arithmetic=dict(r_RBF_not_greater_pairs=reversals,total_pairs=24,
    grouped_identity_to_pi2=float(grp[-1]-grp[0]),interleaved_identity_to_peak=float(bi.max()-bi[0]),
    interleaved_identity_to_pi2=float(bi[-1]-bi[0]),all_observed_grouped_greater=True,
    inference='Source table arithmetic only; no new task effect or reestimated uncertainty.')
assert np.all(grp>bi) and np.all(np.diff(grp)<0)
assert abs(table_arithmetic['interleaved_identity_to_peak']-.0129)<1e-12
assert abs(table_arithmetic['interleaved_identity_to_pi2']+.0078)<1e-12
assert read('next-direction.json')['goal']=='ACTIVE'
out=dict(status='PASS_INDEPENDENT_ALGEBRA_SOURCE_TABLE_AND_RECORD_AUDIT',
    source_identity_and_reading_record_verified=True,source_test_tuning_clauses_checked=True,
    full_native_gradient_parity_record_verified=True,independent_algebra_max_error=algebra_error,
    source_table_arithmetic=table_arithmetic,independent_full_circuit_reimplementation=False,
    external_scientific_review=False,independent_task_confirmation=False,new_method_selected=False,
    task_rows=0,trained_head_forwards=0,new_updates=0,new_gpu_allocations=0,
    goal='ACTIVE',L3='NOT_ACCEPTED',L4T='NOT_ACCEPTED',elapsed_seconds=time.monotonic()-start,
    inputs_sha256={n:sha((r/n).read_bytes()) for n in ['scope.json','synthetic-protocol.json','assignment-audit.py','assignment-results.json','execution.json','source-receipt.json','reading-record.json','source-claim-ledger.json','source-tables.json','derivation.md','analysis.md','history-comparison.json','next-direction.json']},
    verification_code_sha256=sha(Path(__file__).read_bytes()))
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],algebra_error=algebra_error,source_reversal_pairs=len(reversals),elapsed_seconds=out['elapsed_seconds'])))
