"""Independent exact-form checks of compact boundary outputs and provenance."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math
r=Path(__file__).resolve().parent
load=lambda n:json.loads((r/n).read_bytes())
v=load('native-checks.json');e=load('execution.json')
assert e['exit_status']==0 and not e['literal_stderr']
assert json.loads(e['literal_stdout'])==v
assert e['script_sha256']==hashlib.sha256((r/'check_native.py').read_bytes()).hexdigest()
eps=F(1,10**6); count=(1+eps)/(1+eps+F(1,2)+eps)
for row,p in zip(v['coherent_counterexamples'],[F(1),F(1,4)]):
    mass=(1+eps)/(1+eps+p+eps)
    assert abs(row['count_corrected_target_mass']-float(count))<1e-14
    assert abs(row['coherent_target_mass']-float(mass))<1e-14
    assert abs(row['delta']-float(mass-count))<1e-14
expected=F(2,5)/(F(1)+2*eps)
for row,sign in zip(v['signed_utility_counterexamples'],[1,-1]):
    assert row['mean_value_logit']==0
    assert abs(row['count_corrected_logit']-float(sign*expected))<1e-14
for n,m in v['fixed_eps_single_target_upper_bounds'].items():
    assert abs(m-float((1+eps)/(1+eps+int(n)*eps)))<1e-14
for k in ['max_density_linearity_error','max_dephased_linearity_error','max_covariance_identity_error','max_native_state_norm_error']:
    assert math.isfinite(v[k]) and 0<=v[k]<1e-12
assert v['circuit_port_checks']==320 and v['random_trials']==8
assert all(v[k]==0 for k in ['task_rows_read','trained_head_forwards','training_updates','gpu_allocations'])
assert not v['new_theory_claim'] and not v['quantum_advantage_claim'] and not v['source_claim_refuted']
receipt=load('fulltexts/583f2ca9902be227/receipt.json')
assert receipt['identity']=='verified' and receipt['observed_version']=='2610.34682v2'
assert receipt['parsing']=='parsed_text' and not receipt['low_text_pages'] and receipt['pages']==42
assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
pages=json.loads(Path(receipt['parsed_path']).read_bytes())['pages']
assert len(pages)==42 and '2610.34682v2' in pages[0]['text'] and 'HEEHYUN PARK' in pages[0]['text']
assert 'averaging' in pages[5]['text'].lower()
source=Path(e['argv'][e['argv'].index('--native-source')+1])
assert hashlib.sha256(source.read_bytes()).hexdigest()==v['native_source_sha256']
discovery=load('next-source-discovery.json')
assert len(discovery['queries'])==2 and all(q['http_status']==200 for q in discovery['queries'])
assert [len(q['entries']) for q in discovery['queries']]==[6,6]
result=dict(status='PASS',independent_exact_formula_checks=True,all_40_circuit_ports_per_trial=True,
    source_pdf_identity_and_hash_verified=True,native_source_unchanged=True,
    literal_execution_output_and_exit_verified=True,source_theorem_certified=False,
    original_goal_status='ACTIVE',new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',
    direct_transfer='STOP',selected_new_plugin=None,new_task_data_accesses=0,new_fits=0,
    new_gpu_allocations=0,new_cos_objects=0)
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
