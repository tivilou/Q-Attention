"""Offline closeout verification; never repeats either remote scientific calculation."""
from pathlib import Path
import json,hashlib,math
r=Path(__file__).resolve().parent
load=lambda n:json.loads((r/n).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
d=load('objective-diagnostic.json');v=load('verification.json');p=load('diagnostic-protocol.json')
assert v['status']=='PASS_INDEPENDENT_EXISTING_PREDICTION_RECOMPUTATION'
assert d['diagnostic_protocol_sha256']==sha((r/'diagnostic-protocol.json').read_bytes())
assert d['diagnostic_code_sha256']==sha((r/'diagnose.py').read_bytes())
for fname,code in [('execution.json','diagnose.py'),('verification-execution.json','verify_remote.py')]:
    e=load(fname)
    assert e['exit_status']==0 and e['literal_stderr']==''
    assert e['stdin_code_sha256']==sha((r/code).read_bytes())
    literal=json.loads(e['literal_stdout'])
    expected=d if fname=='execution.json' else load('independent-diagnostic.json')
    assert all(expected[k]==val for k,val in literal.items())

errors=[];leaves=0
def compare(a,b):
    global leaves
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):
        assert math.isfinite(a) and math.isfinite(b)
        errors.append(abs(a-b));assert abs(a-b)<1e-12;leaves+=1
    else:assert a==b;leaves+=1
ind=load('independent-diagnostic.json')
compare(d['models'],ind['models']);compare(d['fit_support'],ind['fit_support'])
assert leaves==v['compared_scalar_leaves']==2715 and max(errors)==v['maximum_absolute_numeric_error']
assert len(d['models'])==5 and d['row_count']==4096
for name,m in d['models'].items():
    assert len(m['by_class'])==40 and sum(t['rows'] for t in m['by_class'])==4096
    assert len(m['paired_groups'])==8 and sum(g['rows'] for g in m['paired_groups'].values())==4096
    assert abs(sum(t['brier'] for t in m['by_class'])/40-m['class_macro_brier'])<1e-14
q=d['models']['quantum_distinct'];g=d['models']['gaussian_distinct']
assert q['overall']['brier']>g['overall']['brier'] and q['class_macro_brier']>g['class_macro_brier']
assert q['original_macro_f1']<g['original_macro_f1']
receipt=load('fulltexts/8f220cfb35184edc/receipt.json');reading=load('reading-record.json');source=load('source-receipt.json')
assert source['receipt_sha256']==sha((r/'fulltexts/8f220cfb35184edc/receipt.json').read_bytes())
assert reading['receipt_before_reading_sha256']==sha((r/'receipt.before-reading.json').read_bytes())
assert receipt['identity']=='verified' and receipt['reading']=='read_for_scope'
assert source['pdf_sha256']==sha(Path(receipt['path']).read_bytes())
assert reading['read_pages']==source['read_pages']==receipt['read_pages']==[1,2,3,4,5,6,7,8,12,13]
assert reading['visually_inspected_pages']==[1,3]
for name,s in load('history-comparison.json')['paths_sha256'].items():assert sha(Path(name).read_bytes())==s
disc=load('next-source-discovery.json');public=load('next-source-discovery-public.json')
assert len(disc['queries'])==2 and [q['status'] for q in disc['queries']]==['failed','retrieved']
assert len(public['queries'][1]['entries'])==6
for i,query in enumerate(disc['queries']):
    if query['status']=='retrieved':assert sha((r/f'next-source-query-{i}.xml').read_bytes())==query['sha256']
next_step=load('next-direction.json')
assert next_step['selected_plugin'] is None and next_step['goal_status']=='ACTIVE'
assert '2503.20683v3' in next_step['next_lead']['id'] and next_step['lead_status']=='PREVIOUSLY_COLLECTED_METADATA_ABSTRACT_ONLY'
out=dict(status='PASS_OFFLINE_EVIDENCE_CLOSEOUT',remote_jobs_repeated=0,all5_all40_all8_reconciled=True,
    compared_scalar_leaves=leaves,max_absolute_error=max(errors),source_reading_scope_and_receipts_verified=True,
    execution_literal_outputs_and_code_hashes_verified=True,history_hashes_verified=True,
    discovery_successful_queries=1,discovery_failed_queries=1,discovery_fulltexts_read=0,
    next_lead_metadata_only=True,new_method_selected=False,gen98_decision='STOP_EXACT_CONFIGURATION_UNCHANGED',
    new_updates=0,new_gpu_allocations=0,new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED',goal='ACTIVE')
(r/'evidence-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
