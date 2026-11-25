"""Mechanical source/table integrity checks; not an independent paper review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
def load(n):return json.loads((r/n).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
scope=load('scope.json');source=load('source-receipt.json');read=load('reading-record.json')
rp=r/'fulltexts/63b54eaa5d5cef35/receipt.json';receipt=json.loads(rp.read_bytes())
assert scope['paper_id']==source['paper_id']==receipt['paper_id']=='arxiv:2105.02276v1'
assert source['title']==scope['expected_title']==receipt['title']
assert source['authors']==load('request.json')['papers'][0]['authors']
assert sha(Path(receipt['path']).read_bytes())==source['pdf_sha256']==receipt['sha256']
assert source['pdf_sha256']=='72560b31d90f6cf85bd0436d424210088e7d2091e2afc913779fbe1cdf791a7c'
assert receipt['identity']=='verified' and receipt['reading']=='read_for_scope'
assert source['final_receipt_sha256']==sha(rp.read_bytes())
assert source['reading_record_sha256']==receipt['reading_record_sha256']==sha((r/'reading-record.json').read_bytes())
assert read['receipt_before_reading_sha256']==sha((r/'receipt.before-reading.json').read_bytes())
assert read['read_pages']==source['read_pages']==list(range(1,14))+[15,16,19,20]
assert read['visual_pages']==source['visual_pages']==[1,6,12]
assert not source['author_code_inspected'] and not source['independent_reproduction']
parsed=json.loads(Path(receipt['parsed_path']).read_bytes())
assert sha(Path(receipt['parsed_path']).read_bytes())==receipt['parse_sha256']
assert len(parsed['pages'])==20 and parsed['source_sha256']==receipt['sha256']
for p in read['read_pages']:assert parsed['pages'][p-1]['characters']>0
table=load('source-table1.json');rows=table['rows'];assert len(rows)==6
nonensemble=[z for z in rows if z['minimum'] is not None]
assert len(nonensemble)==5
for row in nonensemble:assert 0<=row['minimum']<=row['maximum']<=1 and 0<=row['trained']<=1
def compare(field):
    return dict(win=sum(z['trained']>z[field] for z in nonensemble),tie=sum(z['trained']==z[field] for z in nonensemble),loss=sum(z['trained']<z[field] for z in nonensemble))
minimum=compare('minimum');maximum=compare('maximum')
assert minimum==dict(win=5,tie=0,loss=0) and maximum==dict(win=3,tie=1,loss=1)
assert rows[3]['maximum'] is None and rows[3]['trained']==1. and rows[3]['total_samples']=='10images,15coordinates each'
interface=load('native-interface-precheck.json')
for x in interface['inputs']:assert sha(Path(x['path']).read_bytes())==x['sha256']
assert interface['inputs'][0]['sha256']=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
assert interface['inputs'][1]['sha256']=='769173b9ab17f76229b0db79b4a7625ac1574427d2d40491be7d5b19d4d0ff25'
card=root/'.ai-progress/workstreams/quantum-projector-v2/refs/20261115-q-rmea-ordinary-alignment-l4-evidence-card.json'
assert sha(card.read_bytes())=='ba10c598e8f12b91668a339cff8d367647025124494afac0f37221c1f023d89e'
previous=Path('D:/CodexHome/task-artifacts/qattn-task-mechanism-discovery-20261126/next-direction.json')
assert sha(previous.read_bytes())==scope['prior_next_sha256']
claims=load('source-claim-ledger.json');history=load('history-comparison.json');next_=load('next-direction.json')
assert next_['selected_plugin'] is None and not claims['new_method_selected'] and not claims['new_theory']
for x in (scope,history,next_):assert x['goal']=='ACTIVE' and x['newL3']==x['newL4T']=='NOT_ACCEPTED'
names=('scope.json','source-receipt.json','reading-record.json','source-claim-ledger.json','source-table1.json',
       'native-interface-precheck.json','history-comparison.json','derivation.md','analysis.md','next-direction.json')
out=dict(status='PASS_MECHANICAL_SOURCE_AND_TABLE_PACKAGE',verified_at=datetime.now(timezone.utc).isoformat(),
    source_pdf_identity_attested_and_hash_verified=True,source_receipt_reading_verified=True,read_page_count=17,
    visual_pages=read['visual_pages'],all_source_table_rows=6,nonensemble_rows=5,trained_vs_minimum=minimum,trained_vs_maximum=maximum,
    original_L4_and_native_source_preserved=True,author_code_reproduced=False,new_method_selected=False,new_theory=False,
    task_rows=0,trained_head_forwards=0,synthetic_circuit_executions=0,new_updates=0,new_gpu_allocations=0,
    server_writes=0,COS_writes=0,model_API_calls=0,external_scientific_review=False,independent_task_confirmation=False,
    limitations=['Mechanical check verifies recorded locators/receipt identities and displayed-row calculations, not human reading or scientific novelty.',
                'No author-code, hardware or task numerical reproduction; gradient formulas are known calculus, not a new T result.',
                'Local fulltext/renders/helper receipt and pre-reading backup are not distributed.'],
    evidence_sha256={n:sha((r/n).read_bytes()) for n in names},goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],read_pages=17,source_table_rows=6,versus_minimum=minimum,versus_maximum=maximum,goal='ACTIVE')))
