"""Mechanical evidence verification; not independent scientific review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,xml.etree.ElementTree as ET
r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
def load(n):return json.loads((r/n).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
profile=load('discovery-profile.json'); discovery=load('next-source-discovery.json')
previous=json.loads(Path('D:/CodexHome/task-artifacts/qattn-trace-kernel-source-20261126/next-direction.json').read_bytes())
assert profile['queries']==[x['query'] for x in previous['queries']]
assert len(discovery['queries'])==profile['max_requests']==2
ns={'a':'http://www.w3.org/2005/Atom','s':'http://a9.com/-/spec/opensearch/1.1/'}
counts=[]; totals=[]
for i,q in enumerate(discovery['queries']):
    raw=(r/f'next-source-query-{i}.xml').read_bytes()
    assert sha(raw)==q['sha256'] and q['http_status']==200
    assert q['query']==profile['queries'][i] and q['requested_max_results']==6
    feed=ET.fromstring(raw); entries=feed.findall('a:entry',ns)
    total=int(feed.find('s:totalResults',ns).text)
    assert len(entries)==len(q['entries'])<=6
    assert [e.find('a:id',ns).text for e in entries]==[e['id'] for e in q['entries']]
    counts.append(len(entries));totals.append(total)
assert counts==totals==[0,1]
public_discovery=load('discovery-summary.json')
assert public_discovery['raw_metadata_receipt_sha256']==sha((r/'next-source-discovery.json').read_bytes())
assert not public_discovery['abstracts_distributed']
for public,private in zip(public_discovery['queries'],discovery['queries']):
    assert public['query']==private['query'] and public['sha256']==private['sha256']
    assert public['entry_count']==len(private['entries'])
    assert public['entries']==[{k:v for k,v in e.items() if k!='summary'} for e in private['entries']]
receipt=load('fulltexts/87aa1f129ab4c438/receipt.json')
source=load('source-receipt.json');read=load('reading-record.json')
assert sha(Path(receipt['path']).read_bytes())==receipt['sha256']==source['pdf_sha256']
assert receipt['identity']=='verified' and receipt['reading']=='read_for_scope'
assert receipt['version']==source['version']=='2609.06016v1'
assert source['title']==discovery['queries'][1]['entries'][0]['title']
assert source['authors']==discovery['queries'][1]['entries'][0]['authors']
assert source['final_receipt_sha256']==sha((r/'fulltexts/87aa1f129ab4c438/receipt.json').read_bytes())
assert source['reading_record_sha256']==sha((r/'reading-record.json').read_bytes())
assert read['receipt_before_reading_sha256']==sha((r/'receipt.before-reading.json').read_bytes())
assert read['read_pages']==source['read_pages']==list(range(1,19))+list(range(22,32))
assert read['visual_pages']==source['visual_pages']==[1,8,9,31]
assert not source['author_code_inspected'] and not source['independent_reproduction']
parsed=json.loads(Path(receipt['parsed_path']).read_bytes())
assert sha(Path(receipt['parsed_path']).read_bytes())==receipt['parse_sha256']
assert len(parsed['pages'])==31 and parsed['source_sha256']==receipt['sha256']
for p in read['read_pages']:assert parsed['pages'][p-1]['characters']>0
tables=load('source-table12.json')
assert tables['datasets']==list('ABCDEFGHIJKLMNOP')
rounding={};ties={};worse={}
for metric in ('acc','nmi'):
    with_=tables[metric+'_with'];without=tables[metric+'_without']
    assert len(with_)==len(without)==16
    assert all(0<=x<=1 for x in with_+without)
    for suffix,values in [('with',with_),('without',without)]:
        key=metric+'_'+suffix
        mean=sum(values)/16
        assert abs(mean-tables['printed_average'][key])<=.000500001
        rounding[key]=dict(mean_of_displayed_rows=mean,printed_average=tables['printed_average'][key])
    ties[metric]=[tables['datasets'][i] for i,(a,b) in enumerate(zip(with_,without)) if a==b]
    worse[metric]=sum(a<b for a,b in zip(with_,without))
assert worse==dict(acc=0,nmi=0)
native=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py')
assert sha(native.read_bytes())=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
card=root/'.ai-progress/workstreams/quantum-projector-v2/refs/20261115-q-rmea-ordinary-alignment-l4-evidence-card.json'
assert sha(card.read_bytes())=='ba10c598e8f12b91668a339cff8d367647025124494afac0f37221c1f023d89e'
next_=load('next-direction.json');history=load('history-comparison.json');claims=load('source-claim-ledger.json')
assert next_['selected_plugin'] is None and not claims['new_method_selected'] and not claims['new_theory']
assert next_['exact_source']=='arxiv:2105.02276v1'
for name,key in [('literature-index.json','prior_index_sha256'),('idea-portfolio.json','prior_portfolio_sha256')]:
    p=root/'.ai-progress/workstreams/quantum-projector-v2/refs/20261026-l3-exploration'/name
    assert sha(p.read_bytes())==next_[key]
for x in (next_,history):assert x['goal']=='ACTIVE' and x['newL3']==x['newL4T']=='NOT_ACCEPTED'
assert history['gen98_thirteen_gate_stop_preserved'] and history['all_positive_negative_history_preserved']
evidence=('scope.json','discovery-profile.json','next-source-discovery.json','discovery-summary.json','source-receipt.json','reading-record.json',
          'source-claim-ledger.json','source-table12.json','history-comparison.json','derivation.md','analysis.md','next-direction.json')
out=dict(status='PASS_MECHANICAL_SOURCE_AND_DISCOVERY_PACKAGE',verified_at=datetime.now(timezone.utc).isoformat(),
    fixed_query_counts=counts,fixed_query_total_results=totals,source_pdf_identity_attested_and_hash_verified=True,
    source_receipt_reading_verified=True,read_page_count=len(read['read_pages']),visual_pages=read['visual_pages'],
    table12_displayed_row_checks=rounding,table12_equal_datasets=ties,table12_worse_count=worse,
    limitations=['Mechanical checks do not certify human reading, mathematical novelty or independent scientific review.',
      'No author-code or numerical-result reproduction; exact ideal product identity is known algebra.',
      'Complete disclosed query responses are not exhaustive coverage of literature.'],
    task_rows=0,trained_head_forwards=0,synthetic_circuit_executions=0,new_updates=0,new_gpu_allocations=0,
    server_writes=0,COS_writes=0,model_API_calls=0,new_method_selected=False,new_theory=False,
    external_scientific_review=False,independent_task_confirmation=False,native_source_and_old_L4_hash_preserved=True,
    evidence_sha256={n:sha((r/n).read_bytes()) for n in evidence},goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],query_counts=counts,read_pages=out['read_page_count'],
                     table12_mean_checks=rounding,table12_ties=ties,goal='ACTIVE')))
