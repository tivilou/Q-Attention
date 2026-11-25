from pathlib import Path
import hashlib
import json
import pymupdf

r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
s=read('source-receipts.json');reading=read('reading-record.json');c=read('comparison.json');n=read('next-direction.json');t=read('reported-tables.json')
assert s['status']=='PARTIAL_ORIGINAL_SOURCE_AUDIT'
assert s['knyazev']['fulltext']=='UNRESOLVED' and not s['knyazev']['algorithm_claims_from_this_paper']
assert reading['knyazev']['fulltext_pages_read']==[]
for kind,dir_,num in [('conference','fulltexts/8fcb56c452e49c5e',9),('arxiv','arxiv-fulltexts/61034008b4c371d1',11)]:
    x=read(dir_+'/receipt.json');pdf=Path(x['path']).read_bytes();parse=Path(x['parsed_path']).read_bytes()
    assert pdf.startswith(b'%PDF-') and sha(pdf)==s['fastweights'][kind]['sha256']==x['sha256']
    assert sha(parse)==x['parse_sha256'] and x['identity']=='verified'
    assert x['reading_record_sha256']==sha((r/'reading-record.json').read_bytes())
    d=pymupdf.open(stream=pdf,filetype='pdf');assert len(d)==num
    text=d[0].get_text(sort=True)
    assert 'Using Fast Weights to Attend to the Recent Past' in text
    assert all(author in text for author in ['Jimmy Ba','Geoffrey Hinton','Volodymyr Mnih','Catalin Ionescu'])
    if kind=='arxiv':assert '1610.06258v1' in text and '20 Oct 2016' in text
    parsed=json.loads(parse);assert len(parsed['pages'])==num and not x['low_text_pages']
    assert x['reading']==('read_for_scope' if kind=='conference' else 'partially_read_for_scope')
    q=read(dir_.split('/')[0]+'-identity-execution.json');assert q['exit_status']==0 and not q['stderr']
for a in s['knyazev']['attempts']:
    assert sha((r/a['receipt']).read_bytes())==a['receipt_sha256']
    assert a['acquisition']=='unresolved' and a['reading']=='not_started'
sp=read('supplement-acquisition.json');b=Path(sp['PDFs'][0]['path']).read_bytes()
assert sha(b)==s['official_supplement']['PDF_sha256']
assert sha(Path(sp['path']).read_bytes())==s['official_supplement']['archive_sha256']
d=pymupdf.open(stream=b,filetype='pdf');assert len(d)==23
p=d[0].get_text(sort=True)
assert 'Upper bounds for non-smooth sums' in p and 'Theorem 9' in p and 'Theorem 10' in p
assert not s['official_supplement']['scientific_use'] and reading['mismatched_supplement']['text_and_visual_pages_inspected']==[1]
for f,h in reading['visual_sha256'].items():assert sha((r/'visual'/f).read_bytes())==h
assert t['table1']['columns_recurrent_units']==[20,50,100]
assert t['table1']['FastWeights']==[1.81,0,0] and t['table1']['LSTM'][-1]==0
assert t['table2']['FastWeights'][0]>t['table2']['ConvNet'][0] and t['table2']['FastWeights'][1]>t['table2']['ConvNet'][1]
assert t['table2']['FastWeights'][2]<t['table2']['ConvNet'][2]
assert round(t['table3']['ConvNet']-t['table3']['FastWeights'],2)==1.89
assert c['status']=='HOLD_NO_NEIGHBOR_CLEARANCE' and not any([c['admission'][k] for k in ['method_selected','new_theory','task_diagnostic','training']])
assert c['admission']['newL3']==c['admission']['newL4T']=='NOT_ACCEPTED'
assert n['goal']=='ACTIVE' and n['minimum']['contexts']==2 and n['minimum']['device']=='CPU'
assert sha(Path(n['source']['path']).read_bytes())==n['source']['sha256']
assert all(v==0 for v in s['resources'].values())
assert s['collection']['ordinary_requests']==18 and s['collection']['public_browser_navigations']==1
assert read('browser-render-result.json')['status']=='ACCESS_CONTROL_STOPPED'
assert read('browser-render-result.json')['challenge_interactions']==0
assert read('proxy-exact-title-search-transport.json')['http_status']==200
assert s['collection']['Google_HTTP200_semantic_status']=='JAVASCRIPT_SHELL_NO_RESULTS'
rb=read('report-build.json');ex=read('report-build-execution.json')
assert ex['exit_status']==0 and ex['stderr']=='' and ex['code_sha256']==sha((r/'build_report.py').read_bytes())
assert all(sha((r/f).read_bytes())==h for f,h in rb['files'].items())
out=dict(status='PASS_PARTIAL_SOURCE_AUDIT_RECORDS',source_audit='PARTIAL',fastweights='READ_FOR_SCOPE',
         knyazev='FULLTEXT_UNRESOLVED',supplement='IDENTITY_MISMATCH_QUARANTINED',
         scientific_test_reruns=0,task_execution=0,method_selected=False,goal='ACTIVE',
         newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',
         verified_hashes=rb['files'],equations='researcher read plus own hand derivation, not automated theorem certification')
(r/'verification.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
