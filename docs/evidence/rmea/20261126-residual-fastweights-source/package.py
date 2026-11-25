from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
v=read('verification.json');ex=read('verify-execution.json')
assert v['status']=='PASS_PARTIAL_SOURCE_AUDIT_RECORDS' and ex['exit_status']==0 and not ex['stderr']
assert ex['code_sha256']==sha((r/'verify.py').read_bytes()) and json.loads(ex['stdout'])==v
entries=[
 ('Crossref exact DOI','metadata-transport.json',0),('NIPS2016 official directory','metadata-transport.json',1),
 ('FastWeights official landing','landing-transport.json',None),
 ('Knyazev publisher PDF','fulltexts/03331f9f4bc03c3f/receipt.json','attempt'),
 ('FastWeights conference PDF','fulltexts/8fcb56c452e49c5e/receipt.json','attempt'),
 ('OpenAlex exact DOI','OA-metadata-transport.json',None),('SemanticScholar exact DOI','semantic-metadata-transport.json',None),
 ('Old indexed author PDF','author-copy-fulltexts/03331f9f4bc03c3f/receipt.json','attempt'),
 ('Unverified author-home candidate','author-home-transport.json',None),
 ('Institutional author-copy candidate','institutional-copy-fulltexts/03331f9f4bc03c3f/receipt.json','attempt'),
 ('Bing exact title RSS','exact-title-search-transport.json',None),
 ('Google exact title direct','second-exact-title-search-transport.json',None),
 ('Official supplemental ZIP','supplement-acquisition.json',None),
 ('Indexed CiteSeerX repository PDF','repository-copy-fulltexts/03331f9f4bc03c3f/receipt.json','attempt'),
 ('Google same title configured-proxy transport correction','proxy-exact-title-search-transport.json',None),
 ('Official arxiv exact ID API','fastweights-arxiv-request.json','arxiv_api'),
 ('Official arxiv fixed-v1 PDF','arxiv-fulltexts/61034008b4c371d1/receipt.json','attempt'),
 ('Institutional www author-copy candidate','institutional-www-copy-fulltexts/03331f9f4bc03c3f/receipt.json','attempt')]
ledger=[]
for i,(label,n,select) in enumerate(entries,1):
 x=read(n)
 if isinstance(select,int):x=x[select]
 elif select=='attempt':assert len(x['attempts'])==1;x=x['attempts'][0]
 elif select=='arxiv_api':x=x['papers'][0]['resolution_attempts'][0]
 ledger.append(dict(ordinal=i,operation=label,record=n,record_sha256=sha((r/n).read_bytes()),
                    status=x.get('status'),http_status=x.get('http_status')))
assert len(ledger)==18
collection=dict(ordinary_requests=ledger,ordinary_count=18,browser_navigation_count=1,
                browser_record='browser-render-result.json',
                semantic_overrides={'Bing200':'irrelevant dictionary results, no paper result',
                                    'Google200':'JS shell, no extracted result'},
                excluded='Raw search HTML, browser challenge URL tokens, profiles, caches and logs remain private.')
(r/'collection-ledger.json').write_bytes((json.dumps(collection,ensure_ascii=False,indent=2)+'\n').encode())
failure=read('verify-execution.before-date-check-fix.json')
assert failure['exit_status']==1 and 'AssertionError' in failure['stderr']
assert failure['code_sha256']==sha((r/'verify.before-date-check-fix.py').read_bytes())
correction=dict(failed_execution='verify-execution.before-date-check-fix.json',
                failed_code='verify.before-date-check-fix.py',
                reason='sort=True splits rotated PDF margin date; version/date correct in raw text order.',
                changed_check='Check arXiv version and date in get_text(sort=False) while retaining sorted identity checks.',
                final_execution='verify-execution.json',scientific_claims_changed=False,new_downloads=0)
(r/'verification-correction.json').write_bytes((json.dumps(correction,indent=2)+'\n').encode())
names=['scope.before-identity.json','acquisition-budget-v2.json','acquisition-budget-v3.json','browser-render-scope.json',
       'browser-render-result.json','source-receipts.json','reading-record.json','comparison.json','reported-tables.json',
       'next-direction.json','analysis.md','collection-ledger.json','verification-correction.json',
       'build_report.py','report-build.json','report-build-execution.json','finish_sources.py','finish_sources-execution.json',
       'fulltexts-identity-execution.json','arxiv-fulltexts-identity-execution.json',
       'fulltexts/8fcb56c452e49c5e/receipt.json','arxiv-fulltexts/61034008b4c371d1/receipt.json',
       'verify.py','verify-execution.json','verification.json',
       'verify.before-date-check-fix.py','verify-execution.before-date-check-fix.json','execute.py']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<512*1024
assert not any(n.endswith(('.pdf','.zip','.html','.png','.pt','.npy','.npz')) or '.private.' in n for n in raw)
out=dict(status='PASS_PARTIAL_SOURCE_EVIDENCE_PACKAGE',public_evidence_sha256={n:sha(b) for n,b in raw.items()},
         source_audit='PARTIAL',task_execution=0,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],public_files=len(raw),public_bytes=sum(map(len,raw.values())))))
