from pathlib import Path
from fractions import Fraction
import hashlib,json,urllib.parse
import numpy as np
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
scope=read('scope.json');person=read('orcid-person-result.json');source=read('source-summary.json');ledger=read('request-ledger.json')
assert len(ledger)==3<=scope['collection']['max_ordinary_requests']
assert [urllib.parse.urlparse(x['url']).hostname for x in ledger]==['pub.orcid.org','en.wikipedia.org','arxiv.org']
assert all(x['method']=='GET' and x['retries']==0 and x['access_mode']=='http' for x in ledger)
assert person['identity_verified'] and person['public_name']['given']=='Andrew' and person['public_name']['family']=='Knyazev'
assert person['transport']['http_status']==200 and sha((r/'orcid-person.private.response').read_bytes())==person['raw_response_sha256']
raw=json.loads((r/'orcid-person.private.response').read_bytes())
assert raw['name']['given-names']['value']=='Andrew' and raw['name']['family-name']['value']=='Knyazev'
arxiv_urls=[x['url'] for x in person['public_researcher_urls'] if x['label']=='arXiv']
assert arxiv_urls==['https://arxiv.org/a/knyazev_a_1'] and ledger[2]['url']==arxiv_urls[0]
assert ledger[1]['status']=='transient_error' and 'handshake' in ledger[1]['failure_reason'].casefold()
assert ledger[2]['http_status']==200
idx=read('index-inspection.json');assert sha((r/'arxiv-author.private.html').read_bytes())==idx['source_sha256']
assert idx['title_prefix_casefold_hits']==idx['exact_DOI_hits']==0
assert source['status']=='ORIGINAL_FULLTEXT_UNRESOLVED_AUTHOR_NAME_MATCHED' and not source['fulltext_read']
assert source['algorithm_claims_from_Knyazev']==[] and source['PDF_transfers']==0
assert not list(r.glob('*.pdf'))
assert not any('epubs.siam.org' in x['url'] or 'google' in x['url'] for x in ledger)
m=read('tangent/results.json');ms=read('tangent/scope.json');receipt=read('tangent/receipt.before-execution.json')
ex=read('tangent/decompose-execution.json');trace=read('tangent/trace.json')
assert ex['exit_status']==0 and not ex['stderr'] and json.loads(ex['stdout'])=={k:v for k,v in m.items() if k not in ['rows','exact_fraction']}
assert ex['code_sha256']==receipt['code_sha256']==sha((r/'tangent/decompose.py').read_bytes())
assert sha((r/'tangent/derivation.md').read_bytes())==receipt['derivation_sha256']
assert sha((r/'tangent/scope.json').read_bytes())==receipt['scope_sha256']==m['scope_sha256']
assert sha((r/'tangent/trace.json').read_bytes())==m['trace_sha256']
for p in ms['source_pins']:assert sha(Path(p['path']).read_bytes())==p['sha256']
assert len(m['rows'])==len(trace['rows'])==len(ms['cases'])==10 and len(m['exact_fraction'])==2
assert all(m['gates'].values()) and m['max_identity_error']<1e-12
by={x['id']:x for x in m['rows']}
for x in trace['rows']:
    row=by[x['id']];assert x['summary']==row and row['PASS']
    v=np.array(x['v']['real'])+1j*np.array(x['v']['imag']);delta=np.array(x['delta']['values'])
    residual=np.array(x['residual']['real'])+1j*np.array(x['residual']['imag'])
    assert np.max(abs(delta@v-residual))<=1e-12
    assert abs(row['g']-2*np.vdot(residual,residual).real/x['D'])<=1e-12
    assert row['g']<=row['upper_bound']+1e-12
for row in m['exact_fraction']:
    n=by[row['id']]
    assert Fraction(row['g'])<=Fraction(row['upper'])
    for key in ['g','g_self','g_cross','Gamma','eta']:assert abs(float(Fraction(row[key]))-n[key])<=1e-12
assert by['retained-native-A']['g_cross']<0<by['retained-native-B']['g_cross']
assert abs(by['retained-native-A']['eta'])<=1e-12 and abs(by['retained-native-B']['eta']-1)<=1e-12
assert by['zero-overlap']['eta'] is None and by['unit-overlap']['eta'] is None
assert all(v==0 for v in m['resources'].values())
next_=read('next-direction.json');assert not next_['source_contract_verified'] and not next_['fresh_server_availability_verified']
for p in next_['local_contract_pins']:assert sha(Path(p['path']).read_bytes())==p['sha256']
assert all(v==0 for v in next_['resource_boundary'].values())
b=read('closeout-build.json');assert b['analysis_sha256']==sha((r/'analysis.md').read_bytes()) and b['next_direction_sha256']==sha((r/'next-direction.json').read_bytes())
out=dict(status='PASS_AUTHOR_CHANNEL_AND_TANGENT_RECORDS',original_source='UNRESOLVED_NOT_READ',
         requests=3,PDFs=0,numeric_algebra_rows=10,exact_moment_certificates=2,
         max_identity_error=m['max_identity_error'],native_reruns=0,task_rows=0,GPU_allocations=0,
         label_alignment=False,new_theory=False,task_method_admitted=False,
         source_summary_sha256=sha((r/'source-summary.json').read_bytes()),tangent_trace_sha256=m['trace_sha256'],
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
