from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time,urllib.request,urllib.error
r=Path(__file__).resolve().parent
assert not (r/'preflight.json').exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
attempts=[];metadata=None
op=urllib.request.build_opener()
def fetch(url,limit=4000000):
 a=dict(url=url,method='GET',recorded_at=datetime.now(timezone.utc).isoformat(),access_mode='public_http',cache='miss')
 try:
  with op.open(urllib.request.Request(url,headers={'User-Agent':'ResearchEvidenceReader/1.0'}),timeout=25) as f:
   b=f.read(limit+1);a.update(status=f.status,bytes=len(b),sha256=sha(b),final_url=f.url)
  assert len(b)<=limit
  attempts.append(a);return json.loads(b)
 except Exception as e:
  a.update(status=getattr(e,'code',None),classification='unavailable',error_type=type(e).__name__)
  attempts.append(a);return None
commit=fetch('https://api.github.com/repos/liyupi/ai-model-world/commits/main')
if commit and 'sha' in commit:
 revision=commit['sha'];time.sleep(2)
 models=fetch('https://raw.githubusercontent.com/liyupi/ai-model-world/'+revision+'/data/models.json')
 if models:
  # Keep only exact candidate metadata, never benchmark hints as a capability ranking.
  rows=models if isinstance(models,list) else models.get('models',[])
  metadata=dict(revision=revision,matches=[x for x in rows if any(n in str(x.get('id','')).lower() for n in ['gpt-6.2-astra','gpt-6.2-sol'])])
direct=urllib.request.build_opener(urllib.request.ProxyHandler({}))
auth={'Authorization':'Bearer '+os.environ['A6API_API_KEY'],'Content-Type':'application/json'}
with direct.open(urllib.request.Request('https://api.a6api.vip/v1/models',headers=auth),timeout=35) as f:catalog=json.load(f)
ids={x['id'] for x in catalog['data']}
assert 'gpt-6.2-astra' in ids
out=dict(status='EXACT_MODEL_PRESENT',model='gpt-6.2-astra',protocol='openai-responses',effort='high',
 metadata_channel='ai-model-world first; optional derived aggregator, no benchmark ranking',metadata=metadata,metadata_attempts=attempts,
 exact_catalog_candidates=[n for n in ['gpt-6.2-astra','gpt-6.2-sol','claude-fable-5-2'] if n in ids],
 selection_basis='Reused explicit owner-selected mechanism/code review profile; no benchmark-superiority claim.',
 limitations=['Catalog presence does not verify inference echo; dispatch must check model and effort.','No capability ranking inferred from catalog order or derived hints.'],
 recorded_at=datetime.now(timezone.utc).isoformat())
(r/'preflight.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],model=out['model'],metadata_attempts=len(attempts),metadata_matches=len((metadata or {}).get('matches',[])),catalog_candidates=out['exact_catalog_candidates'])))
