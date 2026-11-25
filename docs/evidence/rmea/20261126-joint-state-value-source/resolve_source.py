from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, urllib.request, urllib.error, re

r = Path(__file__).resolve().parent
url = 'https://arxiv.org/abs/2205.05625'
request = dict(title='Quantum Self-Attention Neural Networks for Text Classification',
 proposed_arxiv_id='2205.05625', id_status='UNVERIFIED_BEFORE_REQUEST',
 purpose='Check quantum value representation and joint training before any novelty claim',
 budget=dict(metadata_requests=1, pdf_requests=1, retries=0, browser_escalations=0),
 allowed_origins=['https://arxiv.org'], freshness='frozen published version identity',
 resources=dict(GPU=0, task_forward=0, fit=0, new_capacity=0))
(r/'source-request.json').write_bytes((json.dumps(request,indent=2)+'\n').encode())
receipt=dict(url=url,method='GET',access_mode='public_http',cache='miss',
 recorded_at=datetime.now(timezone.utc).isoformat())
try:
 req=urllib.request.Request(url,headers={'User-Agent':'ResearchEvidenceReader/1.0'})
 with urllib.request.urlopen(req,timeout=25) as response:
  raw=response.read(1500000); receipt.update(status=response.status,final_url=response.url,
    bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 text=raw.decode('utf8','replace')
 title=re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"',text)
 receipt['observed_title']=title.group(1) if title else None
 receipt['classification']='metadata' if title else 'unresolved_content'
 receipt['identity_match']=bool(title and title.group(1)==request['title'])
 (r/'source.private.html').write_bytes(raw)
 if receipt['identity_match']:
  authors=re.findall(r'<meta\s+name="citation_author"\s+content="([^"]+)"',text)
  receipt['authors']=authors
  print(json.dumps(dict(title=receipt['observed_title'],authors=authors,identity_match=True)))
 else:
  print(json.dumps(dict(title=receipt['observed_title'],identity_match=False)))
except Exception as e:
 receipt.update(status=getattr(e,'code',None),classification='transport_failure',error=str(e))
 print(json.dumps(dict(classification=receipt['classification'],error=str(e))))
(r/'source-resolution.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
