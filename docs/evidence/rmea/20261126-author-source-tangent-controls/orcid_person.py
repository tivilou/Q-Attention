from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'E:/Claude.Code/本地助手/ai-skills/workflow/research-network-collection/scripts')
from network_policy import PacedHttpClient,FetchPolicy
scope=json.loads((r/'scope.json').read_bytes())
assert not (r/'orcid-person-result.json').exists()
url=scope['first_request']['url']
client=PacedHttpClient(FetchPolicy(min_interval=1.5,max_retries=0,max_browser_escalations=0,cache_dir=r/'http-cache'))
q=client.fetch(url,headers={'Accept':'application/json'},timeout=25)
record=dict(method='GET',url=url,final_url=q.final_url,retrieved_at=q.retrieved_at,status=q.status,
            http_status=q.status_code,response_class=q.response_class,access_mode=q.access_mode,
            cache_state=q.cache_status,retries=q.retry_count,failure_reason=q.failure_reason)
(r/'request-ledger.json').write_bytes((json.dumps([record],ensure_ascii=False,indent=2)+'\n').encode())
raw=q.body.encode();p=r/'orcid-person.private.response';p.write_bytes(raw)
out=dict(status='UNRESOLVED_ORCID_CHANNEL',requests_used=1,transport=record,
         raw_response_sha256=hashlib.sha256(raw).hexdigest(),identity_verified=False,
         public_name=None,public_researcher_urls=[],fulltext_acquired=False,fulltext_read=False)
if q.ok:
    try:
        x=json.loads(q.body)
        name=x.get('name') or {}
        given=(name.get('given-names') or {}).get('value','')
        family=(name.get('family-name') or {}).get('value','')
        credit=(name.get('credit-name') or {}).get('value','')
        matched=(family.casefold().strip()=='knyazev' and given.casefold().strip().split()[0]=='andrew') if given else False
        out.update(public_name=dict(given=given,family=family,credit=credit),identity_verified=matched,
                   status='MATCHED_PUBLIC_ORCID_PERSON' if matched else 'IDENTITY_UNRESOLVED')
        if matched:
            for item in (x.get('researcher-urls') or {}).get('researcher-url',[]):
                u=(item.get('url') or {}).get('value','')
                if u.startswith(('http://','https://')):
                    out['public_researcher_urls'].append(dict(label=item.get('url-name'),url=u,visibility=item.get('visibility')))
    except (ValueError,TypeError,KeyError) as e:
        out['semantic_error']=type(e).__name__
(r/'orcid-person-result.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
