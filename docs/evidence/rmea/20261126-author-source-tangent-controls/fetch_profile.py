from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,sys,urllib.parse
r=Path(__file__).resolve().parent
sys.path.insert(0,'E:/Claude.Code/本地助手/ai-skills/workflow/research-network-collection/scripts')
from network_policy import PacedHttpClient,FetchPolicy
author=json.loads((r/'orcid-person-result.json').read_bytes());assert author['identity_verified']
label=sys.argv[1];filename=sys.argv[2]
if label.startswith('https://'):
    url=label
    approved=json.loads((r/'approved-next-profile.json').read_bytes());assert approved['url']==url
else:
    hits=[x for x in author['public_researcher_urls'] if x['label']==label];assert len(hits)==1
    url=hits[0]['url']
ledger=json.loads((r/'request-ledger.json').read_bytes());assert len(ledger)<4
assert not (r/(filename+'-result.json')).exists()
assert urllib.parse.urlparse(url).hostname not in ['epubs.siam.org','scholar.google.com','www.google.com']
cfg=json.loads((r/'scope.json').read_bytes());assert url not in cfg['collection']['already_failed_targets']
client=PacedHttpClient(FetchPolicy(min_interval=1.5,max_retries=0,max_browser_escalations=0,cache_dir=r/'http-cache'))
q=client.fetch(url,timeout=25)
record=dict(method='GET',url=url,final_url=q.final_url,retrieved_at=q.retrieved_at,status=q.status,
            http_status=q.status_code,response_class=q.response_class,access_mode=q.access_mode,
            cache_state=q.cache_status,retries=q.retry_count,failure_reason=q.failure_reason)
ledger.append(record);(r/'request-ledger.json').write_bytes((json.dumps(ledger,ensure_ascii=False,indent=2)+'\n').encode())
body=q.body.encode();(r/(filename+'.private.html')).write_bytes(body)
class Links(HTMLParser):
    def __init__(self):super().__init__();self.href=None;self.text=[];self.links=[];self.alltext=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.href=dict(attrs).get('href');self.text=[]
    def handle_data(self,data):
        self.alltext.append(data)
        if self.href is not None:self.text.append(data)
    def handle_endtag(self,tag):
        if tag=='a' and self.href is not None:
            self.links.append(dict(label=' '.join(''.join(self.text).split()),url=urllib.parse.urljoin(q.final_url,self.href)));self.href=None
parser=Links()
if q.ok:parser.feed(q.body)
text=' '.join(' '.join(parser.alltext).split())
out=dict(status='PUBLIC_PROFILE_FETCHED' if q.ok else 'PROFILE_UNRESOLVED',transport=record,
         body_sha256=hashlib.sha256(body).hexdigest(),requests_used=len(ledger),
         exact_title_present='Toward the Optimal Preconditioned Eigensolver' in text,
         relevant_links=[x for x in parser.links if any(k in (x['label']+' '+x['url']).casefold() for k in
                       ['merl','knyazev','precondition','lobpcg','36612','s1064827500366124'])],
         fulltext_acquired=False,fulltext_read=False)
(r/(filename+'-links.private.json')).write_bytes((json.dumps(parser.links,ensure_ascii=False,indent=2)+'\n').encode())
(r/(filename+'-result.json')).write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
