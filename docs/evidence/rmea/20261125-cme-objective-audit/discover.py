"""Two bounded official-Atom queries for task-dependent quantum inductive-bias leads."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

r = Path(__file__).resolve().parent
assert not (r/'next-source-discovery.json').exists(), 'Reuse the existing receipt; no repeated discovery.'
queries = __import__('json').loads((r/'discovery-profile.json').read_bytes())['queries']

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
out = []
for i, query in enumerate(queries):
    if i: time.sleep(3.2)
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(dict(
        search_query=query, start=0, max_results=6, sortBy='relevance', sortOrder='descending'))
    v = dict(query=query, url=url, requested_max_results=6,
             retrieved_at=datetime.now(timezone.utc).isoformat())
    start = time.monotonic()
    try:
        with opener.open(urllib.request.Request(url, headers={'User-Agent':'Q-Attention-source-audit/1.0'}), timeout=25) as response:
            data=response.read(2*1024*1024+1)
            assert len(data)<=2*1024*1024
            v.update(http_status=response.status, final_url=response.geturl())
        path=r/f'next-source-query-{i}.xml';path.write_bytes(data)
        root=ET.fromstring(data);ns={'a':'http://www.w3.org/2005/Atom'}
        v.update(status='retrieved',sha256=hashlib.sha256(data).hexdigest(),entries=[])
        for entry in root.findall('a:entry',ns):
            get=lambda name:' '.join((entry.findtext('a:'+name,default='',namespaces=ns)).split())
            v['entries'].append(dict(id=get('id'),title=get('title'),summary=get('summary'),
                published=get('published'),updated=get('updated'),
                authors=[e.findtext('a:name',namespaces=ns) for e in entry.findall('a:author',ns)]))
    except Exception as exc:
        v.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:240])
    v['elapsed_seconds']=time.monotonic()-start;out.append(v)
result=dict(schema='bounded-primary-source-discovery.v1',queries=out, fulltexts_read=0,
    limitations=['Two fixed relevance-ranked top-six queries, not exhaustive novelty coverage.',
        'Metadata/abstract leads only; not a selected mechanism or transfer proof.'])
(r/'next-source-discovery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
