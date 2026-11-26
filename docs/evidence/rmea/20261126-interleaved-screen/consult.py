from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time,urllib.request
r=Path(__file__).resolve().parent
assert not (r/'response.private.json').exists() and not (r/'dispatch.json').exists(),'Inspect prior dispatch, never resend blindly'
selection=json.loads((r/'selection.json').read_bytes());payload=(r/'review-packet.private.json').read_bytes()
assert selection['selected_model_id']=='gpt-6.2-astra' and selection['reasoning_effort']=='high'
assert hashlib.sha256(payload).hexdigest()==selection['packet_sha256']
body=dict(model='gpt-6.2-astra',reasoning=dict(effort='high'),max_output_tokens=6000,input=payload.decode())
headers={'Authorization':'Bearer '+os.environ['A6API_API_KEY'],'Content-Type':'application/json'}
dispatch=dict(model=body['model'],effort='high',protocol='openai-responses',max_output_tokens=6000,attempt=1,retries=0,
 packet_sha256=selection['packet_sha256'],started_at=datetime.now(timezone.utc).isoformat())
(r/'dispatch.json').write_bytes((json.dumps(dispatch,indent=2)+'\n').encode())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));started=time.time()
with opener.open(urllib.request.Request('https://api.a6api.vip/v1/responses',data=json.dumps(body).encode(),headers=headers),timeout=360) as f:raw=json.load(f)
(r/'response.private.json').write_bytes((json.dumps(raw,indent=2)+'\n').encode())
assert raw['model']==body['model'] and raw['reasoning']['effort']=='high'
parts=[c['text'] for x in raw.get('output',[]) for c in x.get('content',[]) if c.get('type')=='output_text']
answers=[]
for text in parts:
 try: answers.append(json.loads(text))
 except json.JSONDecodeError:pass
assert len(answers)==1,'Inspect same sealed response for parse repair; no redispatch'
answer=answers[0];assert answer['verdict'] in ['PASS_WITH_RISK','BLOCK']
out=dict(model=raw['model'],effort='high',elapsed_seconds=time.time()-started,usage=raw.get('usage'),
 packet_sha256=selection['packet_sha256'],response=answer,review_scope='mechanism',review_mode='external',
 raw_sha256=hashlib.sha256((r/'response.private.json').read_bytes()).hexdigest())
(r/'review.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(model=out['model'],effort=out['effort'],elapsed_seconds=out['elapsed_seconds'],verdict=answer['verdict'],substantive_candidate=answer.get('substantive_candidate'),usage=out['usage'])))
