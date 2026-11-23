"""Fixed TRAIN error audit, with two isolated owner-selected semantic readers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
HOST='/hy-tmp/Q-Attention/control/semantic-problem-audit-20261123'
MODEL='gpt-6.1-sol'
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
    (ROOT/n).write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())

SELECT=r'''
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sys
import numpy as np
cfg=json.load(sys.stdin);r=Path('/hy-tmp/Q-Attention');out=Path(cfg['host'])
assert not out.exists(),'Read existing selection; do not resample'
pins=cfg['input_hashes']
for n,h in pins.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
words=json.loads((r/'inputs/native-word-metadata-20261123/train.private.json').read_bytes())
tm=json.loads((r/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:
 logits=z['logits'].copy();gold=z['labels'].copy();fit=z['calibration'].copy();groups=z['groups'].copy()
assert logits.shape==(8192,40) and len(words)==len(tm)==8192 and fit.sum()==4192 and (~fit).sum()==4000
assert not set(groups[fit])&set(groups[~fit])
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):
 g,st,ot=tm[i];assert g==int(gold[i]);support[st,ot].add(g);seen.add(g)
ss=hashlib.sha256(json.dumps(sorted((list(k),sorted(v)) for k,v in support.items()),separators=(',',':')).encode()).hexdigest()
assert ss==cfg['frozen_support_sha256']
allow=np.ones_like(logits,dtype=bool)
for i,(_,st,ot) in enumerate(tm):
 if (st,ot) in support:
  allow[i]=False;allow[i,0]=True
  for k in support[st,ot]|(set(range(40))-seen):allow[i,k]=True
pred=np.where(allow,logits,-np.inf).argmax(1)
population=defaultdict(list);excluded=Counter()
for i in np.flatnonzero(~fit):
 g=int(gold[i]);p=int(pred[i]);w=words[i]
 assert tm[i][0]==g and 0<=w['subject'][0]<w['subject'][1]<=len(w['tokens']) and 0<=w['object'][0]<w['object'][1]<=len(w['tokens'])
 if not allow[i,g]:excluded['gold_excluded_by_rule']+=1;continue
 if g>0 and p==0:family='positive_missed'
 elif g>0 and p==g:family='positive_correct'
 elif g==0 and p>0:family='null_false_positive'
 elif g==0 and p==0:family='null_correct'
 else:excluded['wrong_positive_relation']+=1;continue
 key=hashlib.sha256(json.dumps([w['tokens'],w['subject'],w['object']],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 order=hashlib.sha256(('qattn-semantic-problem-v1:'+key).encode()).hexdigest()
 sentence=hashlib.sha256(json.dumps(w['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 population[family].append((order,key,sentence,int(i)))
chosen=[];sentences=set()
for family in cfg['stratum_order']:
 candidates=sorted(population[family]);take=[]
 for order,key,sentence,i in candidates:
  if sentence in sentences:continue
  take.append(dict(family=family,input_sha256=key,sentence_sha256=sentence,row=i,case_id=hashlib.sha256(('case:'+key).encode()).hexdigest()[:24]));sentences.add(sentence)
  if len(take)==cfg['per_stratum']:break
 assert len(take)==cfg['per_stratum'],family
 chosen.extend(take)
assert len(sentences)==len(chosen)==32
seal=hashlib.sha256(json.dumps(chosen,sort_keys=True,separators=(',',':')).encode()).hexdigest()
packet=[];mapping=[]
for rec in chosen:
 i=rec['row'];w=words[i];g,st,ot=tm[i]
 packet.append(dict(case_id=rec['case_id'],tokens=w['tokens'],subject=w['subject'],object=w['object'],subject_type=st,object_type=ot))
 mapping.append(dict(**rec,gold=labels[g],predicted=labels[int(pred[i])],gold_allowed=True))
out.mkdir(parents=True)
for n,v in [('selection.private.json',mapping),('judge-packet.private.json',dict(inventory=labels,items=packet,selection_sha256=seal)),('design.json',cfg)]:
 (out/n).write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
report=dict(selection_sha256=seal,selected=32,per_stratum=cfg['per_stratum'],distinct_sentences=len(sentences),
 population={k:len(population[k]) for k in cfg['stratum_order']},excluded=dict(excluded),assessment_rows=4000,
 scope='Fixed exposed TRAIN assessment, outcome-stratified hash sample. No representative prevalence, heldout, human-independence or causal claim.',
 valid_test_used=False,model_forwards=0,optimizer_updates=0,cos_writes=0)
(out/'selection.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report))
'''

def prepare():
    assert not (ROOT/'selection.json').exists(),'Reuse existing selection'
    cfg=dict(profile='TRAIN-semantic-error-strata-32-v1',host=HOST,per_stratum=8,
        stratum_order=['positive_missed','positive_correct','null_false_positive','null_correct'],
        frozen_support_sha256='d4c1100df6d306a3fae48d66f7907dfbe2a39d1b30843abb359ba32de7053186',
        input_hashes={
          'inputs/native-word-metadata-20261123/train.private.json':'f5eafb7d51bca46988313c204b6fa5ef19e82fabc32dc9feee9faa70f2285053',
          'control/task-bottleneck-discovery-20261123/cases.private.json':'ad35efee20682e4f42280670de21bfa76a31dcce39d06cfba675348229a93e8c',
          'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz':'ff920146542e7af85f90df5ececfa62f0742be4e90f0db4de8645946e043d833',
          'inputs/native-baseline/recovery-inputs/labels.json':'6e822509a07c5c865130ebe315122a9d83e97b1b11496b9e86c6b2a600f8a429'},
        selection='Exclude rule-excluded golds and wrong-positive-label errors from four prespecified strata; report counts. Sort within strata by SHA256(qattn-semantic-problem-v1:inputhash), take8, one exact sentence globally in stated stratum order. No reselection.',
        annotation='Two isolated gpt-6.1-sol/medium API contexts. Gold/prediction/stratum and other reader hidden. Reader-specific anonymous IDs and order. Use sentence plus given types; cite token indices, preserve uncertainty and invalid/missing rows as unknown. Same-model readers not independent experts.',
        purpose='Distinguish explicit missed relation evidence from insufficient/ambiguous input. Compare structural features against matched outcome strata; no trained method, baseline tuning, quantum attribution or L5 acceptance.',
        stop='No outcome rescue, prompt revision or model redispatch after valid output. If no repeated unambiguous missed structure with counterexamples, do not choose a quantum operation from this audit.',
        execution='R0 read-only cached-array selection and external text annotation. No model/GPU training, formal experiment, TEST, COS or stable-core/lifecycle change.',
        raw_storage='Raw32sentences and per-case mappings/answers remain private on existing compute node. Windows holds design and compact aggregate receipts only. API inputs relayed only in RAM.',
        api_model=MODEL,api_effort='medium',selection_basis='owner_preference',code_sha256=sha(Path(__file__).read_bytes()))
    save('design.json',cfg)
    out=json.loads(ops.remote([ops.PY,'-B','-c',SELECT],data=json.dumps(cfg).encode(),timeout=35));save('selection.json',out);print(json.dumps(out))

def request(path,body=None):
    q=urllib.request.Request('https://api.a6api.vip/v1/'+path,data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['A6API_API_KEY'],'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(q,timeout=420) as response:return json.load(response)

def read_packet():
    code="from pathlib import Path;print(Path("+repr(HOST+'/judge-packet.private.json')+").read_text())"
    packet=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=30));assert len(packet['items'])==32
    assert packet['selection_sha256']==json.loads((ROOT/'selection.json').read_bytes())['selection_sha256']
    return packet

def judge(name,packet):
    assert not (ROOT/(name+'-receipt.json')).exists(),'No duplicate reader dispatch'
    mapping={};items=[]
    for rec in packet['items']:
        identity=sha((name+':'+rec['case_id']).encode())[:20];mapping[identity]=dict(case_id=rec['case_id'],length=len(rec['tokens']))
        items.append({k:v for k,v in rec.items() if k!='case_id'}|dict(item_id=identity))
    items.sort(key=lambda v:v['item_id'])
    task=dict(task='Judge the relation of the marked SUBJECT to marked OBJECT using only this sentence and given entity types. Gold labels, model outputs, outcome strata, and other reader are hidden. No tools. Do not guess from co-occurrence/types/world knowledge. Span endpoints are zero-based HALF-OPEN.',
        instructions='For EACH item choose exactly one inventory label, or uncertain if label semantics/text do not suffice. Use no_relation when the sentence does not assert a supported relation for this particular entity pair. Citation indices must refer to actual predicate/attribute/attachment evidence in this sentence, not entity co-occurrence. Positive confident labels require nonempty evidence_indices. Mark uncertainty for ambiguous reference, indistinct relation inventory, or insufficient context. Never infer gold labels from item order.',
        structures='evidence_form: direct_predicate, apposition, attribute, composed_clauses, coreference, cooccurrence, or none. pair_link: explicit, entailed, missing, or uncertain. competing_entity means another entity could be confused with the marked argument. scope_modifier: none, negated, hypothetical, past_only, mixed, or unclear. Report observable structure, not a theory or model diagnosis. Do not quote sentence text.',
        inventory=packet['inventory'],items=items,
        output='Return ONLY JSON {"judgments":[{"item_id":"...","relation":"inventory_label_or_uncertain","uncertain":false,"evidence_indices":[1,2],"evidence_form":"direct_predicate","pair_link":"explicit","requires_coreference":false,"competing_entity":false,"scope_modifier":"none"}]}. Exactly32 judgments, one per input. No prose or additional keys.')
    encoded=json.dumps(task,ensure_ascii=False).encode();start=time.monotonic()
    raw=request('responses',dict(model=MODEL,reasoning=dict(effort='medium'),max_output_tokens=12000,input=encoded.decode()))
    seconds=time.monotonic()-start
    # Preserve the raw answer on the compute node before parsing, without local raw-data files.
    code='''from pathlib import Path
import sys,json
v=json.load(sys.stdin);r=Path(v['host']);p=r/(v['reader']+'-raw.private.json');assert not p.exists();p.write_bytes((json.dumps(v['raw'],ensure_ascii=False,indent=2)+'\\n').encode());print(json.dumps(dict(saved=True)))
'''
    assert json.loads(ops.remote([ops.PY,'-B','-c',code],data=json.dumps(dict(host=HOST,reader=name,raw=raw),ensure_ascii=False).encode(),timeout=35))['saved']
    profile=raw.get('model')==MODEL and raw.get('reasoning',{}).get('effort')=='medium'
    strings=[''.join(c.get('text','') for c in row.get('content',[]) if c.get('type')=='output_text').strip() for row in raw.get('output',[])]
    if raw.get('output_text') and not strings:strings=[raw['output_text']]
    answers=[];parse_errors=[]
    for text in strings:
        if not text:continue
        try:answers.append(json.loads(text))
        except (ValueError,TypeError):parse_errors.append('malformed_json')
    valid=[];invalid=[];seen=set();inventory=set(packet['inventory'])|{'uncertain'}
    candidates=answers[0].get('judgments',[]) if len(answers)==1 and isinstance(answers[0],dict) else []
    for item in candidates:
        identity=item.get('item_id');reason=[]
        if identity not in mapping or identity in seen:invalid.append(dict(item_id=identity,reason='unknown_or_duplicate_id'));continue
        seen.add(identity);m=mapping[identity]
        if item.get('relation') not in inventory:reason.append('invalid_label')
        if any(type(item.get(k))!=bool for k in ('uncertain','requires_coreference','competing_entity')):reason.append('invalid_bool')
        if item.get('evidence_form') not in ('direct_predicate','apposition','attribute','composed_clauses','coreference','cooccurrence','none'):reason.append('invalid_evidence_form')
        if item.get('pair_link') not in ('explicit','entailed','missing','uncertain'):reason.append('invalid_pair_link')
        if item.get('scope_modifier') not in ('none','negated','hypothetical','past_only','mixed','unclear'):reason.append('invalid_scope')
        indices=item.get('evidence_indices');good=isinstance(indices,list) and all(type(i)==int and 0<=i<m['length'] for i in indices)
        if not good:reason.append('invalid_citation')
        if not item.get('uncertain',True) and item.get('relation') not in ('no_relation','uncertain') and not indices:reason.append('missing_positive_citation')
        if not profile:reason.append('profile_mismatch')
        if reason:invalid.append(dict(item_id=identity,reason=reason));continue
        valid.append({k:v for k,v in item.items() if k!='item_id'}|dict(case_id=m['case_id']))
    result=dict(reader=name,model=MODEL,reasoning_effort='medium',profile_echo_verified=profile,seconds=seconds,packet_sha256=sha(encoded),selection_sha256=packet['selection_sha256'],
        response_status=raw.get('status'),usage=raw.get('usage'),labels_hidden=True,predictions_hidden=True,other_reader_hidden=True,
        judgments=valid,invalid=invalid,missing_ids=[mapping[k]['case_id'] for k in mapping if k not in seen],parse_errors=parse_errors)
    code='''from pathlib import Path
import sys,json
v=json.load(sys.stdin);p=Path(v['host'])/(v['reader']+'.private.json');assert not p.exists();p.write_bytes((json.dumps(v['result'],ensure_ascii=False,indent=2)+'\\n').encode());print(json.dumps(dict(saved=True)))
'''
    assert json.loads(ops.remote([ops.PY,'-B','-c',code],data=json.dumps(dict(host=HOST,reader=name,result=result),ensure_ascii=False).encode(),timeout=35))['saved']
    receipt={k:v for k,v in result.items() if k not in ('judgments','invalid','missing_ids')}
    receipt.update(valid_rows=len(valid),invalid_rows=len(invalid),missing_rows=len(result['missing_ids']))
    save(name+'-receipt.json',receipt);return receipt

def run():
    catalog=request('models');assert MODEL in {m['id'] for m in catalog['data']}
    save('model-selection.json',dict(selection_basis='owner_preference',model=MODEL,reasoning_effort='medium',catalog_verified=True,protocol='OpenAI Responses',two_isolated_contexts=True,independent_experts=False))
    packet=read_packet()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(judge,name,packet) for name in ('reader-a','reader-b')]
        for f in futures:print(json.dumps(f.result()),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
