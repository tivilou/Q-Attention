"""Matched TRAIN inputs, model-blind selection, accepted semantic reader unchanged."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,importlib.util,json,sys
r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('accepted_semantic_reader',r.parent/'semantic-problem-audit-20261123/audit.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
reader.ROOT=r;reader.HOST='/hy-tmp/Q-Attention/control/train-binding-semantics-20261123'

SELECT=r'''
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys
import numpy as np
cfg=json.load(sys.stdin);r=Path('/hy-tmp/Q-Attention');out=Path(cfg['host']);assert not out.exists()
sha=lambda x:hashlib.sha256(x).hexdigest()
p=r/'control/train-binding-census-aligned-20261123/output/census.private.npz'
assert sha(p.read_bytes())==cfg['census_sha256']
with np.load(p,allow_pickle=False) as a:
 # Predictions are deliberately not read until the selection has been sealed.
 order=a['source_order'];y=a['labels'];flag=a['competing'];bins=a['length_bin'];sent=a['sentence_id']
raw=[json.loads(s) for s in (r/'inputs/native-baseline/recovery-inputs/train.jsonl').read_bytes().splitlines() if s.strip()]
rows=[raw[i] for i in order];labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
ident=[sha(str(w['metadata']['id']).encode()) for w in rows]
groups=defaultdict(lambda:[[],[]])
for i,w in enumerate(rows):
 key=('positive',int(y[i]),int(bins[i])) if y[i]>0 else ('null',w['metadata']['subj_type'],w['metadata']['obj_type'],int(bins[i]))
 groups[key][0 if flag[i] else 1].append(i)
groups={k:v for k,v in groups.items() if min(map(len,v))>=10}
chosen=[];used=set();pair_id=0
for family in ('positive','null'):
 candidates=[k for k in groups if k[0]==family]
 candidates.sort(key=lambda k:sha(json.dumps(['blind-binding-semantic-v1',k],separators=(',',':')).encode()))
 seen_labels=set();done=0
 for key in candidates:
  if family=='positive' and key[1] in seen_labels:continue
  selected=[]
  for arm,items in enumerate(groups[key]):
   items=sorted(items,key=lambda i:sha(('binding-case-v1:'+ident[i]).encode()))
   pick=next((i for i in items if sent[i] not in used and all(sent[i]!=sent[j] for j in selected)),None)
   if pick is None:break
   selected.append(pick)
  if len(selected)!=2:continue
  for arm,i in enumerate(selected):
   chosen.append(dict(row=int(i),case_id=sha(('binding-semantic:'+ident[i]).encode())[:24],pair_id=pair_id,family=family,competing=arm==0,gold=labels[int(y[i])],sentence_sha256=str(sent[i])))
   used.add(str(sent[i]))
  if family=='positive':seen_labels.add(key[1])
  pair_id+=1;done+=1
  if done==8:break
 assert done==8,family
assert len(chosen)==len(used)==32
seal=sha(json.dumps(chosen,sort_keys=True,separators=(',',':')).encode())
packet=[]
for rec in chosen:
 w=rows[rec['row']];packet.append(dict(case_id=rec['case_id'],tokens=w['tokens'],subject=w['subject'],object=w['object'],subject_type=w['metadata']['subj_type'],object_type=w['metadata']['obj_type']))
report=dict(selected=32,distinct_sentences=len(used),positive_pairs=8,null_pairs=8,positive_classes=len({q['gold'] for q in chosen if q['family']=='positive'}),
 selection_sha256=seal,model_blind=True,gold_stratified=True,reader_labels_hidden=True,
 scope='Exploratory matched TRAIN sample, not representative population, independent humans, utility or causality. No reselection of closed40pair profile.')
out.mkdir(parents=True)
for name,value in [('design.json',cfg),('selection.private.json',chosen),('judge-packet.private.json',dict(inventory=labels,items=packet,selection_sha256=seal)),('selection.json',report)]:
 (out/name).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(report))
'''
def prepare():
 assert not (r/'selection.json').exists()
 census=json.loads((r.parent/'train-binding-census-aligned-20261123/summary.json').read_bytes())
 assert all(census['gates'].values())
 cfg=dict(profile='model-blind-matched-TRAIN-binding-semantic32-v1',host=reader.HOST,census_sha256=census['private_census_sha256'],
  selection='8goldpositive class/length-matched pairs from8distinct labels and8null type/length-matched pairs, input-ID hashes, globally distinct32sentences; no model output read by selector.',
  purpose='Qualify whether annotated competing-object proxy reflects semantic competing_entity rather than assume binding causality from corpus association.',
  reader='Reuse unchanged accepted semantic judge/parser;2isolated gpt-6.1-sol/medium contexts. Hidden gold,predictions,proxy,stratum,pairing and other answer. Same-model judgments are not independent humans.',
  gates=dict(minimum_both_valid=24,minimum_both_confident_gold_positive=12,minimum_competing_entity_consensus_rate_difference=.25),
  denominators='32all,16goldpositive,16competing and16simple. Missing/uncertain kept in denominator, not repaired or redispatched.',
  next_if_pass='Predeclare one actual consistently encoded matched intervention; no training/plugin or causal/quantum/L5 promotion from annotation.',
  if_fail='Do not promote annotated-object proxy as established semantic binding mechanism; retain whole-corpus association and original results.',
  recovery='Repeatable text audit, no new checkpoints/COS; raw32text/answers stay private compute, Windows compact receipts only.',
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reader_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())
 reader.save('design.json',cfg)
 v=json.loads(reader.ops.remote([reader.ops.PY,'-B','-c',SELECT],data=json.dumps(cfg).encode(),timeout=35));reader.save('selection.json',v);print(json.dumps(v))
def run():
 assert not (r/'reader-a-receipt.json').exists() and not (r/'reader-b-receipt.json').exists()
 catalog=reader.request('models');assert reader.MODEL in {q['id'] for q in catalog['data']}
 reader.save('model-selection.json',dict(selection_basis='owner_preference',model=reader.MODEL,reasoning_effort='medium',catalog_verified=True,protocol='Responses',two_isolated_contexts=True,independent_humans=False))
 packet=reader.read_packet()
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(reader.judge,name,packet) for name in ('reader-a','reader-b')]
  for f in fs:print(json.dumps(f.result()),flush=True)
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
