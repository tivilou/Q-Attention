"""Aggregate both fixed readers without changing labels, sample or denominators."""
from pathlib import Path
import json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code=r'''
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
r=Path('/hy-tmp/Q-Attention');root=r/'control/train-binding-semantics-20261123'
cfg=json.loads((root/'design.json').read_bytes());selection=json.loads((root/'selection.private.json').read_bytes());pub=json.loads((root/'selection.json').read_bytes())
assert len(selection)==32 and pub['selected']==32
reads=[json.loads((root/(name+'.private.json')).read_bytes()) for name in ('reader-a','reader-b')]
assert all(v['selection_sha256']==pub['selection_sha256'] for v in reads)
maps=[{q['case_id']:q for q in v['judgments']} for v in reads]
cases=[]
for row in selection:
 a,b=[m.get(row['case_id']) for m in maps];valid=a is not None and b is not None
 confident=valid and not a['uncertain'] and not b['uncertain'] and a['relation']!='uncertain' and b['relation']!='uncertain'
 cases.append(dict(**row,both_valid=valid,both_confident=bool(confident),
  both_competing_entity=bool(confident and a['competing_entity'] and b['competing_entity']),
  both_coreference=bool(confident and a['requires_coreference'] and b['requires_coreference']),
  both_gold=bool(confident and a['relation']==b['relation']==row['gold']),
  both_positive_gold=bool(confident and a['relation']==b['relation']==row['gold'] and row['gold']!='no_relation' and a['pair_link'] in ('explicit','entailed') and b['pair_link'] in ('explicit','entailed'))))
counts={}
for flag,name in ((True,'competing'),(False,'simple')):
 q=[v for v in cases if v['competing']==flag];assert len(q)==16
 counts[name]=dict(rows=16,both_valid=sum(v['both_valid'] for v in q),both_confident=sum(v['both_confident'] for v in q),both_competing_entity=sum(v['both_competing_entity'] for v in q),both_coreference=sum(v['both_coreference'] for v in q),both_gold=sum(v['both_gold'] for v in q))
positive=[v for v in cases if v['family']=='positive'];assert len(positive)==16
valid=sum(v['both_valid'] for v in cases);gold=sum(v['both_positive_gold'] for v in positive)
delta=(counts['competing']['both_competing_entity']-counts['simple']['both_competing_entity'])/16
gates=dict(both_valid=valid>=cfg['gates']['minimum_both_valid'],positive_gold_support=gold>=cfg['gates']['minimum_both_confident_gold_positive'],semantic_competing_proxy=delta>=cfg['gates']['minimum_competing_entity_consensus_rate_difference'])
modes=['relation_softmax','real_overlap','quantum','dephased'];model_counts={}
for name in modes:
 z=np.load(r/('control/full-model-error-structure-20261123/output/'+name+'-logits.private.npy'),mmap_mode='r',allow_pickle=False);y=np.load(r/'inputs/native-baseline/discovery/cache/train-targets.npy',mmap_mode='r',allow_pickle=False)
 pp=z.argmax(1);values={}
 for family in ('positive','null'):
  for flag,arm in ((True,'competing'),(False,'simple')):
   ids=[v['row'] for v in cases if v['family']==family and v['competing']==flag];assert len(ids)==8
   values[family+'_'+arm]=dict(rows=8,correct=int((pp[ids]==y[ids]).sum()),positive_false_negative=int(((y[ids]>0)&(pp[ids]==0)).sum()),null_false_positive=int(((y[ids]==0)&(pp[ids]>0)).sum()))
 model_counts[name]=values
private=root/'semantic-scored.private.json';assert not private.exists();private.write_bytes((json.dumps(cases,indent=2)+'\n').encode())
summary=dict(selected=32,positive_cases=16,both_valid=valid,both_confident_positive_gold=gold,counts=counts,competing_entity_consensus_rate_difference=delta,gates=gates,
 decision='PROXY_QUALIFIED_FOR_MATCHED_INTERVENTION_DESIGN' if all(gates.values()) else 'DO_NOT_PROMOTE_PROXY_AS_SEMANTIC_BINDING_MECHANISM',
 model_selected_counts=model_counts,source_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('design.json','selection.private.json','judge-packet.private.json','reader-a.private.json','reader-b.private.json','semantic-scored.private.json')},
 model_forwards=0,optimizer_updates=0,valid_rows=0,test_rows=0,cos_writes=0,
 scope='Model-blind gold-stratified matched TRAIN32; same-model isolated readers, not independent humans/causality/population prevalence/deployable classifier or L5.')
(root/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode());print(json.dumps(summary))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=35))
(r/'summary.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
