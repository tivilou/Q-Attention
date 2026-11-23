"""Descriptive same-sentence label-contrast census of existing complete-head outputs."""
from pathlib import Path
import json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code=r'''
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import hashlib,json
import numpy as np
r=Path('/hy-tmp/Q-Attention');out=r/'control/train-binding-census-aligned-20261123/output';assert not (out/'paired-collapse.json').exists()
cfg=json.loads((out.parent/'design.json').read_bytes());summary=json.loads((out/'summary.json').read_bytes())
p=out/'census.private.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==summary['private_census_sha256']
with np.load(p,allow_pickle=False) as a:order=a['source_order'];y=a['labels'];pred=a['predictions'];sent=a['sentence_id']
original=[json.loads(s) for s in (r/'inputs/native-baseline/recovery-inputs/train.jsonl').read_bytes().splitlines() if s.strip()];rows=[original[i] for i in order]
groups=defaultdict(lambda:defaultdict(list))
for i,w in enumerate(rows):groups[(tuple(w['tokens']),tuple(w['subject']),w['metadata']['obj_type'])][tuple(w['object'])].append(i)
posnull=[];positive_positive=0;allpairs=0;ambiguous_spans=0
for objects in groups.values():
 stable={}
 for span,ids in objects.items():
  if len({int(y[i]) for i in ids})!=1:ambiguous_spans+=1;continue
  stable[span]=min(ids,key=lambda i:hashlib.sha256(str(rows[i]['metadata']['id']).encode()).hexdigest())
 for a,b in combinations(sorted(stable),2):
  if not(a[1]<=b[0] or b[1]<=a[0]):continue
  i,j=stable[a],stable[b];allpairs+=1
  if y[i]==y[j]:continue
  if y[i] and y[j]:positive_positive+=1
  else:posnull.append((i,j) if y[i] else (j,i))
pair=np.array(posnull,dtype=np.int64).reshape(-1,2);pos,null=pair.T;assert len(pair)>0
labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
modes=cfg['modes'];report={};all_null=[]
for m,mode in enumerate(modes):
 z=np.load(r/('control/full-model-error-structure-20261123/output/'+mode+'-logits.private.npy'),mmap_mode='r',allow_pickle=False)
 p,n=pred[m,pos],pred[m,null];g=y[pos]
 mp=z[pos,g].astype(float)-z[pos,0].astype(float);mn=z[null,g].astype(float)-z[null,0].astype(float)
 collapse=(p==0)&(n==0);all_null.append(collapse)
 report[mode]=dict(pairs=len(pair),both_correct=int(((p==g)&(n==0)).sum()),both_null=int(collapse.sum()),
  same_prediction=int((p==n).sum()),positive_correct_null_false_positive=int(((p==g)&(n!=0)).sum()),
  relation_margin_orders_pair_correctly=int((mp>mn).sum()),both_null_and_correct_margin_order=int((collapse&(mp>mn)).sum()),
  unique_positive_rows_both_null=len(set(map(int,pos[collapse]))),unique_sentences_both_null=len(set(map(str,sent[pos[collapse]]))))
shared=np.logical_and.reduce(all_null);ids=sorted(set(map(int,pos[shared])))
result=dict(all_disjoint_same_type_pairs=allpairs,positive_null_label_contrast_pairs=len(pair),positive_positive_label_contrast_pairs=positive_positive,
 unique_sentences=len(set(map(str,sent[pos]))),unique_positive_rows=len(set(map(int,pos))),positive_relation_classes=len(set(map(int,y[pos]))),
 ambiguous_input_spans_excluded=ambiguous_spans,results=report,
 all_four_both_null_pairs=int(shared.sum()),all_four_both_null_unique_positive_rows=len(ids),all_four_both_null_unique_sentences=len(set(map(str,sent[ids]))),
 shared_positive_labels={labels[k]:int((y[ids]==k).sum()) for k in sorted(set(map(int,y[ids])))},
 scope='Label-aware within-sentence TRAIN census only; annotation-defined pairs, duplicate inputs collapsed only if labels agree. No representative prevalence, inferred semantics, causal attribution, changed predictions or expected gain. This is not a reselection/rescue of closed40semantic witnesses.',
 model_forwards=0,optimizer_updates=0,valid_rows=0,test_rows=0,cos_writes=0)
np.save(out/'positive-null-pairs.private.npy',pair,allow_pickle=False)
result['private_pair_sha256']=hashlib.sha256((out/'positive-null-pairs.private.npy').read_bytes()).hexdigest()
(out/'paired-collapse.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=35))
(r/'paired-collapse.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
