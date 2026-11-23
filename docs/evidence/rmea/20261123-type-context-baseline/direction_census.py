from pathlib import Path
from collections import Counter,defaultdict
import json,numpy as np,hashlib
r=Path('/hy-tmp/Q-Attention');p=r/'control/native-context-dephasing-20261123/output/predictions.private.npz'
with np.load(p,allow_pickle=False) as a:z=a['logits'][0];y=a['labels']
prev=r/'control/task-bottleneck-discovery-20261123';mt=json.loads((prev/'cases.private.json').read_text())['type_metadata'];mv=json.loads((prev/'validation-cases.private.json').read_text())['metadata']
with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as a:fit=a['calibration']
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):g,st,ot=mt[i];support[(st,ot)].add(g);seen.add(g)
ref=json.loads((prev/'validation-summary.json').read_text());names=['no_relation']+[a['label'] for a in ref['per_class']];idx={s:i for i,s in enumerate(names)}
allow=np.ones(z.shape,bool)
for i,(g,st,ot) in enumerate(mv):
 if (st,ot) in support:
  allow[i]=False;allow[i,0]=True
  for k in support[(st,ot)] | (set(range(40))-seen):allow[i,k]=True
pred=np.where(allow,z,-np.inf).argmax(1)
inverse={'per:children':'per:parents','per:parents':'per:children','org:members':'org:member_of','org:member_of':'org:members','per:employee_of':'org:top_members/employees','org:top_members/employees':'per:employee_of','per:spouse':'per:spouse','per:siblings':'per:siblings','per:identity':'per:identity','per:other_family':'per:other_family','org:alternate_names':'org:alternate_names'}
words=json.loads((r/'inputs/native-word-metadata-20261123/valid.private.json').read_text())
key=lambda w:hashlib.sha256(json.dumps(w['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
lookup=defaultdict(list)
for i,w in enumerate(words):lookup[(key(w),tuple(w['subject']),tuple(w['object']))].append(i)
pairs=[]
for i,w in enumerate(words):
 for j in lookup[(key(w),tuple(w['object']),tuple(w['subject']))]:
  if i<j:pairs.append((i,j))
consistent=[];wrong=[]
for i,j in pairs:
 if inverse.get(names[y[i]])==names[y[j]]:consistent.append((i,j))
 else:wrong.append((i,j))
rows=[]
for a,b in [('per:children','per:parents'),('org:members','org:member_of'),('per:employee_of','org:top_members/employees'),('per:employee_of','per:schools_attended')]:
 rows.append(dict(a=a,b=b,a_to_b=int(((y==idx[a])&(pred==idx[b])).sum()),b_to_a=int(((y==idx[b])&(pred==idx[a])).sum())))
matched={}
for a,b in consistent:matched.setdefault(tuple(sorted((names[y[a]],names[y[b]]))),[]).append((a,b))
summary=dict(profile='cached-fullVALID-residual-semantic-direction-census-v1',all_reversed_input_pairs=len(pairs),declared_inverse_gold_consistent_pairs=len(consistent),other_gold_pairs=len(wrong),consistent_both_correct=sum(pred[i]==y[i] and pred[j]==y[j] for i,j in consistent),consistent_same_predicted_label=sum(pred[i]==pred[j] for i,j in consistent),pair_counts_by_gold=[dict(labels=k,pairs=len(v)) for k,v in sorted(matched.items())],bidirectional_confusions=rows,claim='Exploratory allpopulation census; inverse mapping is a task schema hypothesis, no inference all other pairs are erroneous; no fitting/model calls/utility claim.')
# Python counters only in public output.
summary['consistent_both_correct']=int(summary['consistent_both_correct']);summary['consistent_same_predicted_label']=int(summary['consistent_same_predicted_label'])
(r/'control/type-context-baseline-20261123/direction-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
