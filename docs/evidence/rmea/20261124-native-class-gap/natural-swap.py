"""Bounded read-only metadata/existing-prediction census, no model execution."""
from pathlib import Path
import json
import cpu_ops as ops
r=Path(__file__).resolve().parent
code=r'''from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,time,numpy as np
start=time.monotonic();root=Path('/hy-tmp/Q-Attention')
base=root/'worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13'
cache=base/'discovery/cache';source=base/'recovery-inputs/train.jsonl';old=root/'control/full-train-error-contrast-20261124-cpu/output'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
metadata=json.loads((cache/'metadata.json').read_bytes())['train'];y=np.load(cache/'train-targets.npy',allow_pickle=False)
original=[json.loads(x) for x in source.read_bytes().splitlines() if x.strip()]
lookup={hashlib.sha256(str(v['metadata']['id']).encode()).hexdigest():v for v in original}
rows=[lookup[v['id_hash']] for v in metadata];assert len(rows)==58465
labels=json.loads((old.parent/'inputs/labels.json').read_bytes());assert all(labels[int(k)]==v['label'] for k,v in zip(y,rows))
previous=json.loads((old/'summary.json').read_bytes());pred={}
for mode in ('quantum','dephased','real12','softmax12'):
 p=old/(mode+'-logits.private.npy');assert hashlib.sha256(p.read_bytes()).hexdigest()==previous['results'][mode]['logits_sha256']
 pred[mode]=np.load(p,mmap_mode='r',allow_pickle=False).argmax(-1)
groups=defaultdict(list)
for i,v in enumerate(rows):groups[(tuple(v['tokens']),tuple(v['subject']),tuple(v['object']))].append(i)
inverse={'per:parents':'per:children','per:children':'per:parents','org:members':'org:member_of','org:member_of':'org:members'}
for label in ('no_relation','per:spouse','per:siblings','per:identity'):inverse[label]=label
assert all(k in labels and v in labels for k,v in inverse.items())
totals=Counter();cohorts=Counter();performance={m:Counter() for m in pred};by_relation={};sentence_hashes=set();seen=set()
for key,indices in groups.items():
 reverse=(key[0],key[2],key[1])
 if reverse==key or reverse not in groups or key in seen:continue
 seen.add(key);seen.add(reverse);totals['unique_unordered_span_pairs']+=1
 sentence_hashes.add(hashlib.sha256(json.dumps(key[0]).encode()).hexdigest())
 for i in indices:
  for j in groups[reverse]:
   totals['paired_row_units']+=1;first,second=rows[i]['label'],rows[j]['label']
   kind='unmapped' if first not in inverse or second not in inverse else ('inverse_gold_consistent' if inverse[first]==second else 'inverse_gold_inconsistent')
   cohorts[kind]+=1
   if kind!='inverse_gold_consistent':continue
   pair=' <-> '.join(sorted((first,second)))
   if pair not in by_relation:by_relation[pair]=dict(rows=0,performance={m:Counter() for m in pred})
   by_relation[pair]['rows']+=1
   for m,p in pred.items():
    correct=int(p[i]==y[i])+int(p[j]==y[j]);performance[m][('neither_correct','one_correct','both_correct')[correct]]+=1
    performance[m]['prediction_inverse_consistent']+=int(labels[int(p[i])] in inverse and inverse[labels[int(p[i])]]==labels[int(p[j])])
    by_relation[pair]['performance'][m][('neither_correct','one_correct','both_correct')[correct]]+=1
result=dict(profile='native-TRAIN-natural-exact-token-and-span-reversal-census-v1',rows=58465,totals=dict(totals),distinct_sentences=len(sentence_hashes),
 label_mapping=inverse,cohorts=dict(cohorts),performance={m:dict(v) for m,v in performance.items()},by_relation=by_relation,seconds=time.monotonic()-start,
 limitation='Posthoc descriptive census; Cartesian duplicate row pairs reported separately from unique span pairs; no label rewriting, learned encoder covariance, root cause, deployable repair or utility claim.',
 model_forwards=0,training_updates=0,new_valid=0,new_test=0,cos_writes=0,raw_text_on_windows=False)
print(json.dumps(result))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=30))
if (r/'natural-swap.json').exists() and not (r/'natural-swap-v1-asymmetric-classification.json').exists():
 (r/'natural-swap-v1-asymmetric-classification.json').write_bytes((r/'natural-swap.json').read_bytes())
(r/'natural-swap.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
print(json.dumps(v))
