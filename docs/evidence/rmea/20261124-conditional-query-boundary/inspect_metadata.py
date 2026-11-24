
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,resource,sys,time
start=time.monotonic();cfg=json.load(sys.stdin)
p=Path('/home/Q-Attention/Q-Attention-public/data/relation/retacred/train.jsonl')
assert p.is_file()
memory=Path('/sys/fs/cgroup/memory.max')
current=Path('/sys/fs/cgroup/memory.current')
budget=int(memory.read_text()) if memory.exists() and memory.read_text().strip()!='max' else None
used=int(current.read_text()) if current.exists() else None
assert budget is None or used is None or budget-used>48*1048576,'Insufficient headroom for tiny metadata inspection'
counts=defaultdict(Counter);sentences=defaultdict(lambda:defaultdict(set));digest=hashlib.sha256();n=0
with p.open('rb') as handle:
 for line in handle:
  digest.update(line)
  if not line.strip():continue
  row=json.loads(line);label=row['label'];pair=(row['metadata']['subj_type'],row['metadata']['obj_type'])
  sid=hashlib.sha256(json.dumps(row['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  counts[label][pair]+=1;sentences[label][pair].add(sid);n+=1
assert digest.hexdigest()==cfg['source_sha256']
assert n==58465 and len(counts)==40 and 'no_relation' in counts
per_class=[];eligible=[]
for label in sorted(counts):
 pairs=[dict(subject_type=pair[0],object_type=pair[1],rows=count,distinct_sentences=len(sentences[label][pair]))
        for pair,count in sorted(counts[label].items(),key=lambda item:(-item[1],item[0]))]
 supported=[q for q in pairs if q['distinct_sentences']>=cfg['minimum_sentences_per_pair']]
 total=sum(counts[label].values())
 per_class.append(dict(relation=label,rows=total,type_pairs=pairs,supported_type_pairs=len(supported),majority_pair_fraction=pairs[0]['rows']/total))
 if label!='no_relation' and len(supported)>=2:eligible.append(label)
passed=len(eligible)>=cfg['minimum_positive_relations']
result=dict(profile=cfg['profile'],source_sha256=digest.hexdigest(),rows=n,classes=len(per_class),per_class=per_class,
 eligible_positive_relations=eligible,minimum_sentences_per_pair=cfg['minimum_sentences_per_pair'],
 minimum_positive_relations=cfg['minimum_positive_relations'],support_gate=passed,
 disposition='SUPPORT_ONLY_NOT_A_MECHANISM' if passed else 'CLOSE_EXACT_TYPE_HETEROGENEITY_LEAD',
 claim='Complete TRAIN support census only; no query conflict, model utility, novelty or quantum attribution.',
 heldout_rows_accessed=0,model_calls=0,training_updates=0,elapsed_seconds=time.monotonic()-start,
 execution_kind='short_read_only_owner_metadata_inspection',remote_files_written=0,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,memory_budget_bytes=budget,memory_current_before_bytes=used)
print(json.dumps(result))
