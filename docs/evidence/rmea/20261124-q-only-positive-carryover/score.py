"""Count all fixed blind judgments; no text, model replay or relabeling."""
from pathlib import Path
import json
import cpu_ops as ops

r=Path(__file__).resolve().parent
code=r'''from pathlib import Path
from collections import Counter
import json,hashlib
r=Path('/hy-tmp/Q-Attention/control/q-only-content32-20261124-cpu');o=r/'output'
truth=json.loads((o/'truth.private.json').read_bytes())
readers=[json.loads((o/('reader'+str(i)+'.private.json')).read_bytes()) for i in (1,2)]
assert all(v['parse_error'] is None and len(v['answer']['rows'])==32 for v in readers)
assert readers[0]['packet_sha256']==readers[1]['packet_sha256']
answers=[{row['case_id']:row for row in v['answer']['rows']} for v in readers]
assert all(set(a)==set(truth) for a in answers)
groups={};records=[]
for group in ('q_only_positive','q_only_null','peer_only_positive','peer_only_null'):
 cases=sorted(k for k,v in truth.items() if v['group']==group);assert len(cases)==8
 counts=Counter();agreed={k:Counter() for k in ('category','evidence_structure','distractor_relation')}
 per_reader=[{k:Counter() for k in agreed} for _ in range(2)]
 for case in cases:
  t=truth[case];a,b=(v[case] for v in answers);c1,c2=a['choice'],b['choice'];gold=t['gold_option']
  counts['both_support_gold']+=int(c1==c2==gold)
  counts['both_choose_other']+=int(c1==c2 and c1 in ('A','B') and c1!=gold)
  counts['choice_disagreement']+=int(c1!=c2)
  counts['any_uncertain_or_nonbinary']+=int(c1 not in ('A','B') or c2 not in ('A','B'))
  for k in agreed:
   if a[k]==b[k]:agreed[k][a[k]]+=1
   for i,v in enumerate((a,b)):per_reader[i][k][v[k]]+=1
  records.append(dict(case_id=case,group=group,gold=t['gold'],gold_option=gold,
   reader_choices=[c1,c2],reader_categories=[a['category'],b['category']],
   reader_evidence_structure=[a['evidence_structure'],b['evidence_structure']],
   reader_distractor_relation=[a['distractor_relation'],b['distractor_relation']],
   both_support_gold=c1==c2==gold))
 groups[group]=dict(rows=8,**counts,agreed={k:dict(v) for k,v in agreed.items()},
  per_reader=[{k:dict(v) for k,v in d.items()} for d in per_reader])
summary=dict(profile='q-only-v-all-three-controls-fullTRAIN-fixed-blind32-v1',groups=groups,
 case_records=records,valid_reader_rows=[32,32],packet_sha256=readers[0]['packet_sha256'],
 limitation='Outcome-stratified8pergroup; isolated same-model contexts, not independent humans, prevalence, causal mechanism, relabeling, utility or theory validation.',
 model_forwards=0,training_updates=0,new_valid=0,new_test=0,cos_writes=0)
(o/'blind-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(summary=summary,selection=json.loads((o/'summary.json').read_bytes()),execution=json.loads((r/'execution.json').read_bytes()))))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=25))
for name,key in [('summary.json','summary'),('selection-summary.json','selection'),('execution.json','execution')]:
 (r/name).write_text(json.dumps(v[key],indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(groups={k:{n:w for n,w in d.items() if n!='per_reader'} for k,d in v['summary']['groups'].items()},seconds=v['selection']['seconds'],exit_code=v['execution']['exit_code'])))
