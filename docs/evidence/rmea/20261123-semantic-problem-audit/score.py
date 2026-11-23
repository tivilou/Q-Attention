"""Keep all frozen cases; aggregate two readers without treating them as truth."""
from pathlib import Path
import json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code=r'''
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
r=Path('/hy-tmp/Q-Attention/control/semantic-problem-audit-20261123')
truth=json.loads((r/'selection.private.json').read_bytes());a=json.loads((r/'reader-a.private.json').read_bytes());b=json.loads((r/'reader-b.private.json').read_bytes())
assert a['profile_echo_verified'] and b['profile_echo_verified']
aa={v['case_id']:v for v in a['judgments']};bb={v['case_id']:v for v in b['judgments']};counts=defaultdict(Counter);records=[]
for row in truth:
 x=aa.get(row['case_id']);y=bb.get(row['case_id']);c=counts[row['family']];c['total']+=1
 known=x is not None and y is not None and not x['uncertain'] and not y['uncertain'] and x['relation']!='uncertain' and y['relation']!='uncertain'
 consensus=known and x['relation']==y['relation']
 if not known:c['unknown_or_uncertain']+=1
 elif not consensus:c['confident_disagreement']+=1
 else:
  c['confident_consensus']+=1;c['consensus_equals_stored_label']+=int(x['relation']==row['gold']);c['consensus_equals_model']+=int(x['relation']==row['predicted'])
  c['consensus_differs_stored_label']+=int(x['relation']!=row['gold'])
 both_evidence=bool(consensus and x['relation'] not in ('no_relation','uncertain') and x['evidence_indices'] and y['evidence_indices'])
 if both_evidence:
  c['positive_consensus_with_citations']+=1
  c['both_explicit_or_entailed']+=int(x['pair_link'] in ('explicit','entailed') and y['pair_link'] in ('explicit','entailed'))
 if consensus and x['relation']==row['gold']:
  c['gold_consensus_both_competing_entity']+=int(x['competing_entity'] and y['competing_entity'])
  c['gold_consensus_both_requires_coreference']+=int(x['requires_coreference'] and y['requires_coreference'])
  if x['evidence_form']==y['evidence_form']:c['gold_consensus_same_form_'+x['evidence_form']]+=1
 records.append(dict(case_id=row['case_id'],family=row['family'],gold=row['gold'],prediction=row['predicted'],confident_consensus=bool(consensus),
   consensus=x['relation'] if consensus else None,positive_consensus_with_citations=both_evidence,
   evidence_intersection=sorted(set(x['evidence_indices'])&set(y['evidence_indices'])) if known else [],readers=[x,y]))
report=dict(selection_sha256=a['selection_sha256'],selected=len(truth),strata={k:dict(v) for k,v in counts.items()},
 readers=dict(model=a['model'],effort=a['reasoning_effort'],valid_rows=[len(aa),len(bb)],two_isolated_contexts=True,independent_experts=False),
 limitations=['Outcome-stratified32exposedTRAIN cases do not estimate benchmark prevalence.','Same-model agreement is not human ground truth; do not change labels or infer a new upper bound.','Only label names, not the complete official annotation manual, were supplied; semantic conventions may confound disagreement.','No mechanism/attention trace yet; cue descriptions do not establish attention failure, quantum utility or L5.'],
 summary='Positive missed group: separate both-reader label-supported relation evidence from disputed/uncertain stored labels. No reselection, relabeling, prompt repair, training, VALID/TEST or COS.')
for name,value in [('semantic-results.private.json',records),('summary.json',report)]:
 p=r/name;assert not p.exists();p.write_bytes(json.dumps(value,indent=2).encode()+bytes([10]))
print(json.dumps(report))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=30));(r/'summary.json').write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
