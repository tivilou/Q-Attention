
from pathlib import Path
from collections import defaultdict,Counter
from itertools import combinations
import hashlib,json,sys,numpy as np,time
start=time.monotonic();v=json.load(sys.stdin);r=Path('/hy-tmp/Q-Attention')
out=r/'control/task-bottleneck-discovery-20261123';assert not out.exists(),'No repeated profile/overwrite'
p=r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='ff920146542e7af85f90df5ececfa62f0742be4e90f0db4de8645946e043d833'
with np.load(p,allow_pickle=False) as a:
 z=a['logits'].copy();y=a['labels'].copy();fit=a['calibration'].copy();groups=a['groups'].copy()
words=json.loads((r/'inputs/native-word-metadata-20261123/train.private.json').read_text())
labels=v['labels'];meta=v['rows'];assert len(meta)==len(words)==len(z)==8192
assert z.shape==(8192,40) and fit.dtype==bool and not set(groups[fit])&set(groups[~fit])
for i,m in enumerate(meta):
 if m is not None: assert m[0]==int(y[i]),'Class order / original identity mismatch'
pred=z.argmax(1);assessment=np.flatnonzero(~fit);train=np.flatnonzero(fit)
def metrics(p):
 cm=np.zeros((40,40),dtype=np.int64);np.add.at(cm,(y[assessment],p[assessment]),1)
 tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
 f=np.divide(2*tp,den,out=np.zeros(40),where=den!=0)
 return dict(correct=int(tp.sum()),items=len(assessment),accuracy=float(tp.sum()/len(assessment)),positive39_macro_f1=float(f[1:].mean()))
native=metrics(pred)
# A single input-type constraint is estimated exclusively from the fixed fit partition.
support=defaultdict(set);seen_classes=set()
for i in train:
 if meta[i] is not None:
  _,st,ot=meta[i];support[(st,ot)].add(int(y[i]));seen_classes.add(int(y[i]))
allowed=np.ones(z.shape,dtype=bool);seen_type_pair=np.zeros(len(z),bool)
for i in assessment:
 if meta[i] is None:continue
 _,st,ot=meta[i];pair=(st,ot)
 if pair not in support:continue
 seen_type_pair[i]=True
 allowed[i]=False;allowed[i,0]=True
 for k in support[pair] | (set(range(40))-seen_classes):allowed[i,k]=True
typed=np.where(allowed,z,-np.inf).argmax(1)
gold_excluded=~allowed[np.arange(len(y)),y]
incompatible=~allowed[np.arange(len(y)),pred]
typed_metrics=metrics(typed)
repairs=assessment[(pred[assessment]!=y[assessment])&(typed[assessment]==y[assessment])]
damage=assessment[(pred[assessment]==y[assessment])&(typed[assessment]!=y[assessment])]
projection_positive=typed_metrics['positive39_macro_f1']>native['positive39_macro_f1'] and len(repairs)>len(damage)
# All assessment same-sentence/subject/type disjoint-object pairs, not a new sample of binding40.
by=defaultdict(list)
for i in assessment:
 if meta[i] is None:continue
 w=words[i]
 by[(tuple(w['tokens']),tuple(w['subject']),meta[i][2])].append(int(i))
all_pairs=[];contrast=[]
for indices in by.values():
 for i,j in combinations(indices,2):
  a,b=words[i]['object'],words[j]['object']
  if a[1]<=b[0] or b[1]<=a[0]:
   all_pairs.append((i,j))
   if y[i]!=y[j]:contrast.append((i,j))
collapsed=[];inverted=[];both_correct=0;positive_contrast=0;bound_indices=set()
for i,j in contrast:
 if pred[i]==pred[j]:collapsed.append((i,j));bound_indices.update((i,j))
 if (z[i,y[i]]-z[i,y[j]])-(z[j,y[i]]-z[j,y[j]])<=0:inverted.append((i,j))
 both_correct+=int(pred[i]==y[i] and pred[j]==y[j])
 positive_contrast+=int(y[i]>0 and y[j]>0)
bound=pred.copy()
if bound_indices:
 ids=np.array(sorted(bound_indices));bound[ids]=y[ids]
wrong=assessment[pred[assessment]!=y[assessment]]
errors=Counter((int(y[i]),int(pred[i])) for i in wrong)
top=[dict(gold=labels[a],predicted=labels[b],count=n) for (a,b),n in errors.most_common(15)]
report=dict(profile='task-first-existing-TRAIN8192-prediction-census-v1',scope='Exploratory reused TRAIN assessment, no heldout/independent claim. All fixed4000 assessment rows retained; all contrasts enumerated, not a rescue/resample of closed binding40.',
 data=dict(total=8192,fit=len(train),assessment=len(assessment),types_join_unresolved=v['ambiguous_or_missing'],metadata_alignment_checked=True,source_sha256=v['source_sha256'],prediction_archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
 native=native,type_compatibility=dict(input='Original given subject/object entity-type metadata; additional explicit input relative to native four role coordinates.',rule='Observed fit-only type-pair labels plus no_relation; unseen type-pairs or ambiguous metadata permit all; fit-absent classes retained. Fixed rule/no threshold tuning.',seen_fit_type_pairs=len(support),assessment_seen_type_pair=int(seen_type_pair[assessment].sum()),incompatible_top1=int(incompatible[assessment].sum()),gold_excluded=int(gold_excluded[assessment].sum()),repairs=len(repairs),damage=len(damage),projected=typed_metrics,macro_delta=typed_metrics['positive39_macro_f1']-native['positive39_macro_f1'],decision='POSITIVE_EXPLORATORY_TYPE_REPAIR' if projection_positive else 'NO_POSITIVE_TYPE_REPAIR'),
 role_binding=dict(same_type_disjoint_pairs=len(all_pairs),different_label_pairs=len(contrast),positive_positive_pairs=positive_contrast,same_top1_on_different_gold=len(collapsed),wrong_or_zero_relative_label_contrast=len(inverted),both_correct_pairs=both_correct,collapsed_unique_rows=len(bound_indices),collapsed_sentence_groups=len({groups[i] for ij in collapsed for i in ij}),oracle_fix_all_collapsed_rows=metrics(bound),oracle_caveat='Label-aware ceiling within this exposed cohort, not a deployable repair, causal bottleneck proof or task-independent advantage. Distinct pairs overlap.'),
 error_counts=dict(total=len(wrong),null_false_positive=int(((y[assessment]==0)&(pred[assessment]>0)).sum()),positive_false_negative=int(((y[assessment]>0)&(pred[assessment]==0)).sum()),positive_wrong_relation=int(((y[assessment]>0)&(pred[assessment]>0)&(pred[assessment]!=y[assessment])).sum())),top_confusions=top,
 execution=dict(model_forwards=0,optimizer_updates=0,gpu_compute=0,cos_writes=0,valid_test_used=False,seconds=time.monotonic()-start),
 claim_ceiling='A measurable task problem / cheap repair test only; no chosen quantum solution, novelty, significance, formal utility or L5 acceptance')
out.mkdir(parents=True)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'cases.private.json').write_text(json.dumps(dict(contrast=contrast,collapsed=collapsed,inverted=inverted,type_repairs=repairs.tolist(),type_damage=damage.tolist(),type_metadata=meta),indent=2)+'\n')
print(json.dumps(report))
