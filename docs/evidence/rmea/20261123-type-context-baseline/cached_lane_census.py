
from pathlib import Path
from collections import defaultdict,Counter
import json,sys,numpy as np,hashlib,time
start=time.monotonic();r=Path('/hy-tmp/Q-Attention');root=r/'control/type-context-baseline-20261123'
assert not (root/'summary.json').exists() and not (root/'cases.private.json').exists(),'Preserve completed results'
prev=r/'control/task-bottleneck-discovery-20261123'
t=json.loads((prev/'cases.private.json').read_text())['type_metadata']
v=json.loads((prev/'validation-cases.private.json').read_text())['metadata']
with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as a:
 fit=a['calibration'].copy();ty=a['labels'].copy()
support=defaultdict(set);counts=defaultdict(Counter);seen=set()
for i in np.flatnonzero(fit):
 g,st,ot=t[i];assert g==int(ty[i]);support[(st,ot)].add(g);counts[(st,ot)][g]+=1;seen.add(g)
seal=hashlib.sha256(json.dumps(sorted((list(pair),sorted(ids)) for pair,ids in support.items()),separators=(',',':')).encode()).hexdigest()
assert seal=='d4c1100df6d306a3fae48d66f7907dfbe2a39d1b30843abb359ba32de7053186'
p=r/'control/native-context-dephasing-20261123/output/predictions.private.npz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='eceb0359760d1ff5306665285aa7c96cfe26dde357e3235fd1fac4e47f6fcee8'
with np.load(p,allow_pickle=False) as a:z=a['logits'].copy();y=a['labels'].copy()
assert z.shape==(4,19584,40) and len(v)==19584
names=['no_relation']+[x['label'] for x in json.loads((Path.cwd()/'validation-reference.json').read_text())['per_class']]
assert len(names)==40
allow=np.ones((len(y),40),bool);type_prediction=np.zeros(len(y),int)
for i,(g,st,ot) in enumerate(v):
 assert g==int(y[i]);pair=(st,ot)
 if pair not in support:continue
 allow[i]=False;allow[i,0]=True
 for k in support[pair]|(set(range(40))-seen):allow[i,k]=True
 type_prediction[i]=min(counts[pair],key=lambda k:(-counts[pair][k],k))
def metrics(p):
 cm=np.zeros((40,40),int);np.add.at(cm,(y,p),1);tp=cm.diagonal();d=cm.sum(0)+cm.sum(1)
 f1=np.divide(2*tp,d,out=np.zeros(40),where=d>0)
 return dict(items=len(y),correct=int(tp.sum()),accuracy=float(tp.sum()/len(y)),positive39_macro_f1=float(f1[1:].mean()),null_false_positive=int(((y==0)&(p>0)).sum()),positive_false_negative=int(((y>0)&(p==0)).sum()),positive_wrong_relation=int(((y>0)&(p>0)&(y!=p)).sum()))
lanes=['native','bridge','outer','context'];pred={};results={}
for k,lane in enumerate(lanes):
 raw=z[k].argmax(1);pred[lane]=np.where(allow,z[k],-np.inf).argmax(1)
 results[lane]=dict(raw=metrics(raw),typed=metrics(pred[lane]))
assert abs(results['native']['typed']['positive39_macro_f1']-.5429469429232833)<1e-14
native=pred['native'];comparisons={}
for lane in lanes[1:]:
 other=pred[lane]
 comparisons[lane]=dict(native_correct_other_wrong=int(((native==y)&(other!=y)).sum()),other_correct_native_wrong=int(((other==y)&(native!=y)).sum()),macro_gap=results['native']['typed']['positive39_macro_f1']-results[lane]['typed']['positive39_macro_f1'],claim='Fixed-checkpoint intervention only; no retrained classical opponent or quantum necessity.')
type_groups=defaultdict(list)
for i,(_,st,ot) in enumerate(v):type_groups[(st,ot)].append(i)
type_rows=[];oracle_correct=0;label_impure=0
for pair,ids in type_groups.items():
 hist=Counter(int(y[i]) for i in ids);oracle_correct+=max(hist.values());label_impure+=int(len(hist)>1)
 type_rows.append(dict(subject_type=pair[0],object_type=pair[1],items=len(ids),labels=len(hist),residual_errors=int(sum(native[i]!=y[i] for i in ids)),gold_relations={names[k]:n for k,n in sorted(hist.items())}))
type_rows.sort(key=lambda x:(-x['residual_errors'],x['subject_type'],x['object_type']))
error_pairs=Counter((int(y[i]),int(native[i])) for i in range(len(y)) if native[i]!=y[i])
top=[dict(gold=names[a],predicted=names[b],count=n) for (a,b),n in error_pairs.most_common(15)]
words=json.loads((r/'inputs/native-word-metadata-20261123/valid.private.json').read_text())
families={'null_false_positive':[],'positive_false_negative':[],'wrong_positive_relation':[]}
for i in range(len(y)):
 if native[i]==y[i]:continue
 key='null_false_positive' if y[i]==0 else 'positive_false_negative' if native[i]==0 else 'wrong_positive_relation'
 families[key].append(i)
cases=[]
for family,ids in families.items():
 chosen=sorted(ids,key=lambda i:hashlib.sha256(('type-context-case-v1:'+words[i]['id_hash']).encode()).digest())[:2]
 for i in chosen:
  w=words[i];cases.append(dict(family=family,index=i,id_hash=w['id_hash'],tokens=w['tokens'],subject=w['subject'],object=w['object'],subject_type=v[i][1],object_type=v[i][2],gold=names[y[i]],prediction=names[native[i]],native_prediction=names[z[0,i].argmax()],gold_masked=not bool(allow[i,y[i]])))
report=dict(profile='all-cached-lanes-frozen-type-rule-and-residual-census-v1',scope='Posthoc reused development VALID. All19584/all4oldlanes; no new model call or trainable method.',
 source=dict(prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frozen_support_sha256=seal,train_support_rows=4192,unchanged_rule=True),results=results,typed_native_vs_interventions=comparisons,
 type_only=dict(fit_only_majority=metrics(type_prediction),label_aware_best_possible_correct=oracle_correct,label_aware_maximum_accuracy=oracle_correct/len(y),label_impure_type_pairs=label_impure,total_valid_type_pairs=len(type_groups),bound_scope='Exact upper accuracy of a deterministic TYPE-PAIR-ONLY classifier on this finite labeled population, not a Macro bound, deployable oracle or future-distribution guarantee.'),
 residual_type_groups=type_rows,top_residual_confusions=top,case_selection=dict(rule='First two SHA256(type-context-case-v1:id_hash) residual cases in each of three mutually exclusive error families; outcome-stratified, explanatory only.',case_count=len(cases),indices=[x['index'] for x in cases],ids_sha256=hashlib.sha256(json.dumps([x['id_hash'] for x in cases],separators=(',',':')).encode()).hexdigest(),semantic_status='Primary inspection pending; no blind or prevalence claim.'),
 execution=dict(model_forwards=0,optimizer_updates=0,gpu_compute=0,cos_writes=0,seconds=time.monotonic()-start),claim_ceiling='Task bottleneck and fixed-model intervention association only; no quantum need, retrained comparator, new utility or L5.')
root.mkdir(parents=True,exist_ok=True)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'cases.private.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(summary=report,cases=cases),ensure_ascii=False))
