
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,sys,numpy as np,time
start=time.monotonic();v=json.load(sys.stdin);r=Path('/hy-tmp/Q-Attention')
out=r/'control/task-bottleneck-discovery-20261123';dest=out/'validation-summary.json'
assert not dest.exists(),'One frozen development application, no repeat/retune'
assert (out/'summary.json').exists()
prior=json.loads((out/'cases.private.json').read_text());meta=prior['type_metadata']
with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as a:
 fit=a['calibration'];ty=a['labels']
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):
 if meta[i] is not None:
  g,st,ot=meta[i];assert g==int(ty[i]);support[(st,ot)].add(g);seen.add(g)
support_seal=hashlib.sha256(json.dumps(sorted((list(pair),sorted(ids)) for pair,ids in support.items()),separators=(',',':')).encode()).hexdigest()
p=r/'control/native-context-dephasing-20261123/output/predictions.private.npz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='eceb0359760d1ff5306665285aa7c96cfe26dde357e3235fd1fac4e47f6fcee8'
with np.load(p,allow_pickle=False) as a:z=a['logits'][0].copy();y=a['labels'].copy()
assert z.shape==(19584,40) and len(v['rows'])==19584
labels=v['labels'];metadata=v['rows'];pred=z.argmax(1)
for i,m in enumerate(metadata):
 if m is not None:assert int(y[i])==m[0]
def measures(p):
 cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,p),1)
 tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
 f1=np.divide(2*tp,den,out=np.zeros(40),where=den>0)
 return dict(items=len(y),correct=int(tp.sum()),accuracy=float(tp.sum()/len(y)),positive39_macro_f1=float(f1[1:].mean())),cm,f1
native,cm,fn=measures(pred)
assert abs(native['positive39_macro_f1']-.5164743764)<1e-9,'Native lane/metric does not match the completed source diagnostic'
allow=np.ones(z.shape,bool);pair_seen=np.zeros(len(y),bool)
for i,m in enumerate(metadata):
 if m is None:continue
 _,st,ot=m;pair=(st,ot)
 if pair not in support:continue
 pair_seen[i]=True;allow[i]=False;allow[i,0]=True
 for k in support[pair]|(set(range(40))-seen):allow[i,k]=True
projected=np.where(allow,z,-np.inf).argmax(1)
modified,cf,fp=measures(projected)
repairs=np.flatnonzero((pred!=y)&(projected==y));damage=np.flatnonzero((pred==y)&(projected!=y))
incompatible=~allow[np.arange(len(y)),pred];gold_excluded=~allow[np.arange(len(y)),y]
positive=modified['positive39_macro_f1']>native['positive39_macro_f1'] and len(repairs)>len(damage)
changed=np.flatnonzero(pred!=projected)
type_errors=Counter((metadata[i][1],metadata[i][2],labels[pred[i]]) for i in np.flatnonzero(incompatible) if metadata[i] is not None)
class_rows=[]
for k in range(1,40):
 class_rows.append(dict(label=labels[k],support=int(cm[k].sum()),native_f1=float(fn[k]),projected_f1=float(fp[k]),delta=float(fp[k]-fn[k]),native_tp=int(cm[k,k]),projected_tp=int(cf[k,k]),native_fp=int(cm[:,k].sum()-cm[k,k]),projected_fp=int(cf[:,k].sum()-cf[k,k])))
report=dict(profile='single-application-frozen-TRAIN-type-rule-to-development-VALID-v1',
 data=dict(valid_count=len(y),unresolved_metadata=v['ambiguous_or_missing'],raw_source_sha256=v['source_sha256'],prediction_archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
 fitting=dict(rows=4192,source='Previously fixed TRAIN calibration partition only',support_type_pairs=len(support),support_sha256=support_seal,classes_not_seen=sorted(set(range(40))-seen),validation_labels_used_in_fit=False,validation_labels_used_in_selection=False,rule_changed=False),
 native=native,projected=modified,macro_delta=modified['positive39_macro_f1']-native['positive39_macro_f1'],
 type_compatibility=dict(valid_seen_fit_type_pair=int(pair_seen.sum()),incompatible_top1=int(incompatible.sum()),gold_excluded=int(gold_excluded.sum()),changed_predictions=len(changed),repairs=len(repairs),damage=len(damage),net_correct=len(repairs)-len(damage)),
 positive_class_f1_changes=dict(improved=int((fp[1:]>fn[1:]).sum()),worse=int((fp[1:]<fn[1:]).sum()),unchanged=int((fp[1:]==fn[1:]).sum())),
 per_class=class_rows,top_incompatible_type_predictions=[dict(subject_type=st,object_type=ot,predicted=label,count=n) for (st,ot,label),n in type_errors.most_common(12)],
 decision='POSITIVE_DEVELOPMENT_TASK_BOTTLENECK' if positive else 'NO_TRANSFER_CLOSE_EXACT_TYPE_PROJECTION',
 limitations=['Original entity-type annotations are an explicit additional input compared with the native four-coordinate role interface. Same inputs must be provided to every later comparator.',
 'Compatibility is empirical TRAIN support, not a logical impossibility theorem; gold-excluded counts and harmful changes remain visible.',
 'VALID has been reused in this campaign. This is development problem discovery, not independent confirmation, multi-seed task utility, quantum benefit or L5.',
 'No quantum mechanism has been selected by this analysis.'],
 execution=dict(model_forwards=0,optimizer_updates=0,gpu_compute=0,cos_writes=0,seconds=time.monotonic()-start))
dest.write_text(json.dumps(report,indent=2)+'\n')
(out/'validation-cases.private.json').write_text(json.dumps(dict(repairs=repairs.tolist(),damage=damage.tolist(),changed=changed.tolist(),metadata=metadata),indent=2)+'\n')
print(json.dumps(report))
