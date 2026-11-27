
from pathlib import Path
import hashlib,json,time
import numpy as np
t0=time.monotonic();sha=lambda b:hashlib.sha256(b).hexdigest()
p=Path('/hy-tmp/Q-Attention/control/staged-bottleneck-design-20261125/output/diagnostic.private.npz')
assert sha(p.read_bytes())=='30a2615a0cbb9676b2c985574972d07b54149166790911dd5da64e1659eefa50'
with np.load(p,allow_pickle=False) as z:
 y=np.asarray(z['labels'],dtype=np.int64);groups=z['groups'].copy();ids=z['assessment_indices'].copy()
 modes=['quantum_corroboration','dephased_corroboration','normalized_dephased_corroboration','gaussian_corroboration','gaussian_base','real12','softmax12']
 preds={k:z[k+'_logits'].argmax(1) for k in modes}
assert len(y)==len(groups)==58465 and len(ids)==4096
actual=y[ids];ug,inverse=np.unique(groups[ids],return_inverse=True)
def metric(pred,weights=None):
 cm=np.bincount(40*actual+pred,weights=weights,minlength=1600).reshape(40,40)
 den=cm.sum(0)+cm.sum(1)
 return np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0).mean()
scores={k:float(metric(v)) for k,v in preds.items()}
assert abs(scores[modes[0]]-0.4878746583746036)<1e-12
rng=np.random.default_rng(20261127);deltas={k:[] for k in modes[1:]}
for b in range(500):
 assert time.monotonic()-t0<60,'Frozen CPU budget'
 n=rng.multinomial(len(ug),np.full(len(ug),1/len(ug)))[inverse]
 q=metric(preds[modes[0]],n)
 for k in modes[1:]:deltas[k].append(q-metric(preds[k],n))
intervals={k:{'point_delta':scores[modes[0]]-scores[k],'cluster_bootstrap_90_percentile':np.quantile(v,[.05,.95]).tolist()} for k,v in deltas.items()}
minimum=np.min(np.asarray(list(deltas.values())),axis=0)

# The retained prediction archive only filled group IDs for its own cohort.
# Rebuild every TRAIN group from the pinned original input before new sampling.
cache=Path('/hy-tmp/Q-Attention/inputs/native-baseline/discovery/cache')
source=(cache.parent.parent/'recovery-inputs/train.jsonl').read_bytes()
assert sha(source)=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
metadata=json.loads((cache/'metadata.json').read_bytes())['train']
rows=[json.loads(s) for s in source.splitlines() if s.strip()]
lookup={sha(str(r['metadata']['id']).encode()):r for r in rows}
fullgroups=np.asarray([sha(json.dumps(lookup[m['id_hash']]['tokens'],ensure_ascii=False,separators=(',',':')).encode()) for m in metadata])
assert np.array_equal(groups[ids],fullgroups[ids])
assert len(np.unique(fullgroups))==33477 and np.array_equal(y,np.load(cache/'train-targets.npy',allow_pickle=False))
groups=fullgroups
# New prospective cohort uses only immutable TRAIN labels/groups and a fixed salt.
members={}
for i,g in enumerate(groups.tolist()):members.setdefault(str(g),[]).append(i)
salt='q-rmea-goal-screen-v1-20261127'
ordered=sorted(members,key=lambda g:sha((salt+':group:'+g).encode()))
assignment={g:0 if int(sha((salt+':role:'+g).encode())[:8],16)%20<12 else (1 if int(sha((salt+':role:'+g).encode())[:8],16)%20<15 else 2) for g in ordered}
gl={g:set(map(int,y[members[g]])) for g in ordered}
for _ in range(120):
 missing=[(s,k) for s in range(3) for k in range(40) if not any(assignment[g]==s and k in gl[g] for g in ordered)]
 if not missing:break
 s,k=missing[0]
 opts=[g for g in ordered if assignment[g]!=s and k in gl[g] and all(sum(assignment[h]==assignment[g] and lab in gl[h] for h in ordered)>1 for lab in gl[g])]
 assert opts,('Insufficient independent groups',s,k)
 assignment[opts[0]]=s
assert not missing
def sample(side,n,floor):
 buckets={k:[i for g in ordered if assignment[g]==side for i in members[g] if int(y[i])==k] for k in range(40)}
 counts=np.array([len(buckets[k]) for k in range(40)])
 expected=n*counts/counts.sum();minimum=np.minimum(floor,counts)
 quota=np.maximum(minimum,np.floor(expected).astype(int));quota=np.minimum(quota,counts)
 while quota.sum()<n:
  k=max(np.flatnonzero(quota<counts),key=lambda k:(expected[k]-quota[k],-k));quota[k]+=1
 while quota.sum()>n:
  k=max(np.flatnonzero(quota>minimum),key=lambda k:(quota[k]-expected[k],-k));quota[k]-=1
 chosen=sorted(i for k in range(40) for i in sorted(buckets[k],key=lambda i:sha((salt+':row:'+str(i)).encode()))[:quota[k]])
 assert len(chosen)==n and len(set(chosen))==n and min(quota)>0
 return chosen,quota.tolist(),counts.tolist()
cohort={};supports={};pool_counts={}
for side,(name,n,floor) in enumerate([('fit',4096,1),('selection',1024,8),('qualification',2048,16)]):
 cohort[name],supports[name],pool_counts[name]=sample(side,n,floor)
for a,b in [('fit','selection'),('fit','qualification'),('selection','qualification')]:
 assert not set(cohort[a])&set(cohort[b])
 assert not set(map(str,groups[cohort[a]]))&set(map(str,groups[cohort[b]]))
assert sum(map(len,cohort.values()))==7168
counts=np.bincount(y,minlength=40)
payload={'schema':'q-rmea.goal-aligned-small-cohort.v1','profile_id':'q-rmea-small-screen-v1','sampler_revision':2,'salt':salt,
 'indices':cohort,'supports':supports,'role_pool_class_counts':pool_counts,'train_class_counts':counts.tolist(),
 'source_predictions_sha256':sha(p.read_bytes()),'source_train_sha256':sha(source),'full_sentence_groups':33477,'group_overlap':0,'valid_test_read':False,
 'selection_rule':'TRAIN-only fixed group hash, repair missing label coverage before model outcomes; fixed row hash, proportional quotas with floors fit1/select8/qual16 capped by availability',
 'assessment_estimand':'TRAIN-prior-reweighted class-conditional performance within each assigned role pool; proxy, not an unbiased TEST estimate',
 'assessment_row_weights':{k:(counts/counts.sum()/np.array(supports[k])).tolist() for k in ['selection','qualification']},
 'exposure':'TRAIN is historically exposed. New role split is development and not independent confirmation.'}
raw=(json.dumps(payload,sort_keys=True,indent=2)+'\n').encode()
out=Path('/hy-tmp/Q-Attention/control/goal-aligned-screening-20261127');out.mkdir(exist_ok=True)
f=out/'cohort-v2.private.json'
if f.exists():assert f.read_bytes()==raw,'Preserve concurrent cohort'
else:f.write_bytes(raw)
assert sha(f.read_bytes())==sha(raw)
result={'status':'PASS_CPU_SAVED_PREDICTION_AUDIT_AND_SMALL_COHORT','historical_scores':scores,'paired_intervals':intervals,
 'minimum_control_margin_bootstrap_90_percentile':np.quantile(minimum,[.05,.95]).tolist(),
 'bootstrap_repetitions':500,'bootstrap_unit':'assessment sentence group','assessment_groups':len(ug),
 'calibration_status':'INSUFFICIENT_RETROSPECTIVE_ONE_PAIR','audited_promoted_completed':1,'audited_full_utility_pass':0,
 'new_cohort':{'path':str(f),'sha256':sha(raw),'sizes':{k:len(v) for k,v in cohort.items()},'supports':supports,
 'qualification_classes_below_10':sum(v<10 for v in supports['qualification']),
 'qualification_classes_below_5':sum(v<5 for v in supports['qualification']),
 'group_overlap':0,'total_unique':7168,'old_total_unique':12288,'estimand':payload['assessment_estimand']},
 'new_fits':0,'model_forwards':0,'gpu_initializations':0,'valid_test_read':False,'seconds':time.monotonic()-t0,
 'limitations':['Retrospective design evidence, not prospective validation','Bootstrap excludes seed/scale/adaptation uncertainty','New training subset and selection semantics need future matched reference/candidate evaluation','Rare class has only6 TRAIN rows/5 sentence groups; sparse support remains']}
assert result['seconds']<60
print(json.dumps(result))
