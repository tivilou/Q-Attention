"""Reopen private population and verify rank-AUC by explicit pair counting."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
import numpy as np
r=Path('/hy-tmp/Q-Attention/control/context-pair-diagnostic-20261126');out=r/'output'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
execution=read(r/'execution.json');ev=read(r/'resource-return.json');cfg=read(r/'design.json');spec=cfg['identity_spec']
assert execution['exit_code']==0 and ev['campaign_status']=='completed' and ev['quiescent'] and ev['all_jobs_terminal']
assert execution['runner_attempt_id']==ev['runner_attempt_id']==sys.argv[1]
for n,h in read(out/'RUN_COMPLETE').items():assert sha(out/n)==h
s=read(out/'summary.json');v=read(out/'verification.json');assert s['status']=='COMPLETE' and v['status']=='PASS'
assert sha(out/'population.private.npz')==v['private_population_sha256']
assert sha(out/'trace-cases.private.npz')==v['private_trace_sha256']
with np.load(out/'population.private.npz',allow_pickle=False) as z:
 scores=z['scores'].copy();labels=z['labels'].copy();idx=z['indices'].copy()
 predictions={k:z[k+'_predictions'].copy() for k in spec['models']}
assert scores.shape==(4096,7) and len(idx)==4096
q=predictions['quantum_distinct']==labels;g=predictions['gaussian_covariance']==labels
discord=q!=g;positive=g[discord];f=F
def pair_auc(x,y):
 a=x[y];b=x[~y]
 if not len(a) or not len(b) or not np.isfinite(x).all():return None
 return F(2*int((a[:,None]>b[None,:]).sum())+int((a[:,None]==b[None,:]).sum()),2*len(a)*len(b))
names=['S','sameQ_dephased_minus_G','length','purity_pair_mean']
au={n:pair_auc(scores[discord,j],positive) for j,n in enumerate(names)}
for n,a in au.items():assert (a is None and s['auc'][n] is None) or abs(float(a)-s['auc'][n])<1e-15
missing=int((~np.isfinite(scores[discord,:4]).all(-1)).sum())
sample=int(discord.sum())>=100 and int(positive.sum())>=30 and int((~positive).sum())>=30 and missing==0
adv=au['S']-max(au[n] for n in names[1:]) if all(a is not None for a in au.values()) else None
gates=dict(technical_numeric=True,population=sample,main_auc_at_least_055=au['S'] is not None and au['S']>=F(55,100),advantage_at_least_002=adv is not None and adv>=F(2,100))
assert gates==s['gates']
decision='EXPLORATORY_ASSOCIATION_ONLY_NO_METHOD_ADMISSION' if all(gates.values()) else 'STOP_THIS_EXACT_ALIASING_PREDICTION'
assert decision==s['decision']
assert s['head_forwards']==s['native_batched_circuit_calls']==512 and s['optimizer_updates']==0
assert v['state_before']==v['state_after'] and v['argmax_mismatches']==0
assert len(s['gold_class_length_strata'])==80 and sum(x['rows'] for x in s['gold_class_length_strata'])==4096
for row in s['gold_class_length_strata']:
 mask=(labels==row['gold_class'])&((scores[:,2]>35)==(row['length_band']=='>35'));dd=mask&discord
 assert int(mask.sum())==row['rows'] and int((dd&g).sum())==row['Gcov_only'] and int((dd&q).sum())==row['Qdistinct_only']
 ref=pair_auc(scores[dd,0],g[dd]);actual=row['S_auc']
 assert (ref is None and actual is None) or abs(float(ref)-actual)<1e-15
for mode,pred in predictions.items():
 cm=np.zeros((40,40),np.int64);np.add.at(cm,(labels,pred),1)
 assert cm.tolist()==s['retained_baselines'][mode]['confusion_matrix']
 assert abs(s['retained_baselines'][mode]['all40_macro_f1']-spec['models'][mode]['all40_macro_f1'])<1e-12
for mode,m in spec['models'].items():assert sha(m['checkpoint'])==m['checkpoint_sha256']
result=dict(status='PASS_REOPENED_PRIVATE_POPULATION_FRACTION_AUC',decision=decision,exact_auc={k:str(x) if x is not None else None for k,x in au.items()},
 exact_advantage=str(adv) if adv is not None else None,primary_rows=int(discord.sum()),primary_missing=missing,
 all80strata_verified=True,all4_baselines_verified=True,model_states_unchanged=True,checkpoint_bytes_unchanged=True,
 independent_implementation='Explicit positive/negative pair comparison and Fraction; original runner uses sorted average ranks. Same agent, not external reviewer.',new_model_calls=0,updates=0)
save=dict(files={n:(out/n).read_text('utf8') for n in ('summary.json','verification.json','RUN_COMPLETE')},
 execution=execution,resource_return=ev,independent_verification=result)
print(json.dumps(save))
