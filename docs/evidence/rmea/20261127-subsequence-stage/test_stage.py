"""Synthetic data/selection semantics, never run the old quantum math checker."""
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import goal_aligned_stage as s
H=Path(__file__).resolve().parent
checks={}
y=np.repeat(np.arange(40),2);p=y.copy();p[0]=1
train=np.arange(1,41);supports=np.full(40,2);w=s.row_weights(y,train,supports)
cm=s.confusion(y,p,w)
manual=np.zeros((40,40))
for a,b,c in zip(y,p,w):manual[a,b]+=c
assert np.array_equal(cm,manual) and abs(w.sum()-1)<1e-15
f=[]
for k in range(40):
 d=sum(manual[k])+sum(r[k] for r in manual);f.append(2*manual[k,k]/d if d else 0)
assert s.macro(cm)==sum(f)/40 and s.macro(np.zeros((40,40)))==0
checks['all40_weighted_confusion_and_zero_labels']=True
z=np.eye(40)[p]*3;metric=s.evaluate(z,y,w)
expected=sum(float(ww)*(-zz[int(yy)]+np.log(sum(np.exp(zz)))) for zz,yy,ww in zip(z,y,w))
assert abs(metric['weighted_ce']-expected)<1e-14
checks['fixed_TRAIN_prior_weighted_CE']=True
assert s.choose([dict(step=320,weighted_macro_f1=.5),dict(step=160,weighted_macro_f1=.5)])['step']==160
curves={a:{480:.5,640:.5} for a in 'qdcr'};assert s.stable(curves,'q',['d','c','r'])
curves['q'][640]=.49;assert not s.stable(curves,'q',['d','c','r'])
checks['earliest_tie_and_both_final_checkpoints']=True
roles=dict(fit=list(range(4096)),selection=list(range(4096,5120)),qualification=list(range(5120,7168)));groups=np.arange(7168)
s.validate_roles(roles,groups);bad=groups.copy();bad[4096]=0
try:s.validate_roles(roles,bad)
except AssertionError:pass
else:raise AssertionError('Group leakage not detected')
order,h=s.batches(roles['fit']);assert len(order)==640 and all(len(b)==128 for b in order)
assert all(set(np.concatenate(order[j*32:(j+1)*32]))==set(roles['fit']) for j in range(20))
checks['role_group_separation_and_640_frozen_batches']=True
preds=dict(q=y.copy(),d=p.copy(),c=p.copy(),r=p.copy());g=np.repeat(np.arange(40),2)
a=s.grouped_bootstrap(y,preds,w,g,'q',['d','c','r']);b=s.grouped_bootstrap(y,preds,w,g,'q',['d','c','r']);assert a==b and a['groups']==40
assert len(a['sparse_label_absence_frequency'])==40 and a['point_deltas']['d']>0
checks['paired_groups_fixed_weights_seed500_and_sparse_labels']=True
with tempfile.TemporaryDirectory(prefix='qattn-stage-') as tmp:
 journal=s.SelectionJournal(tmp,'Q','sourcepin');begin=journal.begin(160,'modelpin')
 try:journal.receipts()
 except AssertionError:pass
 else:raise AssertionError('Incomplete selection not fail-closed')
 journal.commit(begin,metric,'predictionpin');assert journal.receipts()[0]['checkpoint_sha']=='modelpin'
 try:journal.begin(160,'modelpin')
 except FileExistsError:pass
 else:raise AssertionError('Repeated checkpoint pass not blocked')
 checks['selection_BEGIN_failure_and_exactly_once_recovery_receipt']=True
result=dict(status='PASS_GOAL_ALIGNED_STAGE_SYNTHETIC_SEMANTICS',checks=checks,source_sha256=hashlib.sha256((H/'goal_aligned_stage.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),task_arrays_read=False,model_created=False,task_forwards=0,updates=0,old_math_rerun=False)
assert not (H/'stage-tests.json').exists();(H/'stage-tests.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
