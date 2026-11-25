"""Map existing native TRAIN relation confusions without fitting/forwarding models."""
from pathlib import Path
import hashlib
import json
import sys

r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122')
import ops
sources={'context-spectral-flow-20261125':['quantum','dephased'],
         'staged-bottleneck-design-20261125':['gaussian_base'],
         'connected-value-pooling-20261125':['gaussian_covariance']}
request=[]
for name,modes in sources.items():
    s=json.loads((r.parent/name/'summary.json').read_bytes())
    request.append(dict(name=name,modes=modes,sha256=s['private_diagnostic_sha256']))
code=r'''from pathlib import Path
import hashlib,json,sys
import numpy as np
request=json.load(sys.stdin);rows={};zs={};reference=None
root=Path('/hy-tmp/Q-Attention')
lp=root/'owner/attempts/9a28320fd4174c8f95544c9ba96eb1ae/source/docs/contracts/q_rmea_entity_labels.json'
raw=lp.read_bytes();names=sorted(json.loads(raw));assert len(names)==40 and names[0]=='no_relation'
cohort_path=root/'control/staged-bottleneck-design-20261125/cohort.private.json'
cb=cohort_path.read_bytes();assert hashlib.sha256(cb).hexdigest()=='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0'
cohort=json.loads(cb)
for item in request:
 p=root/'control'/item['name']/'output/diagnostic.private.npz'
 assert p.stat().st_size<40*1024**2 and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 with np.load(p,allow_pickle=False) as v:
  ids=v['assessment_indices'];y=v['labels'][ids]
  assert np.array_equal(ids,cohort['assessment'])
  assert len(ids)==4096 and np.bincount(y,minlength=40).tolist()==cohort['supports']['assessment']
  assert np.bincount(v['labels'][cohort['fit']],minlength=40).tolist()==cohort['supports']['fit']
  if reference is None:reference=(ids.copy(),y.copy())
  else:assert np.array_equal(ids,reference[0]) and np.array_equal(y,reference[1])
  positive=y>0
  for mode in item['modes']:
   z=v[mode+'_logits'];assert z.shape==(4096,40) and np.isfinite(z).all();zs[mode]=z.copy()
   pred=z.argmax(-1);forced=z[:,1:].argmax(-1)+1
   cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
   pcm=np.bincount(39*(y[positive]-1)+(forced[positive]-1),minlength=39**2).reshape(39,39)
   den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
   rows[mode]=dict(primary_cm=cm.tolist(),primary_f1=f1.tolist(),primary_macro=float(f1.mean()),
     positive39_confusion=pcm.tolist(),positive_forced_correct=int((positive&(forced==y)).sum()))
qpred=zs['quantum'][:,1:].argmax(-1)+1;goodq=qpred==y
pairs={};corrected={}
for mode,z in zs.items():
 if mode=='quantum':continue
 other=z[:,1:].argmax(-1)+1;good=other==y
 qonly=positive&goodq&~good;otheronly=positive&good&~goodq
 corrected[mode]=np.bincount(39*(y[otheronly]-1)+(qpred[otheronly]-1),minlength=39**2).reshape(39,39)
 pairs[mode]=dict(Q_correct_other_wrong=int(qonly.sum()),other_correct_Q_wrong=int(otheronly.sum()),
  per_true_class=[dict(class_id=i,Q_correct_other_wrong=int((qonly&(y==i)).sum()),
                      other_correct_Q_wrong=int((otheronly&(y==i)).sum())) for i in range(1,40)],
  other_correct_Q_error_matrix=corrected[mode].tolist())
classes=[]
for i in range(40):
 item=dict(class_id=i,label=names[i],fit_count=cohort['supports']['fit'][i],assessment_support=cohort['supports']['assessment'][i],
  primary_f1={k:v['primary_f1'][i] for k,v in rows.items()},
  Q_minus_Gaussian_macro_contribution=(rows['quantum']['primary_f1'][i]-rows['gaussian_base']['primary_f1'][i])/40)
 if i:
  item['forced_recall']={k:v['positive39_confusion'][i-1][i-1]/cohort['supports']['assessment'][i] for k,v in rows.items()}
 classes.append(item)
edges=[]
for i in range(1,40):
 for j in range(1,40):
  if i==j:continue
  counts={k:v['positive39_confusion'][i-1][j-1] for k,v in rows.items()}
  if any(counts.values()):
   edges.append(dict(true_class=i,true_label=names[i],predicted_class=j,predicted_label=names[j],
    counts=counts,Q_errors_corrected={k:int(v[i-1,j-1]) for k,v in corrected.items()}))
edges.sort(key=lambda x:(-x['counts']['quantum'],x['true_class'],x['predicted_class']))
print(json.dumps(dict(status='completed',labels=names,label_contract_sha256=hashlib.sha256(raw).hexdigest(),
 cohort_sha256=hashlib.sha256(cb).hexdigest(),rows=rows,classes=classes,paired=pairs,all_observed_directed_errors=edges,
 new_fits=0,new_forward_passes=0,new_heldout_reads=0,cos_writes=0,
 scope='Post-outcome descriptive all-class TRAIN4096 map; positive-conditioned ranking uses gold labels, not a deployed classifier or primary-gate replacement')))
'''
out=json.loads(ops.remote(['/hy-tmp/Q-Attention/env/bin/python','-B','-c',code],data=json.dumps(request).encode(),timeout=30))
assert set(out['rows'])=={'quantum','dephased','gaussian_base','gaussian_covariance'}
maximum=0.
for item in request:
    s=json.loads((r.parent/item['name']/'summary.json').read_bytes())
    for mode in item['modes']:
        row=out['rows'][mode];prior=s['results'][mode]
        assert row['primary_cm']==prior['confusion_matrix']
        maximum=max(maximum,abs(row['primary_macro']-prior['all40_macro_f1']))
        assert sum(map(sum,row['positive39_confusion']))==1365
        assert row['positive_forced_correct']==sum(row['positive39_confusion'][i][i] for i in range(39))
assert maximum<1e-12
assert abs(sum(v['Q_minus_Gaussian_macro_contribution'] for v in out['classes'])-(out['rows']['quantum']['primary_macro']-out['rows']['gaussian_base']['primary_macro']))<1e-12
out['input_private_diagnostic_sha256']={v['name']:v['sha256'] for v in request}
out['analysis_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
save('confusion-map.json',out)
save('verification.json',dict(status='PASS',four_primary_confusion_matrices_equal_source=True,
    max_macro_error=maximum,positive_population=1365,all39_by39_accounted=True,
    identities_match_fixed_cohort=True,no_windows_task_arrays=True))
print(json.dumps(dict(top_Q_directed_confusions=out['all_observed_directed_errors'][:12],
    largest_negative_Q_macro_classes=sorted(out['classes'],key=lambda x:x['Q_minus_Gaussian_macro_contribution'])[:12],
    primary_macros={k:v['primary_macro'] for k,v in out['rows'].items()})))
