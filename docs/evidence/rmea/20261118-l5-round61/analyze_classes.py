"""Descriptive reuse of all frozen report cells; no model, raw data or p-values."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();ROOT=args.root.resolve(strict=True);P=args.output.resolve();P.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(P/n).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
audit=json.loads((ROOT/'docs/evidence/rmea/20261117-l5-round51/completed-cohort-audit.json').read_bytes())
for n,h in audit['hashes'].items():assert sha(ROOT/n)==h,n
path=ROOT/'.codex-artifacts/qattn-rmea-implementation-20261026/fresh5-report-audit-20261030/report/summary.json'
summary=json.loads(path.read_bytes());seeds=[29,53,71,97,113]
labels=summary['per_seed']['29']['labels'];nil=labels.index('no_relation');modes=['quantum','relation_softmax','real_overlap']
rows=[];flows=[]
for split in ['selected_valid','test']:
 for i,label in enumerate(labels):
  metrics={};support=None
  for mode in modes:
   values=[]
   for seed in seeds:
    m=summary['per_seed'][str(seed)]['results'][mode][split];cm=m['confusion_matrix'];n=sum(cm[i]);pred=sum(r[i] for r in cm);tp=cm[i][i]
    assert support is None or support==n;support=n
    f1=2*tp/(n+pred) if n+pred else 0.;assert math.isclose(f1,m['per_class'][i]['f1'],abs_tol=1e-12)
    values.append({'seed':seed,'f1':f1,'recall':tp/n if n else 0.,'precision':tp/pred if pred else 0.,'predicted':pred,
      'miss_to_null':cm[i][nil] if i!=nil else 0,'false_from_null':cm[nil][i] if i!=nil else 0})
   metrics[mode]={'per_seed':values,'means':{k:statistics.mean(v[k] for v in values) for k in ['f1','recall','precision','predicted','miss_to_null','false_from_null']}}
  contrasts={}
  for comparator in modes[1:]:
   delta=[q['f1']-c['f1'] for q,c in zip(metrics['quantum']['per_seed'],metrics[comparator]['per_seed'])]
   contrasts[comparator]={'f1_differences':delta,'mean':statistics.mean(delta),'positive':sum(d>0 for d in delta),'negative':sum(d<0 for d in delta),'ties':sum(d==0 for d in delta)}
  rows.append({'split':split,'label':label,'support':support,'metrics':metrics,'contrasts':contrasts})
 for mode in modes:
  cells=[]
  for seed in seeds:
   m=summary['per_seed'][str(seed)]['results'][mode][split];cm=m['confusion_matrix'];positive=sum(sum(cm[i]) for i in range(40) if i!=nil)
   miss_null=sum(cm[i][nil] for i in range(40) if i!=nil);wrong_positive=sum(cm[i][j] for i in range(40) for j in range(40) if i!=nil and j!=nil and j!=i)
   f1=statistics.mean(next(r for r in rows if r['split']==split and r['label']==label)['metrics'][mode]['per_seed'][seeds.index(seed)]['f1'] for label in labels)
   assert math.isclose(f1,m['macro_f1'],abs_tol=1e-12)
   cells.append({'seed':seed,'positive_count':positive,'positive_miss_to_null':miss_null,'positive_wrong_positive':wrong_positive,'null_false_positive':sum(cm[nil][j] for j in range(40) if j!=nil),'positive_errors':miss_null+wrong_positive,'positive_miss_to_null_fraction':miss_null/positive,'miss_to_null_fraction_of_positive_errors':miss_null/(miss_null+wrong_positive),'macro_f1':f1,'ce_reported_not_recomputed':m['loss']})
  flows.append({'split':split,'mode':mode,'cells':cells,'means':{k:statistics.mean(c[k] for c in cells) for k in cells[0] if k!='seed'}})
stable={}
for split in ['selected_valid','test']:
 relevant=[r for r in rows if r['split']==split and r['support']>0]
 stable[split]={c:{sign:[r['label'] for r in relevant if r['contrasts'][c][sign]==5] for sign in ['positive','negative']} for c in modes[1:]}
stable['both_splits_both_comparators']={sign:[label for label in labels if all(label in stable[split][mode][sign] for split in ['selected_valid','test'] for mode in modes[1:])] for sign in ['positive','negative']}
x={'schema':'qattn.all-class-matched-error-audit.v1','status':'PASS_DESCRIPTIVE_ONLY','analysis_contract':'All40labels/all5seeds/3matched-input-budgetheads; both validation-selected and test; no sorting/exclusion for inferential claims, no p-values, heldout reports already exposed.','input_sha256':sha(path),'prior_audit_sha256':sha(ROOT/'docs/evidence/rmea/20261117-l5-round51/completed-cohort-audit.json'),'rows':rows,'aggregate_flows':flows,'same_sign_all5':stable,'limits':['Validation used for epoch selection, test already exposed; seed consistency is hypothesis generation only.','Confusion aggregates do not identify paired per-example errors, hidden states, calibration or causal mechanisms.','Head parameter counts differ, despite common input/training/value-decoder initialization.','Absent/rare label support limits interpretation; full40 denominator unchanged.','No novel method or L5 follows from class-pattern selection.'],'model_jobs':0,'raw_data_reads':0,'cos_requests':0,'l5':'NOT_ACCEPTED'}
save('class-error-audit.json',x)
print(json.dumps({'rows':len(rows),'aggregate_flows':[{'split':f['split'],'mode':f['mode'],**f['means']} for f in flows],'same_sign_both_splits_both_comparators':stable['both_splits_both_comparators'],'test_stable_counts':{c:{s:len(v) for s,v in d.items()} for c,d in stable['test'].items()}}))
