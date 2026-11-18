"""Recompute the frozen criterion and compact aggregates without raw inputs."""
import json,math
from pathlib import Path
def main():
 p=Path(__file__).resolve().parent;x=json.loads((p/'probe-results.json').read_bytes());rows=x['cells']
 assert len(rows)==5 and [r['seed'] for r in rows]==[29,53,71,97,113]
 for r in rows:
  assert r['original_confusion_exact'] and r['split']=='original_validation_development_only'
  probes=r['probes'];assert set(probes)=={'score_linear','score_quadratic','score_dispersion'}
  assert r['fit_entries']>=100 and r['eval_entries']>=100
  for name,a in probes.items():
   f=a['fit'];assert f['penalty']==.01 and f['fit_entries']==r['fit_entries'] and f['gradient_infinity_norm']<=1e-8
   assert f['input_features']==(40 if name=='score_linear' else 80) and f['fitted_parameters']==f['input_features']+1
   assert all(math.isfinite(v) and v>=0 for v in a['metrics'].values())
   d=a['deployment'];assert d['triggered']==d['correct_positive_recoveries']+d['true_null_false_positives']+d['positive_wrong_label_reassignments']
   assert d['eligible_positive']==r['eval_positive'] and d['true_null']+d['eligible_positive']==r['eval_entries']
   assert d['correct_positive_recoveries']+d['positive_wrong_label_reassignments']<=d['eligible_positive'] and d['true_null_false_positives']<=d['true_null']
  for name,suffix in [('score_linear','linear'),('score_quadratic','quadratic')]:
   assert r['logloss_improvement_vs_'+suffix]==probes[name]['metrics']['logloss']-probes['score_dispersion']['metrics']['logloss']
 expected='ALL5_INCREMENTAL_LOGLOSS' if all(r['logloss_improvement_vs_linear']>0 and r['logloss_improvement_vs_quadratic']>0 for r in rows) else 'INCREMENTAL_CRITERION_NOT_MET'
 assert x['criterion']==expected
 total=sum(r['eval_entries'] for r in rows);assert total==x['eval_seed_example_pairs']
 for name,s in x['summary'].items():
  for metric in ['logloss','brier']:assert s['pooled_'+metric]==sum(r['eval_entries']*r['probes'][name]['metrics'][metric] for r in rows)/total
  for k,v in s['deployment'].items():assert v==sum(r['probes'][name]['deployment'][k] for r in rows)
 assert x['probe_fit'] and not x['head_training'] and not x['original_test_evaluation']
 assert x['candidate']=='NONE' and x['L5']=='NOT_ACCEPTED' and not x['independent_examples']
 print(json.dumps(dict(status='PASS_PUBLIC_PROBE_RECOMPUTATION',criterion=expected,cells=5,L5='NOT_ACCEPTED')))
if __name__=='__main__':main()
