"""Retrospective classwise arithmetic on sealed fresh-five VALID reports only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

r=Path(__file__).resolve().parent
source=r.parents[1]/'qattn-rmea-implementation-20261026/fresh5-report-audit-20261030'
report=source/'report/summary.json';run=json.loads((source/'report/run_summary.json').read_bytes())
assert hashlib.sha256(report.read_bytes()).hexdigest()==run['sha256']['summary.json']
cfg=dict(scope='Retrospective exploratory development VALID, no TEST or new predictions/training',
    seeds=[29,53,71,97,113],conditions=['uniform','relation_softmax','real_overlap','quantum','dephased'],
    endpoint='selected_valid as already frozen independently per existing condition',
    metric='per_class F1 recomputed from stored confusion; positive39 Macro',
    stability='strictly positive / negative in all five seeds, tolerance1e-12; descriptive, not significance',
    comparisons=['quantum-uniform','quantum-relation_softmax','quantum-real_overlap','quantum-dephased'],
    classes='all positive class IDs1..39 without exclusions',input_sha256=run['sha256']['summary.json'],
    historical_audit_sha256=hashlib.sha256((source/'audit.json').read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_at=datetime.now(timezone.utc).isoformat())
assert not (r/'historical-class-protocol.json').exists()
(r/'historical-class-protocol.json').write_bytes((json.dumps(cfg,indent=2)+'\n').encode())
v=json.loads(report.read_bytes());assert sorted(map(int,v['per_seed']))==cfg['seeds']
assert not v['discovery_seed13_included']
fs={};macros={};reference=None;max_error=0.
for seed in cfg['seeds']:
    row=v['per_seed'][str(seed)];assert row['full_data_training'] and not row['test_selection']
    labels=row['labels'];assert labels[0]=='no_relation' and len(labels)==40
    if reference is None:reference=labels
    assert labels==reference
    fs[seed]={};macros[seed]={};support=None
    for name in cfg['conditions']:
        m=row['results'][name]['selected_valid'];cm=np.asarray(m['confusion_matrix'],np.int64)
        assert cm.shape==(40,40) and cm.sum()==19584
        if support is None:support=cm.sum(1)
        assert np.array_equal(support,cm.sum(1))
        den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        error=float(np.max(np.abs(f-np.asarray([c['f1'] for c in m['per_class']]))))
        assert error<1e-12;max_error=max(max_error,error);fs[seed][name]=f;macros[seed][name]=float(f[1:].mean())
classes=[]
for i in range(1,40):
    out=dict(class_id=i,label=reference[i],support=int(support[i]),contrasts={})
    for baseline in cfg['conditions']:
        if baseline=='quantum':continue
        d=[float(fs[s]['quantum'][i]-fs[s][baseline][i]) for s in cfg['seeds']]
        out['contrasts'][baseline]=dict(deltas=d,mean=float(np.mean(d)),positive_seeds=sum(x>1e-12 for x in d),
            negative_seeds=sum(x < -1e-12 for x in d),stable_positive=all(x>1e-12 for x in d),stable_negative=all(x < -1e-12 for x in d))
    classes.append(out)
summary={}
for baseline in cfg['conditions']:
    if baseline=='quantum':continue
    summary[baseline]=dict(stable_positive=[x['label'] for x in classes if x['contrasts'][baseline]['stable_positive']],
        stable_negative=[x['label'] for x in classes if x['contrasts'][baseline]['stable_negative']],
        macro_deltas=[macros[s]['quantum']-macros[s][baseline] for s in cfg['seeds']])
out=dict(protocol=cfg,summary=summary,classes=classes,per_seed_positive_macro=macros,
    verification=dict(status='PASS',stored_confusions_recomputed=25,max_per_class_error=max_error,all_class_supports_match=True),
    claim_ceiling='Descriptive historical VALID stability only. No causal group, new utility claim, independent confirmation, multiple-testing significance or quantum necessity.')
(r/'historical-class-results.json').write_bytes((json.dumps(out,indent=2,ensure_ascii=False)+'\n').encode())
print(json.dumps(dict(summary=summary,verification=out['verification']),ensure_ascii=False))
