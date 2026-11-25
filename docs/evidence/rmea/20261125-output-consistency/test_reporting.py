"""Exercise exact config + summary with synthetic predictions, not task outcomes."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r.parent/'qattn-relation-frame-20261125/source'))
from adapter import summarize,MODES
cfg=json.loads((r/'design.json').read_bytes())
assert len(cfg['source_hashes'])==24
assert set(cfg['code_hashes'])=={'screen.py','plugin.py','adapter.py','gaussian_reference.py','consistency.py'}
for name,sha in cfg['code_hashes'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha
refs=cfg['reused_native_rows']
labels=np.repeat(np.arange(40),refs['quantum']['class_support'])
assert labels.shape==(4096,)
assert all(len(v['class_support'])==40 for v in refs.values())
order=refs['quantum']['order_sha256']
rows={k:dict(order_sha256=order,objective_components={'updates':1024}) for k in MODES}
poor=np.zeros((4096,40));poor[:,0]=1
perfect=np.zeros_like(poor);perfect[np.arange(4096),labels]=1
logits={k:poor.copy() for k in MODES};logits['quantum_vat']=perfect.copy()
passing=summarize(copy.deepcopy(rows),logits,labels,cfg)
assert all(passing['gates'].values()) and len(passing['gates'])==7
logits['quantum_random_consistency']=perfect.copy()
tie=summarize(copy.deepcopy(rows),logits,labels,cfg)
assert not tie['gates']['macro_beats_quantum_random_consistency']
assert tie['gate']=='DEFER_EXACT_OUTPUT_CONSISTENCY_SCREEN'
v=dict(status='PASS',scope='Synthetic predictions on invented order with published supports; no task efficacy',
    exact_design_sha256=hashlib.sha256((r/'design.json').read_bytes()).hexdigest(),
    source_hashes_present=24,reference_support_lengths=[len(v['class_support']) for v in refs.values()],
    all_seven_strict_gates_exercised=True,tie_fails=True,
    review_packet_correction='Previous review-runtime packet mislabeled a compact excerpt as design.json. Actual exact file always included source_hashes and class_support. Preserve BLOCK, send complete exact config next.')
(r/'reporting-verification.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v))
