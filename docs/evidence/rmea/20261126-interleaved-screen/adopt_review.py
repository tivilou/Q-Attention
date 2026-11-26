from pathlib import Path
import difflib,hashlib,json
r=Path(__file__).resolve().parent;root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,x:p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
assert not (r/'review-adoption.json').exists()
review=json.loads((r/'review.json').read_bytes());packet=json.loads((r/'review-packet.private.json').read_bytes())
assert review['response']['verdict']=='PASS_WITH_RISK'
old=next(x for x in packet['files'] if Path(x['path']).name=='adapter.py')
(r/'adapter-reviewed.py').write_bytes(old['content'].encode())
(r/'adapter-checks.diff').write_text(''.join(difflib.unified_diff(old['content'].splitlines(True),(r/'adapter.py').read_text().splitlines(True),fromfile='adapter-reviewed.py',tofile='adapter.py')),encoding='utf-8')
cfg=json.loads((r/'design.json').read_bytes())
folder='semantic-connected-evidence-20261125';base=root/'.codex-artifacts'/folder
cfg['reference_pins'][folder]={n:sha(base/n) for n in ('design.json','plugin.py','screen.py','summary.json','shared_query.py','label-codes.json')}
for folder in ('semantic-connected-evidence-20261125','distinct-token-evidence-20261125'):
 base=root/'.codex-artifacts'/folder
 for n in ('shared_query.py','label-codes.json'):
  cfg['reference_pins'][folder][n]=sha(base/n)
cfg['reference_pins']['distinct-token-evidence-20261125']['moment_head.py']=sha(root/'.codex-artifacts/distinct-token-evidence-20261125/moment_head.py')
for k,row in cfg['references'].items():
 base=root/'.codex-artifacts'/row['source']
 source=json.loads((base/'summary.json').read_bytes())
 arm=k.split('/')[-1]
 original=source['results'][arm]
 for field in ('initial_sha256','order_sha256','updates','parameters','all40_macro_f1','class_support','confusion_matrix','final_sha256'):
  assert row[field]==original[field],(k,field)
 row['source_summary_sha256']=sha(base/'summary.json')
 if k=='context-spectral-flow-20261125/quantum':
  row['comparability']='Exact native-placement ablation: full initial state must equal new heads tensor for tensor. Original terminal Qdag W2 W1 with same prepare/effect/value/decoder/bias, cohort, order and budget.'
cfg['code_hashes']['adapter.py']=sha(r/'adapter.py')
cfg['training_admitted']=False
write(r/'design.json',cfg)
adoption=dict(status='ADOPTED_CONDITIONALLY_FOR_AFFECTED_CHECKS',review_sha256=sha(r/'review.json'),
 plugin_unchanged_since_review=True,adapter_reviewed_sha256=old['sha256'],adapter_after_sha256=sha(r/'adapter.py'),
 delta='Verification only: assert exactly eight rows and actual FP32 parameter change. No change to formulas, model initialization, loss, optimizer or efficacy gates. Add missing source pins and explicit native placement comparability.',
 required=['FP64/FP32 independent operator parity, phase-restored complex amplitude, explicit density D, real8 ordered matrix oracle',
           'Native initial identity/full tensor schema and eight old initializer identities/shared tensor equality',
           'Eight actual TRAIN rows CPU clone updates in both precisions, originals unchanged',
           'Fresh remote exact source/input/cohort/reference binding before fixed-screen admission'],
 training_admitted=False)
write(r/'review-adoption.json',adoption)
print(json.dumps(dict(status=adoption['status'],references_verified_against_original_summaries=len(cfg['references']),adapter_sha256=adoption['adapter_after_sha256'])))
