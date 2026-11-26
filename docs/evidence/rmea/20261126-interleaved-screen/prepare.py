from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,x:p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
assert not (r/'selection.json').exists()
base=root/'.codex-artifacts/distinct-token-evidence-20261125'
old=json.loads((base/'design.json').read_bytes())
shutil.copyfile(base/'screen.py',r/'screen.py')
assert sha(r/'screen.py')=='50ce858f78938151b87ac18711efff388fb0456a06c21df90108a7edea6933ab'
sources={n:root/'.codex-artifacts'/n for n in ('context-spectral-flow-20261125','staged-bottleneck-design-20261125','connected-value-pooling-20261125','distinct-token-evidence-20261125')}
refs={};pins={}
for folder,arms in [
 ('context-spectral-flow-20261125',['quantum','dephased']),
 ('staged-bottleneck-design-20261125',['real12','gaussian_base']),
 ('connected-value-pooling-20261125',['gaussian_covariance','gaussian_mean_product']),
 ('distinct-token-evidence-20261125',['gaussian_distinct'])]:
 d=json.loads((sources[folder]/'design.json').read_bytes())
 s=json.loads((sources[folder]/'summary.json').read_bytes())
 for k in ('seed','updates','batch_size','microbatch_size','lr','weight_decay','betas','optimizer_eps','clip_norm','cohort_sha256','training_dtype','dense_batches'):
  assert d[k]==old[k],(folder,k)
 pins[folder]={n:sha(sources[folder]/n) for n in ('design.json','plugin.py','screen.py','summary.json')}
 for arm in arms:
  row=dict(s['results'][arm]);row.update(source=folder,source_summary_sha256=pins[folder]['summary.json'],
       comparability='Same cohort/order/optimizer/budget/shared value-decoder initial tensors. Router mapping and added moment parameters differ as declared; not equal effective capacity.')
  refs[folder+'/'+arm]=row
s=json.loads((base/'summary.json').read_bytes())
row=dict(old['reused_native_rows']['previous/quantum_shared_mean_product'])
row.update(source='semantic-connected-evidence-20261125',comparability='Same common value/decoder and training profile; shared query plus moment head, 81924 parameters. Strong reference, not exact architecture match.')
refs['semantic-connected-evidence-20261125/quantum_shared_mean_product']=row
cfg={k:v for k,v in old.items() if k in ('seed','updates','batch_size','microbatch_size','lr','weight_decay','betas','optimizer_eps','clip_norm','fit','assessment','precision','bootstrap_replicates','bootstrap_seed','native_manifest_sha256','split_reference_sha256','bootstrap_code_sha256','source_hashes','training_dtype','dense_batches','source','cohort_file','cohort_sha256','cohort_code_sha256')}
cfg.update(profile='interleaved-class-circuit-TRAIN8192-seed13-v1',modes=['quantum_middle','dephased_middle','real8_middle'],
 plugin_module='adapter.py',fixture_real_labels=True,selection='All three fixed fits before final1024/all4096/all40 assessment; ties fail.',
 timeout_seconds=3600,arm_timeout_seconds=900,physical_optimizer_updates=3072,
 expected_trainables=81028,expected_initial_sha256='9248260cf3c68fba44bba6647deeaf31840ccff81bd303e29708f047c2bf4722',
 order_sha256='b12bbb2eaa7d60eab19891ffb7954491e09cc1b319b751d7bc179bea8ca7c67f',
 references=refs,reference_pins=pins,assessment_class_support=next(iter(refs.values()))['class_support'],
 gates='Q strictly above Dmiddle, real8middle and all eight declared retained references on all40 Macro; CE secondary.',
 stop='Any critical failure closes exact placement/ansatz/budget. No position, depth, seed, temperature, cohort, budget or comparator rescue.',
 recovery='Disposable short screen, accepted input/local best/latest; no automatic retry or new COS.',
 public_whitelist=['summary.json','fixtures.json','RUN_COMPLETE'],
 private_outputs=['diagnostic.private.npz','perarm/latest.pt','perarm/best.pt','perarm/final.pt','perarm/result.json'],
 resources=dict(gpus=2,gpu_mb=2048,cpu=4,ram_mb=8192,disk_mb=256),
 initialization='Exact native seed13 RMEAHead FP32 tensors copied into all three heads, including prepare/effect and shared value/decoder/bias. real8 uses same coordinates in the fixed alternate RY layer geometry.',
 effective_capacity='Q last-word three RZ rows have zero route gradient; D also first-word three RZ and final-query-layer three RZ. AdamW may still decay these. real8 changes gate count/geometry; registered 81028 is not effective-capacity equivalence.',
 angle_mapping='Q/D native [layer,wire,RY/RZ]. real8 flattened word first6 -> two RY-ring layers, last6 -> two RY-ring layers; relation12 -> four RY-ring layers. True matrix adjoint/transpose.',
 claim_scope='Development efficacy only; original scoped L4 retained; new L3/L4-T NOT_ACCEPTED',
 training_admitted=False)
cfg['code_hashes']={n:sha(r/n) for n in ('screen.py','plugin.py','adapter.py')}
assert len(refs)==8
write(r/'design.json',cfg)
for n in ('model_preflight.py','consult.py','execute.py'):
 shutil.copyfile(r.parent/n,r/n)
instructions='''Review this implementation of the already provisional middle-placement hypothesis, not a new design. Return one JSON object: verdict PASS_WITH_RISK or BLOCK; findings [{severity, file, location, issue, required_action}]; permitted_next; limits. Check native gate direction, full Q adjoint, terminal bra phase removal, exact D boundary formula, real8 fixed coordinate mapping/true transpose, gradients including null coordinates, original state/utility/EPS schema, common initialization, and adapter checks. No task efficacy claims. This is R2 scientific mechanism review before executing changed numerical code. Previous math-only PASS_WITH_RISK does not cover this code. Stop conditions bind; no fallback on substantive BLOCK. Assess whether code can proceed to affected CPU numerical/gradient checks and a cloned eight-real-TRAIN-row update, then prospective fixed screen admission if checks pass. No new runner lifecycle audit is needed. References are old exposed-TRAIN development; strongest Gcov has 82308 registered parameters versus new81028 and different router parameterization. Old native Q is exact placement ablation with same tensors; other reference schema/key naming differs and must be checked by original initializers before reuse. Screen not yet admitted. Call out any substantive defect precisely; do not invent positive efficacy or family separation.'''
files=[r/'plugin.py',r/'adapter.py',r/'design.json',r.parent/'interleaved_math.py',r.parent/'hypothesis.md',r.parent/'primary-adoption.md',
       Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py'),
       Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea_matched_routers.py')]
content=[]
for p in files:
 if p.name=='design.json':
  x=dict(cfg);x['references']={k:{n:v for n,v in row.items() if n not in ('confusion_matrix','per_class_f1','train_ce_history')} for k,row in refs.items()};body=json.dumps(x,ensure_ascii=False,indent=2)
 else:body=p.read_text(encoding='utf-8')
 content.append(dict(path=str(p),sha256=sha(p),content=body))
packet=dict(instructions=instructions,files=content,previous_review=json.loads((r.parent/'review.json').read_bytes())['response'])
write(r/'review-packet.private.json',packet)
selection=json.loads((r.parent/'selection.json').read_bytes())
selection.update(packet_sha256=sha(r/'review-packet.private.json'),metadata_preflight=str(r/'preflight.json'),
 review_scope='Focused actual scientific plugin/adapter mechanism delta',dispatch_budget=1)
write(r/'selection.json',selection)
print(json.dumps(dict(status='PLUGIN_CONFIG_AND_BOUNDED_REVIEW_PREPARED',plugin_sha256=sha(r/'plugin.py'),
 references=len(refs),packet_bytes=(r/'review-packet.private.json').stat().st_size,training_admitted=False)))
