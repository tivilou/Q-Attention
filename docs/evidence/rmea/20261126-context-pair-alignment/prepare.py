from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
prior=Path('D:/CodexHome/task-artifacts/qattn-native-fastweights-class-audit-20261126')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
pins=read(prior/'next-direction.json')['closed_route_pins']
for x in pins:assert sha(x['path'])==x['sha256']
extra=[
 root/'.codex-artifacts/distinct-token-evidence-20261125/moment_head.py',
 root/'.codex-artifacts/context-frame-training-20261124/router.py',
 root/'.codex-artifacts/context-frame-training-20261124/execution-compact.json',
 root/'.codex-artifacts/context-frame-training-20261124/last-observation.json',
 root/'.codex-artifacts/native-shared-query-learning-20261124/gradients.py',
 root/'.codex-artifacts/native-shared-query-learning-20261124/protocol.md',
 root/'.codex-artifacts/native-shared-query-learning-20261124/summary.json',
 Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py'),
 Path('D:/CodexHome/task-artifacts/qattn-native-phase-construction-20261126/native-stage.json'),
 Path('D:/CodexHome/task-artifacts/qattn-native-mechanism-discovery-20261125/feature-strata.json'),
 prior/'headroom-reuse-audit.json',prior/'results.json',
]
for p in extra:
 assert p.is_file(),str(p)
 if p.suffix=='.py':ast.parse(p.read_text('utf8'))
 pins.append(dict(path=str(p),sha256=sha(p)))
assert pins[7]['sha256']=='88056fd0c023acfa2be868ed8e63a9d00716305b1cd6917d2dc0314a3341f0b6'
assert sha(extra[7])=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
assert sha(extra[8])=='ec3965de5ef4b5b0f7436942bf1a941658d57f133b3cb3ea175dae4d07027389'
scope=dict(status='FROZEN_SOURCE_EXCLUSION_AND_SYNTHETIC_DIAGNOSTIC_QUALIFICATION',
 created_at=datetime.now(timezone.utc).isoformat(),pins=pins,
 action='Read six exact pinned route definitions; distinguish a single input/state pairing diagnostic from stopped interventions, then test its algebraic sufficiency limits.',
 resources=dict(GPU=0,new_task_forward=0,native_head=0,native_circuit=0,independent_circuit=0,optimizer_updates=0,model_API=0,server=0,new_capacity=0),
 fixture='Two predetermined four-word synthetic contexts. No labels, task arrays, checkpoints, circuit evaluation or training. Saved source-bound native plus/minus states reused.',
 tolerance=1e-12,alpha=.1,beta=.5,states='plus,plus,minus,minus on basis0 and2',
 prediction_status='PROVISIONAL_DIAGNOSTIC_ONLY_NOT_A_SELECTED_PLUGIN',
 retained_stops=['gen98','gen102','gen104','all prior exact stopped profiles'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'scope.json').write_text(json.dumps(scope,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=scope['status'],verified_source_pins=len(pins))))
