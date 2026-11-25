from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
r=Path(__file__).resolve().parent
old=Path('D:/CodexHome/task-artifacts/qattn-native-phase-construction-20261126')
source=Path('D:/CodexHome/task-artifacts/qattn-residual-fastweights-source-20261126')
prior=Path('D:/CodexHome/task-artifacts/qattn-tangent-label-diagnostic-20261126-v2')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins=[(Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py'),'9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'),
 (old/'trace.json','f3ec867c781826f6e8fff8fcf8f9f5476e49610b484101cfba863cccb6a26a8b'),
 (old/'native-stage.json','ec3965de5ef4b5b0f7436942bf1a941658d57f133b3cb3ea175dae4d07027389'),
 (old/'independent_reference.py','68d16e11a70afc8fc38ee050db86faa4a2bcaf43ea9e058f05a95cf6080d04dd'),
 (source/'fulltexts/8fcb56c452e49c5e/028b1b47b1d86d0e8e2340cbc07a290f9e477c0a5b1ed439437de6a4a1cecc63.pdf','028b1b47b1d86d0e8e2340cbc07a290f9e477c0a5b1ed439437de6a4a1cecc63')]
for p,h in pins:assert sha(p)==h,p
assert not (r/'scope.json').exists()
(r/'independent_reference.py').write_bytes((old/'independent_reference.py').read_bytes())
(r/'source').mkdir(exist_ok=True)
for n in ('validate_sample_trace_v2.py','sample-trace.v2.schema.json'):(r/'source'/n).write_bytes((prior/'source'/n).read_bytes())
scope=dict(schema='qattn.native-fastweights-class-audit.v1',recorded_at=datetime.now(timezone.utc).isoformat(),
 pins=[dict(path=str(p),sha256=h) for p,h in pins],
 source_facts='Conference pages3/4 Eq1-5 reread; page4 visually reinspected. h0=ReLU(b); hs+1=ReLU(LN(b+A hs)); A=lambda Aold+eta h hT. No original-code reproduction or paper-result claim.',
 reading=dict(PDF_sha256=pins[-1][1],text_pages=[3,4],visual_pages=[4],visual_sha256=sha(r/'fastweights-p4.private.png'),
              prior_full_reading_receipt=str(source/'reading-record.json'),prior_full_reading_receipt_sha256=sha(source/'reading-record.json')),
 frozen=dict(dimension=8,eta=.5,decay=.95,inner_steps=1,layernorm_epsilon=1e-5,
     gamma='sqrt(1+epsilon)/sqrt(7) on all8 units',beta=0,W=0,C='I8',
     cases='Two complete2-write+1-read FW trajectories: stored A=[e0,e2], B=[e0,e0]; same read boundary(e0+e2)/sqrt2. Write boundaries inverse-solved from specified targets, not optimized.',
     bridge='Reuse original native pair, fixed Hadamard on basis0/2 maps plus->e0/minus->e2, querye0->(e0+e2)/sqrt2. No native circuit execution; transformed states not claimed native ansatz closure.',
     algebra='24 deterministic cases: T in1,2,7; 4 normalized complex trig constructions and4 real constructions perT. All40 queries, EPS1e-6, signed utilities; compare direct Born pool, Hermitian64, real16 two-overlap. Native-angle complex counterexample:firstRYpi/2,firstRZ±pi/2,allotherangles0 via previously verified dense circuit.',
     criterion='Numerical identities<=1e-12+1e-11*scale. FW trajectory nonnegative and targetwrite states<=1e-12, memories PSD, finalshared readout e2-e0 opposite nonzero signs. Exact Fraction margins verify this sign analytically. No parameter/epsilon/seed search.',
     budget='Local QT NumPy/stdlib only; <=10 processCPU seconds; no new native code call/task/checkpoint/GPU/network/API/server/COS.'),
 boundaries=['Not an equivalence or containment proof between general recurrent FW and Q-RMEA','A complete Eq5 instantiation, not upstream-code/training reproduction','No task utility/quantum necessity/new T/L3/L4-T from a synthetic separation','Exact classical simulation is not a quantum rejection criterion','gen104 C-exponential self-direction remains stopped; no new flow execution or tuning'],
 scientific_method_admission=False,goal='ACTIVE')
(r/'scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='FROZEN_SOURCE_CLASS_AUDIT',sha256=sha(r/'scope.json'),inputs=len(pins),FW_cases=2,algebra_cases=24,new_task_calls=0)))
