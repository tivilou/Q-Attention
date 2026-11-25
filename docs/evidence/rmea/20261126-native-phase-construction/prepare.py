from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,sys,torch,numpy
r=Path(__file__).resolve().parent
previous=Path('D:/CodexHome/task-artifacts/qattn-residual-fastweights-source-20261126/next-direction.json')
native=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred')
old=Path('D:/CodexHome/task-artifacts/qattn-value-guided-effect-preflight-20261126')
def sha(b):return hashlib.sha256(b).hexdigest()
def write(n,x):(r/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
assert not (r/'receipt.before-execution.json').exists()
n=json.loads(previous.read_bytes());assert n['goal']=='ACTIVE' and n['minimum']['contexts']==2
assert sha((native/'rmea.py').read_bytes())==n['source']['sha256']
assert '2.8.0' in torch.__version__ and numpy.__version__=='1.26.4'
scope=dict(status='FROZEN_BEFORE_NATIVE_EXECUTION',parent=str(previous),parent_sha256=sha(previous.read_bytes()),
 purpose='Joint constructive native implementation reachability only; preserve HOLD and unresolved Knyazev original.',
 synthetic_input=dict(h_shape=[2,2,1024],roles_shape=[2,2,4],mask_shape=[2,2],
                      h_fill=0,h0=[[math.atanh(.5),math.atanh(.5)],[math.atanh(.5),-math.atanh(.5)]],
                      h1=[[-1,1],[-1,1]],h2=[[2,-3],[2,-3]],roles_fill=0,mask_fill=True),
 parameters=dict(all_fill=0,assignments={'prepare.weight[0,0]':1,'value.weight[0,1]':1,
                                      'value.weight[1,2]':1,'decoder[0,0]':1},
                 all_other_class_utilities=0,all_class_biases=0,all_effect_angles=0,
                 requires_grad=False,constructor_seed=0,
                 seed_meaning='Only deterministic constructor bookkeeping; every parameter overwritten, no training seed or selection.'),
 expected=dict(native_basis_support=[0,2],states='contextA plus/plus;contextB plus/minus',
               query='e0 for all40 classes',probability=.5,attention=.5,
               native_all40_logits=0,guided_class0_A=0,guided_class0_B=math.tanh(2)/(1+2e-6),
               guided_other39_logits=0,
               basis_relation_to_prior='Prior generic basis1 is mapped to native basis2; fixed isometry preserving all operator/readout relations.'),
 fixed=dict(EPS=1e-6,strength=1,tolerance=1e-12,device='cpu',real_dtype='float64',complex_dtype='complex128'),
 exact_calls=dict(native_head_forwards=2,native_circuit_calls=4,
                  call1='quantum, batch2contexts x2words, all40classes, details=True',
                  call2='same parameter object switched to dephased, same batch, details=True',
                  independent_dense_circuit_states=44,independent_spectral_rows=80,
                  primary_matrix_exponential_rows=80,optimizer_steps=0),
 budget=dict(max_native_head_forwards=8,max_native_circuit_calls=16,max_internal_CPU_seconds=10,
             measured_interval='Before synthetic parameter/input construction until final gate metrics, after imports/thread setup; artifact serialization excluded.'),
 gates=['all source hashes unchanged','explicit parameter/input construction and constant parameters',
        'CPU FP64/complex128 shapes','exact2head4circuit calls',
        'native state/query vs dense reference and analytic support/norm error<=1e-12',
        'native EPS probability/attention/readout vs independent original ratio<=1e-12',
        'two contexts same native alpha/all64value/populations<=1e-12',
        'Q/D all40 original logits0 and same probability<=1e-12',
        'fixed-C guided0/.9640256520 and native/reference agreement<=1e-12;other39zero',
        'internalCPU<=10seconds'],
 stop='Any scientific gate fail stops this construction claim; no parameter/strength/norm/EPS/rank/seed rescue or task admission.',
 nonclaims=['General gate-family/gradient certification','arbitrary40class query joint reachability',
            'trained checkpoint or natural text reachability/frequency','task Macro/margin benefit',
            'quantum necessity','substantive neighbor clearance','new theory/L3/L4T'],
 resources=dict(task_rows=0,feature_reads=0,checkpoint_reads=0,optimizer_updates=0,GPU_allocations=0,
                remote_server_writes=0,COS=0,model_API=0,source_network_requests=0,package_installs=0),
 preserved=['original limitedL4','all positive/negative history','gen98 STOP','gen102 STOP','original quantum target'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
write('scope.json',scope)
pins=[dict(path=str(p),sha256=sha(p.read_bytes())) for p in
      [native/'rmea.py',native/'frozen_feature_head.py',previous,old/'contrast.py',old/'contrast-results.json']]
write('source-pins.json',pins)
write('environment.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=numpy.__version__,
                              selected_conda_environment='QT',device='cpu',package_installs=[]))
write('receipt.before-execution.json',dict(status='FROZEN_NO_NATIVE_EXECUTION_YET',recorded_at=datetime.now(timezone.utc).isoformat(),
      scope_sha256=sha((r/'scope.json').read_bytes()),source_pins_sha256=sha((r/'source-pins.json').read_bytes()),
      code_sha256={n:sha((r/n).read_bytes()) for n in ['construct.py','independent_reference.py']},
      exact_native_head_calls=2,exact_native_circuit_calls=4))
print(json.dumps(dict(status='CONSTRUCTION_FROZEN',head_calls=2,circuit_calls=4,task_rows=0,device='cpu')))
