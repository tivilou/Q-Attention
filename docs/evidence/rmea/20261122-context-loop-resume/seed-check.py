"""CPU-only seeded-start compatibility, zero fitting or cloud operations."""
from pathlib import Path
import argparse,ast,hashlib,json,random,sys
import numpy as np
import torch

a=argparse.ArgumentParser();a.add_argument('--study',type=Path,required=True);a.add_argument('--before',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
sys.path.insert(0,'/hy-tmp/Q-Attention/source-protected-context-gradient-20261122')
from baselines.retacred.rmea import RMEAHead
from baselines.retacred.rmea_token_transport_study import state_identity
from baselines.retacred.rmea_token_transport_parallel import equal_state
from baselines.retacred.rmea_token_transport_recovery import rng_state as capture_rng,restore_rng

def prelude(path):
 tree=ast.parse(path.read_text());train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 first=next(i for i,n in enumerate(train.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ('random.seed','torch.manual_seed'))
 last=next(i for i,n in enumerate(train.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='order_sha' for x in n.targets))
 return compile(ast.Module(body=train.body[first:last+1],type_ignores=[]),str(path),'exec')

before=prelude(args.before);after=prelude(args.study)

def start(code):
 ns=dict(random=random,np=np,torch=torch,RMEAHead=RMEAHead,state_identity=state_identity,hashlib=hashlib,cfg=dict(seed=13,counts=dict(train=64),epochs=4))
 exec(code,ns)
 return capture_rng(),ns['initial_sha'],ns['order_sha']

torch.set_num_threads(1)
# Distinct predecessor states expose the old process-start defect.
random.seed(101);np.random.seed(101);legacy1=start(before)
random.seed(202);np.random.seed(202);legacy2=start(before)
assert not equal_state(legacy1[0]['python'],legacy2[0]['python'])
assert not equal_state(legacy1[0]['numpy'],legacy2[0]['numpy'])
random.seed(101);np.random.seed(101);fixed1=start(after)
random.seed(202);np.random.seed(202);fixed2=start(after)
assert all(equal_state(fixed1[0][k],fixed2[0][k]) for k in ('python','numpy','torch','cuda'))
assert fixed1[1:]==fixed2[1:]==legacy1[1:]==legacy2[1:]
assert equal_state(legacy1[0]['torch'],fixed1[0]['torch'])
assert equal_state(legacy1[0]['cuda'],fixed1[0]['cuda'])
# Actual existing serializer/restore behavior, with draws in each CPU RNG.
restore_rng(fixed1[0]);expected=(random.random(),float(np.random.random()),torch.rand(8))
random.seed(909);np.random.seed(909);torch.manual_seed(909)
restore_rng(fixed1[0]);actual=(random.random(),float(np.random.random()),torch.rand(8))
assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
args.output.mkdir(parents=True,exist_ok=True)
record=dict(status='PASS',changed='Fresh-run Python and NumPy global RNG initialization; existing Torch seed, model initialization, independent epoch generator and checkpoint restore unchanged',legacy_unseeded_python_numpy_reproduced=True,fixed_all_rng_equal=True,model_initialization_sha256=fixed1[1],epoch_order_sha256=fixed1[2],torch_sequence_unchanged=True,actual_restore_next_draws_equal=True,optimizer_updates=0,cos_operations=0,cuda_scope='CPU scheduler allocation; no CUDA generator instantiated',source_sha256=hashlib.sha256(args.study.read_bytes()).hexdigest())
raw=(json.dumps(record,indent=2)+'\n').encode();(args.output/'seed-validation.json').write_bytes(raw);(args.output/'RUN_COMPLETE').write_text(json.dumps(dict(artifacts={'seed-validation.json':hashlib.sha256(raw).hexdigest()}))+'\n');print(json.dumps(record))
