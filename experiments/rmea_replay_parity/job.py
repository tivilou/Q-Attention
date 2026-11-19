"""Diagnose native historical aggregate mismatch; never accepts the failed case trace."""
import argparse,copy,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import guard,read,sha
from experiments.rmea_training_case_trace.trace import save
from experiments.rmea_replay_parity.differences import differences
DESIGN=ROOT/'docs/evidence/rmea/20261119-l5-round92/parity-design.json'
PATH_ARGUMENTS=['assets','receipt','cohort','donors','checkpoints','original_report','partial_manifest','execution_contract','output']

def arguments():
 p=argparse.ArgumentParser()
 for name in PATH_ARGUMENTS:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
 return p.parse_args()

def checked(a):
 guard();d,c=read(DESIGN),read(a.execution_contract)
 assert d['schema']=='qattn.native-parity.design.v1' and c['schema']=='qattn.native-parity.execution.v1'
 assert c['job_sha256']==sha(Path(__file__)) and c['design_sha256']==sha(DESIGN)
 assert c['budget']==d['proposed_budget'] and c['budget']['optimizer_steps']==0
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and c['review_scope']=='both'
 assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
 for n,h in {**d['source_files'],**d['implementation_files']}.items():assert sha(ROOT/n)==h
 for name in PATH_ARGUMENTS:
  p=getattr(a,name).absolute();assert all(not q.is_symlink() for q in [p,*p.parents])
 assert sha(a.original_report)==d['original_report_sha256'] and sha(a.partial_manifest)==d['partial_manifest_sha256']
 cp=a.checkpoints/d['checkpoint']['filename'];assert cp.is_file() and not cp.is_symlink() and sha(cp)==d['checkpoint']['sha256']
 assert a.output.is_dir() and not list(a.output.iterdir())
 return d,c

def child(a,d):
 import numpy as np
 import torch
 from torch.nn import functional as F
 from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
 from experiments.rmea_ecqp_screen import metrics,runner
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261118)
 started=time.monotonic();identity=d['data']
 data=runner.read_training_prefix(a.assets,a.receipt,a.cohort,identity['cohort_sha256'],a.donors,identity['donors_sha256'],
  expected_receipt_sha256=identity['prefix_receipt_sha256'],expected_plan_sha256=identity['plan_sha256'])
 ids=data['splits']['eval'];assert len(ids)==972
 partial=read(a.partial_manifest);pr=Path(partial['root']);assert partial['status']=='UNACCEPTED_PARTIAL_ONLY'
 assert all(not z.is_symlink() for z in [pr,*pr.parents]) and not(pr/'RUN_COMPLETE').exists()
 assert set(p.name for p in pr.iterdir())==set(partial['files'])
 for n,v in partial['files'].items():
  assert Path(n).name==n
  p=pr/n;assert not p.is_symlink() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
 selection=read(pr/'selection.json');assert sha(pr/'selection.json')==d['original_selection_sha256'] and selection['model_outputs_read_for_selection'] is False
 selected={r['source_index']:r for r in selection['rows']};assert len(selected)==34
 assert set(selected)<=set(map(int,ids))
 for i,r in selected.items():assert r['id_hash']==data['metadata'][i]['id_hash'] and r['label']==int(data['targets'][i])
 conditions=read(ROOT/'docs/evidence/rmea/20261118-l5-round72/screen-design.json')['conditions']
 condition=next(c for c in conditions if c['id']=='native_quantum')
 base=runner.TrainingModel(condition,RMEAHead('quantum'))
 base.load_state_dict(torch.load(a.checkpoints/d['checkpoint']['filename'],map_location='cpu',weights_only=True));base.eval()
 assert sum(p.numel() for p in base.parameters())==d['checkpoint']['parameters']
 variants=d['variants'];assert variants==[
  dict(id='ordinary_trainable_flags',details=False,requires_grad=True),dict(id='details_trainable_flags',details=True,requires_grad=True),
  dict(id='ordinary_frozen_flags',details=False,requires_grad=False),dict(id='details_frozen_flags',details=True,requires_grad=False)]
 models={r['id']:copy.deepcopy(base).requires_grad_(r['requires_grad']) for r in variants}
 logits={r['id']:[] for r in variants};losses={r['id']:0. for r in variants};pairs={};partial_checks={}
 for i,left in enumerate(variants):
  partial_checks[left['id']]=dict(cases=0,exact_cases=0,max_abs=0.,argmax_disagreements=0)
  for right in variants[i+1:]:pairs[left['id']+'__'+right['id']]=dict(elements=0,unequal_elements=0,max_abs=0.,argmax_disagreements=0)
 with torch.no_grad():
  for start in range(0,len(ids),32):
   batch_ids=ids[start:start+32];h,roles,mask,y,donor=runner.batch(data,batch_ids,'eval',False);assert donor is None
   batch_logits={}
   for v in variants:
    out=models[v['id']](h,roles,mask,details=v['details']);z=out['logits'] if v['details'] else out[0]
    assert torch.isfinite(z).all();batch_logits[v['id']]=z
    logits[v['id']].append(z.numpy().copy());losses[v['id']]+=float(F.cross_entropy(z,y,reduction='sum'))
    for j,i in enumerate(map(int,batch_ids)):
     if i not in selected:continue
     with np.load(pr/(selected[i]['id_hash']+'-native_quantum.npz'),allow_pickle=False) as old:
      old_z=torch.from_numpy(old['logits'].copy())
     current=z[j:j+1];row=partial_checks[v['id']];row['cases']+=1;row['exact_cases']+=int(torch.equal(current,old_z))
     row['max_abs']=max(row['max_abs'],float((current-old_z).abs().max()))
     row['argmax_disagreements']+=int(current.argmax(-1).item()!=old_z.argmax(-1).item())
   for i,left in enumerate(variants):
    for right in variants[i+1:]:
     z,w=batch_logits[left['id']],batch_logits[right['id']];row=pairs[left['id']+'__'+right['id']]
     row['elements']+=z.numel();row['unequal_elements']+=int((z!=w).sum());row['max_abs']=max(row['max_abs'],float((z-w).abs().max()))
     row['argmax_disagreements']+=int((z.argmax(-1)!=w.argmax(-1)).sum())
 expected=read(a.original_report)['metrics']['native_quantum'];gold=np.asarray(data['targets'][ids],dtype=np.int64)
 work=a.output.parent/'scientific-work';private=work/'parity-values';private.mkdir(mode=0o700)
 observed={};payload={}
 for v in variants:
  name=v['id'];z=np.concatenate(logits[name]);assert z.shape==(972,40);payload[name]=z
  row=metrics.from_confusion(metrics.confusion(gold,z.argmax(-1)));row['ce']=losses[name]/len(ids)
  diff=differences(expected,row);observed[name]=dict(actual=row,exact_historical_match=not diff,differences=diff)
  assert partial_checks[name]['cases']==34
 np.savez_compressed(private/'native-logits.npz',**payload)
 assert all(sha(pr/n)==v['sha256'] for n,v in partial['files'].items()),'Old partial evidence changed'
 revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
 save(private/'RUN_COMPLETE',dict(schema='qattn.native-parity.values.v1',files={'native-logits.npz':sha(private/'native-logits.npz')},source_revision=revision,training=False))
 report=dict(status='FROZEN_NATIVE_PARITY_DIAGNOSTIC_COMPLETE',design_sha256=sha(DESIGN),job_sha256=sha(Path(__file__)),
  evaluation_examples=972,batches=31,variant_count=4,expected=expected,variants=observed,api_pairwise=pairs,old_partial_logit_checks=partial_checks,
  old_partial_manifest_sha256=sha(a.partial_manifest),old_partial_unchanged=True,old_trace_accepted=False,
  environment=d['environment'],cpu_capability=torch.backends.cpu.get_cpu_capability(),
  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),
  private_closed_sha256=sha(private/'RUN_COMPLETE'),private_values_sha256=sha(private/'native-logits.npz'),
  elapsed_seconds=time.monotonic()-started,optimizer_steps=0,task_training=False,COS_requests=0,cloud_writes=0,new_checkpoints=0,
  parity_tolerance=0,old_parity_gate_relaxed=False,promotion=False,formal_handoff=False,L5='NOT_ACCEPTED')
 assert (private/'native-logits.npz').stat().st_size<=d['proposed_budget']['private_values_bytes_max']
 print(json.dumps(report,allow_nan=False),flush=True)

def main():
 a=arguments();d,c=checked(a)
 if a.child:child(a,d);return
 assert 'torch' not in sys.modules
 work=a.output.parent/'scientific-work';assert not work.exists();work.mkdir()
 command=[sys.executable,'-B',str(Path(__file__).resolve())]
 for name in PATH_ARGUMENTS:command+=['--'+name.replace('_','-'),str(getattr(a,name).absolute())]
 command+=['--child'];save(work/'child-command.json',dict(argv=command))
 stdout,stderr=work/'child-stdout.json',work/'child-stderr.log'
 with stdout.open('xb') as out,stderr.open('xb') as err:
  r=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=c['budget']['timeout_seconds']-60)
 assert r.returncode==0,'Diagnostic failed; private logs retained and no public marker'
 report=read(stdout);assert report['status']=='FROZEN_NATIVE_PARITY_DIAGNOSTIC_COMPLETE' and not report['task_training']
 assert report['design_sha256']==sha(DESIGN) and report['job_sha256']==sha(Path(__file__))
 assert sha(work/'parity-values/RUN_COMPLETE')==report['private_closed_sha256']
 report['parent_child']=dict(command_sha256=sha(work/'child-command.json'),returncode=0,synchronous=True,stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),parent_imported_torch=False)
 assert not list(a.output.iterdir());save(a.output/'compact-report.json',report)
 save(a.output/'RUN_COMPLETE',dict(schema='qattn.native-parity.public.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=False))
 print(json.dumps(dict(status=report['status'],task_training=False,optimizer_steps=0,L5='NOT_ACCEPTED')))

if __name__=='__main__':main()
