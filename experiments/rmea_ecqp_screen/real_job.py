"""Source-bound real-input preflight and the frozen train-only ECQP screen."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import guard,read,save,sha


def arguments():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['preflight','screen'],required=True)
    for name in ['assets','receipt','cohort','donors','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()


def checked_contract(a):
    guard();cpath=Path(__file__).with_name('real_contract.json');c=read(cpath)
    assert c['schema']=='qattn.ecqp.real-train-only-screen.v1' and c['allowed_actions']==['preflight','screen']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert {n:importlib.metadata.version(n) for n in c['environment']}==c['environment']
    for n,h in {**c['dependencies'],**c['design_hashes']}.items():assert sha(ROOT/n)==h
    for path in [a.assets,a.receipt,a.cohort,a.donors,a.output]:
        path=path.absolute();assert all(not p.is_symlink() for p in [path,*path.parents])
    assert a.assets.is_dir() and all(p.is_file() for p in [a.receipt,a.cohort,a.donors])
    assert sha(a.receipt)==c['input']['prefix_receipt_sha256'] and sha(a.cohort)==c['input']['cohort_sha256'] and sha(a.donors)==c['input']['donors_sha256']
    assert a.receipt.resolve().parent==a.assets.resolve().parent
    assert not a.output.exists() or not list(a.output.iterdir())
    return c,cpath


def child(a,c,cpath):
    import resource
    import numpy as np
    import torch
    from experiments.rmea_ecqp_screen import runner,metrics
    from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261118)
    started=time.monotonic()
    data=runner.read_training_prefix(a.assets,a.receipt,a.cohort,c['input']['cohort_sha256'],a.donors,c['input']['donors_sha256'],
        expected_receipt_sha256=c['input']['prefix_receipt_sha256'],expected_plan_sha256=c['input']['plan_sha256'])
    design=read(ROOT/'docs/evidence/rmea/20261118-l5-round72/screen-design.json')
    assert len(design['conditions'])==9 and design['training']['epochs']==4 and design['training']['seed']==20261118
    report=dict(action=a.action,contract_sha256=sha(cpath),prefix_receipt_sha256=sha(a.receipt),
        plan_sha256=c['input']['plan_sha256'],cohort_sha256=sha(a.cohort),donors_sha256=sha(a.donors),
        source_design_sha256=c['design_hashes']['docs/evidence/rmea/20261118-l5-round72/screen-design.json'],
        protocol_addendum_sha256=c['design_hashes']['docs/evidence/rmea/20261118-l5-round72/protocol-addendum.json'],
        input=dict(original_split='train',prefix_examples=4096,prefix_tokens=151571,hidden_size=1024,
            fit=len(data['splits']['fit']),evaluation=len(data['splits']['eval']),valid_or_test_read=False),
        environment=c['environment'],real_data_reads=True,COS_requests=0,cloud_writes=0,formal_handoff=False,L5='NOT_ACCEPTED')
    if a.action=='preflight':
        native=RMEAHead('quantum');checks=[]
        ids=data['splits']['fit'][:8]
        with torch.no_grad():
            for condition in design['conditions']:
                model=runner.TrainingModel(condition,native);model.eval()
                h,r,mask,y,summary=runner.batch(data,ids,'fit',condition['donor'])
                out=model(h,r,mask,summary)[0]
                assert out.shape==(8,40) and torch.isfinite(out).all()
                count=sum(p.numel() for p in model.parameters());assert count==condition['params']
                checks.append(dict(condition=condition['id'],parameters=count,logit_shape=[8,40],finite=True))
        report.update(status='PASS_REAL_INPUT_AND_PARENT_PATH_PREFLIGHT',checks=checks,optimizer_steps=0,
            task_training=False,utility='NOT_RUN',fit_support=np.bincount(data['targets'][data['splits']['fit']],minlength=40).tolist(),
            evaluation_support=np.bincount(data['targets'][data['splits']['eval']],minlength=40).tolist())
    else:
        work=a.output.absolute().parent/'scientific-work'/'checkpoints'
        results=runner.train_conditions(data,design,work)
        report.update(runner.summarize(data,results,metrics))
        report.update(task_training=True,epochs=4,conditions_count=9,seed=20261118,
            checkpoint_policy='Final epoch4 only; private local checkpoints, no best-epoch or outcome-based selection.',
            model_forwards='Fixed train/eval/replay and declared numerical donor interventions only.')
    report.update(elapsed_seconds=time.monotonic()-started,
        peak_child_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ))
    print(json.dumps(report,allow_nan=False),flush=True)


def main():
    a=arguments();c,cpath=checked_contract(a)
    if a.child:child(a,c,cpath);return
    # Parent deliberately imports no torch. Worker leases cover this synchronous child.
    work=a.output.absolute().parent/'scientific-work';assert not work.exists(),'Existing work: preserve and inspect; no implicit rerun'
    work.mkdir();command=[sys.executable,'-B',str(Path(__file__).resolve()),'--action',a.action]
    for name in ['assets','receipt','cohort','donors','output']:command+=['--'+name,str(getattr(a,name).absolute())]
    command+=['--child']
    child_env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run(command,cwd=ROOT,capture_output=True,env=child_env,timeout=c['resources'][a.action]['timeout']-60)
    (work/'child-stdout.json').write_bytes(result.stdout);(work/'child-stderr.log').write_bytes(result.stderr)
    assert result.returncode==0,'Scientific child failed; private stdout/stderr preserved; no automatic retry'
    report=json.loads(result.stdout);assert report['action']==a.action and report['contract_sha256']==sha(cpath)
    assert report['task_training']==(a.action=='screen') and report['COS_requests']==0 and report['L5']=='NOT_ACCEPTED'
    report['parent_child']=dict(command=command,returncode=result.returncode,synchronous=True,
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
        parent_imported_torch='torch' in sys.modules)
    assert not report['parent_child']['parent_imported_torch']
    output=a.output.absolute();output.mkdir(exist_ok=True);assert not list(output.iterdir())
    save(output/'compact-report.json',report)
    save(output/'RUN_COMPLETE',dict(schema='qattn.ecqp.bounded.closed.v1',phase=a.action,
        files={'compact-report.json':sha(output/'compact-report.json')},training=(a.action=='screen')))
    print(json.dumps(dict(status=report['status'],phase=a.action,child_exit=0,
        task_training=report['task_training'],COS_requests=0,L5='NOT_ACCEPTED')),flush=True)


if __name__=='__main__':main()
