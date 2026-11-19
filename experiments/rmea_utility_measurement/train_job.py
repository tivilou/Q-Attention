"""Frozen bounded training child; parent has no Torch allocations."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import guard,read,save,sha
DESIGN=ROOT/'docs/evidence/rmea/20261119-l5-round96/training-design.json'
PATH_ARGUMENTS=['assets','receipt','cohort','donors','execution_contract','output']


def arguments(path_arguments=PATH_ARGUMENTS):
    p=argparse.ArgumentParser()
    for n in path_arguments:p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()


def checked(a,*,design_path=DESIGN,entrypoint=None,schema_prefix='qattn.utility-training',path_arguments=PATH_ARGUMENTS):
    entrypoint=Path(__file__).resolve() if entrypoint is None else Path(entrypoint).resolve()
    guard();d,c=read(design_path),read(a.execution_contract)
    assert d['schema']==schema_prefix+'.design.v1' and c['schema']==schema_prefix+'.execution.v1'
    assert c['job_sha256']==sha(entrypoint) and c['design_sha256']==sha(design_path) and c['budget']==d['budget']
    assert c['review_scope']=='both' and c['training'] is True and c['full_formal_run'] is False
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
    for n,h in d['source_files'].items():assert sha(ROOT/n)==h
    for n in path_arguments:
        p=getattr(a,n).absolute();assert all(not q.is_symlink() for q in [p,*p.parents])
    assert a.output.is_dir() and not list(a.output.iterdir())
    return d,c


def main(*,design_path=DESIGN,entrypoint=None,implementation_module='experiments.rmea_utility_measurement.train',
         schema_prefix='qattn.utility-training',result_status='TRAIN_ONLY_UTILITY_JOINT_OPTIMIZATION_COMPLETE',
         extra_path_arguments=(),precheck=None):
    entrypoint=Path(__file__).resolve() if entrypoint is None else Path(entrypoint).resolve()
    path_arguments=[*PATH_ARGUMENTS,*extra_path_arguments]
    assert len(set(path_arguments))==len(path_arguments)
    a=arguments(path_arguments);d,c=checked(a,design_path=design_path,entrypoint=entrypoint,schema_prefix=schema_prefix,path_arguments=path_arguments)
    if precheck is not None:precheck(a,d,c)
    if a.child:
        run=importlib.import_module(implementation_module).run
        r=run(a,d);r.update(job_sha256=sha(entrypoint),design_sha256=sha(design_path),environment=d['environment'])
        print(json.dumps(r,allow_nan=False),flush=True);return
    assert 'torch' not in sys.modules
    work=a.output.parent/'scientific-work';assert not work.exists();work.mkdir()
    command=[sys.executable,'-B',str(entrypoint)]
    for n in path_arguments:command+=['--'+n.replace('_','-'),str(getattr(a,n).absolute())]
    command+=['--child'];save(work/'child-command.json',dict(argv=command))
    stdout,stderr=work/'child-stdout.json',work/'child-stderr.log'
    with stdout.open('xb') as out,stderr.open('xb') as err:
        done=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=c['budget']['timeout_seconds']-60,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert done.returncode==0,'Training diagnostic failed; private partials retained, no public marker'
    r=read(stdout);assert r['status']==result_status
    assert r['job_sha256']==sha(entrypoint) and r['design_sha256']==sha(design_path)
    assert r['conditions']==d['conditions'] and r['training']==d['training'] and r['optimizer_steps']==d['budget']['optimizer_steps']
    assert set(r['private_checkpoints'])=={n+'-final.pt' for n in d['conditions']} and r['new_checkpoints']==d['budget']['new_checkpoints']==len(d['conditions'])
    assert r['task_training'] and r['full_formal_run'] is False and r['formal_handoff'] is False and r['L5']=='NOT_ACCEPTED'
    cp=work/'checkpoints';marker=read(cp/'RUN_COMPLETE');assert sha(cp/'RUN_COMPLETE')==r['private_checkpoint_marker_sha256']
    assert set(p.name for p in cp.iterdir())==set(r['private_checkpoints'])|{'RUN_COMPLETE'}
    for n,v in r['private_checkpoints'].items():
        assert Path(n).name==n and not (cp/n).is_symlink() and sha(cp/n)==v['sha256'] and (cp/n).stat().st_size==v['bytes']
    assert marker==dict(schema='qattn.utility-training.checkpoints.v1',files={n:v['sha256'] for n,v in r['private_checkpoints'].items()},training=True)
    assert all(r[k]==0 for k in ['COS_requests','cloud_writes']) and r['independent_confirmation'] is False
    r['parent_child']=dict(command_sha256=sha(work/'child-command.json'),returncode=0,synchronous=True,
        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),parent_imported_torch=False)
    assert not list(a.output.iterdir());save(a.output/'compact-report.json',r)
    save(a.output/'RUN_COMPLETE',dict(schema=schema_prefix+'.public.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=True))
    print(json.dumps(dict(status=r['status'],decision=r['decision'],L5='NOT_ACCEPTED')))


if __name__=='__main__':main()
