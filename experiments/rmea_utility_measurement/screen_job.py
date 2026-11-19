"""Single reviewed scheduler child, frozen train-only inputs and compact reports."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import guard,read,save,sha
DESIGN=ROOT/'docs/evidence/rmea/20261119-l5-round95/utility-screen-design.json'
PATH_ARGUMENTS=['assets','receipt','cohort','donors','checkpoint','execution_contract','output']


def arguments():
    p=argparse.ArgumentParser()
    for n in PATH_ARGUMENTS:p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()


def checked(a):
    guard();d,c=read(DESIGN),read(a.execution_contract)
    assert d['schema']=='qattn.utility-measurement.screen-design.v1' and c['schema']=='qattn.utility-measurement.screen-execution.v1'
    assert c['job_sha256']==sha(Path(__file__)) and c['design_sha256']==sha(DESIGN)
    assert c['budget']==d['budget'] and c['review_scope']=='both'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
    for n,h in d['source_files'].items():assert sha(ROOT/n)==h
    for n in PATH_ARGUMENTS:
        p=getattr(a,n).absolute();assert all(not q.is_symlink() for q in [p,*p.parents])
    assert a.output.is_dir() and not list(a.output.iterdir())
    assert sha(a.checkpoint)==d['checkpoint']['sha256'] and a.checkpoint.name==d['checkpoint']['filename']
    return d,c


def main():
    a=arguments();d,c=checked(a)
    if a.child:
        from experiments.rmea_utility_measurement.screen import run
        r=run(a,d);r.update(job_sha256=sha(Path(__file__)),design_sha256=sha(DESIGN),environment=d['environment'])
        print(json.dumps(r,allow_nan=False),flush=True);return
    assert 'torch' not in sys.modules
    work=a.output.parent/'scientific-work';assert not work.exists();work.mkdir()
    command=[sys.executable,'-B',str(Path(__file__).resolve())]
    for n in PATH_ARGUMENTS:command+=['--'+n.replace('_','-'),str(getattr(a,n).absolute())]
    command+=['--child'];save(work/'child-command.json',dict(argv=command))
    stdout,stderr=work/'child-stdout.json',work/'child-stderr.log'
    with stdout.open('xb') as out,stderr.open('xb') as err:
        done=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=c['budget']['timeout_seconds']-60,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert done.returncode==0,'Screen failed; private logs retained and no public completion'
    r=read(stdout);assert r['status']=='TRAIN_ONLY_UTILITY_MEASUREMENT_SCREEN_COMPLETE'
    assert r['job_sha256']==sha(Path(__file__)) and r['design_sha256']==sha(DESIGN)
    assert r['conditions']==d['conditions'] and r['evaluation_examples']==972 and r['groups']==866
    assert not r['task_training'] and not r['formal_handoff'] and r['L5']=='NOT_ACCEPTED'
    r['parent_child']=dict(command_sha256=sha(work/'child-command.json'),returncode=0,synchronous=True,
        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),parent_imported_torch=False)
    assert not list(a.output.iterdir());save(a.output/'compact-report.json',r)
    save(a.output/'RUN_COMPLETE',dict(schema='qattn.utility-measurement.screen.closed.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=False))
    print(json.dumps(dict(status=r['status'],decision=r['decision'],L5='NOT_ACCEPTED')))


if __name__=='__main__':main()
