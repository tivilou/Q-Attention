"""Reviewed scheduler entry; parent remains free of Torch allocations."""
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
DESIGN=ROOT/'docs/evidence/rmea/20261119-l5-round94/canary-design.json'

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--execution-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()

def checked(a):
    guard();d,c=read(DESIGN),read(a.execution_contract)
    assert d['schema']=='qattn.utility-measurement.canary.design.v1' and c['schema']=='qattn.utility-measurement.canary.execution.v1'
    assert c['job_sha256']==sha(Path(__file__)) and c['design_sha256']==sha(DESIGN)
    assert c['budget']==d['budget'] and c['budget']['optimizer_steps']==0 and c['review_scope']=='both'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
    for n,h in d['source_files'].items():assert sha(ROOT/n)==h
    for p in [a.output.absolute(),a.execution_contract.absolute()]:assert all(not q.is_symlink() for q in [p,*p.parents])
    assert a.output.is_dir() and not list(a.output.iterdir())
    return d,c

def main():
    a=arguments();d,c=checked(a)
    if a.child:
        from experiments.rmea_utility_measurement.canary import run
        r=run();r.update(job_sha256=sha(Path(__file__)),design_sha256=sha(DESIGN),environment=d['environment'])
        print(json.dumps(r,allow_nan=False),flush=True);return
    assert 'torch' not in sys.modules
    work=a.output.parent/'scientific-work';assert not work.exists();work.mkdir()
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--execution-contract',str(a.execution_contract.absolute()),'--output',str(a.output.absolute()),'--child']
    save(work/'child-command.json',dict(argv=argv))
    stdout,stderr=work/'child-stdout.json',work/'child-stderr.log'
    with stdout.open('xb') as out,stderr.open('xb') as err:
        done=subprocess.run(argv,cwd=ROOT,stdout=out,stderr=err,timeout=c['budget']['timeout_seconds']-30,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    assert done.returncode==0,'Canary failed; private logs retained, no completion'
    r=read(stdout);assert r['status']=='UTILITY_MEASUREMENT_NUMERICAL_CANARY_COMPLETE' and r['check_count']==len(r['checks'])
    assert r['job_sha256']==sha(Path(__file__)) and r['design_sha256']==sha(DESIGN)
    r['parent_child']=dict(command_sha256=sha(work/'child-command.json'),returncode=0,synchronous=True,
        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),parent_imported_torch=False)
    assert not list(a.output.iterdir());save(a.output/'compact-report.json',r)
    save(a.output/'RUN_COMPLETE',dict(schema='qattn.utility-measurement.canary.closed.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=False))
    print(json.dumps(dict(status=r['status'],task_training=False,L5='NOT_ACCEPTED')))

if __name__=='__main__':main()
