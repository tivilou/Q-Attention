"""Torch-free synchronous CUDA-check parent bound to one immutable contract."""
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

DESIGN=ROOT/'docs/evidence/rmea/20261120-l5-token-transport-gradient/check-design.json'

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--execution-contract',type=Path,required=True)
    p.add_argument('--contract-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--parent-pid',type=int,default=0,help=argparse.SUPPRESS)
    return p.parse_args()

def admitted(a,c):
    """Bind to the launcher's registered digest, durable request and exact output."""
    assert sha(a.execution_contract)==a.contract_sha256
    output=a.output.absolute();assert os.environ.get('RESEARCH_OUTPUT_DIR')==str(output)
    attempt=output.parent;request=read(attempt/'request.json');command=read(attempt/'command.json')
    assert request['project']=='q-attention' and request['attempt']==attempt.name and output.name=='output'
    assert request['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    expected=c['command_template']+['--contract-sha256',a.contract_sha256]
    assert request['recipe']['command']==expected and request['recipe']['outputs']==c['outputs']
    assert request['recipe']['max_attempts']==1 and request['recipe']['replay_safe'] is False
    assert request['recipe']['timeout']==c['budget']['timeout_seconds']
    assert request['recipe']['resources']=={k:c['budget'][k] for k in ('cpu','ram_mb','disk_mb','gpus','gpu_mb')}
    rendered=[x.replace('{python}',sys.executable).replace('{output}',str(output)) for x in expected]
    assert command['argv']==rendered and command['cwd']==str(ROOT) and command['revision']==request['revision']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(request['gpu_ids']) and len(request['gpu_ids'])==1
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip()
    assert revision==request['revision']
    return attempt,dict(attempt=request['attempt'],task=request['task'],revision=revision,contract_sha256=a.contract_sha256)

def claim(path,record):
    with path.open('x',encoding='utf-8') as f:
        json.dump(record,f);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    a=arguments();guard();d=read(DESIGN);c=read(a.execution_contract)
    assert d['schema']=='qattn.token-transport.gradient.design.v1' and c['schema']=='qattn.token-transport.gradient.execution.v1'
    assert c['job_sha256']==sha(Path(__file__)) and c['design_sha256']==sha(DESIGN)
    assert c['budget']==d['budget'] and c['review_scope']=='both' and c['training'] is False
    assert c['max_attempts']==1 and c['replay_safe'] is False
    assert c['outputs']==['compact-report.json','RUN_COMPLETE']
    attempt,identity=admitted(a,c)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');assert visible not in ('','-1') and len(visible.split(','))==1
    assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
    for n,h in d['source_files'].items():assert sha(ROOT/n)==h
    for p in (a.output.absolute(),a.execution_contract.absolute()):assert all(not q.is_symlink() for q in (p,*p.parents))
    assert a.output.is_dir() and not list(a.output.iterdir())
    if a.child:
        parent=read(attempt/'token-gradient-parent-claim.json')
        assert parent['identity']==identity and parent['pid']==a.parent_pid==os.getppid()==os.getpgrp()
        assert Path('/proc/'+str(a.parent_pid)).exists()
        claim(attempt/'token-gradient-science-claim.json',dict(identity=identity,pid=os.getpid(),parent_pid=a.parent_pid))
        from experiments.rmea_utility_measurement.token_transport_gradient import run
        r=run();r.update(job_sha256=sha(Path(__file__)),design_sha256=sha(DESIGN),environment=d['environment'],admitted_identity=identity)
        print(json.dumps(r,allow_nan=False),flush=True);return
    assert 'torch' not in sys.modules
    assert os.getpgrp()==os.getpid(),'Worker must own the complete job process group'
    claim(attempt/'token-gradient-parent-claim.json',dict(identity=identity,pid=os.getpid()))
    work=a.output.parent/'scientific-work';assert not work.exists();work.mkdir()
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--execution-contract',str(a.execution_contract.absolute()),'--contract-sha256',a.contract_sha256,'--output',str(a.output.absolute()),'--child','--parent-pid',str(os.getpid())]
    save(work/'child-command.json',dict(argv=argv));stdout=work/'child-stdout.json';stderr=work/'child-stderr.log'
    with stdout.open('xb') as out,stderr.open('xb') as err:
        done=subprocess.run(argv,cwd=ROOT,stdout=out,stderr=err,timeout=c['budget']['timeout_seconds']-30,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    assert done.returncode==0,'Source check failed; preserve partials and do not retry science'
    assert stdout.stat().st_size<d['max_report_bytes']
    r=read(stdout);assert r['status']=='TOKEN_TRANSPORT_GRADIENT_CHECK_COMPLETE'
    assert r['admitted_identity']==identity
    assert r['job_sha256']==sha(Path(__file__)) and r['design_sha256']==sha(DESIGN)
    assert r['check_count']==len(r['checks'])==5 and {x['name'] for x in r['checks']}==set(d['check_names'])
    assert all(x['status']=='PASS' for x in r['checks']) and r['max_absolute_residual']<1e-9 and r['device']['type']=='cuda'
    assert 0<=r['max_gradient_scaled_error']<=1
    assert all(r[k]==0 for k in ('optimizer_steps','real_dataset_reads','new_checkpoints','COS_requests','cloud_writes'))
    assert all(r[k] is False for k in ('task_training','independent_task_confirmation','formal_handoff')) and r['L5']=='NOT_ACCEPTED'
    r['parent_child']=dict(returncode=0,synchronous=True,parent_imported_torch=False,command_sha256=sha(work/'child-command.json'),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
    raw=(json.dumps(r,allow_nan=False,indent=2)+'\n').encode();assert len(raw)<d['max_report_bytes']
    assert not list(a.output.iterdir());save(a.output/'compact-report.json',r)
    assert (a.output/'compact-report.json').stat().st_size<d['max_report_bytes']
    save(a.output/'RUN_COMPLETE',dict(schema='qattn.token-transport.gradient.closed.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=False))
    print(json.dumps(dict(status=r['status'],task_training=False,L5='NOT_ACCEPTED')))

if __name__=='__main__':main()
