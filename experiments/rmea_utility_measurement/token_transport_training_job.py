"""One fixed four-condition GPU screen;compact reports only,no model files."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import guard,read,save,sha
from experiments.rmea_utility_measurement.token_transport_profile_job import admitted,claim
from experiments.rmea_utility_measurement import token_transport_execution_scope as lifetime
DESIGN=ROOT/'docs/evidence/rmea/20261120-l5-token-transport-training/training-design.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();guard()
    d=read(DESIGN);c=read(a.execution_contract)
    assert d['schema']=='qattn.token-transport.training.design.v1' and c['schema']=='qattn.token-transport.training.execution.v1'
    assert sha(DESIGN)==c['design_sha256'] and sha(Path(__file__))==c['job_sha256']
    assert c['training'] is True and c['budget']==d['budget'] and c['review_scope']=='both'
    assert c['max_attempts']==1 and c['replay_safe'] is False and d['checkpoint_policy']=='none'
    assert set(c['environment'])=={'CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}
    for key,value in c['environment'].items():os.environ[key]=value
    attempt,identity=admitted(a,c);timing=lifetime.arm(attempt,c['budget']['timeout_seconds'])
    claim(attempt/'token-training-parent-claim.json',dict(identity=identity,pid=os.getpid()))
    for n,h in d['source_files'].items():assert sha(ROOT/n)==h,n
    assert {n:importlib.metadata.version(n) for n in d['environment']}==d['environment']
    assert c['outputs']==['compact-report.json','RUN_COMPLETE'] and a.output.is_dir() and not list(a.output.iterdir())
    for name,path in c['input_paths'].items():
        q=Path(path);assert q.is_absolute() and all(not x.is_symlink() for x in (q,*q.parents));setattr(a,name,q)
    a.check_launch=lambda:lifetime.check_launch(attempt)
    a.check_launch()
    from experiments.rmea_utility_measurement.token_transport_training_fixture import run as fixture
    from experiments.rmea_utility_measurement.token_transport_screen import train
    import torch
    torch.cuda.reset_peak_memory_stats();a.check_launch();f=fixture(a,serialization=False)
    assert f['status']=='PASS' and f['optimizer_steps']==0 and f['checkpoint_files']==0
    a.check_launch();r=train(a,d)
    assert r['optimizer_steps']==1568 and r['private_checkpoints']=={} and r['checkpoint_policy']=='none'
    assert all(v['optimizer_steps']==392 and len(v['epochs'])==4 and v['state_replay_first_batch_exact'] and not v['checkpoint_reload_first_batch_exact'] for v in r['results'].values())
    assert not (attempt/'scientific-work').exists()
    r.update(pre_update_fixture=f,admitted_identity=identity,job_sha256=sha(Path(__file__)),design_sha256=sha(DESIGN),environment=d['environment'],lifetime=timing,
        checkpoint_files=0,remaining_model_consumers=[],consumer_policy='All declared final-state comparisons/swaps/ranges complete in RAM;this screen has no retained-model or Case Study consumer. Formal follow-up trains its own model.',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    assert r['peak_reserved_bytes']<=c['budget']['gpu_mb']*1024**2
    raw=(json.dumps(r,allow_nan=False,indent=2)+'\n').encode();assert len(raw)<d['max_report_bytes'];a.check_launch()
    save(a.output/'compact-report.json',r);save(a.output/'RUN_COMPLETE',dict(schema='qattn.token-transport.training.closed.v1',files={'compact-report.json':sha(a.output/'compact-report.json')},training=True,checkpoint_files=0))
    print(json.dumps(dict(status=r['status'],optimizer_steps=r['optimizer_steps'],decision=r['decision'],checkpoint_files=0,L5='NOT_ACCEPTED')))

if __name__=='__main__':main()
