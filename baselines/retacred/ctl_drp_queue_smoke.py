"""Bounded private real mixed-queue CUDA/OOM canary, never a formal runner."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    from .common import atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--cache',required=True);p.add_argument('--paper',required=True)
    p.add_argument('--single',action='store_true')
    a=p.parse_args(argv)
    a.run_dir.mkdir(parents=True,exist_ok=True)
    if a.single:
        import torch
        if torch.cuda.device_count()!=1:raise ValueError('single canary must see one isolated GPU')
        torch.set_num_threads(1)
        started=time.time()
        value=torch.randn(256,256,device='cuda:0',requires_grad=True)
        opt=torch.optim.AdamW([value])
        for _ in range(2):
            opt.zero_grad();loss=value.square().mean();loss.backward();opt.step()
        # Keep this bounded diagnostic resident long enough to observe the
        # simultaneously launched DDP ranks; no formal data or extra updates.
        while time.time()-started<35:
            with torch.no_grad():_product=value@value
            torch.cuda.synchronize();time.sleep(.25)
        atomic_json(a.run_dir/'receipt.json',dict(status='passed',started=started,finished=time.time(),
            visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_updates=2))
        return 0
    from types import SimpleNamespace
    from . import reference_complete,reference_archive
    from .reference_mixed_queue import queue
    from .reference_execution import ledger_path
    events=[]
    def command(job,*_):
        events.append(dict(task=job['id'],group=job['gpu_ids'],tier=job['launch_tier'],time=time.time()))
        atomic_json(a.run_dir/'launches.json',events)
        if job['model'].startswith('ctl_drp_'):
            return [sys.executable,'-u','-m','baselines.retacred.ctl_drp_ddp','--world-size','2',
                '--entry-module','baselines.retacred.ctl_drp_ddp_smoke','--','--cache',a.cache,'--paper',a.paper,
                '--mode','adaptive','--run-dir',str(job['directory'])]
        return [sys.executable,'-u','-m','baselines.retacred.ctl_drp_queue_smoke','--single',
            '--cache',a.cache,'--paper',a.paper,'--run-dir',str(job['directory'])]
    def validate(directory):
        value=json.loads((directory/'receipt.json').read_text())
        if value['status']!='passed':raise ValueError('bounded canary failed')
    reference_complete.command=command
    reference_complete.validate_task=validate
    reference_archive.archive_completed=lambda *_:None  # explicit private diagnostic, no network/archive claim
    args=SimpleNamespace(run_dir=a.run_dir,ctl_gpus_per_task='auto')
    jobs=[dict(id='ctl_supervised_canary',model='ctl_drp_supervised',seed=13,directory=a.run_dir/'ctl'),
          dict(id='single_canary',model='ptr',seed=13,directory=a.run_dir/'single')]
    result=queue(jobs,['0','1','2'],args,None,None)
    if result:return result
    ledger=json.loads(ledger_path(a.run_dir/'ctl').read_text())
    assert ledger['index']==1 and len(ledger['events'])==1
    groups=[e['group'] for e in events if e['task']=='ctl_supervised_canary']
    assert groups==[['0','1'],['0','1']],groups
    single=json.loads((a.run_dir/'single/receipt.json').read_text())
    assert single['visible_devices']=='2'
    starts=json.loads((a.run_dir/'ctl/ddp_starts.json').read_text())
    assert any(single['started']<t<single['finished'] for t in starts)
    atomic_json(a.run_dir/'receipt.json',dict(status='passed',three_gpu_mixed_2_plus_1=True,
        actual_cuda_oom_retry=True,child_exit_76_preserved=True,same_group=groups,
        resumed_updates=2,complete_train_test_values=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
