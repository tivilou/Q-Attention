#!/usr/bin/env python3
"""Bounded synthetic real-parent launch, fresh-process recovery and export canary."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from q_attention.experiments import cfra_protocol as p


def probe():
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable on selected executor GPU')
    x=torch.tensor([1.+2.j],device='cuda',requires_grad=True)
    x.abs().square().sum().backward()
    if not torch.isfinite(x.grad).all():raise ValueError('complex CUDA backward failed')
    print('CUDA complex forward/backward passed',flush=True)


def equal(a,b,*,tolerance,path='state'):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):
        good=a.shape==b.shape and a.dtype==b.dtype
        if good:good=torch.equal(a,b) or (tolerance and (a.is_floating_point() or a.is_complex()) and torch.allclose(a,b,atol=tolerance,rtol=tolerance))
        if not good:raise ValueError('recovery tensor differs: '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('recovery RNG differs: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('recovery keys differ: '+path)
        for key in a:equal(a[key],b[key],tolerance=tolerance,path=path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('recovery sequence differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,tolerance=tolerance,path=path+'.'+str(i))
    elif a!=b:raise ValueError('recovery value differs: '+path)


def execute(output,argv,expected,tag,env=None):
    output.mkdir(parents=True,exist_ok=True)
    with (output/(tag+'.log')).open('w') as stream:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    p.atomic_json(output/(tag+'.command.json'),dict(argv=argv,cwd=str(ROOT),exit_code=result.returncode))
    if result.returncode!=expected:raise RuntimeError(f'{tag}: expected exit{expected}, got{result.returncode}; see canary log')
    print(tag+': passed',flush=True)


def validate(gpu,output):
    from q_attention.experiments.cfra_runner import gpu_ids,WORKER_TARGET
    slots=gpu_ids(gpu,canary=True);output=Path(output).resolve()
    if output.exists():raise ValueError('canary output must be fresh')
    output.mkdir(parents=True)
    for slot in slots:
        if slot=='cpu':continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=slot,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        execute(output,[sys.executable,str(Path(__file__).resolve()),'--probe'],0,'probe-'+slot,env)
    # Serialize the full lifecycle check on one selected device. All selected
    # GPUs were probed; formal conditions still use the real dynamic scheduler.
    selected=slots[0];reference=output/'reference';resumed=output/'resumed'
    entry=[sys.executable,str(WORKER_TARGET),'--canary','--gpu',selected]
    execute(output,entry+['--output-dir',str(reference)],0,'uninterrupted')
    execute(output,entry+['--output-dir',str(resumed),'--stop-job','baseline','--stop-after','2'],p.PAUSED_EXIT,'pause-baseline')
    execute(output,entry+['--resume',str(resumed),'--stop-job','quantum','--stop-after','2'],p.PAUSED_EXIT,'resume-baseline-pause-quantum')
    # A completed condition can need repair while later jobs are still pending
    # and no root summary exists. Keep that route distinct from final-run repair.
    baseline_before=p.sha(resumed/'jobs/baseline/RUN_COMPLETE.json')
    pointer=p.read(resumed/'jobs/disabled/latest.json')
    (resumed/'jobs/disabled/checkpoints'/pointer['latest']['file']).write_bytes(b'injected partial-run completed corruption')
    execute(output,entry+['--resume',str(resumed),'--recover-previous','disabled','--stop-job','quantum','--stop-after','3'],
        p.PAUSED_EXIT,'repair-completed-before-root-summary')
    if p.sha(resumed/'jobs/baseline/RUN_COMPLETE.json')!=baseline_before:raise ValueError('partial repair changed shared baseline')
    preserved={str(f.relative_to(resumed)):p.sha(f) for f in (resumed/'jobs').glob('*/RUN_COMPLETE.json')}
    if len(preserved)!=2:raise ValueError('expected baseline and Disabled completed before quantum pause')
    execute(output,entry+['--resume',str(resumed)],0,'resume-quantum')
    if any(p.sha(resumed/f)!=digest for f,digest in preserved.items()):raise ValueError('completed markers changed across resume')
    completed_before={str(f.relative_to(resumed)):p.sha(f) for f in (resumed/'jobs').glob('*/RUN_COMPLETE.json') if f.parent.name!='quantum'}
    pointer=p.read(resumed/'jobs/quantum/latest.json')
    (resumed/'jobs/quantum/checkpoints'/pointer['latest']['file']).write_bytes(b'injected completed latest corruption')
    execute(output,entry+['--resume',str(resumed)],1,'damaged-completed-rejected')
    execute(output,entry+['--resume',str(resumed),'--recover-previous','quantum'],0,'explicit-completed-repair')
    if any(p.sha(resumed/f)!=digest for f,digest in completed_before.items()):raise ValueError('repair changed other completed jobs')
    # Import Torch only after every real-parent process has exited. This keeps
    # the parent/worker footprint bounded on the development CPU cgroup.
    from q_attention.experiments import cfra_checkpoint as cp,cfra_export as export
    import torch
    torch.set_num_threads(1)
    tolerance=0. if selected=='cpu' else 1e-6
    for root in (reference,resumed):
        if not p.read(root/'run_summary.json')['mechanism_diagnostics']['active']:
            raise ValueError('canary intervention/parameter-group gradients are inactive')
    keys=('model','optimizer','rng','next_update','sampler','best_model','best_metric','best_update','selection_update','logical_batch','base_state_sha256')
    for mode in ('baseline',*p.MODES):
        a,b=(cp.load(path/'jobs'/mode) for path in (reference,resumed))
        equal({k:a[k] for k in keys},{k:b[k] for k in keys},tolerance=tolerance,path=mode)
    markdown=resumed/'run_summary.md';original=markdown.read_bytes();markdown.write_bytes(original+b'injected incorrect result')
    try:export.audit(resumed)
    except ValueError as exc:
        if str(exc)!='human-readable summary differs':raise
    else:raise ValueError('human-readable summary tampering was accepted')
    finally:markdown.write_bytes(original)
    before={str(f.relative_to(resumed)):p.sha(f) for f in (resumed/'jobs').glob('*/RUN_COMPLETE.json')}
    def synthetic_exchange(run,manifest):
        return dict(status='uploaded',synthetic_only=True,fingerprint=manifest['fingerprint'],
            parts=[dict(name='synthetic-proof',verified=True,sha256=p.identity(before),size_bytes=0)])
    destination=output/'report'
    try:export.export_report(resumed,exchange_fn=synthetic_exchange,inject_failure=True,destination=destination)
    except RuntimeError as exc:
        if str(exc)!='injected export validation failure':raise
    else:raise ValueError('export failure injection did not fail')
    if destination.exists() or list(output.glob('.cfra-stage-*')):raise ValueError('partial report survived failed export')
    report=export.export_report(resumed,exchange_fn=synthetic_exchange,destination=destination)
    if before!={str(f.relative_to(resumed)):p.sha(f) for f in (resumed/'jobs').glob('*/RUN_COMPLETE.json')}:
        raise ValueError('export retry retrained or changed completed jobs')
    receipt=p.read(report/'REPORT_COMPLETE.json')
    whitelist={*export.PUBLIC,'run_manifest.json','exchange_manifest.json','case_study.json'}
    if set(receipt['files'])!=whitelist or {x.name for x in report.iterdir()}!=whitelist|{'REPORT_COMPLETE.json'}:
        raise ValueError('report whitelist differs')
    if any(p.sha(report/name)!=digest for name,digest in receipt['files'].items()):raise ValueError('report hash differs')
    if len(p.read(report/'case_study.json')['jobs'])!=8:raise ValueError('incomplete private case index')
    retry=p.read(resumed/'EXPORT_COMPLETE.json');attempts=p.read(resumed/'export_state/attempts.json')
    if retry['retry_count']!=1 or [a['status'] for a in attempts]!=['failed','complete']:raise ValueError('export retry audit absent')
    readiness=dict(schema_version='cfra-executor-readiness.v1',status='passed',source_sha256=p.identity(p.source_contract()),gpus=slots,
        runtime=dict(python=sys.version.split()[0],torch=torch.__version__,device=selected),
        recovery=dict(jobs=8,baseline_and_plugin_fresh_process=True,full_optimizer_rng_sampler_best_state=True,tolerance=tolerance,
            completed_markers_preserved=len(preserved),damaged_completed_rejected=True,explicit_completed_repair_without_training=True,
            other_completed_markers_preserved_during_repair=len(completed_before),completed_repair_before_root_summary=True),
        export=dict(injected_failure_cleaned=True,same_completed_run_retry=True,no_retraining=True,
            source_revision=retry['source_revision'],exporter_sha256=retry['exporter_sha256'],retry_count=1,report_identity=retry['identity'],human_readable_tamper_rejected=True),
        scientific_scope='synthetic12/6/6, seed13, dim8, two epochs; no formal-data training',
        hardware_scope='CPU only; executor GPU canary still required' if selected=='cpu' else 'all selected GPUs probed; lifecycle replay on first selected GPU')
    p.atomic_json(output/'readiness.json',readiness)
    print('Q-CFRA bounded handoff canary passed',flush=True)
    return readiness


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--gpu',default='auto');cli.add_argument('--output',type=Path);cli.add_argument('--probe',action='store_true')
    args=cli.parse_args()
    if args.probe:probe();return 0
    if args.output is None:cli.error('--output required')
    validate(args.gpu,args.output);return 0

if __name__=='__main__':raise SystemExit(main())
