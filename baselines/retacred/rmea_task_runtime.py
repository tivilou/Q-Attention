"""Shared fixed-feature task operations, independent of any scientific condition.

Model factories, traces and scientific result interpretation belong to study adapters.
"""
import os
from pathlib import Path
import sys


def chunks(inputs, size=4):
    """Memory control only: same batch32, same mean CE and optimizer update."""
    for a in range(0,len(inputs[0]),size):
        n=int(inputs[2][a:a+size].sum(-1).max())
        yield a,tuple(x[a:a+size,:n] for x in inputs)



def score_batch(head,inputs):
    import torch
    return torch.cat([head(*x) for _,x in chunks(inputs)],0)



def backward(head,inputs,y):
    import torch
    from torch.nn import functional as F
    total=torch.zeros((),dtype=torch.float64,device=y.device)
    for a,x in chunks(inputs):
        value=F.cross_entropy(head(*x),y[a:a+len(x[0])],reduction='sum')/len(y)
        if not torch.isfinite(value):raise ValueError('Nonfinite microbatch loss')
        value.backward();total+=value.detach()
    return total



def launch_spec(protocol, entry, output, kind, device, *, resume):
    import uuid
    if kind not in protocol.MODES or not str(device).strip():raise ValueError('Invalid condition assignment')
    token=uuid.uuid4().hex
    protocol.atomic_json(Path(output)/'.workers'/kind/'assignment.json',dict(token=token,parent_pid=os.getpid(),condition=kind,device=str(device),source=protocol.identity(protocol.sources())))
    argv=[sys.executable,'-B',str(protocol.ROOT/entry),
          'worker','--output',str(Path(output).resolve()),'--condition',kind,'--parent-pid',str(os.getpid()),'--assignment-token',token]
    if resume:argv.append('--resume')
    return dict(argv=argv,cwd=str(protocol.ROOT),physical_gpu=str(device),condition=kind,revision=protocol.git('rev-parse','HEAD'))



def prepare(protocol, recovery, labels, data_dir):
    """Small metadata delta; full fits reference the existing pinned COS inputs."""
    if not recovery.enabled:return
    import tarfile
    from .rmea_token_transport_recovery import _Parts
    p=protocol;directory=recovery.channel/'000-preparation';directory.mkdir(parents=True,exist_ok=True)
    if not (directory/'request.json').exists():
        for f in directory.glob('preparation.*.tarpart'):f.unlink()
        files=[recovery.parent/'handoff.json',recovery.output/'manifest.json',
               recovery.parent/'recovery-inputs/labels.json',recovery.parent/'recovery-inputs/input-recovery.json',
               recovery.parent/'recovery-inputs/source-manifest.json',recovery.parent/'recovery-inputs/original-manifest.json']
        files += [recovery.parent/n for n in getattr(p, 'EXTRA_RECOVERY_INPUTS', ())]
        if recovery.cfg['profile']=='canary':
            files += [recovery.output/'cache'/n for n in recovery.cache]
        else:
            files += [recovery.parent/'recovery-inputs/admission'/n for n in ('unit/unit-validation.json','unit/RUN_COMPLETE',
                'canary/admission.json','canary/dependency-verification.json','canary/canary-validation.json','canary/RUN_COMPLETE')]
        if any(f.is_symlink() or not f.is_file() or not f.resolve().is_relative_to(recovery.parent) for f in files):
            raise ValueError('Invalid minimal preparation member')
        if sum(f.stat().st_size for f in files)>8*1024**2:raise ValueError('Minimal preparation exceeds8MiB')
        parts=_Parts(directory)
        try:
            with tarfile.open(fileobj=parts,mode='w|',format=tarfile.PAX_FORMAT,dereference=True) as tar:
                for f in sorted(files):tar.add(f,arcname=f.relative_to(recovery.parent).as_posix(),recursive=False)
        finally:parts.close()
    req=directory/'request.json'
    files=[recovery.parent/n for n in p.read(req)['files']] if req.exists() else sorted(directory.glob('*.tarpart'))
    recovery._publish('000-preparation',files,0)



def export(protocol, closed_analysis, output, target, *, inject_failure=False):
    import json,shutil,tempfile
    p=protocol;output=Path(output);target=Path(target)
    if not target.parent.exists():target.parent.mkdir(parents=True)
    manifest=p.read(output/'manifest.json');summary=p.read(output/'summary.json');seal=p.read(output/'RUN_COMPLETE')
    if set(seal.get('artifacts',{}))!={'manifest.json','summary.json','analysis-readiness.json'}:raise ValueError('Completed run artifact set differs')
    for n,r in seal['artifacts'].items():
        if p.digest(output/n)!=r['sha256'] or (output/n).stat().st_size!=r['bytes']:raise ValueError('Immutable run seal differs')
    ready=closed_analysis(output)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        for i,n in enumerate(('summary.json','manifest.json','analysis-readiness.json')):
            shutil.copyfile(output/n,stage/n)
            if inject_failure and i==0:raise OSError('Injected export copy failure')
        counts=manifest['config']['counts']
        (stage/'data.sha256').write_text(''.join(h+'  '+s+'.jsonl\n' for s,(_,h) in p.EXPECTED.items()))
        (stage/'data_counts.txt').write_text(''.join(s+' '+str(n)+'\n' for s,n in counts.items()))
        p.atomic_json(stage/'run_summary.json',dict(data={s:dict(selected_count=n,source_count=p.EXPECTED[s][0],source_sha256=p.EXPECTED[s][1],
            fixture_manifest_sha256=manifest.get('canary_input_manifest') or manifest['source_input_manifest'],
            selected_cache={name:row for name,row in manifest['cache'].items() if name=='metadata.json' or name.startswith(s+'-')}) for s,n in counts.items()},
            config=manifest['config'],results=summary['results'],reference=summary['reference'],L5='NOT_ACCEPTED'))
        p.atomic_json(stage/'receipt.json',dict(source_sha256=manifest['source_sha256'],git_revision=manifest['git_revision'],
            profile=manifest['config']['profile'],completed=True,files={f.name:p.digest(f) for f in sorted(stage.iterdir())},
            experiment=p.EXPERIMENT,whitelist=sorted(p.REPORT_FILES)))
        if {f.name for f in stage.iterdir()}!=p.REPORT_FILES:raise ValueError('Report whitelist differs')
        if any(f.is_symlink() or f.stat().st_size>2*1024**2 for f in stage.iterdir()):raise ValueError('Unsafe report member')
        for n,h in p.read(stage/'receipt.json')['files'].items():
            if p.digest(stage/n)!=h:raise ValueError('Staged report differs')
        from .rmea_token_transport_recovery import sync_directory
        for f in stage.iterdir():
            with f.open('rb') as handle:os.fsync(handle.fileno())
        sync_directory(stage)
        if target.exists():
            if target.is_symlink() or {f.name for f in target.iterdir()}!=p.REPORT_FILES or any(not (target/n).is_file() or (target/n).is_symlink() or p.digest(target/n)!=p.digest(stage/n) for n in p.REPORT_FILES):
                raise ValueError('Existing report differs from immutable completed run')
        else:os.replace(stage,target)
        sync_directory(target.parent)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return dict(path=str(target),files={n:p.digest(target/n) for n in p.REPORT_FILES})
