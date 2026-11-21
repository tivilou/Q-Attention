"""Scientific adapter for the existing training/parallel/full-state machinery."""
import math
import os
from pathlib import Path
import sys
import time
from . import rmea_entity_scattering_protocol as protocol


def make_head(kind):
    from .rmea import RMEAHead
    from .rmea_token_transport import route
    from .rmea_entity_scattering import EntityScatteringHead, anchors, scattering
    if kind in ('complex','real_magnitude'):
        return EntityScatteringHead(kind)
    if kind!='constant_onsite':
        raise ValueError('Unknown scientific condition')
    class ConstantOnsiteHead(RMEAHead):
        def __init__(self):
            super().__init__('quantum')
        def forward(self,h,roles,mask,details=False):
            import torch
            if not torch.isfinite(h).all() or not torch.isfinite(roles).all():
                raise ValueError('Finite features including padding required')
            d=super().forward(h,roles,mask,details=True)
            q=route(d['psi'],d['phi'],mask,kind='linear_coherent')['alpha']
            s,o=anchors(roles,mask)
            mean=(d['utilities'].tanh()*mask[:,None,:]).sum(-1,keepdim=True)/mask.sum(-1)[:,None,None]
            e=mean.expand_as(q)
            readout=scattering(d['psi'],q,e,mask,s,o)
            logits=readout['logits']+self.bias
            if not details:return logits
            return dict(logits=logits,q=q,onsite=e,subject=s,obj=o,psi=d['psi'],phi=d['phi'],utilities=d['utilities'],readout=readout)
    return ConstantOnsiteHead()


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


def launch_spec(output,kind,device,*,resume):
    import uuid
    if kind not in protocol.MODES or not str(device).strip():raise ValueError('Invalid condition assignment')
    token=uuid.uuid4().hex
    protocol.atomic_json(Path(output)/'.workers'/kind/'assignment.json',dict(token=token,parent_pid=os.getpid(),condition=kind,device=str(device),source=protocol.identity(protocol.sources())))
    argv=[sys.executable,'-B',str(protocol.ROOT/'experiments/q_rmea_entity_scattering_train.py'),
          'worker','--output',str(Path(output).resolve()),'--condition',kind,'--parent-pid',str(os.getpid()),'--assignment-token',token]
    if resume:argv.append('--resume')
    return dict(argv=argv,cwd=str(protocol.ROOT),physical_gpu=str(device),condition=kind,revision=protocol.git('rev-parse','HEAD'))


def prepare(recovery,labels,data_dir):
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
    # Keep the tiny sealed canary preparation available for the live restore test.


def unit_inputs(output,split,indices,*,buffers=False):
    import hashlib,numpy as np
    data={k:np.load(output/'cache'/(split+'-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets')}
    rows=[];values=[]
    for i in indices:
        offsets=np.array(data['offsets'][i:i+2],copy=True);a,b=map(int,offsets);row=dict(index=i,start=a,stop=b)
        h=np.array(data['h'][a:b],copy=True);roles=np.array(data['roles'][a:b],copy=True)
        for k,arr in [('offsets',offsets),('h',h),('roles',roles)]:
            row[k]=dict(dtype=str(arr.dtype),shape=list(arr.shape),sha256=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest())
        rows.append(row)
        values.append((h,roles))
    return (rows,values) if buffers else rows


def capture(output,head,kind,split,batch,metadata):
    """Only replay anchors; intermediate states remain recomputable GPU caches."""
    import numpy as np,torch
    p=protocol;folder=output/'analysis';folder.mkdir(exist_ok=True)
    authenticated,values=unit_inputs(output,split,[0,1],buffers=True)
    parameter=next(head.parameters());rows=[]
    with torch.no_grad():
        for h,roles in values:
            inputs=tuple(torch.from_numpy(np.array(x,dtype=np.float64,copy=True))[None].to(parameter.device) for x in (h,roles))
            rows.append(head(*inputs,torch.ones(1,len(h),dtype=torch.bool,device=parameter.device)))
        z=torch.cat(rows,dim=0)
    record=dict(schema='q-rmea.scattering.replay-anchor.v1',condition=kind,split=split,indices=[0,1],
        sample_hashes=[metadata[split][i]['id_hash'] for i in (0,1)],
        inputs=authenticated,
        checkpoint='checkpoints/'+kind+'-final.pt',checkpoint_sha256=p.digest(output/'checkpoints'/(kind+'-final.pt')),
        logits=z.detach().cpu().tolist(),tolerance=1e-8,
        replay=['experiments/q_rmea_entity_scattering_train.py','replay','--output',str(output),'--condition',kind,'--split',split],
        status='Replay anchors only; NOT sample-trace.v2 or materialized Case Study')
    p.atomic_json(folder/(kind+'-'+split+'.json'),record)
    from .rmea_token_transport_recovery import sync_directory
    sync_directory(folder);sync_directory(output)
    return []


def validate_completed(output,kind,row,receipt):
    p=protocol
    if row.get('samples')!=[] or row.get('diagnostics') is not None:
        raise ValueError('Unexpected legacy trace/diagnostic semantics')
    for split in ('train','valid','test'):
        f=output/'analysis'/(kind+'-'+split+'.json');a=p.read(f)
        meta=p.read(output/'cache/metadata.json')
        if (a['condition']!=kind or a['split']!=split or a['indices']!=[0,1]
                or receipt.get('anchors',{}).get(split)!=p.digest(f) or a['tolerance']!=1e-8
                or a['sample_hashes']!=[meta[split][i]['id_hash'] for i in (0,1)]
                or a['checkpoint']!='checkpoints/'+kind+'-final.pt'
                or a['checkpoint_sha256']!=p.digest(output/'checkpoints'/(kind+'-final.pt'))
                or len(a['logits'])!=2 or any(len(z)!=40 or any(not math.isfinite(v) for v in z) for z in a['logits'])):
            raise ValueError('Replay anchor identity differs')


def finish(cfg,output,parts,*,data_dir=None,assets=None):
    p=protocol;results={};initial=set();orders=set()
    for row in parts:
        if set(results)&set(row['results']):raise ValueError('Duplicate condition result')
        results.update(row['results']);initial.add(row['initial_sha']);orders.add(row['order_sha'])
    if set(results)!=set(p.MODES) or initial!={p.BASE_INITIAL} or len(orders)!=1:
        raise ValueError('Initialization/order/condition coverage differs')
    m=p.read(output/'manifest.json')
    if cfg['profile']=='discovery' and next(iter(orders))!=m['reference']['order_sha256']:
        raise ValueError('Training order differs from pinned linear baseline')
    reference=m['reference']['result']
    contrasts={};promotion=None
    if cfg['profile']=='discovery':
        baseline=reference['test'];candidate=results['complex']['test']
        for name,control in [('linear_reference',baseline)]+[(k,results[k]['test']) for k in p.MODES[1:]]:
            delta=candidate['positive39_macro_f1']-control['positive39_macro_f1']
            denom=abs(control['positive39_macro_f1'])
            contrasts[name]=dict(absolute_macro_delta=delta,relative_macro_delta=delta/denom if denom else None,ce_delta=candidate['ce']-control['ce'])
        promotion=all(x['absolute_macro_delta']>0 for x in contrasts.values()) and contrasts['linear_reference']['ce_delta']<=0
    summary=dict(schema='q-rmea.entity-scattering.training.result.v1',results=results,reference=reference,
        contrasts=contrasts,promotion_gate=promotion,seed=13,full_data_training=cfg['profile']=='discovery',
        common_initial_state_sha256=next(iter(initial)),common_epoch_order_sha256=next(iter(orders)),
        source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg),
        selection=cfg['selection'],L5='NOT_ACCEPTED',
        limitations='One training seed and repeatedly exposed cohort; no independent confirmation, novelty or quantum/hardware advantage inferred')
    p.atomic_json(output/'summary.json',summary)
    readiness=dict(schema='q-rmea.entity-scattering.analysis-readiness.v1',selection='Fixed final epoch',
        replay_preflight='Canary must verify this exact source before formal launch',
        checkpoints={k:dict(path='checkpoints/'+k+'-final.pt',sha256=results[k]['checkpoint_sha256']) for k in p.MODES},
        anchors={k+'/'+s:p.digest(output/'analysis'/(k+'-'+s+'.json')) for k in p.MODES for s in ('train','valid','test')},
        status='Final-model replay anchors prepared; sample-trace.v2 and materialized Case Study not claimed')
    p.atomic_json(output/'analysis-readiness.json',readiness)
    names=('manifest.json','summary.json','analysis-readiness.json')
    from .rmea_token_transport_recovery import sync_directory
    sync_directory(output)
    p.atomic_json(output/'RUN_COMPLETE',dict(artifacts={n:dict(sha256=p.digest(output/n),bytes=(output/n).stat().st_size) for n in names}))
    sync_directory(output)


def export(output,target,*,inject_failure=False):
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
            experiment='entity-scattering',whitelist=sorted(p.REPORT_FILES)))
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


def closed_analysis(output):
    p=protocol;seal=p.read(output/'RUN_COMPLETE');ready=p.read(output/'analysis-readiness.json')
    if p.digest(output/'analysis-readiness.json')!=seal['artifacts']['analysis-readiness.json']['sha256']:
        raise ValueError('Sealed replay inventory differs')
    if set(ready['checkpoints'])!=set(p.MODES) or set(ready['anchors'])!={k+'/'+s for k in p.MODES for s in ('train','valid','test')}:
        raise ValueError('Closed replay inventory coverage differs')
    for k in p.MODES:
        row=ready['checkpoints'][k];expected='checkpoints/'+k+'-final.pt'
        if row['path']!=expected or p.digest(output/expected)!=row['sha256']:raise ValueError('Sealed final checkpoint differs')
        for s in ('train','valid','test'):
            if p.digest(output/'analysis'/(k+'-'+s+'.json'))!=ready['anchors'][k+'/'+s]:raise ValueError('Sealed replay anchor differs')
    return ready
