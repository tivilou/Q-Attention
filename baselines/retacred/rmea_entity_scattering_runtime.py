"""Scientific adapter for the existing training/parallel/full-state machinery."""
import math
import hashlib
import os
from pathlib import Path
import sys
import time
from . import rmea_entity_scattering_protocol as protocol
from . import rmea_task_runtime as task
from .rmea_task_runtime import chunks, score_batch, backward


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


def launch_spec(output, kind, device, *, resume):
    return task.launch_spec(protocol, 'experiments/q_rmea_entity_scattering_train.py',
                            output, kind, device, resume=resume)


def prepare(recovery, labels, data_dir, *, protocol_override=None):
    return task.prepare(protocol_override or protocol, recovery, labels, data_dir)
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


def export(output, target, *, inject_failure=False, protocol_override=None, closed_analysis_override=None):
    return task.export(protocol_override or protocol, closed_analysis_override or closed_analysis,
                       output, target, inject_failure=inject_failure)


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


def require(ok,message):
    if not ok:raise ValueError(message)

def unit(out):
    import torch
    p=protocol;ROOT=p.ROOT;runtime=sys.modules[__name__]
    from experiments import q_rmea_entity_scattering as old
    from baselines.retacred.rmea_token_transport_study import state_identity
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'),'CPU unit recipe required')
    from experiments.q_rmea_query_alignment import fresh_output
    fresh_output(out)
    import subprocess
    regression=subprocess.run([sys.executable,'-B','-m','unittest','tests.test_rmea_token_transport_parallel','tests.test_rmea_token_transport_recovery'],cwd=ROOT,capture_output=True,text=True)
    (out/'default-regression.txt').write_text(regression.stdout+regression.stderr)
    require(regression.returncode==0,'Default runner regression failed')
    checks=out/'identities';checks.mkdir();old.unit(checks)
    torch.set_num_threads(1);torch.manual_seed(20261122)
    h=torch.randn(8,5,1024,dtype=torch.float64)*.1
    roles=torch.zeros(8,5,4,dtype=torch.float64);roles[:,0,2]=1;roles[:,2,3]=1
    mask=torch.ones(8,5,dtype=torch.bool);mask[-3:,3:]=False;y=torch.arange(8)%40
    errors={}
    for kind in p.MODES:
        torch.manual_seed(13);a=runtime.make_head(kind).double();b=runtime.make_head(kind).double();b.load_state_dict(a.state_dict())
        require(sum(x.numel() for x in a.parameters())==81028,'Parameter budget differs')
        full=torch.nn.functional.cross_entropy(a(h,roles,mask),y);full.backward()
        chunk=runtime.backward(b,(h,roles,mask),y)
        error=max(float((x.grad-z.grad).abs().max()) for x,z in zip(a.parameters(),b.parameters()))
        require(error<1e-9 and abs(float(full.detach()-chunk))<1e-10,'Microbatch/full objective or gradient differs')
        errors[kind]=error
        if kind=='constant_onsite':
            d=a(h,roles,mask,details=True)
            expected=((d['utilities'].tanh()*mask[:,None,:]).sum(-1,keepdim=True)/mask.sum(-1)[:,None,None]).expand_as(d['onsite'])
            require(torch.equal(d['onsite'],expected),'Constant onsite control differs')
    torch.manual_seed(13)
    from baselines.retacred.rmea import RMEAHead
    require(state_identity(RMEAHead('quantum').double().state_dict())==p.BASE_INITIAL,'Initialization differs from reused baseline')
    p.atomic_json(out/'unit-validation.json',dict(gate=True,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),
        gradient_errors=errors,numeric_fixture_sha256=p.digest(checks/'unit-validation.json'),baseline_initial=p.BASE_INITIAL))
    p.atomic_json(out/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(out/'unit-validation.json')}))

def replay(output,condition,split):
    import io
    import numpy as np
    import torch
    from baselines.retacred.rmea_token_transport_study import state_identity
    p=protocol;runtime=sys.modules[__name__];ready=closed_analysis(output);af=output/'analysis'/(condition+'-'+split+'.json');a=p.read(af)
    receipt=p.read(output/'.workers'/condition/'WORKER_COMPLETE');cp=output/'checkpoints'/(condition+'-final.pt')
    metadata=p.read(output/'cache/metadata.json')
    authenticated,values=runtime.unit_inputs(output,split,[0,1],buffers=True)
    require(cp.stat().st_size<16*1024**2,'Bounded replay checkpoint required')
    checkpoint=cp.read_bytes();checkpoint_sha=hashlib.sha256(checkpoint).hexdigest()
    require(receipt['anchors'][split]==p.digest(af) and a['tolerance']==1e-8 and a['indices']==[0,1]
        and a['condition']==condition and a['split']==split and a['checkpoint']=='checkpoints/'+condition+'-final.pt'
        and a['sample_hashes']==[metadata[split][i]['id_hash'] for i in (0,1)]
        and a['inputs']==authenticated
        and checkpoint_sha==a['checkpoint_sha256']==receipt['checkpoint_sha256'],'Replay model/anchor identity differs')
    head=runtime.make_head(condition).double();head.load_state_dict(torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True));head=head.to('cuda').eval()
    result=[]
    with torch.no_grad():
        for h,roles in values:
            inputs=tuple(torch.from_numpy(np.array(x,dtype=np.float64,copy=True))[None].to('cuda') for x in (h,roles))
            result.append(head(*inputs,torch.ones(1,len(h),dtype=torch.bool,device='cuda')).cpu())
    error=float((torch.cat(result)-torch.tensor(a['logits'],dtype=torch.float64)).abs().max())
    require(error<=a['tolerance'],'Replay numerical tolerance failed')
    return dict(condition=condition,split=split,error=error,checkpoint_sha256=a['checkpoint_sha256'])

def worker_receipt(output,condition):
    return dict(anchors={s:protocol.digest(output/'analysis'/(condition+'-'+s+'.json')) for s in ('train','valid','test')})

def replay_all(output):
    return [replay(output,k,s) for k in protocol.MODES for s in ('train','valid','test')]
