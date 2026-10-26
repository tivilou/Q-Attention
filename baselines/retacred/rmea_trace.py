"""Private complete-value traces; public export contains aggregate readiness only."""
from pathlib import Path
import numpy as np
import torch
from . import rmea_protocol as p
from .rmea import RMEAHead,replay
from scripts.validate_sample_trace import validate,STAGES

def arrays(detail,intervention,h,roles,mask,head):
    values=dict(detail,**intervention,h=h,roles=roles,mask=mask,decoder=head.decoder,bias=head.bias)
    return {k:v.detach().cpu().numpy().copy() for k,v in values.items()}

def capture(root,mode,checkpoint,split,selected,batch,metadata,head,checkpoint_file):
    root=Path(root);directory=root/'trace';directory.mkdir(exist_ok=True)
    h,roles,mask,y,ids=batch(split,selected)
    with torch.no_grad():
        d=head(h,roles,mask,True);r=replay(head,d,mask,roles,ids)
    all_arrays=arrays(d,r,h,roles,mask,head);records=[]
    for local,index in enumerate(selected):
        sample=metadata[split][index];name=f'{mode}-{checkpoint}-{split}-{local}.npz'
        # Global effect/decoder tensors are shared; every other tensor has batch first.
        shared={'phi','effect_angles','decoder','bias'}
        numeric={k:(v if k in shared else v[local:local+1]) for k,v in all_arrays.items()}
        np.savez(directory/name,**numeric)
        specs={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in numeric.items()}
        identifier=f'{mode}:{checkpoint}:{split}:{sample["id_hash"]}'
        stages=[]
        fields={'data':dict(split=split,token_count=sample['words'],sample_id_hash=sample['id_hash']),
                'preprocess':dict(source='cache',sample=sample),
                'scoring':dict(numeric_file='trace/'+name,sha256=p.digest(directory/name),representations=specs,checkpoint_file=checkpoint_file),
                'evaluation':dict(target=int(y[local]),prediction=int(d['logits'][local].argmax()),target_access='post_generation_only'),
                'diagnosis':dict(intervention='context-weight-permutation-v1',target_access='not_available')}
        for stage in STAGES:
            stages.append(dict(stage=stage,status='observed' if stage in fields else 'not_applicable',outputs=fields.get(stage,{})))
        records.append(dict(sample_id=identifier,split=split,condition=mode,checkpoint=checkpoint,cache_index=index,stages=stages))
    return records

def close(root,samples):
    value=dict(schema_version='sample-trace.v1',trace_id='q-rmea-private',
        experiment=dict(config_sha256=p.identity(p.read(root/'manifest.json')['config']),source_sha256=p.identity(p.sources())),
        sample_selection=dict(selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples],
                              rule='first2 stable-ID-hash ordered samples per split; same samples all5 modes and both checkpoints'),
        coverage={s:('observed' if s in ('data','preprocess','scoring','evaluation','diagnosis') else 'not_applicable') for s in STAGES},samples=samples)
    p.atomic_json(root/'sample-trace.json',value)
    return audit(root)

def audit(root):
    root=Path(root);trace=p.read(root/'sample-trace.json');errors=validate(trace)
    if errors:raise ValueError(errors)
    manifest=p.read(root/'manifest.json');cfg=manifest['config']
    if trace['experiment']!=dict(config_sha256=p.identity(cfg),source_sha256=manifest['source_sha256']):raise ValueError('Trace provenance differs')
    expected={(m,c,s,i) for m in p.MODES for c in cfg['trace_checkpoints'] for s in cfg['counts'] for i in range(2)}
    seen=set();identities={};checked=0;max_residual=0.
    metadata=p.read(root/'cache/metadata.json')
    for record in trace['samples']:
        mode,checkpoint,split,index=(record[k] for k in ('condition','checkpoint','split','cache_index'))
        key=(mode,checkpoint,split,index)
        if key in seen or key not in expected:raise ValueError('Trace coordinate duplicate/unexpected')
        seen.add(key);stages={s['stage']:s['outputs'] for s in record['stages']};score=stages['scoring']
        sid=stages['data']['sample_id_hash'];identities.setdefault((split,index),set()).add(sid)
        path=root/score['numeric_file']
        if path.parent!=root/'trace' or p.digest(path)!=score['sha256']:raise ValueError('Trace path/hash differs')
        with np.load(path,allow_pickle=False) as values:
            saved={k:values[k] for k in values.files}
        if set(saved)!=set(score['representations']):raise ValueError('Representation inventory differs')
        for name,v in saved.items():
            if dict(shape=list(v.shape),dtype=str(v.dtype))!=score['representations'][name] or not np.isfinite(v).all():raise ValueError('Shape/dtype/finite check failed')
        state_path=root/score['checkpoint_file']
        if state_path.parent!=root/'checkpoints':raise ValueError('Unexpected checkpoint path')
        head=RMEAHead(mode);head.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True));head.eval()
        h=torch.from_numpy(saved['h']);roles=torch.from_numpy(saved['roles']);mask=torch.from_numpy(saved['mask'])
        meta=metadata[split][index]
        if stages['preprocess']['sample']!=meta or sid!=meta['id_hash']:raise ValueError('Input lineage differs')
        offset=np.load(root/'cache'/f'{split}-offsets.npy',mmap_mode='r');lo,hi=map(int,offset[index:index+2])
        cached=np.load(root/'cache'/f'{split}-h.npy',mmap_mode='r')[lo:hi].astype(np.float32)
        cached_roles=np.load(root/'cache'/f'{split}-roles.npy',mmap_mode='r')[lo:hi]
        target=int(np.load(root/'cache'/f'{split}-targets.npy',mmap_mode='r')[index])
        if target!=stages['evaluation']['target']:raise ValueError('Posthoc target reader differs')
        np.testing.assert_array_equal(saved['h'][0,:hi-lo],cached)
        np.testing.assert_array_equal(saved['roles'][0,:hi-lo],cached_roles)
        encoder=np.load(root/'cache'/f'{split}-encoder-{index}.npy',mmap_mode='r')
        recomputed=np.stack([encoder[pieces].mean(0) for pieces in meta['word_pieces']]).astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(cached,recomputed,atol=.002,rtol=.002)
        if len(meta['tokens'])!=hi-lo or len(meta['word_pieces'])!=hi-lo or not mask[0,:hi-lo].all() or mask[0,hi-lo:].any():raise ValueError('Coordinate coverage differs')
        with torch.no_grad():
            d=head(h,roles,mask,True);r=replay(head,d,mask,roles,[sid])
        actual=arrays(d,r,h,roles,mask,head)
        if set(actual)!=set(saved):raise ValueError('Key states missing or invented')
        for name,v in saved.items():
            np.testing.assert_allclose(actual[name],v,atol=5e-5,rtol=3e-4,err_msg=name);checked+=1
        max_residual=max(max_residual,float(np.abs(saved['identity_residual']).max()))
        if int(d['logits'][0].argmax())!=stages['evaluation']['prediction']:raise ValueError('Prediction reader differs')
        if not torch.allclose(d['alpha'].sum(-1),torch.ones(1,40),atol=1e-5):raise ValueError('Attention normalization')
    if seen!=expected or any(len(v)!=1 for v in identities.values()):raise ValueError('Trace coverage/parity incomplete')
    report=dict(schema='q-rmea.trace-readiness.v1',structural='pass',semantic='pass',replay='pass',samples=len(seen),complete_numeric_arrays_checked=checked,
                max_identity_residual=max_residual,target_boundary='labels posthoc only; no labels accepted by model or permutation',
                training_trajectory='separate first-update parameter/gradient files; inference traces are initial and valid-selected only')
    p.atomic_json(root/'trace_validation.json',report);return report
