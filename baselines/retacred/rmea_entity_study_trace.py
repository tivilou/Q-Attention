"""Producer-owned complete values and independent fixed-checkpoint replay."""
from pathlib import Path
import numpy as np
import torch
from . import rmea_entity_study_protocol as p
from .rmea_entity import make_head,entity_inputs
from .rmea_entity_trace import AXES,SHARED,details,display,read_value
from scripts.validate_sample_trace_v2 import STAGES,validate

EXTRA_AXES=dict(input_ids=['subword'],attention_mask=['subword'],encoder_hidden=['subword','hidden'],
    role_means=['subject_or_object','feature'],role_mean_count=['singleton'],
    branch_mean_pair=['sample','subject_or_object','feature'],branch_swapped_pair=['sample','subject_or_object','feature'],
    branch_mismatch_pair=['sample','subject_or_object','feature'],diagnostic_logits=['intervention','sample','relation'],target=['singleton'])
GROUPS=('tokenization','encoder','word_input','entities','values','weights','prediction','diagnosis','evaluation')


def owner(name):
    if name in ('input_ids','attention_mask'):return 'tokenization'
    if name=='encoder_hidden':return 'encoder'
    if name in ('h','roles','mask','features'):return 'word_input'
    if name in ('entity_pair','entity_raw','subject_angles','object_angles','role_means','role_mean_count'):return 'entities'
    if name in ('values','utilities','decoder','bias'):return 'values'
    if name=='logits':return 'prediction'
    if name=='target':return 'evaluation'
    if name.startswith('branch_') or name=='diagnostic_logits':return 'diagnosis'
    return 'weights'


def capture(root,head,split,checkpoint,batch,metadata,overrides):
    inputs,y=batch(split,[0,1]);samples=[]
    with torch.no_grad():
        d=details(head,*inputs)
        branches=overrides(split,[0,1],inputs) if head.variant=='ordered' else {}
        diagnostic=torch.stack([head(*inputs,entity_override=pair)[0] for pair in branches.values()]) if branches else None
    for i in range(2):
        meta=metadata[split][i]
        arrays={k:(v if k in SHARED else v[i:i+1]).detach().cpu().numpy().copy() for k,v in d.items()}
        arrays.update(input_ids=np.asarray(meta['input_ids'],np.int64),attention_mask=np.asarray(meta['attention_mask'],np.int64),
            encoder_hidden=np.load(root/'cache'/f'{split}-encoder-{i}.npy',allow_pickle=False),target=np.asarray([int(y[i])],np.int64))
        if head.variant=='no_content':arrays.update(role_means=head.role_means.cpu().numpy().copy(),role_mean_count=np.asarray([int(head.role_mean_count)],np.int64))
        for name,pair in branches.items():arrays['branch_'+name+'_pair']=pair[i:i+1].cpu().numpy().copy()
        if diagnostic is not None:arrays['diagnostic_logits']=diagnostic[:,i:i+1].cpu().numpy().copy()
        name=f'{head.variant}-{checkpoint}-{split}-{i}.npz';path=root/'trace'/name;path.parent.mkdir(exist_ok=True)
        np.savez(path,**arrays)
        ref=dict(path='trace/'+name,sha256=p.digest(path),bytes=path.stat().st_size,visibility='private')
        groups={g:[] for g in GROUPS};reps=[];sid=f'{head.variant}:{checkpoint}:{split}:{i}'
        for key,a in sorted(arrays.items()):
            g=owner(key);groups[g].append(key);axes=EXTRA_AXES.get(key,AXES.get(key))
            if axes is None or len(axes)!=a.ndim:raise ValueError('Undeclared coordinate semantics '+key)
            reps.append(dict(id=key,stage_id=g,status='observed',meaning=key+' complete actual detached producer value',shape=list(a.shape),dtype=str(a.dtype),
                axis_semantics=axes,portal_preview=dict(array_key=key,owner_stage=g,direction='output',visibility='private'),artifact_refs=[ref],display=display(key),
                value_capture=dict(status='complete',shape=list(a.shape),logical_element_count=int(a.size),stored_element_count=int(a.size),artifact_refs=[ref],
                    reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',reader_ref='/api/value?sample='+sid+'&representation='+key,source_sha256=[ref['sha256']]))))
        links={'tokenization':['source:tokens','source:entity_spans'],
            'encoder':['input_ids','attention_mask','external:pinned_roberta_large'],
            'word_input':['encoder_hidden','source:word_pieces','source:entity_spans'],
            'entities':['h','roles','mask','external:checkpoint_prepare']+(['external:train_only_means'] if head.variant=='no_content' else []),
            'values':['h','external:checkpoint_value_decoder'],
            'weights':['features','mask','external:checkpoint_scorer']+(['entity_pair','subject_angles','object_angles'] if 'entity_pair' in arrays else [])+(['utilities'] if 'sensitivity' in arrays else []),
            'prediction':['alpha','utilities','bias'],
            'diagnosis':['h','roles','mask','logits','external:train_only_means','external:whole_split_id_derangement'],
            'evaluation':['logits','source:held_out_target']}
        canonical=dict(tokenization='data',encoder='preprocess',word_input='preprocess',entities='scoring',values='scoring',weights='scoring',prediction='generation',diagnosis='diagnosis',evaluation='evaluation')
        stages=[]
        for step,g in enumerate(GROUPS):
            if not groups[g]:continue
            stages.append(dict(stage_id=g,stage=g,canonical_stage=canonical[g],lane='replay',status='observed',step=step,checkpoint=checkpoint,
                purpose=g+' source-owned computation; labels only in posthoc evaluation',input_refs=links[g],output_refs=groups[g],artifact_refs=[ref],
                display=display(g),target_access='post_generation_only' if g=='evaluation' else 'not_available'))
        cp=root/'checkpoints'/f'{head.variant}-{checkpoint}.pt'
        samples.append(dict(sample_id=sid,lane='replay',split=split,checkpoint=checkpoint,
            roles=dict(source_sample=dict(tokens=meta['tokens'],id_hash=meta['id_hash']),sequence_or_features=dict(word_pieces=meta['word_pieces']),
                task_objects=dict(subject=meta['subject'],object=meta['object']),target_or_gold=dict(value=int(y[i]),boundary='post_generation_only'),
                prediction=dict(value=int(arrays['logits'].argmax())),
                provenance=dict(condition=head.variant,cache_index=i,checkpoint_file=str(cp.relative_to(root)),checkpoint_sha256=p.digest(cp))),
            stages=stages,representations=reps))
    return samples


def close(root,samples,cfg,metadata,donors):
    manifest=p.read(root/'manifest.json')
    required=['input_ids','attention_mask','encoder_hidden','h','roles','mask','features','alpha','values','utilities','logits','target']
    payload=dict(schema_version='sample-trace.v2',trace_id='q-rmea-entity-'+str(cfg['seed']),trace_completeness='complete',
        experiment=dict(run_id=root.parent.name,dataset='Re-TACRED official splits; pinned tiny canary when profile=canary',code_revision=manifest['git_revision'],
            config_sha256=p.identity(cfg),model_identity='Frozen RoBERTa-large last states and five entity relation heads',seed=cfg['seed']),
        sample_selection=dict(rule='first2 hash-selected samples per split; identical all modes and checkpoints',population_scope='bounded fixed inference replay; no full training trajectory',
            selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='entity-conditioned relation extraction inference replay',required_roles=['source_sample','sequence_or_features','task_objects','prediction','target_or_gold'],
            required_representations=required,required_value_representations=required,required_splits=list(cfg['counts']),required_checkpoints=cfg['trace_checkpoints'],required_lanes=['replay'],
            checkpoint_policy='initial and valid-selected; saved encoder final states, no live encoder replay or internal-layer claim; first-update evidence separate'),
        coverage={s:('observed' if s in ('data','preprocess','scoring','generation','evaluation','diagnosis') else 'not_applicable') for s in STAGES},samples=samples)
    errors,warnings=validate(payload,require_complete=True)
    if errors:raise ValueError(errors)
    checked=coordinates=0;seen=set();parity={}
    cache=root/'cache'
    train_means=np.load(root/'train-calibration.npz',allow_pickle=False)['role_means']

    def pooled(split,index):
        offsets=np.load(cache/f'{split}-offsets.npy',mmap_mode='r');lo,hi=map(int,offsets[index:index+2])
        h=np.load(cache/f'{split}-h.npy',mmap_mode='r')[lo:hi].astype(np.float32)
        r=np.load(cache/f'{split}-roles.npy',mmap_mode='r')[lo:hi].copy()
        return entity_inputs(torch.from_numpy(h[None]),torch.from_numpy(r[None]),torch.ones(1,hi-lo,dtype=torch.bool))

    for sample in samples:
        prov=sample['roles']['provenance'];mode=prov['condition'];i=prov['cache_index'];split=sample['split'];checkpoint=sample['checkpoint']
        key=(mode,checkpoint,split,i)
        if key in seen:raise ValueError('Duplicate sample')
        seen.add(key);cp=root/prov['checkpoint_file']
        if p.digest(cp)!=prov['checkpoint_sha256']:raise ValueError('Checkpoint differs')
        head=make_head(mode);head.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));head.eval()
        ref=sample['representations'][0]['artifact_refs'][0]
        with np.load(root/ref['path'],allow_pickle=False) as f:saved={k:f[k].copy() for k in f.files}
        meta=metadata[split][i]
        if sample['roles']['source_sample']!=dict(tokens=meta['tokens'],id_hash=meta['id_hash']):raise ValueError('Source lineage differs')
        if sample['roles']['task_objects']!=dict(subject=meta['subject'],object=meta['object']):raise ValueError('Role lineage differs')
        np.testing.assert_array_equal(saved['input_ids'],meta['input_ids']);np.testing.assert_array_equal(saved['attention_mask'],meta['attention_mask'])
        if not np.isin(saved['attention_mask'],[0,1]).all() or len(saved['encoder_hidden'])!=len(saved['input_ids']):raise ValueError('Encoder coordinate mapping')
        np.testing.assert_array_equal(saved['encoder_hidden'],np.load(cache/f'{split}-encoder-{i}.npy',allow_pickle=False))
        offset=np.load(cache/f'{split}-offsets.npy',mmap_mode='r');lo,hi=map(int,offset[i:i+2]);n=hi-lo
        cached=np.load(cache/f'{split}-h.npy',mmap_mode='r')[lo:hi].astype(np.float32)
        if n!=len(meta['tokens']) or len(meta['word_pieces'])!=n:raise ValueError('Original word coverage differs')
        for pieces in meta['word_pieces']:
            if not pieces or any(type(k)!=int or k<0 or k>=len(saved['input_ids']) or saved['attention_mask'][k]!=1 for k in pieces):raise ValueError('Word subpiece map invalid')
        recomputed=np.stack([saved['encoder_hidden'][pieces].mean(0) for pieces in meta['word_pieces']]).astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(recomputed,cached,atol=.002,rtol=.002)
        np.testing.assert_array_equal(saved['h'][0,:n],cached)
        np.testing.assert_array_equal(saved['roles'][0,:n],np.load(cache/f'{split}-roles.npy',mmap_mode='r')[lo:hi])
        if not saved['mask'][0,:n].all() or saved['mask'][0,n:].any() or np.any(saved['h'][0,n:]) or np.any(saved['roles'][0,n:]):raise ValueError('Padding coverage')
        target=int(np.load(cache/f'{split}-targets.npy',mmap_mode='r')[i])
        if target!=int(saved['target'][0]) or target!=sample['roles']['target_or_gold']['value']:raise ValueError('Posthoc target differs')
        inputs=tuple(torch.from_numpy(saved[k]) for k in ('h','roles','mask'))
        with torch.no_grad():
            actual={k:v.detach().numpy() for k,v in details(head,*inputs).items()}
            if mode=='no_content':
                np.testing.assert_array_equal(head.role_means.numpy(),train_means)
                if int(head.role_mean_count)!=cfg['counts']['train']:raise ValueError('Calibration count differs')
                actual.update(role_means=head.role_means.numpy(),role_mean_count=np.asarray([int(head.role_mean_count)],np.int64))
            if mode=='ordered':
                branches=dict(mean=torch.from_numpy(train_means[None]),swapped=pooled(split,i).flip(1),mismatch=pooled(split,donors[split][i]))
                actual.update({'branch_'+k+'_pair':v.numpy() for k,v in branches.items()})
                actual['diagnostic_logits']=torch.stack([head(*inputs,entity_override=pair)[0] for pair in branches.values()]).numpy()
        extras={'input_ids','attention_mask','encoder_hidden','target'}
        if set(saved)!=set(actual)|extras:raise ValueError('Missing or extra actual producer states')
        for name,a in actual.items():np.testing.assert_allclose(a,saved[name],atol=5e-5,rtol=3e-4,err_msg=name)
        if int(saved['logits'].argmax())!=sample['roles']['prediction']['value']:raise ValueError('Prediction differs')
        if not np.allclose(saved['alpha'].sum(-1),1,atol=2e-5):raise ValueError('Normalized attention')
        for rep in sample['representations']:
            a=saved[rep['id']]
            coords=[list(map(int,np.unravel_index(k,a.shape))) for k in sorted({0,a.size//2,a.size-1})]
            values=read_value(root,sample,rep['id'],coords)
            for coord,value in zip(coords,values):
                v=complex(value['real'],value['imag']) if isinstance(value,dict) else value
                if v!=a[tuple(coord)].item():raise ValueError('Reader coordinate differs')
                coordinates+=1
            checked+=1
        parity.setdefault((split,i),set()).add(p.identity({k:hashlib_array(saved[k]) for k in ('h','roles','mask','input_ids','attention_mask','encoder_hidden')}))
    expected={(v,c,s,i) for v in p.MODES for c in cfg['trace_checkpoints'] for s in cfg['counts'] for i in range(2)}
    if seen!=expected or any(len(v)!=1 for v in parity.values()):raise ValueError('Trace coverage/input parity differs')
    p.atomic_json(root/'sample-trace.json',payload)
    report=dict(schema='q-rmea.entity.trace-readiness.v2',structural='pass',semantic='pass',replay='pass',coordinate_reader='pass',samples=len(samples),
        complete_numeric_arrays_checked=checked,source_linked_coordinates_checked=coordinates,warnings=warnings,
        boundary='Actual cached subword encoder output to original words and head; no live encoder/internal layers/training-trajectory claim')
    p.atomic_json(root/'trace_validation.json',report)
    return report


def hashlib_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()
