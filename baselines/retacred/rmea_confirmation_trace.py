"""Private complete-value traces; public export contains aggregate readiness only."""
from pathlib import Path
import numpy as np
import torch
from . import rmea_confirmation_protocol as p
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

def legacy_close(root,samples):
    value=dict(schema_version='sample-trace.v1',trace_id='q-rmea-private',
        experiment=dict(config_sha256=p.identity(p.read(root/'manifest.json')['config']),source_sha256=p.identity(p.sources())),
        sample_selection=dict(selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples],
                              rule='first2 stable-ID-hash ordered samples per split; same samples all5 modes and both checkpoints'),
        coverage={s:('observed' if s in ('data','preprocess','scoring','evaluation','diagnosis') else 'not_applicable') for s in STAGES},samples=samples)
    p.atomic_json(root/'sample-trace-legacy.json',value)
    return legacy_audit(root)

def legacy_audit(root):
    root=Path(root);trace=p.read(root/'sample-trace-legacy.json');errors=validate(trace)
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

# v2 is emitted by this producer while the actual checkpoint and inputs exist.
# The legacy document is retained privately for the independent replay above.
from . import rmea_confirmation_values as values
from scripts.validate_sample_trace_v2 import validate as validate_v2

GROUPS=('tokenization','encoder','word_input','values','weights','prediction','diagnosis')
PURPOSES={
 'tokenization':'Inspect the actual marked subword IDs and attention mask.',
 'encoder':'Inspect the frozen encoder final states; internal layers are outside this study.',
 'word_input':'Map encoder subwords to original-word means and concatenate ordered entity-role features.',
 'values':'Project word values and compute each relation decoder contribution.',
 'weights':'Compute the declared uniform, classical, or binary-measurement relation word weights.',
 'prediction':'Combine word weights and values into relation logits without target input.',
 'diagnosis':'Replay the fixed checkpoint with context-permuted and relation-tied weights.'}

def display(summary,field=None,value=None):
    return dict(summary=summary,default_view='human_summary',visible_fields=([] if field is None else [dict(id=field,label=field,help=summary,value=value)]),visualizations=[])

_capture_legacy=capture
def capture(root,mode,checkpoint,split,selected,batch,metadata,head,checkpoint_file):
    records=_capture_legacy(root,mode,checkpoint,split,selected,batch,metadata,head,checkpoint_file)
    h,roles,mask,y,ids=batch(split,selected)
    with torch.no_grad():
        features=torch.cat((h,roles),-1)
        extra=dict(features=features)
        if mode in ('quantum','dephased'):
            extra.update(prepare_raw=head.prepare(features),effect_raw=head.effect)
        else:
            _,alpha=head(h,roles,mask)
            extra['pooled']=torch.einsum('brt,btd->brd',alpha,head.value(h))
            if mode!='uniform':
                key=head.key(features);query=head.query
                extra.update(key=key,query=query)
                if mode=='relation_softmax':extra['scores']=torch.einsum('btd,rd->brt',key,query)/(8**.5)
                else:
                    nk=torch.nn.functional.normalize(key,dim=-1);nq=torch.nn.functional.normalize(query,dim=-1)
                    extra.update(normalized_key=nk,normalized_query=nq,scores=torch.einsum('btd,rd->brt',nk,nq).square())
    shared={'effect_raw','query','normalized_query'}
    for local,record in enumerate(records):
        score=next(s['outputs'] for s in record['stages'] if s['stage']=='scoring')
        with np.load(Path(root)/score['numeric_file'],allow_pickle=False) as old:numeric={k:old[k] for k in old.files}
        numeric.update({k:(v if k in shared else v[local:local+1]).detach().cpu().numpy().copy() for k,v in extra.items()})
        meta=metadata[split][selected[local]]
        numeric.update(input_ids=np.asarray(meta['input_ids'],dtype=np.int64),attention_mask=np.asarray(meta['attention_mask'],dtype=np.int64),encoder_hidden=np.load(Path(root)/'cache'/f'{split}-encoder-{selected[local]}.npy',allow_pickle=False))
        if set(numeric)!=values.expected(mode):raise ValueError('Implemented head dataflow inventory differs')
        name=score['numeric_file'].replace('.npz','-v2.npz');np.savez(Path(root)/name,**numeric)
        record['v2_numeric_file']=name
    return records

def sample_v2(root,record):
    mode=record['condition'];checkpoint=record['checkpoint'];split=record['split'];sid=record['sample_id']
    fields={s['stage']:s['outputs'] for s in record['stages']};meta=fields['preprocess']['sample']
    ref=values.artifact(root,record['v2_numeric_file']);reps=[];groups={g:[] for g in GROUPS}
    with np.load(Path(root)/ref['path'],allow_pickle=False) as numeric:
        for name in sorted(numeric.files):
            array=numeric[name];meaning,group,axes=values.MEANINGS[name]
            if array.ndim!=len(axes):raise ValueError('Axis meaning rank differs')
            groups[group].append(name)
            count=int(array.size)
            reps.append(dict(id=name,stage_id=group,status='observed',meaning=meaning,shape=list(array.shape),dtype=str(array.dtype),axis_semantics=axes,
                portal_preview=dict(array_key=name,owner_stage=group,direction='output',reader_label=meaning,plain_language_meaning=meaning,
                   unit='radians' if 'angles' in name else 'unitless',visibility='reader_and_audit',logical_transition_group=group,
                   missing_state='observed',missing_reason=None,coordinate_policy='private loopback arbitrary coordinates; max64 values/request; no public projection'),
                artifact_refs=[ref],value_capture=dict(status='complete',shape=list(array.shape),logical_element_count=count,stored_element_count=count,artifact_refs=[ref],
                    reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',reader_ref='/api/value?sample='+sid+'&representation='+name,source_sha256=[ref['sha256']])),
                display=display(meaning)))
    # Explicit source links bind all transitions to producer-owned values.
    inputs={
      'tokenization':['source:tokens','source:entity_spans'],
      'encoder':['input_ids','attention_mask','external:pinned_roberta_large'],
      'word_input':['encoder_hidden','source:word_piece_mapping','source:entity_spans'],
      'values':['h','external:checkpoint_value_decoder'],
      'weights':['features','mask','external:checkpoint_scorer'],
      'prediction':['alpha','values','decoder','bias'],
      'diagnosis':['alpha','utilities','logits','mask','roles','source:sample_id_hash']}
    stages=[]
    canon={'tokenization':'data','encoder':'preprocess','word_input':'preprocess','values':'scoring','weights':'scoring','prediction':'generation','diagnosis':'diagnosis'}
    for i,g in enumerate(GROUPS):
        stages.append(dict(stage_id=g,stage=g,canonical_stage=canon[g],lane='replay',status='observed',step=i,checkpoint=checkpoint,
          purpose=PURPOSES[g],input_refs=inputs[g],output_refs=groups[g],display=display(PURPOSES[g]),
          outputs=dict(representations=groups[g],identity_mapping=({'word_pieces':meta['word_pieces'],'operation':'mean selected encoder subwords in float32, round to float16 cache, reload float32; preserve original word order and subject/object spans'} if g=='word_input' else None)),artifact_refs=[ref],target_access='outside_generation_input'))
    stages.append(dict(stage_id='evaluation',stage='evaluation',canonical_stage='evaluation',lane='replay',status='observed',step=7,checkpoint=checkpoint,
       purpose='Compare emitted logits against the target only after prediction.',input_refs=['logits','source:held_out_target'],output_refs=['evaluation_result'],
       outputs=fields['evaluation'],display=display('Posthoc target/prediction comparison','comparison',fields['evaluation']),target_access='post_generation_only'))
    return dict(sample_id=sid,lane='replay',split=split,checkpoint=checkpoint,
      roles=dict(source_sample=dict(tokens=meta['tokens']),sequence_or_features=dict(input_ids=meta['input_ids'],attention_mask=meta['attention_mask'],word_pieces=meta['word_pieces']),
       task_objects=dict(subject=meta['subject'],object=meta['object']),target_or_gold=dict(value=fields['evaluation']['target'],boundary='post_generation_only'),
       prediction=dict(value=fields['evaluation']['prediction']),diagnosis=dict(intervention='context-weight-permutation-v1'),
       provenance=dict(condition=mode,cache_index=record['cache_index'],sample_id_hash=meta['id_hash'],checkpoint_file=fields['scoring']['checkpoint_file'])),
      representations=reps,stages=stages)

def close(root,samples):
    legacy_samples=[{k:v for k,v in s.items() if k!='v2_numeric_file'} for s in samples]
    legacy=legacy_close(root,legacy_samples)
    manifest=p.read(Path(root)/'manifest.json');cfg=manifest['config']
    payload=dict(schema_version='sample-trace.v2',trace_id='q-rmea-confirmation-'+str(cfg['seed']),trace_completeness='complete',
       experiment=dict(run_id=Path(root).parent.name,dataset='Re-TACRED full official splits',code_revision=manifest['git_revision'],
        config_sha256=p.identity(cfg),model_identity='RoBERTa-large frozen output head; five same-interface conditions',seed=cfg['seed']),
       sample_selection=dict(selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples],rule=p.contract()['trace']['selection'],population_scope='same2 per train/valid/test for all5 modes, initial and valid-selected'),
       task_contract=dict(task_type='relation extraction frozen-head inference replay',required_roles=['source_sample','sequence_or_features','task_objects','target_or_gold','prediction','diagnosis'],
        required_representations=sorted(values.BASE|values.EXTRA),required_value_representations=sorted(values.BASE|values.EXTRA),required_splits=list(cfg['counts']),required_checkpoints=cfg['trace_checkpoints'],required_lanes=['replay'],
        checkpoint_policy='Initial and valid-selected inference replay; first-update batch evidence separate, full optimization trajectory outside declared scope.'),
       coverage={s:('observed' if s in ('data','preprocess','scoring','generation','evaluation','diagnosis') else 'not_applicable') for s in STAGES},
       samples=[sample_v2(root,s) for s in samples])
    p.atomic_json(Path(root)/'sample-trace.json',payload)
    errors,warnings=validate_v2(payload,require_complete=True)
    if errors:raise ValueError(errors)
    checked=0
    for sample in payload['samples']:
        mode=sample['roles']['provenance']['condition']
        if {r['id'] for r in sample['representations']}!=values.expected(mode):raise ValueError('Missing source-backed representation')
        for rep in sample['representations']:
            a=values.load(root,rep)
            coords=[list(np.unravel_index(k,a.shape)) for k in sorted({0,a.size//2,a.size-1})]
            result=values.coordinates(root,sample,rep['id'],coords)
            for c,v in zip(coords,result['values']):
                expected=a[tuple(c)].item();actual=complex(v['real'],v['imag']) if isinstance(v,dict) else v
                if actual!=expected:raise ValueError('Coordinate reader/source differs')
                checked+=1
    report=dict(legacy,schema='q-rmea.trace-readiness.v2',structural='pass',semantic='pass',coordinate_reader='pass',
        source_linked_coordinates_checked=checked,private_reader='scripts/view_rmea_confirmation_trace.py --run-dir PHASE; loopback only, no public raw values',warnings=warnings)
    p.atomic_json(Path(root)/'trace_validation.json',report);return report
