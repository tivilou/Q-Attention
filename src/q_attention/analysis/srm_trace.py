"""Producer-owned factual Q-SRM traces and independent NumPy readiness checks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from q_attention.analysis.ecpa_trace import display, read_values
from q_attention.analysis.ecpa_contract_validator import validate as structural_validate
from q_attention.analysis.cfra_trace import sha, state_sha, vector
from q_attention.experiments.cfra_protocol import atomic_json

COMMON = ('input_ids','attention_mask','subject_mask','object_mask','token_embedding','position_embedding',
          'embedding','first_hidden','query','key','value','base_scores','attention','null_probability',
          'context','hidden','pooled','logits','probabilities','gold','loss_components',
          'parameter_before','parameter_gradient','parameter_after','parameter_delta')
STATES = ('query_state','key_state','encoded_key_state','key_density','permutation','query_angles','key_angles',
          'query_circuit_states','key_circuit_states')
POVM = ('gram','inverse_root','spectrum','effects','null_effect','completeness')


def inventory(mode):
    return list(COMMON) + ([] if mode == 'disabled' else ['plugin_'+x for x in
        STATES + (('fidelity','denominator') if mode == 'fidelity' else POVM)])


def stage_for(name):
    if name.startswith('parameter_'):return 'training'
    if name in ('gold','loss_components'):return 'evaluation'
    if name in ('pooled','logits','probabilities'):return 'generation'
    if name in ('context','hidden'):return 'context'
    if name in COMMON[:8]:return 'preprocess'
    return 'scoring'


def axes_for(name):
    if name in ('input_ids','attention_mask','subject_mask','object_mask','plugin_permutation'):return ['token']
    if name in ('token_embedding','position_embedding','embedding','first_hidden','hidden'):return ['token','feature']
    if name in ('query','key','value','context'):return ['head','token','head_feature']
    if name in ('base_scores','attention','plugin_fidelity'):return ['head','query_token','key_token']
    if name in ('null_probability','plugin_denominator'):return ['head','query_token']
    if name in ('logits','probabilities'):return ['relation']
    if name=='gold':return []
    if name=='pooled':return ['pooled_feature']
    if name=='loss_components':return ['factual_CE']
    if name.startswith('parameter_'):return ['named_trainable_parameter']
    if name in ('plugin_query_state','plugin_key_state','plugin_encoded_key_state'):return ['head','token','basis']
    if name in ('plugin_key_density','plugin_effects'):return ['head','token','basis_row','basis_col']
    if name in ('plugin_query_angles','plugin_key_angles'):return ['head','token','angle']
    if name in ('plugin_query_circuit_states','plugin_key_circuit_states'):return ['head','token','circuit_stage','basis']
    if name=='plugin_spectrum':return ['head','eigenvalue']
    return ['head','basis_row','basis_col']


def values(model,gold,loss,before,gradient,after):
    result={k:v[0].detach().cpu().clone() for k,v in model.observed.items()}
    result.update(gold=gold[0].detach().cpu(),loss_components=loss.detach().cpu().reshape(1),
                  parameter_before=before,parameter_gradient=gradient,parameter_after=after,parameter_delta=after-before)
    return result


class Trace:
    def __init__(self,root,mode,experiment,labels,parameter_names):
        self.root,self.mode,self.labels=Path(root),mode,labels
        path=self.root/'sample_trace.json'
        if path.exists():
            self.payload=json.loads(path.read_text())
            if self.payload['experiment']!=experiment:raise ValueError('trace identity changed')
        else:
            names=inventory(mode)
            self.payload=dict(schema_version='sample-trace.v2',trace_id=f'srm-{mode}-{experiment["seed"]}',
                trace_completeness='partial',experiment=experiment,
                sample_selection=dict(rule='First selected source record per split, fixed before training.',
                    population_scope=experiment['profile'],selected_count=0,selected_sample_ids=[]),
                coverage={s:('not_applicable' if s in ('retrieval','selection','diagnosis') else 'observed') for s in
                    ('data','preprocess','scoring','context','generation','evaluation','training','retrieval','selection','diagnosis')},
                task_contract=dict(task_type='relation classification',
                    required_roles=['sentence','tokens','subject','object','correct_relation','predicted_relation','checkpoint_identity','input_provenance'],
                    required_representations=names,required_value_representations=names,
                    required_splits=['train','valid','test'],required_checkpoints=['initial','best','final'],required_lanes=['replay','training'],
                    checkpoint_policy='Each split at initial/best/final; selected train record in actual update0; replay never steps optimizer.'),
                parameter_order=parameter_names,label_map=labels,samples=[])

    def add(self,row,split,checkpoint,lane,tensors,updates,model_sha):
        names=inventory(self.mode)
        if set(tensors)!=set(names):raise ValueError('trace inventory mismatch: '+str(set(tensors)^set(names)))
        sid=f'{split}-selected0:{checkpoint}:{lane}';reps=[];by_stage={}
        for name in names:
            arr=tensors[name].numpy()
            if arr.dtype.hasobject or not np.isfinite(arr).all():raise ValueError('invalid trace values: '+name)
            relative=f'values/{sid.replace(":","_")}/{name}.npy'
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.save(stream,arr,allow_pickle=False)
            temporary.replace(path)
            ref=dict(path=relative,sha256=sha(path),bytes=path.stat().st_size,visibility='exchange_only')
            stage=stage_for(name);by_stage.setdefault(stage,[]).append(name)
            meaning=f'Producer-owned factual Q-SRM {name}; training gradient/loss covers the logical minibatch; replay covers this sample.'
            reps.append(dict(id=name,stage_id=stage,status='observed',meaning=meaning,shape=list(arr.shape),dtype=str(arr.dtype),
                axis_semantics=axes_for(name),artifact_refs=[ref],display=display(name,meaning,name),
                value_capture=dict(status='complete',shape=list(arr.shape),logical_element_count=arr.size,stored_element_count=arr.size,
                    artifact_refs=[ref],reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                    reader_ref=f'q_attention.analysis.ecpa_trace:read_values#{sid}/{name}',source_sha256=[ref['sha256']]))))
        consumed=dict(data=['source:sentence','source:entity_spans'],preprocess=['tokens','entity_spans','external:model_checkpoint'],
            scoring=['first_hidden','attention_mask','external:model_checkpoint'],context=['attention','null_probability','value','first_hidden','external:model_checkpoint'],
            generation=['hidden','subject_mask','object_mask','attention_mask','external:model_checkpoint'],
            evaluation=['logits','source:correct_relation'],training=['loss_components','external:optimizer_checkpoint'])
        stages=[]
        for stage in ('data','preprocess','scoring','context','generation','evaluation','training'):
            purpose=f'Q-SRM factual {stage}; labels are unavailable to the attention module.'
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=stage,lane=lane,status='observed',purpose=purpose,
                input_refs=consumed[stage],output_refs=['tokens','entity_spans'] if stage=='data' else by_stage.get(stage,[]),checkpoint=model_sha,
                target_access='post_generation_only' if stage=='generation' else 'training_only' if stage=='training' else 'outside_generation_input' if stage=='evaluation' else 'not_available',
                display=display(stage,purpose)))
        def entity(role):
            span=getattr(row,role);field='subj_type' if role=='subject' else 'obj_type'
            return dict(text=' '.join(row.tokens[span[0]:span[1]]),span=list(span),type=row.metadata.get(field,'UNKNOWN'))
        inverse={v:k for k,v in self.labels.items()}
        roles=dict(sentence=' '.join(row.tokens),tokens=list(row.tokens),subject=entity('subject'),object=entity('object'),
            correct_relation=row.label,predicted_relation=inverse[int(tensors['logits'].argmax())],
            input_provenance=dict(policy='factual input only; no neutralization or prediction-derived selection',mode=self.mode),
            checkpoint_identity=dict(model_state_sha256=model_sha,optimizer_updates_completed=updates,
                capture='actual update0 before optimizer step' if lane=='training' else 'diagnostic replay; no optimizer update'))
        item=dict(sample_id=sid,lane=lane,split=split,checkpoint=checkpoint,roles=roles,representations=reps,stages=stages)
        self.payload['samples']=[s for s in self.payload['samples'] if s['sample_id']!=sid]+[item]
        self.save()

    def save(self):
        self.payload['sample_selection'].update(selected_count=len(self.payload['samples']),selected_sample_ids=[s['sample_id'] for s in self.payload['samples']])
        atomic_json(self.root/'sample_trace.json',self.payload)


def _close(actual,expected,label,atol=3e-6):
    if not np.allclose(actual,expected,atol=atol,rtol=2e-5):raise ValueError(label+' lineage differs')


def _circuit(features,params,prefix):
    projected=features@params[prefix+'.projection.weight'].T+params[prefix+'.projection.bias']
    # Projection and tanh are float32 in the base model; gate arithmetic is float64.
    angles=np.pi*np.tanh(projected).astype(np.float64)
    state=np.zeros((*features.shape[:-1],8),dtype=np.complex128);state[...,0]=1
    checkpoints=[state.copy()]
    def rotate(angle,axis,wire):
        nonlocal state
        if axis=='y':
            c,s=np.cos(angle/2),np.sin(angle/2)
            gate=np.stack([c,-s,s,c],-1).reshape(*np.shape(angle),2,2)
        else:
            zero=np.zeros_like(angle)
            gate=np.stack([np.exp(-.5j*angle),zero,zero,np.exp(.5j*angle)],-1).reshape(*np.shape(angle),2,2)
        shape=state.shape;parts=state.reshape(*shape[:-1],2,2,2)
        parts=np.moveaxis(parts,-3+wire,-1)
        parts=np.einsum('...ij,...kj->...ki',gate,parts.reshape(*shape[:-1],4,2))
        state=np.moveaxis(parts.reshape(*shape[:-1],2,2,2),-1,-3+wire).reshape(shape)
    for wire in range(3):
        rotate(angles[...,wire*2],'y',wire);rotate(angles[...,wire*2+1],'z',wire)
    checkpoints.append(state.copy())
    sign=np.array([(-1)**(((i>>2&1)*(i>>1&1))+((i>>1&1)*(i&1))+((i&1)*(i>>2&1))) for i in range(8)])
    for layer in range(2):
        for wire in range(3):
            for idx,axis in enumerate(('z','y','z')):rotate(float(params[prefix+'.rotations'][layer,wire,idx]),axis,wire)
        state=state*sign;checkpoints.append(state.copy())
    return angles,np.stack(checkpoints,-2)


def validate_trace(path,*,complete=True):
    path=Path(path);payload=json.loads(path.read_text())
    errors,warnings=structural_validate(payload,require_complete=complete)
    if errors:raise ValueError('structural trace: '+'; '.join(errors[:8]))
    mode=payload['experiment']['model_identity'];expected=set(inventory(mode))
    if set(payload['task_contract']['required_value_representations'])!=expected:raise ValueError('omitted frozen inventory')
    coverage=set();seen=set();count=0
    for sample in payload['samples']:
        sid=sample['sample_id']
        if sid in seen:raise ValueError('duplicate sample')
        seen.add(sid);coverage.add((sample['split'],sample['checkpoint'],sample['lane']))
        reps={r['id']:r for r in sample['representations']}
        if set(reps)!=expected or len(reps)!=len(sample['representations']):raise ValueError('representation inventory differs')
        a={}
        for name,rep in reps.items():
            ref=rep['artifact_refs'][0];file=(path.parent/ref['path']).resolve()
            if not file.is_relative_to(path.parent.resolve()) or sha(file)!=ref['sha256']:raise ValueError('unsafe/mutated numeric artifact')
            arr=np.load(file,allow_pickle=False);cap=rep['value_capture']
            if (not np.isfinite(arr).all() or list(arr.shape)!=rep['shape'] or str(arr.dtype)!=rep['dtype']
                or arr.size!=cap['stored_element_count'] or arr.size!=cap['logical_element_count'] or file.stat().st_size!=ref['bytes']):
                raise ValueError('incomplete numeric artifact')
            for coords in (tuple(0 for _ in arr.shape),tuple(n-1 for n in arr.shape)):
                for component,fn in (('real',np.real),('imag',np.imag)):
                    if not np.array_equal(read_values(path,sid,name,coords,component=component),fn(arr[coords])):raise ValueError('reader coordinate mismatch')
            a[name]=arr;count+=1
        roles=sample['roles'];identity=roles['checkpoint_identity']['model_state_sha256']
        if any(stage['checkpoint']!=identity for stage in sample['stages']):raise ValueError('checkpoint lineage differs')
        params={};cursor=0
        for item in payload['parameter_order']:
            end=cursor+item['numel'];params[item['name']]=a['parameter_before'][cursor:end].reshape(item['shape']);cursor=end
        if any(a[n].size!=cursor for n in ('parameter_before','parameter_gradient','parameter_after','parameter_delta')):raise ValueError('parameter layout differs')
        if int(a['gold'])!=payload['label_map'][roles['correct_relation']]:raise ValueError('target differs')
        if int(a['logits'].argmax())!=payload['label_map'][roles['predicted_relation']]:raise ValueError('prediction differs')
        mask=a['attention_mask'].astype(bool)
        for role in ('subject','object'):
            start,end=roles[role]['span']
            if not a[role+'_mask'][start:end].all() or int(a[role+'_mask'].sum())!=end-start:raise ValueError('entity mask differs')
        _close(a['embedding'],a['token_embedding']+a['position_embedding'],'embedding')
        _close(a['attention']@a['value'],a['context'],'context')
        _close(a['parameter_after']-a['parameter_before'],a['parameter_delta'],'update',atol=1e-7)
        _close(a['attention'].sum(-1)+a['null_probability'],1,'token/null normalization',atol=1e-9)
        if a['attention'].min() < -1e-12 or a['null_probability'].min() < -1e-12 or np.any(a['attention'][...,~mask]!=0):raise ValueError('probability support fails')
        prob=np.exp(a['logits']-a['logits'].max());prob/=prob.sum()
        _close(a['probabilities'],prob,'classifier probability')
        if mode=='disabled':
            scores=a['base_scores'].copy();scores[...,~mask]=-1e30
            ex=np.exp(scores-scores.max(-1,keepdims=True))
            _close(a['attention'],ex/ex.sum(-1,keepdims=True),'disabled softmax')
            _close(a['null_probability'],0,'disabled null')
            continue
        for prefix in ('query','key'):
            angles,states=_circuit(a[prefix],params,'kernel.'+prefix+'_map')
            _close(a['plugin_'+prefix+'_angles'],angles,prefix+' input angles')
            _close(a['plugin_'+prefix+'_circuit_states'],states,prefix+' circuit',atol=1e-5)
        q=a['plugin_query_state'];encoded=a['plugin_encoded_key_state'];key=a['plugin_key_state']
        _close(q,a['plugin_query_circuit_states'][...,-1,:],'query encoding')
        _close(encoded,a['plugin_key_circuit_states'][...,-1,:],'key encoding')
        perm=np.arange(len(mask));valid=np.flatnonzero(mask)
        if mode=='key_mismatch':perm[valid]=np.roll(valid,1)
        if not np.array_equal(a['plugin_permutation'],perm):raise ValueError('mismatch permutation differs')
        _close(key,encoded[:,perm],'permuted key')
        rho=key[..., :,None]*key.conj()[...,None,:]
        if mode=='diagonal':rho=rho*np.eye(8)
        _close(a['plugin_key_density'],rho,'key density')
        if mode=='fidelity':
            f=np.abs(q.conj()@key.swapaxes(-1,-2))**2*mask[None,None,:]
            denom=f.sum(-1)+.01;attention=f/denom[...,None];null=.01/denom
            _close(a['plugin_fidelity'],f,'pairwise fidelity');_close(a['plugin_denominator'],denom,'fidelity denominator')
        else:
            gram=(rho*mask[None,:,None,None]).sum(-3)+.01*np.eye(8)
            ev,u=np.linalg.eigh(gram);g=(u/np.sqrt(ev)[...,None,:])@u.conj().swapaxes(-1,-2)
            effects=g[:,None]@(rho*mask[None,:,None,None])@g[:,None]
            null_effect=.01*g@g
            attention=np.einsum('hqi,hkij,hqj->hqk',q.conj(),effects,q).real
            null=np.einsum('hqi,hij,hqj->hq',q.conj(),null_effect,q).real
            for name,reference in [('gram',gram),('spectrum',ev),('inverse_root',g),('effects',effects),('null_effect',null_effect),('completeness',effects.sum(-3)+null_effect-np.eye(8))]:
                _close(a['plugin_'+name],reference,name,atol=1e-9)
            if np.linalg.eigvalsh(effects).min() < -1e-9:raise ValueError('effects not PSD')
            _close(effects.sum(-3)+null_effect,np.eye(8),'POVM completeness',atol=1e-9)
        attention*=mask[None,:,None];null=np.where(mask[None,:],null,1)
        _close(a['attention'],attention,'Born probability',atol=1e-9)
        _close(a['null_probability'],null,'null probability',atol=1e-9)
    if complete:
        wanted={(s,c,'replay') for s in ('train','valid','test') for c in ('initial','best','final')}
        wanted.add(('train','update0','training'))
        if coverage!=wanted:raise ValueError('split/checkpoint coverage differs')
    return dict(status='passed',samples=len(seen),full_value_artifacts=count,structural_warnings=warnings)
