"""Complete private Q-CFRA sample values and independent readiness checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from q_attention.analysis.ecpa_trace import atomic_json,display,read_values
from q_attention.analysis.ecpa_contract_validator import validate as structural_validate

COMMON = ('view_ids','attention_mask','subject_mask','object_mask','neutral_ids',
          'token_embedding','position_embedding','embedding','first_hidden','query','key','value',
          'base_scores','attention_logits','attention','context','hidden','pooled','logits','probabilities',
          'gold','loss_components','parameter_gradient','parameter_before','parameter_after','parameter_delta')
GEOMETRY = ('projected','fidelity','covariance','geometry_features','readout_inputs','readout_angles','logits')
QUANTUM = ('amplitudes','pure','density','paired_density','channel_weights','channel_unitaries','bloch','commutator','circuit_states')
CLASSICAL = ('real_states','real_readout')


def inventory(mode):
    names = list(COMMON)
    if mode != 'disabled':
        names += ['plugin_'+x for x in GEOMETRY+(CLASSICAL if mode.startswith('classical') else QUANTUM)]
    return names


def stage_for(name):
    if name in ('view_ids','attention_mask','subject_mask','object_mask','neutral_ids','token_embedding','position_embedding','embedding','first_hidden'):
        return 'preprocess'
    if name in ('context','hidden'):
        return 'context'
    if name in ('pooled','logits','probabilities'):
        return 'generation'
    if name in ('gold','loss_components'):
        return 'evaluation'
    if name.startswith('parameter_'):
        return 'training'
    return 'scoring'


def axes_for(name):
    if name in ('attention_mask','subject_mask','object_mask'):
        return ['token']
    if name=='view_ids':
        return ['view','token']
    if name=='neutral_ids':
        return ['ordered_entity_role']
    if name in ('token_embedding','position_embedding','embedding','first_hidden','hidden'):
        return ['view','token','feature']
    if name in ('query','key','value','context'):
        return ['view','head','token','head_feature']
    if name in ('base_scores','attention_logits','attention','plugin_logits'):
        return ['view','head','query_token','key_token']
    if name=='pooled':
        return ['view','pooled_feature']
    if name in ('logits','probabilities'):
        return ['view','relation']
    if name=='gold':
        return []
    if name=='loss_components':
        return ['loss_term_CE_predictionJS_attentionJS_total']
    if name.startswith('parameter_'):
        return ['named_trainable_parameter']
    suffix=name.removeprefix('plugin_')
    return {
        'projected':['view','token','projected_component'],
        'amplitudes':['view','token','qubit_basis'],
        'pure':['view','token','basis_row','basis_col'],
        'density':['view','token','basis_row','basis_col'],
        'paired_density':['factual_subjectCF_objectCF','token','basis_row','basis_col'],
        'channel_weights':['channel_component'],
        'channel_unitaries':['channel_component','basis_row','basis_col'],
        'bloch':['factual_subjectCF_objectCF','token','Pauli_axis'],
        'commutator':['token'],'fidelity':['token','subject_object_contrast'],
        'covariance':['token'],'geometry_features':['token','geometry_feature'],
        'readout_inputs':['view','head','query_token','key_token','input_feature'],
        'readout_angles':['view','head','query_token','key_token','input_feature'],
        'circuit_states':['view','head','query_token','key_token','circuit_checkpoint','two_qubit_basis'],
        'real_states':['view','token','real_feature'],
        'real_readout':['view','head','query_token','key_token','readout_feature']
    }[suffix]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str((tuple(t.shape),t.dtype)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def vector(model,gradient=False):
    tensors = []
    for _,p in model.named_parameters():
        if p.requires_grad:
            t = (p.grad if p.grad is not None else torch.zeros_like(p)) if gradient else p
            tensors.append(t.detach().flatten().cpu())
    return torch.cat(tensors)


def values(model,gold,components,before,gradient,after):
    # Channel weights/unitaries are producer-owned global parameters, not a batch axis.
    result = {k:(t if k in ('plugin_channel_weights','plugin_channel_unitaries') else t[0]).detach().cpu().clone()
              for k,t in model.observed.items()}
    result.update(gold=gold[0].detach().cpu(),loss_components=components.detach().cpu(),
                  parameter_gradient=gradient,parameter_before=before,parameter_after=after,parameter_delta=after-before)
    return result


class Trace:
    def __init__(self,root,mode,experiment,labels,parameter_names):
        self.root,self.mode,self.labels = Path(root),mode,labels
        self.names = inventory(mode)
        path = self.root/'sample_trace.json'
        if path.exists():
            self.payload = json.loads(path.read_text())
            if self.payload['experiment'] != experiment:
                raise ValueError('trace experiment identity changed')
        else:
            self.payload = dict(schema_version='sample-trace.v2',trace_id=f'cfra-{mode}-{experiment["seed"]}',
                trace_completeness='partial',experiment=experiment,
                sample_selection=dict(rule='First hash-ordered selected record per split, frozen before training.',
                    population_scope='bounded Re-TACRED subset; NOT formal data evidence',selected_count=0,selected_sample_ids=[]),
                coverage={s:('not_applicable' if s in ('retrieval','selection','diagnosis') else 'observed') for s in
                          ('data','preprocess','scoring','context','generation','evaluation','training','retrieval','selection','diagnosis')},
                task_contract=dict(task_type='relation classification',required_roles=['sentence','tokens','subject','object','correct_relation','predicted_relation','checkpoint_identity','counterfactual_provenance'],
                    required_representations=self.names,required_value_representations=self.names,
                    required_splits=['train','valid','test'],required_checkpoints=['initial','best','final'],required_lanes=['replay','training'],
                    checkpoint_policy='Every split at initial/best/final; first selected train record in actual optimizer update0.'),
                parameter_order=parameter_names,label_map=labels,samples=[])

    def add(self,row,split,checkpoint,lane,tensors,updates,model_sha):
        if set(tensors) != set(self.names):
            raise ValueError(f'trace inventory mismatch {set(tensors)^set(self.names)}')
        sample_id = f'{split}-selected0:{checkpoint}:{lane}'
        reps,by_stage = [],{}
        for name in self.names:
            arr = tensors[name].numpy()
            if arr.dtype.hasobject or not np.isfinite(arr).all():
                raise ValueError(f'invalid complete values {name}')
            rel = f'values/{sample_id.replace(":","_")}/{name}.npy'
            path = self.root/rel
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.save(f,arr,allow_pickle=False)
            temporary.replace(path)
            ref = dict(path=rel,sha256=sha(path),bytes=path.stat().st_size,visibility='exchange_only')
            stage = stage_for(name)
            by_stage.setdefault(stage,[]).append(name)
            meaning = f'Actual producer-owned {name}; views are factual, subject-neutralized, object-neutralized. Training gradients describe the complete logical minibatch; replay gradients describe this sample.'
            reps.append(dict(id=name,stage_id=stage,status='observed',meaning=meaning,shape=list(arr.shape),dtype=str(arr.dtype),
                axis_semantics=axes_for(name),artifact_refs=[ref],display=display(name,meaning,name),
                value_capture=dict(status='complete',shape=list(arr.shape),logical_element_count=arr.size,stored_element_count=arr.size,
                    artifact_refs=[ref],reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                        reader_ref=f'q_attention.analysis.ecpa_trace:read_values#{sample_id}/{name}',source_sha256=[ref['sha256']]))))
        consumed = dict(data=['source:sentence','source:entity_spans','source:entity_types'],
            preprocess=['tokens','entity_spans','entity_types','external:model_checkpoint'],
            scoring=['first_hidden','attention_mask','subject_mask','object_mask','external:model_checkpoint'],
            context=['first_hidden','attention','value','external:model_checkpoint'],
            generation=['hidden','subject_mask','object_mask','attention_mask','external:model_checkpoint'],
            evaluation=['logits','attention','source:correct_relation'],training=['loss_components','external:optimizer_checkpoint'])
        stages = []
        for stage in ('data','preprocess','scoring','context','generation','evaluation','training'):
            purpose = f'Producer-owned Q-CFRA {stage}; targets only enter post-prediction loss/evaluation.'
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=stage,lane=lane,status='observed',purpose=purpose,
                input_refs=consumed[stage],output_refs=['tokens','entity_spans','entity_types'] if stage=='data' else by_stage.get(stage,[]),
                checkpoint=model_sha,target_access='post_generation_only' if stage=='generation' else 'training_only' if stage=='training' else 'outside_generation_input' if stage=='evaluation' else 'not_available',display=display(stage,purpose)))
        def entity(role,field):
            span = getattr(row,role)
            return dict(text=' '.join(row.tokens[span[0]:span[1]]),span=list(span),type=row.metadata.get(field,'UNKNOWN'))
        inverse = {v:k for k,v in self.labels.items()}
        roles = dict(sentence=' '.join(row.tokens),tokens=list(row.tokens),subject=entity('subject','subj_type'),object=entity('object','obj_type'),
            correct_relation=row.label,predicted_relation=inverse[int(tensors['logits'][0].argmax())],
            counterfactual_provenance=dict(policy='one token replaced per entity position by role/type ID; no target',mode=self.mode),
            checkpoint_identity=dict(model_state_sha256=model_sha,optimizer_updates_completed=updates,
                capture='actual minibatch update; captured state is before optimizer step' if lane=='training' else 'diagnostic replay, no optimizer update'))
        sample = dict(sample_id=sample_id,lane=lane,split=split,checkpoint=checkpoint,roles=roles,representations=reps,stages=stages)
        self.payload['samples'] = [s for s in self.payload['samples'] if s['sample_id'] != sample_id]+[sample]
        self.save()

    def save(self):
        self.payload['sample_selection'].update(selected_count=len(self.payload['samples']),selected_sample_ids=[s['sample_id'] for s in self.payload['samples']])
        atomic_json(self.root/'sample_trace.json',self.payload)


def validate_trace(path,*,complete=True):
    path = Path(path)
    payload = json.loads(path.read_text())
    errors,warnings = structural_validate(payload,require_complete=complete)
    if errors:
        raise ValueError('structural trace: '+'; '.join(errors[:8]))
    expected = set(inventory(payload['experiment']['model_identity']))
    if set(payload['task_contract']['required_value_representations']) != expected:
        raise ValueError('omitted frozen trace representations')
    coverage,set_ids,count = set(),set(),0
    for sample in payload['samples']:
        if sample['sample_id'] in set_ids:
            raise ValueError('duplicate sample')
        set_ids.add(sample['sample_id'])
        coverage.add((sample['split'],sample['checkpoint'],sample['lane']))
        reps = {r['id']:r for r in sample['representations']}
        if set(reps) != expected or len(reps) != len(sample['representations']):
            raise ValueError('representation inventory differs')
        arrays = {}
        identity = sample['roles']['checkpoint_identity']['model_state_sha256']
        if any(s['checkpoint'] != identity for s in sample['stages']):
            raise ValueError('checkpoint lineage differs')
        for name,rep in reps.items():
            artifact = rep['artifact_refs'][0]
            file = (path.parent/artifact['path']).resolve()
            if not file.is_relative_to(path.parent.resolve()) or sha(file) != artifact['sha256']:
                raise ValueError('unsafe/mutated numeric artifact')
            a = np.load(file,allow_pickle=False)
            cap = rep['value_capture']
            if list(a.shape)!=rep['shape'] or str(a.dtype)!=rep['dtype'] or not np.isfinite(a).all() or a.size!=cap['stored_element_count'] or a.size!=cap['logical_element_count'] or file.stat().st_size!=artifact['bytes']:
                raise ValueError('incomplete numeric artifact')
            for coords in (tuple(0 for _ in a.shape),tuple(n-1 for n in a.shape)):
                for component,fn in (('real',np.real),('imag',np.imag)):
                    if not np.array_equal(read_values(path,sample['sample_id'],name,coords,component=component),fn(a[coords])):
                        raise ValueError('reader/source coordinate mismatch')
            arrays[name]=a; count+=1
        a = arrays
        expected_parameters=sum(p['numel'] for p in payload['parameter_order'])
        if any(a[n].size!=expected_parameters for n in ('parameter_before','parameter_after','parameter_delta','parameter_gradient')):
            raise ValueError('parameter layout/capture differs')
        if int(a['gold'])!=payload['label_map'][sample['roles']['correct_relation']]:
            raise ValueError('target differs from declared training label map')
        for role in ('subject','object'):
            start,end=sample['roles'][role]['span']
            if not np.all(a[role+'_mask'][start:end]) or int(a[role+'_mask'].sum())!=end-start:
                raise ValueError('actual entity mask differs from source role')
        if not np.allclose(a['embedding'],a['token_embedding']+a['position_embedding'],atol=1e-6):
            raise ValueError('embedding lineage differs')
        if not np.allclose(a['attention'] @ a['value'],a['context'],atol=2e-6):
            raise ValueError('attention/context lineage differs')
        if not np.allclose(a['parameter_after']-a['parameter_before'],a['parameter_delta'],atol=1e-7):
            raise ValueError('optimizer update lineage differs')
        mask = a['attention_mask'].astype(bool)
        if not np.allclose(a['attention'].sum(-1),1,atol=1e-5) or np.any(a['attention'][...,~mask] != 0):
            raise ValueError('attention normalization/padding fails')
        scores=a['attention_logits'].copy()
        scores[...,~mask]=-1e30
        exp=np.exp(scores-scores.max(-1,keepdims=True))
        if not np.allclose(exp/exp.sum(-1,keepdims=True),a['attention'],atol=2e-6):
            raise ValueError('logit/masked-softmax lineage differs')
        mode = payload['experiment']['model_identity']
        if mode=='disabled' and not np.array_equal(a['base_scores'],a['attention_logits']):
            raise ValueError('disabled score parity fails')
        if mode!='disabled' and not mode.startswith('classical'):
            rho = a['plugin_density']
            if not np.allclose(rho,np.swapaxes(rho.conj(),-1,-2),atol=1e-6) or not np.allclose(np.trace(rho,axis1=-2,axis2=-1),1,atol=1e-5) or np.linalg.eigvalsh(rho).min() < -1e-6:
                raise ValueError('density is not PSD, Hermitian and unit trace')
            amplitudes=a['plugin_amplitudes']
            if not np.allclose(a['plugin_pure'],amplitudes[..., :,None]*amplitudes.conj()[...,None,:],atol=1e-6):
                raise ValueError('amplitude/pure-state lineage differs')
            weights,unitaries=a['plugin_channel_weights'],a['plugin_channel_unitaries']
            if not np.allclose(weights.sum(),1,atol=1e-6) or np.any(weights<0):
                raise ValueError('invalid CPTP channel weights')
            if not np.allclose(unitaries @ np.swapaxes(unitaries.conj(),-1,-2),np.eye(2),atol=1e-6):
                raise ValueError('invalid channel unitary')
            reproduced=sum(weights[i]*(unitaries[i] @ a['plugin_pure'] @ unitaries[i].conj().T) for i in range(3))
            reproduced=(1-1e-4)*reproduced+1e-4*np.eye(2)/2
            if mode=='diagonal':
                reproduced=reproduced*np.eye(2)
            if not np.allclose(reproduced,rho,atol=2e-6):
                raise ValueError('CPTP channel output lineage differs')
            first,second,third=a['plugin_paired_density'].astype(np.complex128)
            eigen,basis=np.linalg.eigh(first)
            root=(basis*np.sqrt(np.maximum(eigen,0))[...,None,:]) @ np.swapaxes(basis.conj(),-1,-2)
            reference=[]
            for other in (second,third):
                ev=np.linalg.eigvalsh(root @ other @ root)
                reference.append(np.square(np.sqrt(np.maximum(ev,0)).sum(-1)).clip(1e-5,1-1e-5))
            if not np.allclose(np.stack(reference,-1),a['plugin_fidelity'],atol=2e-5,rtol=2e-5):
                raise ValueError('independent spectral fidelity lineage differs')
            r,rs,ro=a['plugin_bloch']
            x,y=r-rs,r-ro
            covariance=(x*y).sum(-1)-(r*x).sum(-1)*(r*y).sum(-1)
            if not np.allclose(covariance,a['plugin_covariance'],atol=2e-6):
                raise ValueError('noncommuting covariance lineage differs')
            if not np.allclose((np.abs(a['plugin_circuit_states'])**2).sum(-1),1,atol=1e-5):
                raise ValueError('circuit normalization fails')
            logits = ((np.abs(a['plugin_circuit_states'][...,-1,:])**2)*np.array([3,-1,-1,-1])).sum(-1)/.7
            if not np.allclose(logits,a['attention_logits'],atol=2e-6):
                raise ValueError('circuit observable/logit lineage differs')
        if mode!='disabled':
            inputs=a['plugin_readout_inputs']
            if not np.allclose(inputs[...,0],a['base_scores'],atol=1e-6) or not np.allclose(inputs[...,1:],a['plugin_geometry_features'][None,None,None,:,:],atol=1e-6):
                raise ValueError('counterfactual geometry/readout input lineage differs')
            if not np.allclose(np.pi*np.tanh(inputs*np.array([1,.25,.25,1])),a['plugin_readout_angles'],atol=1e-6):
                raise ValueError('angle encoding lineage differs')
    if complete:
        wanted = {(s,c,'replay') for s in ('train','valid','test') for c in ('initial','best','final')}
        wanted.add(('train','update0','training'))
        if coverage != wanted:
            raise ValueError(f'checkpoint/split coverage differs: {coverage^wanted}')
    return dict(status='passed',samples=len(set_ids),full_value_artifacts=count,structural_warnings=warnings)
