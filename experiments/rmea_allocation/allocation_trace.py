"""Post-training, selected-checkpoint replay at the frozen-feature boundary.

Traces are disposable private inference products, never training activations.
The selected checkpoint, fixed input NPZ, recipe and their identities are the
replay dependencies. This module does not certify encoder/data provenance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from legacy.rmea import MODES as VARIANTS, RMEAHead as make_head
from scripts.validate_sample_trace_v2 import STAGES, validate


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned(root, ref):
    root = Path(root).resolve()
    name = ref['path']
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Expected a package-relative artifact')
    p = root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or sha(p) != ref['sha256']:
        raise ValueError('Artifact path or hash mismatch')
    return p


def ref(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                bytes=path.stat().st_size, visibility='private')


def display(text):
    return dict(summary=text, default_view='human_summary', visible_fields=[], visualizations=[])


def source_files():
    root=Path(__file__).resolve().parents[2]
    names=['experiments/rmea_allocation/'+n for n in ('allocation_trace.py','hybrid_trace.py','legacy/rmea.py','legacy/frozen_feature_head.py','scripts/validate_sample_trace_v2.py','runner.py','contract.json')]
    return {n:sha(root/n) for n in names}


def check_manifest(root,m):
    if m.get('schema')!='q-rmea.allocation.selected-replay.v1' or m.get('lane')!='replay':
        raise ValueError('Selected replay manifest required')
    for k in ('source_revision','scope','sample_selection_rule'):
        if not isinstance(m.get(k),str) or not m[k].strip():raise ValueError('Missing manifest '+k)
    if m.get('source_files')!=source_files():raise ValueError('Replay implementation identity differs')
    models=m.get('models');samples=m.get('samples')
    if not isinstance(models,list) or not models or not all(isinstance(x,dict) for x in models):raise ValueError('Model list required')
    if not all(isinstance(x.get('variant'),str) and x['variant'] in VARIANTS and isinstance(x.get('selection_rule'),str) and x['selection_rule'].strip() for x in models):raise ValueError('Model identity/selection rule required')
    if len({x['variant'] for x in models})!=len(models):raise ValueError('Duplicate model')
    if not isinstance(samples,list) or not samples or not all(isinstance(s,dict) and isinstance(s.get('sample_id'),str) and s['sample_id'].strip() and s.get('split') in ('train','valid','test') for s in samples):raise ValueError('Fixed sample/split identities required')
    if len({s['sample_id'] for s in samples})!=len(samples):raise ValueError('Duplicate sample')
    for item in models:pinned(root,item['checkpoint'])
    pinned(root,m['inputs'])
    return models,samples


def stage_inputs(arrays):
    prep=[k for k in arrays if owner(k)=='preparation']
    return {'inputs':['external:pinned_features','external:fixed_sample_manifest'],
        'preparation':['features','h','roles','mask','external:selected_checkpoint'],
        'values':['h','external:selected_checkpoint'],
        'measurement':prep+['features','mask','external:selected_checkpoint']+(['utilities'] if 'sensitivity' in arrays else []),
        'prediction':['alpha','utilities','bias']}


def head_from_checkpoint(root, item):
    if item['variant'] not in VARIANTS or not item.get('selection_rule'):
        raise ValueError('Explicit variant and evaluation-model selection rule required')
    state = torch.load(pinned(root, item['checkpoint']), map_location='cpu', weights_only=True)
    head = make_head(item['variant']).to(dtype=state['value.weight'].dtype)
    head.load_state_dict(state, strict=True)
    return head.eval()


@torch.no_grad()
def numeric(head, inputs):
    d = dict(head(*inputs, details=True))
    for name, value in list(d.items()):
        if isinstance(value, dict):
            del d[name]
            d.update({name+'_'+k: v for k, v in value.items()})
    h, roles, mask = inputs
    features=torch.cat((h,roles),-1)
    if head.mode in ('quantum','dephased'):
        d.update(prepare_raw=head.prepare(features),effect_raw=head.effect)
    elif head.mode!='uniform':
        key=head.key(features);query=head.query
        d.update(key=key,query=query)
        if head.mode=='relation_softmax': d['scores']=torch.einsum('btd,rd->brt',key,query)/(8**.5)
        else:
            nk=torch.nn.functional.normalize(key,dim=-1);nq=torch.nn.functional.normalize(query,dim=-1)
            d.update(normalized_key=nk,normalized_query=nq,scores=torch.einsum('btd,rd->brt',nk,nq).square())
    d.update(h=h, roles=roles, mask=mask, features=torch.cat((h, roles), -1),
             decoder=head.decoder, bias=head.bias)
    return {k: v.detach().cpu().numpy().copy() for k, v in d.items()}


AXES = {
    'h': ['sample','word','hidden'], 'roles': ['sample','word','role'],
    'mask': ['sample','word'], 'features': ['sample','word','feature'],
    'entity_inputs': ['sample','subject_or_object','feature'],
    'word_angles': ['sample','word','flat_rotation'],
    'subject_angles': ['sample','flat_rotation'], 'object_angles': ['sample','flat_rotation'],
    'angles': ['sample','word','layer','wire','rotation'],
    'subject': ['sample','basis'], 'object': ['sample','basis'],
    'psi': ['sample','word','basis'], 'phi': ['relation','basis'],
    'conditional': ['sample','relation','basis'],
    'values': ['sample','word','value'], 'decoder': ['relation','value'],
    'bias': ['relation'], 'logits': ['sample','relation'],
    'cp': ['relation','rank','role','flat_rotation'],
}
for _name in ('probability','alpha','utilities','sensitivity','amplitude','separable_null_q','separable_null_direction','separable_null_alpha'):
    AXES[_name] = ['sample','relation','word']
for _name in ('normalizer','separable_null_c','separable_null_q_mass','separable_null_mixture'):
    AXES[_name] = ['sample','relation','singleton']


AXES.update(prepare_raw=['sample','word','flat_rotation'],effect_raw=['relation','layer','wire','rotation'],
            key=['sample','word','key'],query=['relation','key'],normalized_key=['sample','word','key'],
            normalized_query=['relation','key'],scores=['sample','relation','word'])


def axes(name, value):
    if name == 'effect_angles':
        return ['relation','layer','wire','rotation'] if value.ndim == 4 else ['relation','wire','rotation']
    return AXES[name]


def owner(name):
    if name in ('h','roles','mask','features'): return 'inputs'
    if name in ('entity_inputs','word_angles','subject_angles','object_angles','angles','prepare_raw','effect_raw','key','query','normalized_key','normalized_query'): return 'preparation'
    if name in ('values','decoder','utilities','bias'): return 'values'
    if name == 'logits': return 'prediction'
    return 'measurement'


def read_value(root, sample, name, coords):
    rep = next(x for x in sample['representations'] if x['id'] == name)
    if not isinstance(coords, list) or not 1 <= len(coords) <= 64:
        raise ValueError('Expected 1..64 explicit coordinates')
    with np.load(pinned(root, rep['artifact_refs'][0]), allow_pickle=False) as f:
        a = f[name]
    if list(a.shape) != rep['shape'] or str(a.dtype) != rep['dtype'] or not np.isfinite(a).all():
        raise ValueError('Array metadata/nonfinite mismatch')
    result = []
    for c in coords:
        if not isinstance(c, list) or len(c) != a.ndim or any(type(i) is not int or not 0 <= i < n for i,n in zip(c,a.shape)):
            raise ValueError('Coordinate outside declared axes')
        v = a[tuple(c)].item()
        result.append(dict(real=v.real,imag=v.imag) if isinstance(v,complex) else v)
    return result


def replay(root, manifest):
    """Create a new cache directory only. Never mutate selected dependencies."""
    root = Path(root).resolve()
    check_manifest(root,manifest)
    if manifest['schema'] != 'q-rmea.allocation.selected-replay.v1' or manifest['lane'] != 'replay':
        raise ValueError('Selected post-training replay manifest required')
    if not manifest.get('source_revision') or not manifest.get('scope'):
        raise ValueError('Explicit source revision and evidence scope required')
    items = manifest['models']
    if not items or len({i['variant'] for i in items}) != len(items):
        raise ValueError('Unique selected models required')
    source = pinned(root, manifest['inputs'])
    selected = manifest['samples']
    if not selected or len({s['sample_id'] for s in selected}) != len(selected):
        raise ValueError('Frozen unique sample identities required')
    with np.load(source, allow_pickle=False) as f:
        if set(f.files) != {'h','roles','mask'}:
            raise ValueError('Feature boundary accepts no labels/extra fields')
        data = {k: f[k].copy() for k in f.files}
    if len(selected) != len(data['h']) or any(s['split'] not in ('train','valid','test') for s in selected):
        raise ValueError('Sample identity/split coverage mismatch')
    out = root/'trace';out.mkdir(exist_ok=False)
    samples = []
    try:
        for item in items:
            head = head_from_checkpoint(root, item)
            for i, identity in enumerate(selected):
                length=int(data['mask'][i].sum()); assert data['mask'][i,:length].all() and not data['mask'][i,length:].any()
                inputs = tuple(torch.from_numpy(data[k][i:i+1,:length]) for k in ('h','roles','mask'))
                arrays = numeric(head, inputs)
                path = out/(item['variant']+'-'+str(i)+'.npz')
                np.savez_compressed(path, **arrays)
                artifact = ref(root,path)
                sid = item['variant']+':selected:'+identity['sample_id']
                reps = []
                for name,a in sorted(arrays.items()):
                    if len(axes(name,a)) != a.ndim or not np.isfinite(a).all():
                        raise ValueError('Undeclared/nonfinite representation: '+name)
                    reps.append(dict(id=name,stage_id=owner(name),status='observed',
                        meaning=name+' at the selected frozen-feature head replay boundary',shape=list(a.shape),
                        dtype=str(a.dtype),axis_semantics=axes(name,a),artifact_refs=[artifact],display=display(name),
                        value_capture=dict(status='complete',shape=list(a.shape),logical_element_count=int(a.size),
                            stored_element_count=int(a.size),artifact_refs=[artifact],reader_projection=dict(
                                status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                                reader_ref='experiments/rmea_allocation/runner.py value --sample '+sid+' --representation '+name,
                                source_sha256=[artifact['sha256']]))))
                stages = []
                groups = {g: [k for k in arrays if owner(k)==g] for g in ('inputs','preparation','values','measurement','prediction')}
                incoming = stage_inputs(arrays)
                for g, names in groups.items():
                    if not names: continue
                    stages.append(dict(stage_id=g,stage=g,canonical_stage='data' if g=='inputs' else ('generation' if g=='prediction' else 'scoring'),
                        lane='replay',status='observed',purpose=g+' from the selected evaluation checkpoint',
                        input_refs=incoming[g],output_refs=names,artifact_refs=[artifact],display=display(g),target_access='not_available'))
                samples.append(dict(sample_id=sid,lane='replay',split=identity['split'],checkpoint='selected',
                    roles=dict(source_sample=dict(id=identity['sample_id']),sequence_or_features=dict(boundary='frozen original-word features'),
                        task_objects=dict(subject_role=2,object_role=3),prediction=dict(value=int(arrays['logits'].argmax())),
                        provenance=dict(condition=item['variant'],checkpoint=item['checkpoint'],selection_rule=item['selection_rule'],scope=manifest['scope'])),
                    representations=reps,stages=stages))
        required = sorted(set.intersection(*(set(r['id'] for r in s['representations']) for s in samples)))
        payload = dict(schema_version='sample-trace.v2',trace_id=root.name,trace_completeness='complete',
            experiment=dict(run_id=root.name,dataset=manifest['scope'],code_revision=manifest['source_revision'],
                config_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),model_identity='Five selected fresh5 seed29 frozen-feature heads'),
            sample_selection=dict(rule=manifest['sample_selection_rule'],population_scope=manifest['scope'],
                selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples]),
            task_contract=dict(task_type='frozen-feature relation classification',required_roles=['source_sample','sequence_or_features','task_objects','prediction','provenance'],
                required_representations=required,required_value_representations=required,
                required_splits=sorted({s['split'] for s in selected}),required_checkpoints=['selected']),
            coverage={s:'observed' if s in ('data','scoring','generation') else 'unavailable' for s in STAGES},samples=samples)
        # Different model variants have different applicable intermediate names.
        # Full semantic validation below requires every actual key, not just this
        # generic common representation set.
        errors,warnings=validate(payload,require_complete=True)
        if errors: raise ValueError(errors)
        (root/'sample-trace.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
        return payload,warnings
    except BaseException:
        # Keep failed evidence for inspection; a subsequent run uses a fresh
        # directory. Never delete selected checkpoint/input dependencies here.
        raise


def audit(root, manifest, payload):
    """Fresh model replay plus full array/coordinate and exact inventory checks."""
    root=Path(root).resolve()
    check_manifest(root,manifest)
    exp=payload['experiment'];selection=payload['sample_selection']
    if exp['code_revision']!=manifest['source_revision'] or exp['dataset']!=manifest['scope'] or exp['config_sha256']!=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest():
        raise ValueError('Source/config/scope provenance differs')
    if selection['rule']!=manifest['sample_selection_rule'] or selection['population_scope']!=manifest['scope']:
        raise ValueError('Sample selection contract differs')
    with np.load(pinned(root,manifest['inputs']),allow_pickle=False) as f:
        if set(f.files)!={'h','roles','mask'}:raise ValueError('No labels or extra feature keys accepted')
        inputs={k:f[k].copy() for k in f.files}
    if len(inputs['h'])!=len(manifest['samples']):raise ValueError('Input/sample count differs')
    models={m['variant']:m for m in manifest['models']}
    expected={(v,s['sample_id']) for v in models for s in manifest['samples']}
    seen=set();checked=0;coordinates=0
    errors,warnings=validate(payload,require_complete=True)
    if errors: raise ValueError(errors)
    for sample in payload['samples']:
        prov=sample['roles']['provenance'];v=prov['condition'];sid=sample['roles']['source_sample']['id']
        if (v,sid) not in expected or (v,sid) in seen: raise ValueError('Unexpected/duplicate sample')
        seen.add((v,sid));i=next(i for i,s in enumerate(manifest['samples']) if s['sample_id']==sid)
        if sample['split']!=manifest['samples'][i]['split'] or sample['lane']!='replay' or sample['checkpoint']!='selected' or prov['checkpoint']!=models[v]['checkpoint'] or prov['selection_rule']!=models[v]['selection_rule'] or prov['scope']!=manifest['scope']:
            raise ValueError('Selected replay provenance differs')
        h=head_from_checkpoint(root,models[v])
        length=int(inputs['mask'][i].sum()); assert inputs['mask'][i,:length].all() and not inputs['mask'][i,length:].any()
        actual=numeric(h,tuple(torch.from_numpy(inputs[k][i:i+1,:length]) for k in ('h','roles','mask')))
        if sample['roles']['prediction']!={'value':int(actual['logits'].argmax())}:raise ValueError('Prediction projection differs')
        incoming=stage_inputs(actual)
        expected_stages={g for g in incoming if any(owner(k)==g for k in actual)}
        if {s['stage_id'] for s in sample['stages']}!=expected_stages or len(sample['stages'])!=len(expected_stages):raise ValueError('Stage inventory differs')
        for stage in sample['stages']:
            g=stage['stage_id'];canonical='data' if g=='inputs' else ('generation' if g=='prediction' else 'scoring')
            if stage['input_refs']!=incoming[g] or set(stage['output_refs'])!={k for k in actual if owner(k)==g} or stage['canonical_stage']!=canonical or stage['lane']!='replay' or stage['status']!='observed' or stage['target_access']!='not_available':raise ValueError('Producer-owned lineage differs')
        reps=sample['representations']
        if {r['id'] for r in reps}!=set(actual) or len(reps)!=len(actual): raise ValueError('Key inventory differs')
        for r in reps:
            name=r['id'];a=actual[name]
            with np.load(pinned(root,r['artifact_refs'][0]),allow_pickle=False) as f: saved=f[name]
            if r['axis_semantics']!=axes(name,a) or r['stage_id']!=owner(name): raise ValueError('Semantic axes/owner differs')
            if r['shape']!=list(a.shape) or r['dtype']!=str(a.dtype): raise ValueError('Representation shape/dtype differs')
            np.testing.assert_allclose(saved,a,atol=2e-6,rtol=2e-5,err_msg=name)
            coords=[list(map(int,np.unravel_index(k,a.shape))) for k in sorted({0,a.size//2,a.size-1})]
            values=read_value(root,sample,name,coords)
            for c,value in zip(coords,values):
                value=complex(value['real'],value['imag']) if isinstance(value,dict) else value
                if value!=saved[tuple(c)].item(): raise ValueError('Coordinate reader differs')
            coordinates+=len(coords);checked+=1
    if seen!=expected: raise ValueError('Selected replay incomplete')
    return dict(structural='pass',semantic='pass',fresh_model_replay='pass',coordinate_reader='pass',
        samples=len(seen),arrays=checked,coordinates=coordinates,warnings=warnings,
        boundary='frozen features/head only; no encoder/tokenization or task-utility certification')
