"""Complete private values at the frozen-feature/head boundary, sample-trace.v2.

This technical fixture does not certify tokenization, encoder replay, or a full
formal handoff. The same reader opens and hashes actual values, including complex
states; no public projection contains raw tensors.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .rmea_entity import entity_inputs, make_head
from scripts.validate_sample_trace_v2 import STAGES, validate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display(text):
    return dict(summary=text, default_view='human_summary', visible_fields=[], visualizations=[])


def details(head, h, roles, mask):
    d = dict(head(h, roles, mask, True))
    d.update(h=h, roles=roles, mask=mask, decoder=head.decoder, bias=head.bias,
             features=torch.cat((h, roles), -1), effect_raw=head.effect)
    if head.variant == 'static':
        d['prepare_raw'] = head.prepare(d['features'])
    return d


SHARED = {'decoder', 'bias', 'phi', 'effect_angles', 'effect_raw', 'query'}
AXES = {
    'h': ['sample', 'word', 'hidden'], 'roles': ['sample', 'word', 'role'],
    'mask': ['sample', 'word'], 'features': ['sample', 'word', 'feature'],
    'entity_pair': ['sample', 'subject_or_object', 'feature'],
    'entity_raw': ['sample', 'subject_or_object', 'flat_angle'],
    'subject_angles': ['sample', 'layer', 'wire', 'rotation'],
    'object_angles': ['sample', 'layer', 'wire', 'rotation'],
    'prepare_raw': ['sample', 'word', 'flat_angle'],
    'angles': ['sample', 'word', 'layer', 'wire', 'rotation'],
    'effect_raw': ['relation', 'layer', 'wire', 'rotation'],
    'effect_angles': ['relation', 'layer', 'wire', 'rotation'],
    'phi': ['relation', 'basis'], 'psi': ['sample', 'word', 'basis'],
    'effective_phi': ['sample', 'relation', 'basis'],
    'reverse_effective_phi': ['sample', 'relation', 'basis'],
    'probability': ['sample', 'relation', 'word'],
    'ordered_probability': ['sample', 'relation', 'word'],
    'reverse_probability': ['sample', 'relation', 'word'],
    'normalizer': ['sample', 'relation', 'singleton'],
    'sensitivity': ['sample', 'relation', 'word'],
    'alpha': ['sample', 'relation', 'word'], 'values': ['sample', 'word', 'value'],
    'utilities': ['sample', 'relation', 'word'], 'decoder': ['relation', 'value'],
    'bias': ['relation'], 'logits': ['sample', 'relation'],
    'query': ['relation', 'feature12'], 'conditioned_query': ['sample', 'relation', 'feature12'],
    'scores': ['sample', 'relation', 'word'],
}


def group(name):
    if name in ('h', 'roles', 'mask', 'features'):
        return 'inputs'
    if name in ('entity_pair', 'entity_raw', 'subject_angles', 'object_angles'):
        return 'entities'
    if name in ('values', 'utilities', 'decoder', 'bias'):
        return 'values'
    if name == 'logits':
        return 'prediction'
    return 'weights'


def capture(root, head, data, split, checkpoint):
    root = Path(root)
    h, roles, mask = data
    head.eval()
    with torch.no_grad():
        actual = details(head, h[:2], roles[:2], mask[:2])
    cp = root/'checkpoints'/f'{head.variant}-{checkpoint}.pt'
    cp.parent.mkdir(exist_ok=True)
    if not cp.exists():
        torch.save(head.state_dict(), cp)
    samples = []
    for i in range(min(2, len(h))):
        arrays = {k: (v if k in SHARED else v[i:i+1]).detach().cpu().numpy().copy()
                  for k, v in actual.items()}
        name = f'{head.variant}-{checkpoint}-{split}-{i}.npz'
        path = root/'trace'/name
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **arrays)
        ref = dict(path='trace/'+name, sha256=digest(path), bytes=path.stat().st_size, visibility='private')
        sid = f'{head.variant}:{checkpoint}:{split}:{i}'
        reps = []
        groups = {g: [] for g in ('inputs', 'entities', 'values', 'weights', 'prediction')}
        for key, value in sorted(arrays.items()):
            owner = group(key)
            groups[owner].append(key)
            if value.ndim != len(AXES[key]):
                raise ValueError('Undeclared axis meaning: '+key)
            reps.append(dict(id=key, stage_id=owner, status='observed', meaning=key+' actual detached head state',
                shape=list(value.shape), dtype=str(value.dtype), axis_semantics=AXES[key],
                portal_preview=dict(array_key=key, owner_stage=owner, direction='output', visibility='private'),
                artifact_refs=[ref], display=display(key+' complete private numeric array'),
                value_capture=dict(status='complete', shape=list(value.shape), logical_element_count=int(value.size),
                    stored_element_count=int(value.size), artifact_refs=[ref], reader_projection=dict(
                        status='available', kind='coordinate_inspector', coordinate_access='arbitrary',
                        reader_ref='experiments/q_rmea_entity_dev.py value --sample '+sid+' --representation '+key,
                        source_sha256=[ref['sha256']]))))
        stages = []
        incoming = {'inputs': ['external:frozen_features', 'external:ordered_roles', 'external:valid_words'],
                    'entities': ['h', 'roles', 'mask', 'external:checkpoint_prepare'] + (
                        ['external:train_only_role_means'] if head.variant == 'no_content' else []),
                    'values': ['h', 'external:checkpoint_value_decoder'],
                    'weights': ['features', 'mask', 'external:checkpoint_scorer'] + (
                        ['entity_pair', 'subject_angles', 'object_angles'] if 'entity_pair' in arrays else []) + (
                        ['utilities'] if 'sensitivity' in arrays else []),
                    'prediction': ['alpha', 'utilities', 'bias']}
        for i_stage, owner in enumerate(groups):
            if not groups[owner]:
                continue
            canonical = 'data' if owner == 'inputs' else ('generation' if owner == 'prediction' else 'scoring')
            purpose = owner+' at the frozen-feature head boundary; target not available'
            stages.append(dict(stage_id=owner, stage=owner, canonical_stage=canonical, lane='replay', status='observed',
                step=i_stage, checkpoint=checkpoint, purpose=purpose, input_refs=incoming[owner],
                output_refs=groups[owner], artifact_refs=[ref], display=display(purpose), target_access='not_available'))
        samples.append(dict(sample_id=sid, lane='replay', split=split, checkpoint=checkpoint,
            roles=dict(source_sample=dict(id=f'{split}:{i}'), sequence_or_features=dict(boundary='frozen features'),
                task_objects=dict(subject_role_index=2, object_role_index=3),
                prediction=dict(value=int(arrays['logits'].argmax())),
                provenance=dict(condition=head.variant, checkpoint_file=str(cp.relative_to(root)),
                                checkpoint_sha256=digest(cp))),
            representations=reps, stages=stages))
    return samples


def read_value(root, sample, representation, coords):
    root = Path(root).resolve()
    rep = next(r for r in sample['representations'] if r['id'] == representation)
    ref = rep['value_capture']['artifact_refs'][0]
    path = root/ref['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path) != ref['sha256']:
        raise ValueError('Unsafe or changed private value artifact')
    if not isinstance(coords, list) or not 1 <= len(coords) <= 64:
        raise ValueError('Expected 1..64 coordinates')
    with np.load(path, allow_pickle=False) as data:
        a = data[representation].copy()
    if list(a.shape) != rep['shape'] or str(a.dtype) != rep['dtype'] or not np.isfinite(a).all():
        raise ValueError('Numeric representation mismatch')
    result = []
    for coord in coords:
        if len(coord) != a.ndim or any(type(v) != int or v < 0 or v >= n for v, n in zip(coord, a.shape)):
            raise ValueError('Coordinate outside array axes')
        v = a[tuple(coord)].item()
        result.append(dict(real=v.real, imag=v.imag) if isinstance(v, complex) else v)
    return result


def close(root, samples, config, source_revision):
    root = Path(root)
    required = ['h', 'roles', 'mask', 'features', 'alpha', 'values', 'utilities', 'logits']
    payload = dict(schema_version='sample-trace.v2', trace_id='rmea-entity-head-dev', trace_completeness='complete',
        experiment=dict(run_id=root.name, dataset=config['data_scope'], code_revision=source_revision,
            config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            model_identity='Five frozen-feature heads; head-boundary technical fixture only', seed=config['seed']),
        sample_selection=dict(rule='first two stable selected examples per train/valid, all modes, initial/trained checkpoints',
            population_scope='head-boundary technical fixture; no test; no encoder/tokenization audit',
            selected_count=len(samples), selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='frozen-feature relation head technical replay',
            required_roles=['source_sample', 'sequence_or_features', 'task_objects', 'prediction'],
            required_representations=required, required_value_representations=required,
            required_splits=['train', 'valid'], required_checkpoints=['initial', 'trained'], required_lanes=['replay'],
            checkpoint_policy='Initial and bounded-budget final state, not a formal valid-selected checkpoint'),
        coverage={s: ('observed' if s in ('data', 'scoring', 'generation') else 'not_applicable') for s in STAGES},
        samples=samples)
    errors, warnings = validate(payload, require_complete=True)
    if errors:
        raise ValueError(errors)
    checked = 0
    coords_checked = 0
    parity = {}
    for sample in samples:
        variant = sample['roles']['provenance']['condition']
        path = root/sample['roles']['provenance']['checkpoint_file']
        if digest(path) != sample['roles']['provenance']['checkpoint_sha256']:
            raise ValueError('Checkpoint identity differs')
        head = make_head(variant)
        head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        ref = sample['representations'][0]['artifact_refs'][0]
        with np.load(root/ref['path'], allow_pickle=False) as f:
            saved = {k: f[k].copy() for k in f.files}
        with torch.no_grad():
            replay = details(head, *(torch.from_numpy(saved[k]) for k in ('h', 'roles', 'mask')))
        if set(replay) != set(saved):
            raise ValueError('Missing/extra model representation')
        for rep in sample['representations']:
            name = rep['id']
            a = saved[name]
            b = replay[name].detach().cpu().numpy()
            np.testing.assert_allclose(a, b, atol=2e-5, rtol=1e-4, err_msg=name)
            coords = [list(map(int, np.unravel_index(k, a.shape))) for k in sorted({0, a.size//2, a.size-1})]
            values = read_value(root, sample, name, coords)
            for coord, value in zip(coords, values):
                actual = complex(value['real'], value['imag']) if isinstance(value, dict) else value
                if actual != a[tuple(coord)].item():
                    raise ValueError('Coordinate reader mismatch')
                coords_checked += 1
            checked += 1
        key = (sample['split'], sample['sample_id'].rsplit(':', 1)[1])
        inputs = b''.join(saved[k].tobytes() for k in ('h', 'roles', 'mask'))
        parity.setdefault(key, set()).add(hashlib.sha256(inputs).hexdigest())
    expected = {(v, c, s, i) for v in config['variants'] for c in ('initial', 'trained')
                for s in ('train', 'valid') for i in ('0', '1')}
    observed = {(s['roles']['provenance']['condition'], s['checkpoint'], s['split'], s['sample_id'].rsplit(':', 1)[1]) for s in samples}
    if observed != expected or len(samples) != len(expected) or any(len(v) != 1 for v in parity.values()):
        raise ValueError('Sample selection/coverage/parity differs')
    (root/'sample-trace.json').write_text(json.dumps(payload, indent=2)+'\n')
    report = dict(structural='pass', semantic='pass', replay='pass', coordinate_reader='pass',
                  samples=len(samples), arrays_checked=checked, coordinates_checked=coords_checked, warnings=warnings,
                  boundary='frozen features/head only; no formal or encoder certification')
    (root/'trace_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
