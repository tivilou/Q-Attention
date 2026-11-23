"""Full-value final-head replay from pinned frozen features; no encoder claim."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, digest
from .rmea import EPS, RMEAHead
from .rmea_protected_context import KINDS, ProtectedContextHead
from .rmea_token_transport_trace import display, read_value
from scripts.validate_sample_trace_v2 import STAGES, validate

ROOT = Path(__file__).resolve().parents[2]
READER = 'scripts/view_q_rmea_protected_context_trace.py'
AXES = dict(
    h=['sample', 'word', 'hidden'], roles=['sample', 'word', 'role'],
    mask=['sample', 'word'], features=['sample', 'word', 'feature'],
    prepare_raw=['sample', 'word', 'flat_angle'], angles=['sample', 'word', 'layer', 'wire', 'rotation'],
    effect_raw=['relation', 'layer', 'wire', 'rotation'], effect_angles=['relation', 'layer', 'wire', 'rotation'],
    psi=['sample', 'word', 'basis'], phi=['relation', 'basis'], mention=['sample', 'word'],
    projector=['sample', 'basis', 'basis2'], perp=['sample', 'basis', 'basis2'],
    covariance=['sample', 'basis', 'basis2'], used_covariance=['sample', 'basis', 'basis2'],
    singular_values=['sample', 'singular'], system=['sample', 'basis', 'basis2'],
    rhs=['sample', 'basis', 'relation'], correction=['sample', 'relation', 'basis'],
    corrected=['sample', 'relation', 'basis'], amplitude=['sample', 'relation', 'word'],
    probability=['sample', 'relation', 'word'], raw=['sample', 'relation', 'word'],
    normalizer=['sample', 'relation', 'singleton'], alpha=['sample', 'relation', 'word'],
    native_alpha=['sample', 'relation', 'word'], native_logits=['sample', 'relation'],
    values=['sample', 'word', 'value'], utilities=['sample', 'relation', 'word'],
    decoder=['relation', 'value'], bias=['relation'], logits=['sample', 'relation'],
    sensitivity=['sample', 'relation', 'word'], target=['singleton'])
COMMON = ['h', 'roles', 'mask', 'features', 'prepare_raw', 'angles', 'effect_raw',
          'effect_angles', 'psi', 'phi', 'probability', 'raw', 'normalizer', 'alpha',
          'native_alpha', 'native_logits', 'values', 'utilities', 'decoder', 'bias', 'logits', 'target']
EXTRA = ['mention', 'projector', 'perp', 'covariance', 'used_covariance',
         'singular_values', 'system', 'rhs', 'correction', 'corrected', 'amplitude']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def owner(name):
    if name in ('h', 'roles', 'mask'):
        return 'feature_input'
    if name == 'features':
        return 'feature_join'
    if name in ('prepare_raw', 'angles', 'effect_raw', 'effect_angles', 'psi', 'phi'):
        return 'state_preparation'
    if name in ('mention', 'projector', 'perp', 'covariance', 'used_covariance', 'singular_values'):
        return 'context_geometry'
    if name in ('values', 'utilities', 'decoder', 'bias'):
        return 'values'
    if name == 'logits':
        return 'prediction'
    if name == 'target':
        return 'evaluation'
    if name in ('native_alpha', 'native_logits'):
        return 'paired_native'
    return 'measurement'


def lineage(keys):
    specifications = [
        ('feature_input', 'data', ['artifact:pinned_feature_cache', 'source:cache_index']),
        ('feature_join', 'preprocess', ['h', 'roles']),
        ('state_preparation', 'scoring', ['features', 'external:checkpoint_prepare_effect']),
        ('context_geometry', 'scoring', ['psi', 'roles', 'mask', 'external:fixed_condition_and_DFT8']),
        ('measurement', 'scoring', ['psi', 'phi', 'mask'] +
         ['external:fixed_ridge_and_source_floor'] + (['perp', 'used_covariance'] if 'perp' in keys else [])),
        ('values', 'scoring', ['h', 'external:checkpoint_value_decoder_bias']),
        ('prediction', 'generation', ['alpha', 'utilities', 'bias']),
        ('paired_native', 'diagnosis', ['h', 'roles', 'mask',
         'external:checkpoint_prepare_effect', 'external:checkpoint_value_decoder_bias']),
        ('evaluation', 'evaluation', ['logits', 'source:pinned_split_target']),
    ]
    return [(name, canonical, inputs, sorted(k for k in keys if owner(k) == name))
            for name, canonical, inputs in specifications if any(owner(k) == name for k in keys)]


@torch.no_grad()
def details(head, h, roles, mask):
    d = head(h, roles, mask, details=True)
    # Explicit disabled routing at the SAME checkpoint, not a separately fit model.
    native = RMEAHead.forward(head, h, roles, mask, details=True)
    arrays = {k: v for k, v in d.items() if torch.is_tensor(v) and k != 'sensitivity'}
    features = torch.cat((h, roles), -1)
    arrays.update(h=h, roles=roles, mask=mask, features=features,
                  prepare_raw=head.prepare(features), effect_raw=head.effect,
                  decoder=head.decoder, bias=head.bias,
                  native_alpha=native['alpha'], native_logits=native['logits'])
    if 'raw' not in arrays:
        arrays['raw'] = (d['probability']+EPS)*mask[:, None, :]
    return {k: v.detach().cpu().numpy().copy() for k, v in arrays.items()}


def capture(root, head, kind, split, batch, metadata, *, reuse_identical=False):
    """First two cache units per split, frozen final model only; private values."""
    root = Path(root)
    require(kind in KINDS and head.kind == kind, 'Head condition mismatch')
    checkpoint = root/'checkpoints'/(kind+'-final.pt')
    require(checkpoint.is_file() and not checkpoint.is_symlink(), 'Regular final checkpoint required')
    samples = []
    head.eval()
    for index in (0, 1):
        inputs, target = batch(split, [index])
        arrays = details(head, *inputs)
        arrays['target'] = np.asarray([int(target[0])], dtype=np.int64)
        path = root/'trace'/f'{kind}-final-{split}-{index}.npz'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            require(reuse_identical and not path.is_symlink() and path.is_file(), 'Preserve existing private trace')
            with np.load(path, allow_pickle=False) as prior:
                require(set(prior.files) == set(arrays), 'Existing trace inventory mismatch')
                for name, array in arrays.items():
                    old = prior[name]
                    require(old.dtype == array.dtype and old.shape == array.shape
                            and np.array_equal(old, array), 'Existing trace producer values differ: '+name)
        else:
            pending = path.with_name('.'+path.name+'.pending')
            with pending.open('wb') as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(pending, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        ref = dict(path=path.relative_to(root).as_posix(), sha256=digest(path),
                   bytes=path.stat().st_size, visibility='private')
        sid = f'{kind}:final:{split}:{index}'
        representations = []
        for name, array in sorted(arrays.items()):
            require(name in AXES and len(AXES[name]) == array.ndim and np.isfinite(array).all(),
                    'Unspecified or invalid complete array '+name)
            stage = owner(name)
            representations.append(dict(id=name, stage_id=stage, status='observed',
                meaning=name+' complete detached producer value', shape=list(array.shape),
                dtype=str(array.dtype), axis_semantics=AXES[name], artifact_refs=[ref], display=display(name),
                portal_preview=dict(array_key=name, owner_stage=stage, direction='output', visibility='private'),
                value_capture=dict(status='complete', shape=list(array.shape),
                    logical_element_count=int(array.size), stored_element_count=int(array.size), artifact_refs=[ref],
                    reader_projection=dict(status='available', kind='coordinate_inspector',
                        coordinate_access='arbitrary', reader_ref=READER+' --sample '+sid+' --representation '+name,
                        source_sha256=[ref['sha256']]))))
        stages = [dict(stage_id=name, stage=name, canonical_stage=canonical, lane='replay',
            status='observed', step=i, checkpoint='final',
            purpose=name+' from pinned frozen-feature inputs and immutable final model',
            input_refs=inputs, output_refs=outputs, artifact_refs=[ref], display=display(name),
            target_access='post_generation_only' if name == 'evaluation' else 'not_available')
            for i, (name, canonical, inputs, outputs) in enumerate(lineage(arrays))]
        meta = metadata[split][index]
        samples.append(dict(sample_id=sid, lane='replay', split=split, checkpoint='final',
            roles=dict(source_sample=dict(id_hash=meta['id_hash'], cache_index=index),
                sequence_or_features=dict(input='pinned frozen word h and role features'),
                task_objects=dict(subject=meta['subject'], object=meta['object']),
                prediction=dict(value=int(arrays['logits'].argmax())),
                target_or_gold=dict(value=int(target[0]), boundary='post_generation_only'),
                provenance=dict(condition=kind, cache_index=index,
                    checkpoint_file=checkpoint.relative_to(root).as_posix(), checkpoint_sha256=digest(checkpoint),
                    value_reader_source_sha256=digest(ROOT/READER))),
            stages=stages, representations=representations))
    return samples


def validate_trace(root, payload, *, cfg, metadata, cache_manifest, checkpoint_hashes):
    """Open every full value, verify live cache, source replay and reader binding."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_bytes())
    require(manifest['config'] == cfg and manifest['cache'] == cache_manifest
            and payload['experiment']['code_revision'] == manifest['git_revision']
            and payload['experiment']['config_sha256'] == manifest.get('config_sha256',
                hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()),
            'Source/config/cache contract mismatch')
    require(metadata == json.loads((root/'cache/metadata.json').read_bytes()), 'Pinned metadata mismatch')
    errors, warnings = validate(payload, require_complete=True)
    require(not errors, str(errors))
    expected = {(kind, split, index) for kind in cfg.get('trace_conditions', cfg['conditions'])
                for split in cfg['counts'] for index in (0, 1)}
    seen, parity = set(), {}
    checked = elements = coordinates = 0
    for sample in payload['samples']:
        prov = sample['roles']['provenance']
        kind, split, index = prov['condition'], sample['split'], prov['cache_index']
        require((kind, split, index) in expected and (kind, split, index) not in seen, 'Trace population mismatch')
        seen.add((kind, split, index))
        sid = f'{kind}:final:{split}:{index}'
        cp = root/'checkpoints'/(kind+'-final.pt')
        path = root/'trace'/f'{kind}-final-{split}-{index}.npz'
        require(sample['sample_id'] == sid and sample['lane'] == 'replay' and sample['checkpoint'] == 'final', 'Trace identity mismatch')
        require(not cp.is_symlink() and cp.is_file() and prov['checkpoint_file'] == cp.relative_to(root).as_posix()
                and prov['checkpoint_sha256'] == digest(cp) == checkpoint_hashes[kind], 'Reported frozen model binding mismatch')
        require(not path.is_symlink() and path.is_file() and prov['value_reader_source_sha256'] == digest(ROOT/READER), 'Private trace/reader binding mismatch')
        ref = dict(path=path.relative_to(root).as_posix(), sha256=digest(path), bytes=path.stat().st_size, visibility='private')
        with np.load(path, allow_pickle=False) as archive:
            saved = {name: archive[name].copy() for name in archive.files}
        require(set(saved) == {r['id'] for r in sample['representations']}, 'Full representation inventory mismatch')
        require(set(COMMON).issubset(saved) and (kind == 'native' or set(EXTRA).issubset(saved)), 'Missing source mechanism values')
        for name in ('h', 'roles', 'offsets', 'targets'):
            filename = split+'-'+name+'.npy'
            require(filename in cache_manifest, 'Missing cache identity')
            cache = root/'cache'/filename
            require(cache.is_file() and not cache.is_symlink() and cache.stat().st_size == cache_manifest[filename]['bytes'], 'Cache member mismatch')
        # Full cache hashes are checked once by the caller before this bounded replay.
        lo, hi = map(int, np.load(root/'cache'/f'{split}-offsets.npy', mmap_mode='r')[index:index+2])
        replay_inputs = []
        for name in ('h', 'roles'):
            source = np.load(root/'cache'/f'{split}-{name}.npy', mmap_mode='r')[lo:hi].astype(np.float64)
            require(saved[name].dtype == source.dtype and saved[name].shape == source[None].shape,
                    'Source input dtype/shape mismatch: '+name)
            np.testing.assert_array_equal(saved[name], source[None])
            replay_inputs.append(torch.from_numpy(source[None]))
        mask = np.ones((1, hi-lo), dtype=np.bool_)
        require(saved['mask'].dtype == mask.dtype and saved['mask'].shape == mask.shape
                and np.array_equal(saved['mask'], mask), 'Unpadded source mask mismatch')
        replay_inputs.append(torch.from_numpy(mask))
        target = int(np.load(root/'cache'/f'{split}-targets.npy', mmap_mode='r')[index])
        expected_target = np.asarray([target], dtype=np.int64)
        require(saved['target'].dtype == expected_target.dtype and saved['target'].shape == expected_target.shape
                and np.array_equal(saved['target'], expected_target)
                and sample['roles']['target_or_gold'] == dict(value=target, boundary='post_generation_only'),
                'Posthoc evaluation identity mismatch')
        meta = metadata[split][index]
        require(sample['roles']['source_sample'] == dict(id_hash=meta['id_hash'], cache_index=index)
                and sample['roles']['task_objects'] == dict(subject=meta['subject'], object=meta['object']), 'Source metadata identity mismatch')
        positions = np.arange(hi-lo, dtype=np.float32)
        subject, obj = meta['subject'], meta['object']
        roles = np.stack([np.clip((positions-(subject[0]+subject[1]-1)/2)/32, -1, 1),
            np.clip((positions-(obj[0]+obj[1]-1)/2)/32, -1, 1),
            ((positions >= subject[0]) & (positions < subject[1])).astype(np.float32),
            ((positions >= obj[0]) & (positions < obj[1])).astype(np.float32)], -1)
        np.testing.assert_array_equal(saved['roles'][0], roles)
        head = ProtectedContextHead(kind).double().eval()
        head.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True), strict=True)
        actual = details(head, *replay_inputs)
        require(set(saved) == set(actual)|{'target'}, 'Producer field inventory mismatch')
        for name, array in actual.items():
            require(saved[name].dtype == array.dtype and saved[name].shape == array.shape,
                    'Replayed producer dtype/shape mismatch: '+name)
            np.testing.assert_allclose(array, saved[name], atol=1e-8, rtol=1e-8, err_msg=name)
        require(int(saved['logits'].argmax()) == sample['roles']['prediction']['value'], 'Prediction replay mismatch')
        expected_lineage = lineage(saved)
        require(len(sample['stages']) == len(expected_lineage), 'Stage coverage mismatch')
        for step, (record, (name, canonical, inputs, outputs)) in enumerate(zip(sample['stages'], expected_lineage)):
            require(record['stage_id'] == name and record['stage'] == name and record['canonical_stage'] == canonical
                    and record['lane'] == 'replay' and record['status'] == 'observed'
                    and record['checkpoint'] == 'final' and record['step'] == step
                    and record['input_refs'] == inputs and record['output_refs'] == outputs
                    and record['artifact_refs'] == [ref]
                    and record['target_access'] == ('post_generation_only' if name == 'evaluation' else 'not_available'), 'Producer lineage/target boundary mismatch')
        for rep in sample['representations']:
            name = rep['id']; array = saved[name]; value = rep['value_capture']
            require(rep['artifact_refs'] == [ref] and value['artifact_refs'] == [ref], 'Artifact binding mismatch')
            reader = value['reader_projection']
            require(reader['reader_ref'] == READER+' --sample '+sid+' --representation '+name
                    and reader['source_sha256'] == [ref['sha256']], 'Advertised reader mismatch')
            require(rep['shape'] == list(array.shape) and rep['dtype'] == str(array.dtype)
                    and rep['axis_semantics'] == AXES[name] and rep['stage_id'] == owner(name)
                    and rep['portal_preview'] == dict(array_key=name, owner_stage=owner(name), direction='output', visibility='private')
                    and value['stored_element_count'] == int(array.size), 'Complete coordinate/owner mismatch')
            coords = [list(map(int, np.unravel_index(i, array.shape))) for i in sorted({0, array.size//2, array.size-1})]
            for coordinate, value in zip(coords, read_value(root, sample, name, coords)):
                scalar = complex(value['real'], value['imag']) if isinstance(value, dict) else value
                require(scalar == array[tuple(coordinate)].item(), 'Source-linked coordinate mismatch')
                coordinates += 1
            checked += 1; elements += int(array.size)
        parity.setdefault((split, index), set()).add(tuple(hashlib.sha256(saved[name].tobytes()).hexdigest() for name in ('h', 'roles', 'mask', 'target')))
    require(seen == expected and all(len(v) == 1 for v in parity.values()), 'Condition input/selection parity mismatch')
    return dict(structural='pass', semantic='pass', replay='pass', coordinate_reader='pass', samples=len(seen),
                complete_numeric_arrays_checked=checked, complete_numeric_elements_checked=elements,
                source_linked_coordinates_checked=coordinates, warnings=warnings,
                boundary='Pinned frozen word features to final head; no encoder/internal-layer/training-trajectory claim')


def close(root, samples, cfg, metadata, *, cache_manifest, checkpoint_hashes):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_bytes())
    for name, expected in cache_manifest.items():
        path = root/'cache'/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes']
                and digest(path) == expected['sha256'], 'Pinned input hash mismatch: '+name)
    payload = dict(schema_version='sample-trace.v2', trace_id='rmea-protected-context-final', trace_completeness='complete',
        experiment=dict(run_id=root.parent.name, dataset='Pinned Re-TACRED frozen-feature cohort; preflight cache when declared',
            code_revision=manifest['git_revision'], config_sha256=manifest['config_sha256'],
            model_identity='Immutable final81028parameter head; paired disabled routing uses same checkpoint', seed=13),
        sample_selection=dict(rule='First2 cache units per split, fixed before outcomes, every declared condition final only',
            population_scope='Post-training frozen-feature inference only; preflight checkpoint is not a future trained model',
            selected_count=len(samples), selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='Protected-context final-head frozen-feature inference',
            required_roles=['source_sample', 'sequence_or_features', 'task_objects', 'prediction', 'target_or_gold', 'provenance'],
            required_representations=COMMON, required_value_representations=COMMON,
            required_splits=list(cfg['counts']), required_checkpoints=['final'], required_lanes=['replay'],
            checkpoint_policy='Fixed final epoch; preflight uses existing immutable native model'),
        coverage={stage: 'observed' if stage in ('data', 'preprocess', 'scoring', 'generation', 'diagnosis', 'evaluation') else 'not_applicable' for stage in STAGES},
        samples=samples)
    report = validate_trace(root, payload, cfg=cfg, metadata=metadata,
                            cache_manifest=cache_manifest, checkpoint_hashes=checkpoint_hashes)
    atomic_json(root/'sample-trace.json', payload)
    atomic_json(root/'trace-validation.json', report)
    return report
