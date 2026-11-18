"""Private full-value trace writer/reader for a future reviewed CPU diagnostic.

The boundary begins at the already frozen word features. It does not claim to
observe encoder execution, training, semantic relevance, or a causal intervention.
"""
import hashlib
import json
from pathlib import Path

from experiments.rmea_semantic_probe.scripts.validate_sample_trace_v2 import STAGES, validate

AXES = {
    'h': ['sample', 'word', 'hidden'], 'roles': ['sample', 'word', 'role'],
    'mask': ['sample', 'word'], 'features': ['sample', 'word', 'feature'],
    'angles': ['sample', 'word', 'layer', 'wire', 'rotation'],
    'effect_angles': ['relation', 'layer', 'wire', 'rotation'],
    'psi': ['sample', 'word', 'basis'], 'phi': ['relation', 'basis'],
    'prepare_raw': ['sample', 'word', 'flat_rotation'],
    'values': ['sample', 'word', 'value'], 'decoder': ['relation', 'value'],
    'bias': ['relation'], 'logits': ['sample', 'relation'],
    'entities': ['sample', 'subject_or_object', 'hidden'],
    'entity_summary': ['sample', 'hidden'],
    'projected_word': ['sample', 'word', 'projection'],
    'projected_entity': ['sample', 'projection'],
    'interaction': ['sample', 'word', 'projection'],
}
for _name in ('alpha', 'utilities', 'probability', 'sensitivity', 'routing_scores'):
    AXES[_name] = ['sample', 'relation', 'word']
AXES['normalizer'] = ['sample', 'relation', 'singleton']

# Independent source-defined inventories: completeness is not inferred only
# from the intersection of whichever representations a producer happened to emit.
INPUT_TENSORS = {'h', 'roles', 'mask', 'features'}
CORE = {'logits', 'alpha', 'values', 'utilities', 'decoder', 'bias', 'angles', 'effect_angles'}
ECQP = CORE | {'prepare_raw', 'routing_scores', 'entities', 'entity_summary',
               'projected_word', 'projected_entity', 'interaction'}
EXPECTED_TENSORS = {
    'native_quantum': CORE | {'psi', 'phi', 'probability', 'normalizer', 'sensitivity'},
    'ecqp_quantum': ECQP | {'psi', 'phi', 'probability', 'normalizer'},
    'softmax12_entity': ECQP,
    'real_overlap12_entity': ECQP | {'probability', 'normalizer'},
}



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('xb') as f:
        f.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def ref(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                bytes=path.stat().st_size, visibility='private')


def owner(name):
    if name in ('h', 'roles', 'mask', 'features'):
        return 'inputs'
    if name in ('values', 'utilities', 'decoder', 'bias'):
        return 'values'
    if name == 'logits':
        return 'prediction'
    if name in ('alpha', 'probability', 'normalizer', 'sensitivity', 'routing_scores'):
        return 'measurement'
    return 'preparation'


def display(text):
    return dict(summary=text, default_view='human_summary', visible_fields=[], visualizations=[])


def read_value(root, sample, name, coordinates):
    import numpy as np
    root = Path(root).absolute()
    assert all(not p.is_symlink() for p in [root, *root.parents])
    rep = next(r for r in sample['representations'] if r['id'] == name)
    item = rep['artifact_refs'][0]
    relative = Path(item['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = root / relative
    assert all(not p.is_symlink() for p in [path, *path.parents])
    assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    with np.load(path, allow_pickle=False) as f:
        a = f[name]
    assert list(a.shape) == rep['shape'] and str(a.dtype) == rep['dtype'] and np.isfinite(a).all()
    assert isinstance(coordinates, list) and 1 <= len(coordinates) <= 64
    values = []
    for c in coordinates:
        assert isinstance(c, list) and len(c) == a.ndim
        assert all(type(i) is int and 0 <= i < n for i, n in zip(c, a.shape))
        v = a[tuple(c)].item()
        values.append(dict(real=v.real, imag=v.imag) if isinstance(v, complex) else v)
    return values


def build(root, records, design, design_sha256, revision):
    import numpy as np
    samples = []
    for record in records:
        reps = []
        for path in (root / record['input_file'], root / record['head_file']):
            artifact = ref(root, path)
            with np.load(path, allow_pickle=False) as arrays:
                expected = INPUT_TENSORS if path.name.endswith('-inputs.npz') else EXPECTED_TENSORS[record['condition']]
                assert set(arrays.files) == expected
                for name in arrays.files:
                    a = arrays[name]
                    assert name in AXES and len(AXES[name]) == a.ndim and np.isfinite(a).all()
                    reps.append(dict(id=name, stage_id=owner(name), status='observed',
                        meaning='Detached actual frozen forward boundary: ' + name,
                        shape=list(a.shape), dtype=str(a.dtype), axis_semantics=AXES[name],
                        artifact_refs=[artifact], display=display(name), value_capture=dict(
                            status='complete', shape=list(a.shape), logical_element_count=int(a.size),
                            stored_element_count=int(a.size), artifact_refs=[artifact],
                            reader_projection=dict(status='available', kind='coordinate_inspector',
                                coordinate_access='arbitrary', reader_ref=(
                                    'experiments/rmea_training_case_trace/job.py value --run-dir PRIVATE_TRACE '
                                    '--sample ' + record['sample_id'] + ' --representation ' + name +
                                    ' --coordinates JSON_COORDINATES'), source_sha256=[artifact['sha256']],
                                implementation_sha256=[sha(Path(__file__))]))))
        incoming = {
            'inputs': ['external:pinned_train_prefix', 'external:selection_manifest'],
            'preparation': ['h', 'roles', 'external:frozen_checkpoint'],
            'values': ['h', 'external:frozen_checkpoint'],
            'measurement': ['mask', 'utilities'] + [r['id'] for r in reps if r['stage_id'] == 'preparation'],
            'prediction': ['alpha', 'utilities', 'bias'],
        }
        stages = []
        for stage, inputs in incoming.items():
            outputs = [r['id'] for r in reps if r['stage_id'] == stage]
            assert outputs
            stages.append(dict(stage_id=stage, stage=stage, canonical_stage=(
                'data' if stage == 'inputs' else 'generation' if stage == 'prediction' else 'scoring'),
                lane='replay', status='observed', purpose='Frozen-feature matched-case ' + stage,
                input_refs=inputs, output_refs=outputs, target_access='not_available',
                artifact_refs=[ref(root, root / record['input_file']), ref(root, root / record['head_file'])],
                display=display(stage)))
        samples.append(dict(sample_id=record['sample_id'], lane='replay', split='train', checkpoint='final4',
            roles=dict(source_sample=dict(id=record['id_hash']),
                sequence_or_features=dict(boundary='Pinned fp16 cache -> fp32, original eval batching/padding retained'),
                task_objects=dict(subject_role=2, object_role=3),
                prediction=dict(value=record['prediction']), provenance=dict(condition=record['condition'],
                    checkpoint=design['checkpoints'][record['condition']],
                    selection_manifest=ref(root, root / 'selection.json'),
                    case_inputs=ref(root, root / 'case-inputs.json'))),
            representations=reps, stages=stages))
    common = sorted(set.intersection(*(set(r['id'] for r in s['representations']) for s in samples)))
    payload = dict(schema_version='sample-trace.v2', trace_id=root.parent.parent.name,
        trace_completeness='complete', experiment=dict(run_id=root.parent.parent.name,
            dataset='Original training-prefix group-separated eval partition; exploratory cases',
            code_revision=revision, config_sha256=design_sha256, model_identity=design['checkpoints']),
        sample_selection=dict(rule=design['selection']['rule'], population_scope=design['selection']['scope'],
            selected_count=len(samples), selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='Frozen-feature relation classification replay',
            required_roles=['source_sample', 'sequence_or_features', 'task_objects', 'prediction', 'provenance'],
            required_representations=common, required_value_representations=common,
            required_splits=['train'], required_checkpoints=['final4']),
        coverage={s: 'observed' if s in ('data', 'scoring', 'generation') else 'not_applicable' for s in STAGES},
        samples=samples)
    errors, warnings = validate(payload, require_complete=True)
    assert not errors, errors
    # Exercise arbitrary-coordinate access on every captured representation.
    for sample in samples:
        for rep in sample['representations']:
            read_value(root, sample, rep['id'], [[0] * len(rep['shape']), [n - 1 for n in rep['shape']]])
    save(root / 'sample-trace.json', payload)
    return dict(samples=len(samples), representations=sum(len(s['representations']) for s in samples),
                structural_errors=0, warnings=warnings, coordinate_reader_exercised=True)


def value_cli(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--representation', required=True)
    p.add_argument('--coordinates', required=True)
    a = p.parse_args(argv)
    root = a.run_dir.absolute()
    assert all(not q.is_symlink() for q in [root, *root.parents])
    closed = json.loads((root / 'RUN_COMPLETE').read_bytes())
    assert closed['files']['sample-trace.json'] == sha(root / 'sample-trace.json')
    payload = json.loads((root / 'sample-trace.json').read_bytes())
    sample = next(s for s in payload['samples'] if s['sample_id'] == a.sample)
    print(json.dumps(read_value(root, sample, a.representation, json.loads(a.coordinates)), allow_nan=False))
