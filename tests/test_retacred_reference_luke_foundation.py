"""Source-interface preparation tests; never certify full-weight/runtime fidelity."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.retacred import luke
from baselines.retacred.luke_fidelity import report, source_head_fixture
from baselines.retacred.common import cohort, select_models

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.external/retacred/luke-foundation-20261008'
LABELS = ['no_relation'] + ['relation_' + str(i) for i in range(39)]


class TokenizerFixture:
    """Deterministic tiny tokenizer, NOT measured BPE/tokenizer equivalence."""
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    entity_vocab = {'[MASK]': 1, '[MASK2]': 2}
    def convert_tokens_to_ids(self, value):
        return {'<ent>': 50265, '<ent2>': 50266}[value]
    def __call__(self, sentence, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        tokens = re.findall(r'<ent2>|<ent>|[^\s<>]+', sentence)
        return {'input_ids': [0] + [self.convert_tokens_to_ids(t) if t.startswith('<ent') else 100 for t in tokens] + [2]}


def record(reverse=False):
    return {'id': 'toy', 'token': ['Alice', 'joined', 'Acme', '.'], 'relation': LABELS[1],
            'subj_start': 2 if reverse else 0, 'subj_end': 2 if reverse else 0,
            'obj_start': 0 if reverse else 2, 'obj_end': 0 if reverse else 2}


def test_actual_cached_source_hashes_and_provider_weight_identity():
    assert CACHE.exists(), 'required pinned source fixture must not be skipped'
    assert len(luke.verify_source(CACHE)) == 15
    checkpoint = luke.lock()['checkpoint']
    assert checkpoint['revision'] == '0729d044dfe301d9ecabc222d60633f92ac450eb'
    assert checkpoint['files']['pytorch_model.bin']['sha256'] == '9ea73dbb7e0625e88c646f26230902c3df000e71320139e99b0703e1654f7bcc'
    assert checkpoint['files']['pytorch_model.bin']['size'] == 2246566835
    assert not checkpoint['local_weights_verified']


@pytest.mark.parametrize('reverse', [False, True])
def test_actual_source_reader_no_caller_mutation_and_role_order(reverse):
    raw = record(reverse)
    before = copy.deepcopy(raw)
    features = luke.encode_records(CACHE, TokenizerFixture(), [raw], LABELS)
    assert raw == before
    assert features[0]['entity_ids'] == [1, 2]
    spans = features[0]['entity_spans']
    assert (spans[0][0] > spans[1][0]) == reverse
    batch = luke.collate_features(features, 1)
    for span, positions in zip(spans, batch['entity_position_ids'][0]):
        assert positions == list(range(span[0], span[1] + 1))
    assert batch['entity_attention_mask'] == [[1, 1]]


def test_gold_changes_do_not_change_any_model_input():
    a, b = record(), record()
    b['relation'] = LABELS[2]
    x, y = (luke.encode_records(CACHE, TokenizerFixture(), [r], LABELS)[0] for r in (a, b))
    assert x['labels'] != y['labels']
    assert {k:v for k,v in x.items() if k != 'labels'} == {k:v for k,v in y.items() if k != 'labels'}


def test_long_entity_keeps_every_position_including_markers():
    raw = {'id': 'long', 'token': ['word'] * 35 + ['met', 'object'], 'relation': LABELS[1],
           'subj_start': 0, 'subj_end': 34, 'obj_start': 36, 'obj_end': 36}
    encoded = luke.encode_records(CACHE, TokenizerFixture(), [raw], LABELS)
    batch = luke.collate_features(encoded, 1)
    assert len(batch['entity_position_ids'][0][0]) == 37
    assert -1 not in batch['entity_position_ids'][0][0]
    assert batch['entity_position_ids'][0][1].count(-1) == 34


def test_variable_word_padding_does_not_shift_entity_positions():
    first = luke.encode_records(CACHE, TokenizerFixture(), [record()], LABELS)[0]
    second = copy.deepcopy(first)
    second['input_ids'] = second['input_ids'][:-1] + [101, 102, 2]
    data = luke.collate_features([first, second], 1)
    assert data['input_ids'][0][-2:] == [1, 1]
    assert data['attention_mask'][0][-2:] == [0, 0]
    assert data['entity_position_ids'][0] == data['entity_position_ids'][1]


@pytest.mark.parametrize('mutate', [
 lambda r: r.update(subj_start=-1), lambda r: r.update(subj_end=100),
 lambda r: r.update(obj_start=0, obj_end=0), lambda r: r.update(relation='unknown'),
 lambda r: r['token'].__setitem__(0, '<ent>'), lambda r: r.pop('id'),
 lambda r: r.update(subj_start=True),
])
def test_invalid_input_rejected_not_repaired(mutate):
    raw = record(); mutate(raw)
    with pytest.raises(ValueError):
        luke.encode_records(CACHE, TokenizerFixture(), [raw], LABELS)


def test_no_silent_word_truncation_or_marker_repair():
    raw = record(); raw['token'] += ['word'] * 510
    with pytest.raises(ValueError, match='exceeds512'):
        luke.encode_records(CACHE, TokenizerFixture(), [raw], LABELS)
    class BadTokenizer(TokenizerFixture):
        def __call__(self, *args, **kwargs): return {'input_ids':[0, 100, 2]}
    with pytest.raises(ValueError, match='entity markers'):
        luke.encode_records(CACHE, BadTokenizer(), [record()], LABELS)


@pytest.mark.parametrize('labels', [LABELS[:-1], ['no_relation'] * 40, ['other_' + str(i) for i in range(40)]])
def test_label_mapping_explicit_and_canonical40(labels):
    with pytest.raises(ValueError, match='canonical40'):
        luke.validate_records([], labels)


def test_source_tampering_before_execution(tmp_path):
    spec = luke.lock()['source']['files']['examples/relation_classification/reader.py']
    (tmp_path / spec['file']).write_text('raise RuntimeError("should not run")')
    with pytest.raises(ValueError, match='hash mismatch'):
        luke.parse_records(tmp_path, [record()])


def test_actual_source_head_fixture_honest_scope():
    data = source_head_fixture(CACHE)
    assert data['logit_shape'] == [2, 40]
    assert data['feature_vectors'][0] == [1, 2, 3, 4]
    assert data['loss'] > 0
    assert not data['backbone_runtime_checked'] and not data['autograd_checked']


@pytest.mark.parametrize('change', [
 {'missing_keys':['classifier.weight','classifier.bias','luke.encoder.missing']},
 {'missing_keys':[]}, {'unexpected_keys':['alien.weight']},
 {'mismatched_keys':['shape mismatch']}, {'error_msgs':['load failed']},
])
def test_checkpoint_key_negatives(change):
    info = {'missing_keys':['classifier.weight','classifier.bias'], 'unexpected_keys':['lm_head.bias']}
    info.update(change)
    with pytest.raises(ValueError): luke.loading_keys_checked(info)


def test_checkpoint_allowed_new_head_and_discarded_pretraining_heads():
    luke.loading_keys_checked({'missing_keys':['classifier.weight','classifier.bias','luke.pooler.dense.weight'],
                              'unexpected_keys':['lm_head.bias','entity_predictions.bias']})


def test_streaming_checkpoint_asset_integrity_without_torch(tmp_path, monkeypatch):
    import hashlib
    data = b'tiny-fixture'
    (tmp_path/'weights.bin').write_bytes(data)
    spec = {'checkpoint':{'files':{'weights.bin':{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}}
    monkeypatch.setattr(luke, 'lock', lambda:spec)
    assert luke.verify_checkpoint(tmp_path)
    (tmp_path/'weights.bin').write_bytes(b'x' * len(data))
    with pytest.raises(ValueError, match='hash mismatch'): luke.verify_checkpoint(tmp_path)


def test_symlink_asset_rejected(tmp_path):
    source = tmp_path/'source'; source.write_text('fixture')
    target = tmp_path/'target'; target.symlink_to(source)
    with pytest.raises(ValueError, match='symlink'): luke.checked_file(tmp_path, 'target')


def test_preparation_not_admission_and_existing_five_unchanged():
    output = report(CACHE)
    assert output['adapter_implemented'] and output['jobs_launched'] == 0
    assert output['status'] == 'not_verifiable' and not output['strict_comparison_eligible']
    assert len(select_models('ready')) == 5
    assert 'luke_foundation' not in [row['id'] for row in cohort()['models']]
    for name in ('luke_core','luke_cfie','luke_resample','luke_focal'):
        with pytest.raises(ValueError): select_models(name)


def test_actual_cli_imports_no_torch_or_numpy():
    code = ('import runpy,sys; sys.argv=["gate","--source-cache",'+repr(str(CACHE))+']; '
            '\ntry: runpy.run_module("baselines.retacred.luke_fidelity",run_name="__main__")'
            '\nexcept SystemExit as e:'
            '\n assert "torch" not in sys.modules and "numpy" not in sys.modules; raise')
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)['jobs_launched'] == 0


def test_existing_cgroup_memory_guard_prevents_weight_load():
    maximum = Path('/sys/fs/cgroup/memory.max')
    if maximum.exists() and maximum.read_text().strip() != 'max' and int(maximum.read_text()) < 12*1024**3:
        with pytest.raises(RuntimeError, match='runtime gate pending'):
            luke.require_runtime_memory()
