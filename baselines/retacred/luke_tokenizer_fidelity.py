"""Offline real-LUKE tokenizer/interface gate, not a model fidelity certificate.

Runs in a dedicated process. The actual pinned author's span method is executed
with list-only field carriers and a real HF tokenizer. This is NOT an AllenNLP
runtime equivalence claim. No Torch/model import or weight read is permitted.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from .common import ROOT
from .luke import ENT, ENT2, checked_file, collate_features, encode_records, file_digest, lock, parse_records, verify_source
from .luke_assets import REVISION, verify_original

HEAVY = frozenset(('torch', 'tensorflow', 'jax', 'jaxlib'))
LABELS = ['no_relation'] + ['toy_relation_' + str(i) for i in range(39)]


class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in HEAVY:
            raise ImportError('tokenizer-only gate blocked a model backend import')
        return None


def assert_no_backend():
    if any(name.split('.')[0] in HEAVY for name in sys.modules):
        raise ValueError('run tokenizer gate in a fresh process without model backends')


def make_record(name, tokens, subject, obj, relation=None):
    return {'id': name, 'token': tokens[:], 'subj_start': subject[0], 'subj_end': subject[1],
            'obj_start': obj[0], 'obj_end': obj[1], 'relation': relation or LABELS[1]}


def fixtures():
    return [
        make_record('subject_first', ['Alice', 'joined', 'Acme', '.'], (0, 0), (2, 2)),
        make_record('object_first', ['Acme', 'hired', 'Alice', '.'], (2, 2), (0, 0)),
        make_record('adjacent', ['Alice', 'Acme'], (0, 0), (1, 1)),
        make_record('adjacent_reverse', ['Acme', 'Alice'], (1, 1), (0, 0)),
        make_record('punctuation', ['(', 'Alice', ')', 'joined', 'Acme', ',', 'Inc.', '.'], (1, 1), (4, 6)),
        make_record('unicode', ['Zoë', 'met', '李明', 'in', 'Montréal', '.'], (0, 0), (2, 2)),
        make_record('multisubword', ['antidisestablishmentarianism', 'met', 'Acme'], (0, 0), (2, 2)),
        make_record('long_entity', ['word'] * 35 + ['met', 'object'], (0, 34), (36, 36)),
        make_record('exact512', ['word'] * 505, (0, 0), (504, 504)),
    ]


def source_reader(source, tokenizer):
    """Exact pinned text_to_instance logic; carriers expose only spans/fields.

    The token carrier adds BOS/EOS to real HF BPE. Actual AllenNLP wrapper,
    indexer and tensor fields are deliberately NOT imported or certified.
    """
    spec = lock()['source']['files']['examples/relation_classification/reader.py']
    path = checked_file(source, spec['file'])
    if file_digest(path) != spec['sha256']:
        raise ValueError('pinned source reader identity mismatch')
    tree = ast.parse(path.read_bytes())
    methods = [method for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'RelationClassificationReader' for method in node.body
               if isinstance(method, ast.FunctionDef) and method.name == 'text_to_instance']
    if len(methods) != 1:
        raise ValueError('source span method identity mismatch')
    utility = lock()['source']['files']['examples/utils/util.py']
    utilpath = checked_file(source, utility['file'])
    if file_digest(utilpath) != utility['sha256']:
        raise ValueError('pinned source utility identity mismatch')
    functions = [node for node in ast.parse(utilpath.read_bytes()).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'list_rindex']
    if len(functions) != 1:
        raise ValueError('source reverse-index identity mismatch')
    namespace = {'ENT': ENT, 'ENT2': ENT2,
                 'Token': lambda text: SimpleNamespace(text=text),
                 'TextField': lambda tokens, token_indexers: tokens,
                 'SpanField': lambda start, end, text_field: [start, end],
                 'MetadataField': lambda value: value, 'LabelField': lambda value: value,
                 'TensorField': lambda value: value, 'Instance': lambda fields: fields,
                 'np': SimpleNamespace(array=lambda values: list(values))}
    exec(compile(ast.Module(body=functions + methods, type_ignores=[]), str(path), 'exec'), namespace)
    def tokenize(sentence):
        return [SimpleNamespace(text=t) for t in
                [tokenizer.bos_token] + tokenizer.tokenize(sentence) + [tokenizer.eos_token]]
    reader = SimpleNamespace(tokenizer=SimpleNamespace(tokenize=tokenize), token_indexers={},
                             use_entity_feature=True, head_entity_id=tokenizer.entity_vocab['[MASK]'],
                             tail_entity_id=tokenizer.entity_vocab['[MASK2]'])
    return lambda sentence, label: namespace['text_to_instance'](reader, sentence, label)


def rejected(tokenizer, source, raw, expected):
    try:
        encode_records(source, tokenizer, [raw], LABELS)
    except ValueError as error:
        if expected not in str(error):
            raise AssertionError('wrong rejection reason: ' + str(error)) from error
        return {'case': raw['id'], 'reason': str(error), 'accepted': False}
    raise AssertionError('invalid fixture accepted: ' + raw['id'])


def report(source, checkpoint):
    assert_no_backend()
    if 'transformers' in sys.modules:
        raise ValueError('run tokenizer gate before importing transformers in a dedicated process')
    # The guard applies even if a dependency tries to enable a backend later.
    os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', USE_TORCH_XLA='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    guard = NoModelImports()
    sys.meta_path.insert(0, guard)
    try:
        source_hashes = verify_source(source)
        assets = {name: verify_original(checked_file(checkpoint, name), name)
                  for name in lock()['checkpoint']['files'] if name != 'pytorch_model.bin'}
        # Tokenizer source import only; no AutoModel/torch/weights/tensor return.
        from transformers.models.luke.tokenization_luke import LukeTokenizer
        tokenizer = LukeTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                                   additional_special_tokens=[ENT, ENT2], task=None)
        if len(tokenizer) != 50267 or tokenizer.task is not None:
            raise AssertionError('tokenizer vocabulary/task identity mismatch')
        if [tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id] != [0, 1, 2]:
            raise AssertionError('tokenizer BOS/pad/EOS identity mismatch')
        raw = fixtures()
        untouched = copy.deepcopy(raw)
        features = encode_records(source, tokenizer, raw, LABELS)
        if raw != untouched or len(features) != len(raw):
            raise AssertionError('caller mutation or record loss')
        reference = source_reader(source, tokenizer)
        measured = []
        for item, feature in zip(raw, features):
            original = reference(feature['input_sentence'], item['relation'])
            reference_ids = tokenizer.convert_tokens_to_ids([t.text for t in original['word_ids']])
            spans = [original['entity1_span'], original['entity2_span']]
            if feature['input_ids'] != reference_ids or feature['entity_spans'] != spans:
                raise AssertionError('actual source span/BPE-ID routing mismatch')
            if feature['entity_ids'] != original['entity_ids']:
                raise AssertionError('source subject/object mask routing mismatch')
            individual = collate_features([feature], tokenizer.pad_token_id)
            for span, positions in zip(spans, individual['entity_position_ids'][0]):
                valid = [p for p in positions if p != -1]
                if valid != list(range(span[0], span[1] + 1)):
                    raise AssertionError('mention position loss or truncation')
            measured.append({'case': item['id'], 'raw_tokens': item['token'],
                             'subject_raw_span': [item['subj_start'], item['subj_end']],
                             'object_raw_span': [item['obj_start'], item['obj_end']],
                             'source_sentence': feature['input_sentence'], 'input_ids': feature['input_ids'],
                             'bpe_tokens': tokenizer.convert_ids_to_tokens(feature['input_ids']),
                             'entity_spans': spans, 'entity_ids': feature['entity_ids'],
                             'entity_position_ids': individual['entity_position_ids'][0],
                             'token_count': len(feature['input_ids']), 'source_span_match': True})
        by_name = {row['case']: row for row in measured}
        if by_name['subject_first']['source_sentence'] != '<ent> Alice<ent> joined<ent2> Acme<ent2> .':
            raise AssertionError('pinned raw sentence fixture changed')
        if by_name['object_first']['entity_spans'][0][0] <= by_name['object_first']['entity_spans'][1][0]:
            raise AssertionError('subject/object roles sorted by physical position')
        if by_name['multisubword']['entity_spans'][0][1] - by_name['multisubword']['entity_spans'][0][0] <= 2:
            raise AssertionError('fixture failed to exercise subword expansion')
        if len([p for p in by_name['long_entity']['entity_position_ids'][0] if p != -1]) <= 30:
            raise AssertionError('long mention fixture not exercised')
        if by_name['exact512']['token_count'] != 512:
            raise AssertionError('512-token boundary fixture changed')
        mixed = collate_features(features[:2] + features[5:6] + features[7:8], tokenizer.pad_token_id)
        for feature, mask, positions in zip(features[:2] + features[5:6] + features[7:8],
                                            mixed['attention_mask'], mixed['entity_position_ids']):
            if sum(mask) != len(feature['input_ids']):
                raise AssertionError('word mask does not preserve input length')
            for span, row in zip(feature['entity_spans'], positions):
                if [p for p in row if p >= 0] != list(range(span[0], span[1] + 1)):
                    raise AssertionError('batch padding shifted entity positions')
        changed = copy.deepcopy(raw[0]); changed['relation'] = LABELS[2]
        alternate = encode_records(source, tokenizer, [changed], LABELS)[0]
        if alternate['labels'] == features[0]['labels'] or any(alternate[k] != value for k, value in features[0].items() if k != 'labels'):
            raise AssertionError('gold label affects tokenizer/model input')
        negatives = [rejected(tokenizer, source, make_record('over512', ['word'] * 506, (0, 0), (505, 505)), 'exceeds512')]
        for name, change, reason in (
            ('overlap', {'obj_start': 0, 'obj_end': 0}, 'overlapping'),
            ('marker_collision', {'token': ['<ent>', 'joined', 'Acme', '.']}, 'collide'),
            ('bad_span', {'subj_end': 99}, 'span'),
            ('unknown_label', {'relation': 'not_in_mapping'}, 'unknown')):
            bad = copy.deepcopy(raw[0]); bad.update(change, id=name)
            negatives.append(rejected(tokenizer, source, bad, reason))
        assert_no_backend()
        import resource
        return {'schema': 'luke-tokenizer-interface.v1', 'status': 'tokenizer_interface_verified',
                'checkpoint_revision': REVISION, 'source_revision': lock()['source']['revision'],
                'source_files_verified': len(source_hashes), 'tokenizer_asset_sha256': assets,
                'python_version': sys.version.split()[0], 'transformers_version': importlib.metadata.version('transformers'),
                'tokenizer_implementation_sha256': file_digest(Path(sys.modules[LukeTokenizer.__module__].__file__)),
                'vocab_size': len(tokenizer), 'marker_ids': [tokenizer.convert_tokens_to_ids(t) for t in (ENT, ENT2)],
                'mask_ids': [tokenizer.entity_vocab[t] for t in ('[MASK]', '[MASK2]')],
                'positive_cases': measured, 'negative_cases': negatives,
                'batch_padding_verified': True, 'gold_input_isolation_verified': True,
                'source_span_method_executed': True, 'allen_nlp_runtime_checked': False,
                'comparison_boundary': 'real_HF_BPE_plus_actual_pinned_reader_span_method_on_list_carriers',
                'torch_imported': False, 'weight_file_read': False, 'model_loaded': False,
                'network_requests': 0, 'jobs_launched': 0, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'strict_comparison_eligible': False,
                'pending': ['actual_AllenNLP_wrapper_indexer_runtime_equivalence', 'backbone_forward',
                            'checkpoint_load', 'autograd_update_resume_numeric_trace', 'CUDA_common_loop_recipe']}
    finally:
        sys.meta_path.remove(guard)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, default=ROOT / '.external/retacred/luke-foundation-20261008')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '.external/assets/luke' / REVISION / 'model')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        # Always perform this read-only check; never overwrite old evidence.
        if args.output is not None and args.output.exists():
            raise ValueError('existing tokenizer evidence preserved; choose a fresh output path')
        result = report(args.source_cache, args.checkpoint)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as target:
                json.dump(result, target, indent=2, ensure_ascii=False, allow_nan=False)
                target.write('\n')
        summary = {key: value for key, value in result.items() if key not in ('positive_cases', 'negative_cases')}
        summary.update(positive_cases=len(result['positive_cases']), negative_cases=len(result['negative_cases']))
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, AssertionError, ImportError) as error:
        print(json.dumps({'status': 'invalid_tokenizer_evidence', 'reason': str(error), 'jobs_launched': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
