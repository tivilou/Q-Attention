"""Dependency-light LUKE preparation gate. Full Torch/runtime gates stay pending."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .luke import checked_file, file_digest, lock, parse_records, require_runtime_memory, verify_source


class ToyTensor:
    """Only supports the author's two entity selections; not a torch substitute."""
    def __init__(self, data, ndim):
        self.data, self.ndim = data, ndim

    def __getitem__(self, index):
        if len(index) != 3 or index[0] != slice(None) or index[2] != slice(None) or index[1] not in (0, 1):
            raise ValueError('unexpected head source slice')
        return ToyTensor([row[index[1]][:] for row in self.data], 2)


def source_head_fixture(source):
    """Execute actual author forward on tiny fixed vectors, dropout identity.

    Tests head concatenation/order and scalar CE algebra only, not backbone,
    embeddings, autograd, stochastic dropout, optimizer or weight load.
    """
    spec = lock()['source']['files']['examples/relation_classification/luke_for_entity_pair_classification.py']
    path = checked_file(source, spec['file'])
    if file_digest(path) != spec['sha256']:
        raise ValueError('pinned classifier source hash mismatch')
    tree = ast.parse(path.read_bytes())
    classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'LukeForEntityPairClassification']
    nodes = [x for x in classes[0].body if isinstance(x, ast.FunctionDef) and x.name == 'forward']
    if len(nodes) != 1:
        raise ValueError('classifier forward identity mismatch')
    captured = {}
    weights = [[(i + 1) * 0.01, -0.02, 0.03, -0.04] for i in range(40)]
    bias = [i * 0.001 for i in range(40)]
    vectors = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]

    def backbone(**kwargs):
        captured['backbone_inputs'] = kwargs
        return SimpleNamespace(entity_last_hidden_state=ToyTensor(vectors, 3),
                               hidden_states=None, entity_hidden_states=None, attentions=None)

    def concatenate(values, dim):
        if dim != 1 or len(values) != 2:
            raise ValueError('unexpected concat')
        return ToyTensor([a + b for a, b in zip(values[0].data, values[1].data)], 2)

    def classifier(tensor):
        captured['feature_vectors'] = tensor.data
        return ToyTensor([[sum(a * b for a, b in zip(row, w)) + c for w, c in zip(weights, bias)]
                          for row in tensor.data], 2)

    def cross_entropy(logits, labels):
        losses = []
        for row, target in zip(logits.data, labels.data):
            pivot = max(row)
            losses.append(pivot + math.log(sum(math.exp(value - pivot) for value in row)) - row[target])
        return sum(losses) / len(losses)

    namespace = {'torch': SimpleNamespace(cat=concatenate),
                 'nn': SimpleNamespace(functional=SimpleNamespace(cross_entropy=cross_entropy)),
                 'EntityPairClassificationOutput': lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    self = SimpleNamespace(config=SimpleNamespace(use_return_dict=True), luke=backbone,
                           dropout=lambda value: value, classifier=classifier)
    gold = ToyTensor([0, 2], 1)
    positions = [[[1, 2, 3], [6, 7, 8]], [[6, 7, 8], [1, 2, 3]]]
    output = namespace['forward'](self, entity_ids=[[1, 2], [1, 2]], entity_position_ids=positions, labels=gold)
    if captured['feature_vectors'] != [[1., 2., 3., 4.], [5., 6., 7., 8.]]:
        raise AssertionError('subject/object concatenation mismatch')
    if captured['backbone_inputs']['entity_position_ids'] != positions or 'labels' in captured['backbone_inputs']:
        raise AssertionError('position routing or gold-input contamination')
    expected = [[sum(a * b for a, b in zip(row, w)) + c for w, c in zip(weights, bias)]
                for row in captured['feature_vectors']]
    if output.logits.data != expected or abs(output.loss - cross_entropy(ToyTensor(expected, 2), gold)) > 1e-12:
        raise AssertionError('head/CE algebra mismatch')
    return {'scope': 'actual_source_head_algebra_on_tiny_vectors_only',
            'feature_vectors': captured['feature_vectors'], 'logit_shape': [2, 40], 'loss': output.loss,
            'backbone_runtime_checked': False, 'autograd_checked': False}


def report(source):
    hashes = verify_source(source)
    fixture = {'id': 'toy', 'token': ['Alice', 'joined', 'Acme', '.'],
               'subj_start': 0, 'subj_end': 0, 'obj_start': 2, 'obj_end': 2, 'relation': 'toy_relation'}
    parsed = parse_records(source, [fixture])
    if parsed != [{'example_id': 'toy', 'sentence': '<ent> Alice<ent> joined<ent2> Acme<ent2> .', 'label': 'toy_relation'}]:
        raise AssertionError('source parser fixture mismatch')
    return {'schema': 'luke-foundation-preparation.v1', 'status': 'not_verifiable',
            'adapter_implemented': True, 'strict_comparison_eligible': False,
            'source_files_verified': len(hashes), 'checkpoint_identity': lock()['checkpoint']['revision'],
            'weights_downloaded': False, 'source_parser_fixture': parsed,
            'source_head_fixture': source_head_fixture(source),
            'jobs_launched': 0, 'pending': lock()['pending_runtime_gates']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--runtime-memory-gate', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = report(args.source_cache)
        if args.runtime_memory_gate:
            try:
                result['runtime_memory_headroom'] = require_runtime_memory()
                result['memory_gate'] = 'passed_not_a_runtime_certificate'
            except RuntimeError as error:
                result['memory_gate'] = 'pending'
                result['memory_gate_reason'] = str(error)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2
    except (ValueError, OSError, AssertionError) as error:
        print(json.dumps({'status': 'invalid_evidence', 'reason': str(error), 'jobs_launched': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
