"""Inspect canonical Re-TACRED inputs without loading a model or emitting text.

This produces observations, not an automatic qualification or efficacy verdict.
The ordered entity spans are zero-based and end-exclusive. No data is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

EXPECTED = {
    # The reference manifest separately hashes source JSONL and adapter output.
    # This study reads the source JSONL; never substitute its output_sha256.
    'train': (58465, 'f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'),
    'valid': (19584, '9d6965f984d7485e2e81d380a0b0de2922d3da1e2685393ea3e46869805c2b55'),
    'test': (13418, '2c1cfac1f3c0d18ed8a1c9f9d49808dacc3a0f03519978b58242a7e79d4cdf40'),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf8')).digest()


def inspect_split(path):
    labels, lengths, metadata_fields = Counter(), Counter(), Counter()
    identities = {name: set() for name in ('id', 'document', 'sentence', 'input', 'surface_pair')}
    input_labels = {}
    repeated = Counter()
    errors = Counter()
    entity_overlap = eligible_lt2 = missing_ids = missing_docids = conflicting_labels = 0
    file_hash = hashlib.sha256()
    count = 0
    original_stat = path.stat()
    with path.open('rb') as handle:
        for raw in handle:
            file_hash.update(raw)
            if not raw.strip():
                errors['blank_line'] += 1
                continue
            count += 1
            try:
                row = json.loads(raw)
                tokens, subject, obj, label = (row[k] for k in ('tokens', 'subject', 'object', 'label'))
                assert isinstance(tokens, list) and tokens and all(isinstance(t, str) and t for t in tokens)
                assert isinstance(label, str) and label
                for span in (subject, obj):
                    assert isinstance(span, list) and len(span) == 2
                    assert all(type(i) is int for i in span)
                    assert 0 <= span[0] < span[1] <= len(tokens)
                meta = row.get('metadata', {})
                assert isinstance(meta, dict)
            except (ValueError, KeyError, TypeError, AssertionError):
                errors['invalid_record'] += 1
                continue
            labels[label] += 1
            lengths[len(tokens)] += 1
            metadata_fields.update(meta.keys())
            entity_positions = set(range(*subject)) | set(range(*obj))
            entity_overlap += bool(set(range(*subject)) & set(range(*obj)))
            eligible_lt2 += len(tokens) - len(entity_positions) < 2
            sample_id = meta.get('id')
            missing_ids += sample_id is None
            keys = {
                'sentence': fingerprint(tokens),
                'input': fingerprint([tokens, subject, obj]),
                # Surface equality is a correlation proxy, not entity coreference.
                'surface_pair': fingerprint([
                    [t.casefold() for t in tokens[subject[0]:subject[1]]],
                    [t.casefold() for t in tokens[obj[0]:obj[1]]],
                ]),
            }
            if sample_id is not None:
                keys['id'] = fingerprint(sample_id)
            docid = meta.get('docid')
            missing_docids += docid is None or docid == ''
            if docid is not None and docid != '':
                keys['document'] = fingerprint(docid)
            for name, key in keys.items():
                repeated[name] += key in identities[name]
                identities[name].add(key)
            input_key = keys['input']
            conflicting_labels += input_key in input_labels and input_labels[input_key] != label
            input_labels[input_key] = label
    final_stat = path.stat()
    if (original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns) != (
            final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError('Input changed during inspection')
    return {
        'records': count, 'sha256': file_hash.hexdigest(), 'label_counts': dict(sorted(labels.items())),
        'word_length_histogram': dict(sorted(lengths.items())),
        'word_tokens': sum(k*v for k, v in lengths.items()),
        'max_words': max(lengths, default=0), 'metadata_fields': sorted(metadata_fields),
        'errors': dict(errors), 'missing_ids': missing_ids, 'missing_document_ids': missing_docids,
        'repeated_rows_within_split': dict(repeated), 'conflicting_input_label_rows': conflicting_labels,
        'entity_overlap_rows': entity_overlap, 'fewer_than_two_context_words': eligible_lt2,
    }, identities, input_labels


def inspect(data_dir):
    results, indices, label_maps = {}, {}, {}
    for split, (count, sha) in EXPECTED.items():
        result, indices[split], label_maps[split] = inspect_split(data_dir / (split + '.jsonl'))
        result['matches_audited_reference_count'] = result['records'] == count
        result['matches_audited_reference_sha256'] = result['sha256'] == sha
        results[split] = result
    labels = sorted(results['train']['label_counts'])
    overlaps = {}
    for a, b in itertools.combinations(EXPECTED, 2):
        overlap = {name: len(indices[a][name] & indices[b][name]) for name in indices[a]}
        common = label_maps[a].keys() & label_maps[b].keys()
        overlap['input_label_conflicts'] = sum(label_maps[a][key] != label_maps[b][key] for key in common)
        overlaps[a + '/' + b] = overlap
    identity_ok = all(r['matches_audited_reference_count'] and r['matches_audited_reference_sha256']
                      and not r['errors'] for r in results.values())
    return {
        'schema': 'qattn.rmea.input-inspection.v1',
        'status': 'observations_only', 'dataset_identity_matches_reference': identity_ok,
        'splits': results, 'cross_split_shared_unique_keys': overlaps,
        'train_derived_labels': labels, 'num_labels': len(labels),
        'unseen_labels': {s: sorted(set(r['label_counts']) - set(labels)) for s, r in results.items()},
        'fixed_label_macro_policy': 'all training labels, including zero-support labels; zero division = 0',
        'reference': 'retacred_reference_suite/20261012T112015Z/luke_core_seed13/data_manifest.json',
        'limitations': [
            'Sentence and entity-surface overlaps are correlations, not proof of contamination or document disjointness.',
            'Shared input hashes require explicit interpretation before qualification; official splits are preserved.',
            'No tokenizer, frozen encoder, learnability, model test, training or efficacy evidence is produced.',
            'Input construction must whitelist tokens/ordered spans; metadata and labels never enter encoder features.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior inspection; select a new output path')
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        raise ValueError('Keep inspection outputs outside source data')
    result = inspect(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted write cannot look like a complete JSON result.
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    with args.output.open('xb') as handle:
        handle.write(payload)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(payload).hexdigest(),
                      'identity_matches': result['dataset_identity_matches_reference']}))
    if not result['dataset_identity_matches_reference']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
