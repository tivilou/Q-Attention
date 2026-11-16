"""Deterministic train-only private evidence-ownership pack; no model or readers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

PREFIX = b'qattn-evidence-ownership-v1\n'
PACK_KEYS = {'schema', 'instructions', 'groups'}
INSTRUCTIONS = (
    'Judge each query using only this unchanged text and resolved references. '
    'Mark supported with explicit token intervals beyond entity identities, '
    'unsupported_in_text, or ambiguous_or_context_missing. Compare fragments '
    'as supports_A_only/supports_B_only/supports_both/neither/uncertain. '
    'Multiple/discontinuous spans are allowed. Do not assume inverse relations '
    'or use world knowledge. Give one concise evidence-linked justification. '
    'Keep every group, including ambiguity; do not see another reader\'s answer.'
)


def encoded(x):
    return json.dumps(x, ensure_ascii=True, separators=(',', ':'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build(lines, directory, budget):
    """Label values are never looked up; source bytes still contain labels."""
    digest = hashlib.sha256()
    rows = 0
    with tempfile.TemporaryDirectory(prefix='ownership-', dir=directory) as scratch:
        db = sqlite3.connect(str(Path(scratch) / 'groups.sqlite3'))
        try:
            db.execute('PRAGMA cache_size=-2048')
            db.execute('PRAGMA temp.cache_size=-2048')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE q(t TEXT COLLATE BINARY,s0 INT,s1 INT,o0 INT,o1 INT,PRIMARY KEY(t,s0,s1,o0,o1)) WITHOUT ROWID')
            for line in lines:
                digest.update(line)
                row = json.loads(line)
                tokens, subject, obj = row['tokens'], row['subject'], row['object']
                if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
                    raise ValueError('invalid original tokens')
                for span in (subject, obj):
                    if not isinstance(span, list) or len(span) != 2 or not all(type(x) is int for x in span) or not 0 <= span[0] < span[1] <= len(tokens):
                        raise ValueError('invalid entity span')
                db.execute('INSERT OR IGNORE INTO q VALUES(?,?,?,?,?)', (encoded(tokens), *subject, *obj))
                rows += 1
            db.commit()
            db.create_function('groupkey', 1, lambda t: sha(PREFIX + t.encode('utf8')))
            eligible = db.execute('SELECT COUNT(*) FROM (SELECT t FROM q GROUP BY t HAVING COUNT(*)>=2)').fetchone()[0]
            groups = []
            selected = db.execute('SELECT t,groupkey(t) FROM q GROUP BY t HAVING COUNT(*)>=2 ORDER BY groupkey(t),t COLLATE BINARY LIMIT ?', (budget,)).fetchall()
            for tokens, key in selected:
                queries = db.execute('SELECT s0,s1,o0,o1 FROM q WHERE t=? ORDER BY s0,s1,o0,o1 LIMIT 2', (tokens,)).fetchall()
                groups.append(dict(id='g-' + key, tokens=json.loads(tokens), queries=[dict(id='q-' + sha((key + ':' + encoded(q)).encode()), subject=list(q[:2]), object=list(q[2:])) for q in queries]))
            pack = dict(schema='qattn.evidence-ownership.private-pack.v1', instructions=INSTRUCTIONS, groups=groups)
            validate_pack(pack, budget)
            return pack, dict(train_sha256=digest.hexdigest(), rows=rows, eligible_groups=eligible, selected_groups=len(groups), selected_queries=sum(len(g['queries']) for g in groups))
        finally:
            db.close()


def validate_pack(pack, budget):
    assert set(pack) == PACK_KEYS and pack['instructions'] == INSTRUCTIONS
    assert pack['schema'] == 'qattn.evidence-ownership.private-pack.v1'
    assert len(pack['groups']) == budget
    keys = []
    for g in pack['groups']:
        assert set(g) == {'id', 'tokens', 'queries'}
        t = g['tokens']
        assert isinstance(t, list) and t and all(isinstance(x, str) for x in t)
        key = sha(PREFIX + encoded(t).encode('utf8'))
        assert g['id'] == 'g-' + key
        keys.append((key, encoded(t)))
        assert len(g['queries']) == 2
        spans = []
        for q in g['queries']:
            assert set(q) == {'id', 'subject', 'object'}
            for s in (q['subject'], q['object']):
                assert isinstance(s, list) and len(s) == 2 and all(type(x) is int for x in s) and 0 <= s[0] < s[1] <= len(t)
            coords = (*q['subject'], *q['object'])
            assert q['id'] == 'q-' + sha((key + ':' + encoded(coords)).encode())
            spans.append(coords)
        assert spans == sorted(set(spans))
    assert keys == sorted(set(keys))


def validate_public_report(report):
    """Reject unknown/narrative fields before any compact report transfer."""
    fixed = dict(schema='qattn.evidence-ownership.extraction-result.v1', status='PASS', private_pack_mode='0o600', output_directory_mode='0o700', private_pack_validated=True, label_values_consulted=False, label_values_exported=False, source_bytes_include_train_labels=True, raw_examples_exported_to_collector=False, valid_or_test_files_read=False, reader_launches=0, new_model_forwards=0, training=0, l5_accepted=False)
    hashes = {'private_pack_sha256', 'protocol_sha256', 'source_sha256'}
    assert set(report) == set(fixed) | hashes | {'counts', 'private_pack_bytes', 'peak_rss_kib_linux'}
    assert all(type(report[k]) is type(v) and report[k] == v for k, v in fixed.items())
    def digest(x):
        return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)
    assert all(digest(report[k]) for k in hashes)
    counts = report['counts']
    assert set(counts) == {'train_sha256', 'rows', 'eligible_groups', 'selected_groups', 'selected_queries'}
    assert digest(counts['train_sha256'])
    assert all(type(v) is int and v >= 0 for k, v in counts.items() if k != 'train_sha256')
    assert 0 < counts['selected_groups'] <= 32 and counts['selected_queries'] == 2 * counts['selected_groups'] and counts['eligible_groups'] >= counts['selected_groups']
    assert type(report['private_pack_bytes']) is int and 0 < report['private_pack_bytes'] <= 2 * 1024**2
    peak = report['peak_rss_kib_linux']
    assert peak is None or type(peak) is int and 0 < peak


def validate_canary_report(report):
    fixed = dict(schema='qattn.evidence-ownership.lifecycle-canary.v1', status='PASS', injected_failure_observed=True, partial_output_not_complete=True, partial_output_cleaned=True, retry_same_immutable_source_inputs=True, private_pack_retry_byteexact=True, permissions_verified=True, real_data_read=False, reader_launches=0, training=0)
    assert set(report) == set(fixed) | {'fixture_counts', 'command_results'}
    assert all(type(report[k]) is type(v) and report[k] == v for k, v in fixed.items())
    c = report['fixture_counts']
    assert set(c) == {'train_sha256', 'rows', 'eligible_groups', 'selected_groups', 'selected_queries'}
    assert isinstance(c['train_sha256'], str) and len(c['train_sha256']) == 64 and all(x in '0123456789abcdef' for x in c['train_sha256'])
    assert all(type(c[k]) is int and c[k] == v for k, v in dict(rows=8, eligible_groups=3, selected_groups=3, selected_queries=6).items())
    assert len(report['command_results']) == 2
    for r, code in zip(report['command_results'], [1, 0]):
        assert set(r) == {'exit_status', 'stdout_sha256', 'stderr_sha256'} and type(r['exit_status']) is int and r['exit_status'] == code
        for k in ['stdout_sha256', 'stderr_sha256']:
            assert isinstance(r[k], str) and len(r[k]) == 64 and all(x in '0123456789abcdef' for x in r[k])


def validate_transport(report_bytes, marker_bytes, mode):
    """Called on owner before report bytes leave that trust boundary."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate report object key')
            result[key] = value
        return result
    report = json.loads(report_bytes, object_pairs_hook=unique_object)
    if mode == 'audit':
        validate_public_report(report)
        assert marker_bytes == b'private blind ownership pack validated\n'
    else:
        assert mode == 'canary'
        validate_canary_report(report)
        assert marker_bytes == b'private-pack lifecycle canary complete\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    raw = a.protocol.read_bytes()
    p = json.loads(raw)
    assert p['schema'] == 'qattn.evidence-ownership.extractor.v1'
    assert type(p['group_budget']) is int and 0 < p['group_budget'] <= 32
    assert a.input.name == 'train.jsonl' and a.input.is_file() and not a.input.is_symlink() and a.input.stat().st_size <= 64 * 1024**2
    assert not a.output.is_symlink() and (not a.output.exists() or a.output.is_dir() and not any(a.output.iterdir()))
    a.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(a.output, 0o700)
    with a.input.open('rb') as lines:
        pack, counts = build(lines, str(a.output.parent), p['group_budget'])
    assert counts['train_sha256'] == p['expected_sha256'] and counts['rows'] == p['expected_rows'] and counts['eligible_groups'] == p['expected_eligible_groups'], 'wrong source identity/population'
    data = (encoded(pack) + '\n').encode('utf8')
    assert len(data) <= 2 * 1024**2
    # The worker hashes this explicitly declared private output, but the owner
    # collector transfers only report+marker. Raw pack and IDs stay on owner.
    target = a.output / 'private-pack.json'
    with target.open('xb') as f:
        os.chmod(target, 0o600)
        f.write(data)
    assert target.read_bytes() == data
    validate_pack(json.loads(target.read_bytes()), p['group_budget'])
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        peak = None
    result = dict(schema='qattn.evidence-ownership.extraction-result.v1', status='PASS', counts=counts, private_pack_sha256=sha(data), private_pack_bytes=len(data), private_pack_mode=oct(target.stat().st_mode & 0o777), output_directory_mode=oct(a.output.stat().st_mode & 0o777), private_pack_validated=True, protocol_sha256=sha(raw), source_sha256=sha(Path(__file__).read_bytes()), peak_rss_kib_linux=peak, label_values_consulted=False, label_values_exported=False, source_bytes_include_train_labels=True, raw_examples_exported_to_collector=False, valid_or_test_files_read=False, reader_launches=0, new_model_forwards=0, training=0, l5_accepted=False)
    validate_public_report(result)
    (a.output / 'compact-report.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    (a.output / 'RUN_COMPLETE').write_bytes(b'private blind ownership pack validated\n')
    print(json.dumps(dict(status='PASS', counts=counts, private_pack_bytes=len(data), reader_launches=0)))


if __name__ == '__main__':
    main()
