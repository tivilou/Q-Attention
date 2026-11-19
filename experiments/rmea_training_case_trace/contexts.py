"""Authenticate text context separately from the intentionally sparse feature metadata."""
import hashlib
import json
import unicodedata


def paired_inputs(metadata, selected, contexts, *, train_sha256, selection_sha256, cohort):
    assert set(contexts) == {'schema', 'train_sha256', 'source_selection_sha256', 'rows'}
    assert contexts['schema'] == 'qattn.training-case-contexts.v1'
    assert contexts['train_sha256'] == train_sha256
    assert contexts['source_selection_sha256'] == selection_sha256
    wanted = {int(i): int(label) for label, i in selected.items()}
    assert len(wanted) == len(selected) == len(contexts['rows'])
    rows = {}
    for row in contexts['rows']:
        assert set(row) == {'source_index', 'id_hash', 'label', 'tokens', 'subject', 'object'}
        i = row['source_index']
        assert type(i) is int and i in wanted and i not in rows
        m = metadata[i]
        assert row['id_hash'] == m['id_hash'] == cohort[i]['id_hash']
        assert type(row['label']) is int and row['label'] == wanted[i]
        tokens = row['tokens']
        assert isinstance(tokens, list) and len(tokens) == m['words'] > 0
        assert all(isinstance(t, str) and t for t in tokens)
        for role in ('subject', 'object'):
            span = row[role]
            assert isinstance(span, list) and len(span) == 2
            assert all(type(v) is int for v in span) and 0 <= span[0] < span[1] <= len(tokens)
        for key in ('tokens', 'subject', 'object'):
            if key in m:
                assert m[key] == row[key]
        def normalized(role):
            text = ' '.join(tokens[slice(*row[role])])
            return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
        pair = sorted([normalized('subject'), normalized('object')])
        assert all(pair)
        group = hashlib.sha256(json.dumps(pair, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        assert group == cohort[i]['group']
        rows[i] = row
    assert set(rows) == set(wanted)
    return [dict(rows[i]) for i in selected.values()]
