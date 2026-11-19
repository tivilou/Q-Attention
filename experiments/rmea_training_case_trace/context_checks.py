"""Stdlib schema checks: sparse metadata is valid; mismatched original text is not."""
import copy
import hashlib
import json
from experiments.rmea_training_case_trace.contexts import paired_inputs


def main():
    group = hashlib.sha256(b'["alice","company"]' ).hexdigest()
    minimal = {'id_hash': 'synthetic-id', 'words': 4}
    row = dict(source_index=0, id_hash='synthetic-id', label=3,
               tokens=['Alice', 'joined', 'Company', '.'], subject=[0, 1], object=[2, 3])
    contexts = dict(schema='qattn.training-case-contexts.v1', train_sha256='train',
                    source_selection_sha256='selection', rows=[row])
    cohort = [dict(id_hash='synthetic-id', group=group)]
    def check(c=contexts, m=None, g=None):
        return paired_inputs([minimal if m is None else m], {3: 0}, c,
            train_sha256='train', selection_sha256='selection', cohort=cohort if g is None else g)
    assert check() == [row]
    assert check(m=dict(minimal, **{k:row[k] for k in ('tokens','subject','object')})) == [row]
    failures = []
    for name, mutate in [
        ('wrong_original_train', lambda c: c.update(train_sha256='other')),
        ('wrong_original_selection', lambda c: c.update(source_selection_sha256='other')),
        ('missing_context', lambda c: c.update(rows=[])),
        ('duplicate_context', lambda c: c['rows'].append(copy.deepcopy(c['rows'][0]))),
        ('wrong_id', lambda c: c['rows'][0].update(id_hash='other')),
        ('unselected_index', lambda c: c['rows'][0].update(source_index=2)),
        ('wrong_label', lambda c: c['rows'][0].update(label=4)),
        ('wrong_token_count', lambda c: c['rows'][0]['tokens'].append('extra')),
        ('empty_entity_span', lambda c: c['rows'][0].update(subject=[1,1])),
        ('bool_entity_index', lambda c: c['rows'][0].update(subject=[False,1])),
        ('out_of_bounds_span', lambda c: c['rows'][0].update(object=[2,7])),
        ('wrong_entity_surface', lambda c: c['rows'][0]['tokens'].__setitem__(0,'Bob')),
    ]:
        c=copy.deepcopy(contexts);mutate(c)
        try: check(c)
        except AssertionError: failures.append(name)
        else: raise AssertionError('Bad original context accepted: '+name)
    try: check(m=dict(minimal, tokens=['other']*4))
    except AssertionError: failures.append('rich_metadata_disagrees')
    else: raise AssertionError('Contradictory rich metadata accepted')
    print(json.dumps(dict(status='PASS_MIXED_METADATA_CONTEXT_CHECKS', sparse_and_rich_pass=True,
        rejected=failures, rejected_count=len(failures), torch_imported=False)))


if __name__ == '__main__':
    main()
