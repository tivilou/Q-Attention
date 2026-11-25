"""Read-only source-port tests; no encoder/model/cache or dataset execution."""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--source-pins', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    marker = args.source_dir / 'audit_rmea_tokenization.py'
    extractor = args.source_dir / 'rmea_run.py'
    assert digest(marker) == '5f7860e47a7a0979976e5acc435ed34acede4c96bfac799d79d3018366b6aef8'
    assert digest(extractor) == '769173b9ab17f76229b0db79b4a7625ac1574427d2d40491be7d5b19d4d0ff25'
    marker_tree = ast.parse(marker.read_text(encoding='utf8'))
    fn = next(n for n in marker_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'marked_words')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(marker), 'exec'), namespace)
    marked_words = namespace['marked_words']
    tree = ast.parse(extractor.read_text(encoding='utf8'))
    extract = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract')
    assignments = [n for n in ast.walk(extract) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == 'role' for t in n.targets)]
    assert len(assignments) == 1
    role_expression = compile(ast.Expression(body=assignments[0].value), str(extractor), 'eval')

    def roles(row):
        return eval(role_expression, {'np': np},
                    {'positions': np.arange(len(row['tokens']), dtype=np.float32),
                     's': row['subject'], 'o': row['object']})

    def swap(row):
        return dict(row, subject=row['object'], object=row['subject'])

    cases = changed = involutions = mapping_equal = metadata_ignored = 0
    role_error = 0.
    # Exhaustive non-overlapping, nonempty ordered half-open spans, lengths 2..12.
    for n in range(2, 13):
        spans = [(a, b) for a in range(n) for b in range(a + 1, n + 1)]
        for s in spans:
            for o in spans:
                if not (s[1] <= o[0] or o[1] <= s[0]):
                    continue
                row = {'tokens': ['w' + str(i) for i in range(n)],
                       'subject': s, 'object': o, 'label': 'SYNTHETIC_UNUSED',
                       'metadata': {'fixture': True}}
                words, mapping = marked_words(row)
                reverse_words, reverse_mapping = marked_words(swap(row))
                cases += 1
                changed += words != reverse_words
                mapping_equal += mapping == reverse_mapping
                assert [w for w, index in zip(words, mapping) if index is not None] == row['tokens']
                assert [index for index in mapping if index is not None] == list(range(n))
                involutions += marked_words(swap(swap(row))) == (words, mapping)
                metadata_ignored += marked_words(dict(row, label='OTHER_UNUSED', metadata={})) == (words, mapping)
                role_error = max(role_error, float(np.max(np.abs(roles(swap(row)) - roles(row)[:, [1, 0, 3, 2]]))))
    assert cases > 0 and changed == mapping_equal == involutions == metadata_ignored == cases
    assert role_error == 0

    old = json.loads(args.historical.read_text(encoding='utf8'))
    assert old['totals']['unique_unordered_span_pairs'] == old['totals']['paired_row_units'] == 1242
    assert old['distinct_sentences'] == 837
    assert old['cohorts'] == {'inverse_gold_consistent': 1056, 'unmapped': 141, 'inverse_gold_inconsistent': 45}
    assert sum(old['cohorts'].values()) == 1242
    by_relation = old['by_relation']
    directed = [by_relation['per:children <-> per:parents'], by_relation['org:member_of <-> org:members']]
    assert [v['rows'] for v in directed] == [27, 6]
    assert sum(v['rows'] for v in by_relation.values()) == 1056
    assert by_relation['per:identity <-> per:identity']['rows'] == 878
    assert all(sum(v['performance'][m].values()) == v['rows']
               for v in by_relation.values() for m in ['quantum', 'dephased', 'real12', 'softmax12'])
    assert all(old[k] == 0 for k in ['model_forwards', 'training_updates', 'new_valid', 'new_test', 'cos_writes'])
    pins = json.loads(args.source_pins.read_text(encoding='utf8'))
    assert pins['aggregate_matches'] is True
    assert pins['source_files'] == 31
    assert all(v['matches_current_component'] for v in pins['components'].values())
    result = dict(
        status='PASS_BOUNDED_AUDIT',
        synthetic_source_port=dict(cases=cases, marked_encoder_inputs_changed=changed,
            original_word_mappings_equal=mapping_equal, involutions=involutions,
            label_metadata_ignored=metadata_ignored, role_permutation_max_error=role_error),
        historical_reuse=dict(source_sha256=digest(args.historical), new_census_executed=False,
            pairs=1242, sentences=837, mapped_consistent=1056, unmapped=141, mapped_inconsistent=45,
            identity_pairs=878, directed_pairs=33, directed_fraction_all_pairs=33/1242,
            directed_fraction_consistent=33/1056, cohort_selected_using_observed_labels=True,
            label_mapping_semantically_recertified=False),
        provenance=dict(source_pins_sha256=digest(args.source_pins),
            component_matches_declared_git_revision=True,
            manifest_schema_matched_source_aggregate_verified=True, source_contract_file_count=31),
        decision='STOP_ROLE_EXCHANGE_SHORTCUT_BEFORE_MODEL_DESIGN',
        scope=['Changed marked words do not prove numerically unequal encoder hidden states.',
               'Synthetic tests do not validate a frozen encoder/cache transformation.',
               'Historical label-consistent pairs are descriptive, not fresh task qualification.',
               'An earlier 22-file aggregate comparison used the wrong schema; the correct 31-file contract matches.',
               'A universal label involution and quantum-specific prediction remain unestablished.'],
        encoder_forwards=0, task_head_forwards=0, training_updates=0,
        gpu_allocations=0, new_dataset_valid_test_access=0, cos_writes=0,
        original_pdf_source_examples_used_in_candidate_design=False,
        newT=False, L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
