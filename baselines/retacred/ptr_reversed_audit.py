"""Offline PTR Reversed task/provenance audit, not an implemented adapter."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from .assets import reject_symlinks
from .common import PACKAGE, digest, git, pin_source


def evidence_lock() -> dict:
    return json.loads((PACKAGE / 'ptr_reversed_evidence.lock.json').read_text())


def audit(source: Path) -> dict:
    lock = evidence_lock()
    for name, expected in lock['active_file_sha256'].items():
        path = source / name
        reject_symlinks(path)
        if digest(path) != expected:
            raise ValueError('PTR pinned file drift: ' + name)
    if (git('rev-parse', 'HEAD', cwd=source) != lock['active_commit']
            or git('status', '--porcelain', '--untracked-files=no', cwd=source)):
        raise ValueError('PTR source commit/cleanliness mismatch')
    if git('remote', 'get-url', 'origin', cwd=source).removesuffix('.git') != lock['repository']:
        raise ValueError('PTR source origin mismatch')
    tree = ast.parse((source / 'src/data_prompt.py').read_text())
    dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'REPromptDataset')
    tokenize = next(node for node in dataset.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'tokenize')
    assignments = {ast.unparse(node.targets[0]): ast.unparse(node.value)
                   for node in tokenize.body if isinstance(node, ast.Assign)}
    # Inspect the pinned active path; never infer a reversed method from names.
    if assignments['pos_head'] != "item['h']" or assignments['pos_tail'] != "item['t']":
        raise ValueError('PTR normal entity-order anchor changed')
    if any(isinstance(node, ast.If) for node in ast.walk(tokenize)):
        raise ValueError('PTR conditional entity-order path needs review')
    return {
        'schema': 'ptr-reversed-readiness.v1', 'baseline_id': 'ptr_reversed',
        'status': 'not_verifiable', 'source_commit': lock['active_commit'],
        'active_entity_order': ['h', 't'],
        'active_source_variant': 'normal_ptr_not_reversed',
        'published_setting': 'partially_reoriented_relation_task',
        'author_evidence': lock['author_evidence'],
        'paper': lock['paper'], 'reported_f1': .919,
        'reported_score_rank_eligible': False, 'benchmark_comparable': False,
        'strict_comparison_eligible': False,
        'target_input_contract': 'canonical_input_must_not_depend_on_gold_relation',
        'blockers': [
            'author_marks_reversed_setting_non_comparable_with_conventional_task',
            'complete_reversed_relation_subset_and_inverse_label_map_not_recovered',
            'reversed_verbalizers_and_original_transform_entrypoint_not_recovered',
            'paper_dev_best_and_10_percent_warmup_vs_source_final_and_500_steps',
        ],
        'source_modified': False, 'upstream_imported': False,
        'model_jobs_launched': 0, 'normal_ptr_substituted': False,
    }


def main() -> None:
    print(json.dumps(audit(pin_source('ptr', fetch=False)), indent=2))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
