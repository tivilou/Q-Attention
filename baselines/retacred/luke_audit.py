"""Offline audit of published LUKE variants; not a training adapter."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .assets import reject_symlinks
from .common import PACKAGE, cohort, digest, git, metrics, pin_source


def evidence_lock() -> dict:
    return json.loads((PACKAGE / 'luke_evidence.lock.json').read_text(encoding='utf-8'))


def validate_lock(lock: dict) -> None:
    collection = lock['collection']
    requests = collection['requests']
    if (lock['schema'] != 'luke-variants-evidence.v1'
            or len(requests) != collection['requests_used']
            or len(requests) > collection['max_requests']
            or collection['max_retries'] != 0 or collection['max_browser_escalations'] != 0):
        raise ValueError('LUKE evidence/collection contract drift')
    accepted = {}
    for item in requests:
        if item['status'] != 200:
            continue
        name = Path(item['file'])
        if (name.is_absolute() or len(name.parts) != 1 or name.name in ('', '.', '..')
                or '\\' in item['file'] or item['file'] in accepted):
            raise ValueError('unsafe or duplicate evidence filename')
        accepted[item['file']] = item
    for name, source in lock['cfie_source']['selected_files'].items():
        item = accepted.get(source['file'])
        if item is None or item['sha256'] != source['sha256']:
            raise ValueError('CFIE selected-file evidence binding drift')
        if len(source['git_blob']) != 40 or any(c not in '0123456789abcdef' for c in source['git_blob']):
            raise ValueError('invalid CFIE Git blob hash')
        expected = f"https://raw.githubusercontent.com/HeyyyyyyG/CFIE/{lock['cfie_source']['commit']}/{name}"
        if item['url'] != expected:
            raise ValueError('CFIE pinned URL binding drift')


def verify_core(source: Path, lock: dict) -> ast.Module:
    spec = lock['core_source']
    for name, expected in spec['files'].items():
        path = source / name
        reject_symlinks(path)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('CoRE pinned file drift: ' + name)
    if (git('rev-parse', 'HEAD', cwd=source) != spec['commit']
            or git('status', '--porcelain', '--untracked-files=no', cwd=source)):
        raise ValueError('CoRE commit/cleanliness mismatch')
    if git('remote', 'get-url', 'origin', cwd=source).removesuffix('.git') != spec['repository']:
        raise ValueError('CoRE source origin mismatch')
    return ast.parse((source / 'core.py').read_text(encoding='utf-8'))


class _Vector:
    """Six-value fixture only: no numpy/Torch import and no tensor-model claim."""
    def __init__(self, values):
        self.values = tuple(values)

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        values = other.values if isinstance(other, _Vector) else (other,) * len(self.values)
        return _Vector(a == b for a, b in zip(self.values, values))

    def __and__(self, other):
        return _Vector(a and b for a, b in zip(self.values, other.values))

    def astype(self, _dtype):
        return self

    def sum(self):
        return sum(self.values)


class _FixtureNP:
    int32 = 'fixture_int32'


def source_slice_fixture(tree: ast.Module) -> dict:
    """Run only the already hash-checked scalar scorer and list comprehension.

    Full module imports/model/dataset loading are never executed. These fixed,
    reviewed AST slices demonstrate the released evaluation scope, not neural
    model fidelity or a project empirical score.
    """
    select = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'challenge_set' for t in node.targets))
    expected = '[i_ for i_ in range(len(luke_prob)) if luke_preds_tde[i_] != keys[i_]]'
    if ast.unparse(select) != expected:
        raise ValueError('CoRE challenge selection anchor changed')
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'getF1')
    # Whitelist the reviewed scalar function's calls before its bounded fixture.
    names = {'min', 'max', 'range', 'float', 'sum', 'len'}
    for node in ast.walk(function):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.With)):
            raise ValueError('unexpected executable scorer operation')
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in names
                or isinstance(node.func, ast.Attribute) and node.func.attr in {'astype', 'sum'}
                or isinstance(node.func, ast.Attribute) and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {'f1_ls', 'recall_ls', 'prec_ls'}):
            raise ValueError('unexpected scorer call')
    context = {'__builtins__': {name: getattr(__import__('builtins'), name) for name in names}, 'np': _FixtureNP}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<locked-CoRE-scorer-slice>', 'exec'), context)
    gold, predicted = [0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 1, 1]
    source_macro = context['getF1'](_Vector(gold), _Vector(predicted))
    common = metrics(gold, predicted, ['no_relation', 'fixture_a', 'fixture_b'], .5)
    absent_class = context['getF1'](_Vector([0, 1, 3]), _Vector([0, 1, 3]))
    baseline = {'luke_prob': [None] * 4, 'luke_preds_tde': [0, 1, 2, 1]}
    code = compile(ast.Expression(select), '<locked-CoRE-challenge-slice>', 'eval')
    restricted = {'__builtins__': {'range': range, 'len': len}}
    selected_a = eval(code, {**restricted, **baseline, 'keys': [0, 1, 1, 2]})
    selected_b = eval(code, {**restricted, **baseline, 'keys': [1, 1, 2, 1]})
    return {'fixture_only': True, 'source_macro_f1': source_macro,
            'benchmark_non_null_micro_f1': common['micro_f1'],
            'source_absent_class_fixture_f1': absent_class,
            'challenge_indices_gold_a': selected_a, 'challenge_indices_gold_b': selected_b,
            'challenge_denominator_depends_on_gold': selected_a != selected_b,
            'upstream_ast_slices_executed': ['getF1', 'challenge_set'],
            'full_upstream_module_imported': False}


def verify_cache(cache: Path, lock: dict) -> dict:
    validate_lock(lock)
    reject_symlinks(cache)
    if not cache.is_dir():
        raise ValueError('public evidence cache directory missing')
    accepted = [r for r in lock['collection']['requests'] if r['status'] == 200]
    verified = []
    for item in accepted:
        name = Path(item['file'])
        if name.is_absolute() or len(name.parts) != 1 or '..' in name.parts or '\\' in item['file']:
            raise ValueError('unsafe evidence filename')
        path = cache / name
        reject_symlinks(path)
        if (not path.is_file() or path.stat().st_size != item['size_bytes']
                or digest(path) != item['sha256']):
            raise ValueError('public evidence hash drift: ' + item['file'])
        verified.append(name.name)
    for source in lock['cfie_source']['selected_files'].values():
        body = (cache / source['file']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if blob != source['git_blob']:
            raise ValueError('CFIE pinned Git blob mismatch')
    return {'verified': True, 'accepted_files': len(verified), 'network_requests': 0,
            'cfie_source_files_git_verified': len(lock['cfie_source']['selected_files'])}


def audit(source: Path | None = None, cache: Path | None = None) -> dict:
    lock = evidence_lock()
    validate_lock(lock)
    registry = {row['id']: row for row in cohort()['models']}
    for name, paper_row in lock['reported_luke_rows'].items():
        row = registry[name]
        if (row['status'] != 'blocked' or row['reported_f1'] != paper_row['micro_f1']
                or row['reported_macro_f1'] != paper_row['macro_f1']
                or row['strict_comparison_eligible'] is not False
                or row['reported_score_rank_eligible'] is not False):
            raise ValueError('LUKE registry/evidence drift: ' + name)
    source = pin_source('core', fetch=False) if source is None else source
    tree = verify_core(source, lock)
    fixture = source_slice_fixture(tree)
    checked = verify_cache(cache, lock) if cache is not None else {'verified': False, 'network_requests': 0}
    return {'schema': 'luke-variants-readiness.v1', 'status': 'not_verifiable',
            'reported_luke_rows': lock['reported_luke_rows'],
            'paper_aggregation': 'median_five_training_runs_not_mean_or_single_seed',
            'released_core_scope': 'TACRED_checkpoint_and_gold_conditioned_challenge_macro',
            'source_slice_fixture': fixture, 'original_cfie_repository_identity_verified': True,
            'cfie_source_recovery': 'partial_selected_files_not_full_importability',
            'cfie_repository': lock['cfie_source']['repository'],
            'cfie_branch': 'RE', 'cfie_commit': lock['cfie_source']['commit'],
            'luke_cfie_retacred_port_verified': False,
            'luke_official_discovery': lock['luke_official_discovery'],
            'cache_validation': checked, 'blockers': lock['blockers'],
            'strict_comparison_eligible': False, 'model_jobs_launched': 0,
            'adapter_implemented': False, 'checkpoint_loaded': False,
            'precomputed_dataset_or_pickle_loaded': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-cache', type=Path, help='verify existing public evidence; no download')
    args = parser.parse_args()
    try:
        result = audit(cache=args.verify_cache)
    except (ValueError, KeyError, StopIteration, OSError) as error:
        print(json.dumps({'status': 'invalid_evidence', 'error': str(error), 'model_jobs_launched': 0}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
