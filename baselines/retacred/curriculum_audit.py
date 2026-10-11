"""Read-only Curriculum paper/provenance audit; this is not a model adapter."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from urllib.parse import urlparse

from .assets import reject_symlinks
from .common import PACKAGE, cohort, digest


def evidence_lock() -> dict:
    return json.loads((PACKAGE / 'curriculum_evidence.lock.json').read_text(encoding='utf-8'))


def validate_lock(lock: dict) -> None:
    if (lock['schema'] != 'curriculum-paper-evidence.v1'
            or lock['paper']['version'] != 'v2'
            or lock['model_implemented'] is not False
            or lock['strict_comparison_eligible'] is not False):
        raise ValueError('Curriculum evidence/admission contract drift')
    collection = lock['bounded_collection']
    requests = collection['requests']
    if (len(requests) != collection['requests_used']
            or len(requests) > collection['max_requests']
            or collection['max_retries'] != 0
            or collection['max_browser_escalations'] != 0):
        raise ValueError('collection budget contract drift')
    names = set()
    for item in requests:
        parsed = urlparse(item['url'])
        if parsed.scheme != 'https' or parsed.hostname not in collection['allowed_origins']:
            raise ValueError('evidence origin outside declared collection')
        if item['status'] != 200:
            continue
        name = Path(item['file'])
        if (name.is_absolute() or len(name.parts) != 1 or name.name in ('', '.', '..')
                or name.name in names or '\\' in item['file']):
            raise ValueError('unsafe or duplicate evidence filename')
        names.add(name.name)
        sha = item['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid evidence hash')
    pdf = next(item for item in requests if item.get('file') == 'curriculum-paper.pdf')
    if (pdf['url'] != lock['paper']['url']
            or pdf['sha256'] != lock['paper']['pdf_sha256']):
        raise ValueError('paper hash/URL binding drift')


def verify_cache(cache: Path, lock: dict) -> dict:
    """Verify only public cached evidence bytes, never import downloaded code."""
    validate_lock(lock)
    reject_symlinks(cache)
    if not cache.is_dir():
        raise ValueError('evidence cache directory missing')
    verified = []
    for item in lock['bounded_collection']['requests']:
        if item['status'] != 200:
            continue
        path = cache / item['file']
        reject_symlinks(path)
        if (not path.is_file() or path.stat().st_size != item['size_bytes']
                or digest(path) != item['sha256']):
            raise ValueError('evidence missing or hash drift: ' + item['file'])
        verified.append(item['file'])
    if (cache / 'curriculum-paper.pdf').read_bytes()[:5] != b'%PDF-':
        raise ValueError('locked paper is not a PDF')
    return {'verified': True, 'files': verified, 'network_requests': 0}


def audit(cache: Path | None = None) -> dict:
    lock = evidence_lock()
    validate_lock(lock)
    row = next(row for row in cohort()['models'] if row['id'] == 'curriculum')
    if (row['status'] != 'blocked' or row['reported_f1'] != lock['paper']['reported']['curriculum']['f1']
            or row['strict_comparison_eligible'] is not False
            or row['reported_score_rank_eligible'] is not False):
        raise ValueError('Curriculum registry/evidence drift')
    reported = lock['paper']['reported']
    delta = Decimal(str(reported['curriculum']['f1'])) - Decimal(str(reported['own_baseline']['f1']))
    verified = verify_cache(cache, lock) if cache is not None else {'verified': False, 'network_requests': 0}
    return {
        'schema': 'curriculum-readiness.v1', 'baseline_id': 'curriculum',
        'status': 'not_verifiable', 'available_provenance': 'paper',
        'paper': lock['paper'], 'paper_facts': lock['paper_facts'],
        'reported_gain_over_own_baseline': float(delta),
        'reported_gain_percentage_points': float(delta * 100),
        'benchmark_comparability': 'unverified_evaluator_binding',
        'reported_score_rank_eligible': False, 'strict_comparison_eligible': False,
        'source_recovery': lock['source_recovery'],
        'blockers': lock['unresolved'], 'cache_validation': verified,
        'model_implemented': False, 'model_jobs_launched': 0,
        'upstream_imported': False, 'marker_only_substituted': False,
        'review_scope': 'human_audited_paper_and_bounded_public_entries_not_runtime_fidelity',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-cache', type=Path, help='optional public evidence cache; never downloaded here')
    args = parser.parse_args()
    try:
        result = audit(args.verify_cache)
    except (ValueError, KeyError, StopIteration, OSError) as error:
        print(json.dumps({'status': 'invalid_evidence', 'error': str(error), 'model_jobs_launched': 0}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
