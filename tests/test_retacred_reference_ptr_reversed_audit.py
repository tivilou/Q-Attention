from __future__ import annotations

import json
import subprocess
import sys

import pytest

from baselines.retacred import ptr_reversed_audit as reversed_audit
from baselines.retacred.common import ROOT, cohort, digest, pin_source, select_models


@pytest.fixture(scope='module')
def source():
    return pin_source('ptr', fetch=False)


def test_member_evidence_preserves_subset_uncertainty_and_non_comparability():
    lock = reversed_audit.evidence_lock()
    member = lock['author_evidence']
    assert member['author_association'] == 'MEMBER'
    assert member['main_reversed_relation'] == 'per:origin'
    assert member['complete_subset_verified'] is False
    assert member['conventional_setting_comparable'] is False
    assert member['comparability_comment'].endswith('#issuecomment-900011357')
    assert lock['paper']['reported_reversed_f1'] == .919


def test_actual_pinned_normal_path_stays_read_only_and_is_not_a_reversed_adapter(source):
    before = {name: digest(source/name) for name in reversed_audit.evidence_lock()['active_file_sha256']}
    result = reversed_audit.audit(source)
    assert result['active_entity_order'] == ['h', 't']
    assert result['active_source_variant'] == 'normal_ptr_not_reversed'
    assert result['benchmark_comparable'] is result['strict_comparison_eligible'] is False
    assert result['normal_ptr_substituted'] is False and result['model_jobs_launched'] == 0
    assert before == {name: digest(source/name) for name in before}


def test_actual_offline_cli_does_not_load_torch_or_launch_training(source):
    code = ('import runpy,sys\n'
            'try: runpy.run_module("baselines.retacred.ptr_reversed_audit",run_name="__main__")\n'
            'finally: assert "torch" not in sys.modules\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)['model_jobs_launched'] == 0


def test_registry_preserves_historical_row_but_blocks_rank_and_admission():
    row = next(row for row in cohort()['models'] if row['id'] == 'ptr_reversed')
    assert row['rank'] == 3 and row['reported_f1'] == .919
    assert row['reported_score_rank_eligible'] is row['benchmark_comparable'] is False
    assert row['comparison_role'] == 'changed_task_reference_only'
    with pytest.raises(ValueError, match='no jobs were launched'):
        select_models('ptr_reversed')
    assert [row['id'] for row in select_models('ready')] == [
        'typed_marker_punct', 'typed_marker', 'entity_marker_punct', 'entity_marker', 'ptr']


def test_wrong_source_stops_before_interpretation(tmp_path):
    first = next(iter(reversed_audit.evidence_lock()['active_file_sha256']))
    (tmp_path/first).parent.mkdir(parents=True)
    (tmp_path/first).write_text('# another source\n')
    with pytest.raises(ValueError, match='pinned file drift'):
        reversed_audit.audit(tmp_path)


def test_wrong_origin_and_dirty_source_fail(source, monkeypatch):
    real_git = reversed_audit.git
    def other_origin(*args, **kwargs):
        if args[:2] == ('remote', 'get-url'):
            return 'https://example.invalid/PTR.git'
        return real_git(*args, **kwargs)
    monkeypatch.setattr(reversed_audit, 'git', other_origin)
    with pytest.raises(ValueError, match='origin mismatch'):
        reversed_audit.audit(source)
    monkeypatch.setattr(reversed_audit, 'git', lambda *args, **kwargs:
        ' M src/data_prompt.py' if args[0] == 'status' else real_git(*args, **kwargs))
    with pytest.raises(ValueError, match='cleanliness mismatch'):
        reversed_audit.audit(source)


def test_selective_gold_based_reorientation_is_not_target_invariant():
    # Negative control only, not a production transform or a guessed original
    # recipe: identical country/person types cannot disambiguate the two labels.
    entities = ('fixture_person', 'fixture_country')
    orders = {label: entities[::-1] if label == 'per:origin' else entities
              for label in ('per:origin', 'per:countries_of_residence')}
    assert orders['per:origin'] != orders['per:countries_of_residence']
    assert reversed_audit.evidence_lock()['author_evidence']['complete_subset_verified'] is False
