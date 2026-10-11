from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys

import pytest

from baselines.retacred import luke_audit as luke
from baselines.retacred.common import ROOT, cohort, digest, pin_source, select_models


@pytest.fixture(scope='module')
def source():
    return pin_source('core', fetch=False)


def test_original_cfie_is_author_linked_re_branch_not_ner_or_luke():
    lock = luke.evidence_lock()
    assert lock['cfie_source']['branch'] == 'RE'
    assert lock['cfie_source']['default_branch'] == 'NER'
    assert lock['cfie_source']['commit'] == '17583143a4d70eb4c12e07461620b24dfe376198'
    assert lock['cfie_source']['root_license'] == 'Apache-2.0'
    assert lock['cfie_source']['author_link_binding'] == 'original_paper_section1_exact_repository_url'
    assert lock['cfie_source']['luke_retacred_port_verified'] is False
    assert len(lock['cfie_source']['selected_files']) == 12


def test_reported_micro_macro_and_five_run_median_are_separate():
    lock = luke.evidence_lock()
    assert lock['core_paper']['aggregation'] == 'median_five_training_runs'
    assert lock['reported_luke_rows']['luke_cfie'] == {'micro_f1': .905, 'macro_f1': .804}
    assert lock['reported_luke_rows']['luke_core'] == {'micro_f1': .909, 'macro_f1': .816}
    assert lock['reported_luke_rows']['luke_focal'] == {'micro_f1': .904, 'macro_f1': .803}
    assert lock['reported_luke_rows']['luke_resample'] == {'micro_f1': .905, 'macro_f1': .805}


def test_actual_source_slices_expose_denominator_and_metric_difference(source):
    before = {name: digest(source/name) for name in luke.evidence_lock()['core_source']['files']}
    result = luke.audit(source)
    fixture = result['source_slice_fixture']
    assert fixture['source_macro_f1'] == pytest.approx(1/3)
    assert fixture['benchmark_non_null_micro_f1'] == pytest.approx(4/7)
    assert fixture['source_absent_class_fixture_f1'] == 1
    assert fixture['challenge_indices_gold_a'] == [2, 3]
    assert fixture['challenge_indices_gold_b'] == [0]
    assert fixture['challenge_denominator_depends_on_gold'] is True
    assert not result['precomputed_dataset_or_pickle_loaded'] and not result['checkpoint_loaded']
    assert before == {name: digest(source/name) for name in before}


def test_registry_keeps_four_historical_rows_outside_admission():
    rows = {row['id']: row for row in cohort()['models']}
    for name, expected in luke.evidence_lock()['reported_luke_rows'].items():
        assert rows[name]['reported_f1'] == expected['micro_f1']
        assert rows[name]['reported_macro_f1'] == expected['macro_f1']
        assert rows[name]['reported_aggregation'] == 'median_five_training_runs'
        assert not rows[name]['strict_comparison_eligible']
        with pytest.raises(ValueError, match='no jobs were launched'):
            select_models(name)
    assert len(select_models('ready')) == 5


def test_actual_offline_cli_does_not_import_torch_or_numpy(source):
    code = ('import runpy,sys\n'
            'try: runpy.run_module("baselines.retacred.luke_audit",run_name="__main__")\n'
            'finally: assert "torch" not in sys.modules and "numpy" not in sys.modules\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2, result.stderr
    audit = json.loads(result.stdout)
    assert audit['model_jobs_launched'] == 0 and audit['adapter_implemented'] is False


def test_actual_public_cache_and_twelve_source_blob_hashes(source):
    cache = ROOT / '.external/retacred/luke-recovery-20261007'
    if not cache.is_dir():
        pytest.skip('public source collection cache not in this checkout')
    result = luke.audit(source, cache)
    assert result['cache_validation'] == {'verified': True, 'accepted_files': 22,
                                         'network_requests': 0, 'cfie_source_files_git_verified': 12}


def test_official_luke_foundation_does_not_certify_core_authors_recipe():
    info = luke.evidence_lock()['luke_official_discovery']
    assert info['commit'] == '251654548a15b9a8ddc708a53b715bdf3a112102'
    assert info['training_stack'] == 'AllenNLP_entity_aware_attention'
    assert info['base_recipe']['physical_batch'] == 4
    assert info['base_recipe']['accumulation'] == 8
    assert info['core_authors_retacred_recipe_verified'] is False


def test_tls_failures_are_retained_not_retried_or_converted_to_absence():
    lock = luke.evidence_lock()
    assert len(lock['collection']['requests']) == 24
    failed = [r for r in lock['collection']['requests'] if r['status'] != 200]
    assert len(failed) == 2
    assert {r['url'].split('/')[-1] for r in failed} == {'eval.py', 'loss.py'}
    assert lock['collection']['max_retries'] == 0


def test_wrong_core_source_fails_before_interpretation(tmp_path):
    (tmp_path / 'core.py').write_text('# other source\n')
    with pytest.raises(ValueError, match='pinned file drift'):
        luke.audit(tmp_path)


def test_wrong_origin_and_dirty_core_source_stop(source, monkeypatch):
    real_git = luke.git
    monkeypatch.setattr(luke, 'git', lambda *args, **kwargs:
        'https://example.invalid/CoRE.git' if args[0] == 'remote' else real_git(*args, **kwargs))
    with pytest.raises(ValueError, match='origin mismatch'):
        luke.audit(source)
    monkeypatch.setattr(luke, 'git', lambda *args, **kwargs:
        ' M core.py' if args[0] == 'status' else real_git(*args, **kwargs))
    with pytest.raises(ValueError, match='cleanliness mismatch'):
        luke.audit(source)


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'folder/file', 'folder\\file'])
def test_unsafe_evidence_path_never_read(name):
    lock = deepcopy(luke.evidence_lock())
    next(r for r in lock['collection']['requests'] if r['status'] == 200)['file'] = name
    with pytest.raises(ValueError, match='filename'):
        luke.validate_lock(lock)


def test_missing_cache_and_unbound_source_file_fail(tmp_path):
    lock = luke.evidence_lock()
    with pytest.raises(ValueError, match='hash drift'):
        luke.verify_cache(tmp_path, lock)
    tampered = deepcopy(lock)
    next(iter(tampered['cfie_source']['selected_files'].values()))['file'] = 'absent.py'
    with pytest.raises(ValueError, match='binding drift'):
        luke.validate_lock(tampered)


def test_registry_drift_is_not_promoted(source, monkeypatch):
    registry = deepcopy(cohort())
    next(row for row in registry['models'] if row['id'] == 'luke_cfie')['reported_macro_f1'] = .905
    monkeypatch.setattr(luke, 'cohort', lambda: registry)
    with pytest.raises(ValueError, match='registry/evidence drift'):
        luke.audit(source)
