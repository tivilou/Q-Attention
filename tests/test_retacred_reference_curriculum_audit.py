from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import curriculum_audit as curriculum
from baselines.retacred.common import ROOT, cohort, select_models


def test_paper_mechanism_is_not_marker_only_or_ordinary_oof():
    facts = curriculum.evidence_lock()['paper_facts']
    assert facts['forward_components'] == [
        'roberta_large', 'punctuation_entity_encoding', 'dependency_graph_gat',
        'concatenate_graph_and_entity_encodings', 'bilinear_classifier']
    assert facts['reviewer_training'] == 'each_model_on_its_own_disjoint_subset'
    assert facts['reviewer_evaluation'] == 'other_subsets_only'
    assert facts['difficulty_signal'] == 'reviewer_prediction_correctness'


def test_literal_recipe_and_material_unknowns_are_preserved():
    lock = curriculum.evidence_lock()
    assert lock['paper_facts']['reported_recipe'] == {
        'hidden_size': 1024, 'dropout': .1, 'learning_rate': .0005,
        'layer_decay': .7, 'optimizer': 'AdamW'}
    assert {'gat_and_bilinear_equations', 'reviewer_count_index_ambiguity',
            'difficulty_aggregation_and_pacing', 'layerwise_lr_group_mapping',
            'loss_batch_epoch_seed_checkpoint_recipe', 'evaluator_binding'} <= set(lock['unresolved'])


def test_gain_is_against_papers_own_gat_baseline():
    result = curriculum.audit()
    assert result['paper']['reported']['own_baseline']['f1'] == .912
    assert result['reported_gain_over_own_baseline'] == .002
    assert result['reported_gain_percentage_points'] == .2
    assert result['model_jobs_launched'] == 0
    assert not result['model_implemented'] and not result['marker_only_substituted']
    assert not result['cache_validation']['verified']


def test_candidate_handle_is_not_promoted_to_official_source():
    recovery = curriculum.evidence_lock()['source_recovery']
    assert recovery['official_executable_source_verified'] is False
    assert recovery['candidate_account_author_identity'] == 'unverified_handle_match_only'
    assert recovery['candidate_repository_count'] == 19
    assert recovery['no_global_absence_claim'] is True
    assert len(recovery['inspected_candidate_trees']) == 2


def test_registry_blocks_unverified_recipe_but_preserves_selection_row():
    row = next(row for row in cohort()['models'] if row['id'] == 'curriculum')
    assert row['rank'] == 6 and row['reported_f1'] == .914
    assert row['reported_own_baseline_f1'] == .912
    assert row['benchmark_comparability'] == 'unverified_evaluator_binding'
    with pytest.raises(ValueError, match='no jobs were launched'):
        select_models('curriculum')
    with pytest.raises(ValueError, match='no jobs were launched'):
        select_models('all')
    assert len(select_models('ready')) == 5
    legacy = json.loads((ROOT / 'configs/retacred_reference_baselines.json').read_text())
    legacy_row = next(row for row in legacy['candidates'] if row['method_family'] == 'curriculum_learning')
    assert legacy_row['status'] == 'blocked_source_and_protocol'
    assert legacy_row['reported_score_rank_eligible'] is False
    assert legacy_row['reported_in_paper']['metric_convention_verified'] is False
    assert 'reviewer_membership_and_schedule_contract' in legacy_row['required_gates']


def test_actual_offline_cli_is_stdlib_only():
    code = ('import runpy,sys\n'
            'try: runpy.run_module("baselines.retacred.curriculum_audit",run_name="__main__")\n'
            'finally: assert "torch" not in sys.modules\n')
    process = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 2, process.stderr
    assert json.loads(process.stdout)['status'] == 'not_verifiable'


def test_actual_collected_public_evidence_hashes():
    # Optional network-free integration: local tests do not need private cache.
    cache = ROOT / '.external/retacred/curriculum-recovery-20261007'
    if not cache.is_dir():
        pytest.skip('public collection cache not present in this checkout')
    result = curriculum.audit(cache)
    assert result['cache_validation']['verified']
    assert len(result['cache_validation']['files']) == 15
    assert result['cache_validation']['network_requests'] == 0


def test_tampered_or_missing_cache_is_not_evidence(tmp_path):
    lock = curriculum.evidence_lock()
    first = next(item for item in lock['bounded_collection']['requests'] if item['status'] == 200)
    (tmp_path / first['file']).write_bytes(b'wrong bytes')
    with pytest.raises(ValueError, match='hash drift'):
        curriculum.verify_cache(tmp_path, lock)
    with pytest.raises(ValueError, match='directory missing'):
        curriculum.verify_cache(tmp_path / 'absent', lock)


def test_symlink_cache_is_not_read(tmp_path):
    link = tmp_path / 'cache-link'
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        curriculum.verify_cache(link, curriculum.evidence_lock())


@pytest.mark.parametrize('name', ['../paper.pdf', '/tmp/paper.pdf', 'folder/paper.pdf', 'folder\\paper.pdf'])
def test_unsafe_evidence_path_rejected(name):
    lock = deepcopy(curriculum.evidence_lock())
    next(item for item in lock['bounded_collection']['requests'] if item['status'] == 200)['file'] = name
    with pytest.raises(ValueError, match='filename'):
        curriculum.validate_lock(lock)


def test_invalid_cache_cli_fails_without_starting_model(tmp_path):
    process = subprocess.run([sys.executable, '-m', 'baselines.retacred.curriculum_audit',
                              '--verify-cache', str(tmp_path)], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert result['status'] == 'invalid_evidence' and result['model_jobs_launched'] == 0


def test_paper_and_registry_drift_stop_before_interpretation(monkeypatch):
    lock = deepcopy(curriculum.evidence_lock())
    lock['paper']['pdf_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='binding drift'):
        curriculum.validate_lock(lock)
    registry = deepcopy(cohort())
    next(row for row in registry['models'] if row['id'] == 'curriculum')['status'] = 'implemented_runtime_gate_required'
    monkeypatch.setattr(curriculum, 'cohort', lambda: registry)
    with pytest.raises(ValueError, match='registry/evidence drift'):
        curriculum.audit()
