"""Fixed class-centered entry; shared parent stays Torch-free."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import read, sha
from experiments.rmea_utility_measurement.train_job import main


def require_managed_canary(arguments, design, contract):
    required = design['synthetic_canary']
    proof = read(arguments.canary_acceptance)
    assert sha(arguments.canary_acceptance) == contract['synthetic_canary_acceptance_sha256']
    assert proof['schema'] == 'qattn.class-centered.canary.owner-acceptance.v1'
    assert proof['status'] == 'ACCEPTED_CLASS_CENTERED_SYNTHETIC_CANARY'
    assert proof['attempt_count'] == 1 and proof['returncode'] == 0 and proof['quiescent']
    assert proof['source_revision'] == required['source_revision']
    assert proof['contract_sha256'] == required['contract_sha256']
    assert proof['task'] == contract['synthetic_canary_task'] and proof['attempt'] == contract['synthetic_canary_attempt']
    report = proof['report']
    assert report['status'] == 'CLASS_CENTERED_NUMERICAL_CANARY_COMPLETE'
    assert report['check_count'] == len(report['checks']) == 8
    assert {row['name'] for row in report['checks']} == set(required['check_names'])
    assert all(row['status'] == 'PASS' for row in report['checks'])
    assert all(report[k] == 0 for k in ['optimizer_steps', 'real_dataset_reads', 'new_checkpoints', 'COS_requests', 'cloud_writes'])
    assert report['task_training'] is False and report['independent_confirmation'] is False
    assert proof['observer_model_forwards'] == 0 and proof['source_clean']
    assert report['job_sha256'] == required['job_sha256'] and report['design_sha256'] == required['design_sha256']
    assert report['L5'] == 'NOT_ACCEPTED' and not report['formal_handoff']


if __name__ == '__main__':
    main(design_path=ROOT / 'docs/evidence/rmea/20261119-l5-round103/executable-training-design.json',
         entrypoint=Path(__file__).resolve(), implementation_module='experiments.rmea_utility_measurement.class_centered_train',
         schema_prefix='qattn.class-centered.training', result_status='CLASS_CENTERED_JOINT_TRAINING_COMPLETE',
         extra_path_arguments=('canary_acceptance',), precheck=require_managed_canary)
