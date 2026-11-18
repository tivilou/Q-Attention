"""Only managed synthetic unit execution is exposed in this revision."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import complete,guard,read,sha


def main():
    guard();parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['unit'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    contract_path=Path(__file__).with_name('contract.json');c=read(contract_path)
    for n,h in {**c['dependencies'],**c['source_hashes'],**c['design_hashes']}.items():assert sha(ROOT/n)==h
    assert {n:importlib.metadata.version(n) for n in c['environment']}==c['environment']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and c['allowed_actions']==['unit']
    import torch
    from experiments.rmea_ecqp_screen import runner,metrics,synthetic_checks,prefix_reader_checks,recovery_lifecycle_checks
    from experiments.rmea_ecqp_recovery import restore
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    design=read(ROOT/'docs/evidence/rmea/20261118-l5-round72/screen-design.json')
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ecqp-screen-unit-') as directory:
        folder=Path(directory)
        metric=synthetic_checks.metric_checks(metrics)
        prefix=prefix_reader_checks.checks(runner,folder/'prefix-reader')
        training=synthetic_checks.training_checks(runner,design,metrics)
        recovery=recovery_lifecycle_checks.checks(restore,folder/'recovery-lifecycle')
    report=dict(status='PASS_ECQP_SCREEN_AND_RECOVERY_LIFECYCLE_SYNTHETIC_UNIT',
        metric_checks=metric,prefix_reader_checks=prefix,training_checks=training,recovery_lifecycle_checks=recovery,
        contract_sha256=sha(contract_path),environment=c['environment'],elapsed_seconds=time.monotonic()-started,
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        synthetic_prefix_shape=[151571,1024],synthetic_prefix_values_bytes=310417408,
        synthetic_optimizer_steps=36,synthetic_conditions=9,synthetic_epochs=4,
        actual_restore_run_invoked=True,restore_boundaries_mocked=['validate_plan','download','extract'],
        task_training=False,real_data_reads=0,COS_requests=0,utility='NOT_RUN',L5='NOT_ACCEPTED')
    complete(args.output,args.action,report)
    print(json.dumps(dict(status=report['status'],metric_checks=len(metric),prefix_reader_checks=len(prefix),
        training_checks=len(training),recovery_lifecycle_checks=len(recovery),real_data_reads=0,COS_requests=0)))


if __name__=='__main__':main()
