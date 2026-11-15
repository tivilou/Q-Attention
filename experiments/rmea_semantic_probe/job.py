"""Trusted synchronous scheduler entry; one immutable private forward attempt."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import worker

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();worker.require(root.is_dir() and not any(root.iterdir()),'Empty scheduler output required')
    repo=Path(__file__).resolve().parents[2]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    contract=json.loads((Path(__file__).parent/'contract.json').read_bytes())
    inputs=Path(contract['private_input_directory'])
    args=SimpleNamespace(inputs=inputs/'licensed-inputs.json',annotations=inputs/'probe-annotations.json',run_dir=root/'private',source_revision=revision)
    worker.run(args)
    closed=json.loads((args.run_dir/'RUN_COMPLETE').read_bytes())
    worker.require(closed['schema']=='qattn.semantic-forward.closed.v1','Private closure differs')
    for name,digest in closed['files'].items():worker.require(worker.sha(args.run_dir/name)==digest,'Private closed file changed')
    shutil.copyfile(args.run_dir/'compact-report.json',root/'compact-report.json')
    worker.save(root/'RUN_COMPLETE',dict(schema='qattn.semantic-forward.scheduler-closed.v1',source_revision=revision,
        compact_report_sha256=worker.sha(root/'compact-report.json'),private_closure_sha256=worker.sha(args.run_dir/'RUN_COMPLETE'),
        private_files=len(closed['files']),formal_runs=0,training=False,test_loaded=False))
if __name__=='__main__':main()
