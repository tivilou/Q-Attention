"""Registered owner canary through the real parent; injected export and Git retry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_entity_study_protocol as p
from baselines.retacred import rmea_entity_handoff as h


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-archive',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() or output.is_relative_to(p.ROOT):raise ValueError('Fresh private canary output required')
    output.mkdir(parents=True)
    root=output/'20261031T000000Z_seed13';target=output/(root.name+'-safe-report')
    command=[sys.executable,'-B',str(p.ROOT/'experiments/q_rmea_entity_handoff.py')]
    fail=subprocess.run(command+['run','--run-dir',str(root),'--canary-only','--development-canary','--development-cache',str(args.cache_archive),'--inject-export-failure'],cwd=p.ROOT)
    if fail.returncode==0 or target.exists() or list(output.glob('.'+target.name+'-*')):raise ValueError('Expected clean injected export failure')
    p.audit(root/'canary')
    before={f.relative_to(root).as_posix():(p.digest(f),f.stat().st_mtime_ns) for f in root.rglob('*') if f.is_file()}
    for _ in range(2):subprocess.run(command+['export','--run-dir',str(root),'--development-canary'],cwd=p.ROOT,check=True)
    after={f.relative_to(root).as_posix():(p.digest(f),f.stat().st_mtime_ns) for f in root.rglob('*') if f.is_file()}
    if before!=after:raise ValueError('Export retrained or mutated immutable source')
    h.validate_report(target)
    import urllib.request
    from urllib.parse import urlencode
    reader=subprocess.Popen([sys.executable,'-B',str(p.ROOT/'scripts/view_rmea_entity_trace.py'),'--run-dir',str(root/'canary'),'--port','0'],stdout=subprocess.PIPE,text=True)
    try:
        url=reader.stdout.readline().strip().split()[-1]
        if not url.startswith('http://127.0.0.1:'):raise ValueError('Reader must bind loopback')
        query=urlencode(dict(sample='ordered:selected:valid:0',representation='subject_angles',coords='[[0,0,0,0]]'))
        with urllib.request.urlopen(url+'/api/value?'+query,timeout=20) as response:
            value=json.load(response)
        sample=next(s for s in p.read(root/'canary/sample-trace.json')['samples'] if s['sample_id']=='ordered:selected:valid:0')
        from baselines.retacred.rmea_entity_trace import read_value
        if value['values']!=read_value(root/'canary',sample,'subject_angles',[[0,0,0,0]]):raise ValueError('HTTP reader differs from complete values')
    finally:
        reader.terminate();reader.wait(timeout=20)
    # Exercise actual add/commit/push and second-publish idempotency with a local
    # bare origin. There is no permission to push this test to GitHub.
    bare=output/'origin.git';checkout=output/'checkout'
    subprocess.run(['git','clone','--bare',str(p.ROOT),str(bare)],check=True,capture_output=True)
    rev=p.git('rev-parse','HEAD')
    for ref in ('main','1.1'):subprocess.run(['git','--git-dir',str(bare),'update-ref','refs/heads/'+ref,rev],check=True)
    subprocess.run(['git','clone','--branch','1.1',str(bare),str(checkout)],check=True,capture_output=True)
    for key in ('user.name','user.email'):
        subprocess.run(['git','-C',str(checkout),'config',key,p.git('config',key)],check=True)
    original=p.ROOT
    try:
        p.ROOT=checkout
        h.publish(root);commit=p.git('rev-parse','HEAD');h.publish(root)
        if p.git('rev-parse','HEAD')!=commit or p.git('rev-list','--count',rev+'..HEAD')!='1':raise ValueError('Duplicate report commit')
        if p.git('ls-remote','origin','refs/heads/1.1').split()[0]!=commit:raise ValueError('Local push missing')
        if p.git('status','--porcelain'):raise ValueError('Dirty post-push checkout')
    finally:p.ROOT=original
    if before!={f.relative_to(root).as_posix():(p.digest(f),f.stat().st_mtime_ns) for f in root.rglob('*') if f.is_file()}:raise ValueError('Git retry changed run')
    p.atomic_json(output/'lifecycle-acceptance.json',dict(status='pass',parent_child='real',injected_failure='clean',export_retries=2,
        immutable_files=len(before),training_repeated=False,report_files=sorted(h.REPORT_FILES),actual_local_git_push='pass',second_publish_same_commit=True,
        private_http_reader='pass',revision=rev,source_sha256=p.identity(p.sources()),formal_runs=0))


if __name__=='__main__':main()
