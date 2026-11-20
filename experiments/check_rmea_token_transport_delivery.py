"""Exercise report Git delivery on a disposable local remote, without training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_token_transport_handoff as h
from baselines.retacred import rmea_token_transport_protocol as p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    manifest,_=p.audit(root/'canary');base=manifest['git_revision']
    original=p.ROOT
    if p.read(root/'handoff.json')['canary_only'] is not True:
        raise ValueError('Delivery regression is technical canary-only')
    with tempfile.TemporaryDirectory(prefix='transport-delivery-',dir=root.parent) as temporary:
        temp=Path(temporary);bare=temp/'origin.git';repo=temp/'checkout'
        def command(*argv,cwd=None):
            return subprocess.check_output(list(argv),cwd=cwd,text=True,stderr=subprocess.PIPE).strip()
        command('git','clone','--bare','--shared',str(original),str(bare))
        for branch in ('main','1.1'):
            command('git','--git-dir',str(bare),'update-ref','refs/heads/'+branch,base)
        command('git','--git-dir',str(bare),'symbolic-ref','HEAD','refs/heads/1.1')
        command('git','clone','--shared',str(bare),str(repo))
        for key in ('user.name','user.email'):
            command('git','config',key,p.git('config',key),cwd=repo)
        def git(*argv):return command('git',*argv,cwd=repo)
        with patch.object(p,'ROOT',repo),patch.object(p,'git',git):
            if git('rev-parse','HEAD')!=base:raise ValueError('Disposable baseline differs')
            # Both publish paths are real, including commit/blob verification and
            # ordinary push. origin is this private local bare repo, never GitHub.
            h.publish(root);first=git('rev-parse','HEAD')
            h.publish(root);second=git('rev-parse','HEAD')
            remote=command('git','--git-dir',str(bare),'rev-parse','1.1')
            if first==base or second!=first or remote!=first or git('rev-list','--count',base+'..HEAD')!='1':
                raise ValueError('Fresh commit or idempotent export retry failed')
        # A sealed canary substituted for discovery must fail before a report is
        # created. Preserve the real source; all copied bytes live in this scope.
        substituted=temp/'substituted'
        for phase in ('canary','discovery'):
            shutil.copytree(root/'canary',substituted/phase)
        p.atomic_json(substituted/'handoff.json',dict(git_revision=base,canary_only=False))
        try:
            h.export(substituted,temp/'forged-report')
        except ValueError as error:
            if 'phase/revision' not in str(error) and 'nonfixture' not in str(error):raise
        else:
            raise ValueError('Sealed canary masqueraded as discovery')
        if (temp/'forged-report').exists():raise ValueError('Rejected substitution created a report')
        # A bad publication ID is rejected before even probing or launching.
        ns=argparse.Namespace(run_dir=str(temp/'invalid-id'),development_canary=False)
        with patch.object(h,'isolated_probe',side_effect=AssertionError('probe reached before ID validation')):
            try:h.run(ns)
            except ValueError as error:
                if 'basename before training' not in str(error):raise
            else:raise ValueError('Invalid run basename accepted')
        print(json.dumps(dict(status='PASS',fresh_local_report_commit=True,idempotent_same_commit=True,
            sealed_canary_substitution_rejected=True,bad_basename_rejected_before_probe=True,
            training_calls=0,external_GitHub_writes=0,source_run_complete_sha256=p.digest(root/'canary/RUN_COMPLETE'))))


if __name__=='__main__':
    main()
