"""Exercise report Git delivery on a disposable local remote, without training."""
import argparse
import json
from pathlib import Path
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
        for key,field in (('user.name','%an'),('user.email','%ae')):
            command('git','config',key,p.git('show','-s','--format='+field,base),cwd=repo)
        def git(*argv):return command('git',*argv,cwd=repo)
        with patch.object(p,'ROOT',repo),patch.object(p,'git',git):
            if git('rev-parse','HEAD')!=base:raise ValueError('Disposable baseline differs')
            target=repo/'reports/q_rmea_token_transport'/root.name
            before_index=p.digest(repo/'.git/index')
            with patch.object(p,'review_gate',return_value={'verdict':'PASS'}) as gate:
                try:h.publish(root)
                except ValueError as error:
                    if 'live nonfixture canary' not in str(error):raise
                else:raise ValueError('Ordinary publisher admitted the development fixture')
                gate.assert_called_once_with(formal=True)
            if git('rev-parse','HEAD')!=base or p.digest(repo/'.git/index')!=before_index or target.exists() or git('status','--porcelain'):
                raise ValueError('Rejected fixture publication changed HEAD, index or report tree')
            # Exercise shared delivery mechanics on a private disposable origin.
            # This direct test export supplies no production admission evidence.
            h.export(root,target)
            h._commit_report_and_push(target);first=git('rev-parse','HEAD')
            h._commit_report_and_push(target);second=git('rev-parse','HEAD')
            remote=command('git','--git-dir',str(bare),'rev-parse','1.1')
            if first==base or second!=first or remote!=first or git('rev-list','--count',base+'..HEAD')!='1':
                raise ValueError('Fresh commit or idempotent export retry failed')
        # The real sealed fixture must be rejected by the formal nonfixture gate.
        # Override only the handoff read; do not copy or relabel sealed evidence.
        actual_read=p.read
        formal_handoff=dict(git_revision=base,canary_only=False)
        def handoff_read(path):
            return formal_handoff if Path(path)==root/'handoff.json' else actual_read(path)
        with patch.object(p,'read',side_effect=handoff_read):
            try:h.export(root,temp/'fixture-formal-report')
            except ValueError as error:
                if 'nonfixture' not in str(error):raise
            else:raise ValueError('Formal export admitted the development fixture')
        if (temp/'fixture-formal-report').exists():raise ValueError('Rejected fixture export created a report')
        # Independently reach the second audit with an explicitly synthetic audit
        # boundary double. This checks control flow, not live source readiness.
        substituted=temp/'substituted'
        substituted.mkdir()
        p.atomic_json(substituted/'handoff.json',dict(git_revision=base,canary_only=False))
        audit_double=dict(config=dict(profile='canary'),git_revision=base,environment=dict(audit_test_double=True))
        with patch.object(p,'audit',side_effect=[(audit_double,{}),(audit_double,{})]) as audits:
            try:h.export(substituted,temp/'forged-report')
            except ValueError as error:
                if 'phase/revision' not in str(error):raise
            else:raise ValueError('Discovery profile substitution accepted')
            if [c.args[0] for c in audits.call_args_list]!=[substituted/'canary',substituted/'discovery']:
                raise ValueError('Discovery profile regression did not reach the second audit')
        if (temp/'forged-report').exists():raise ValueError('Rejected substitution created a report')
        # A bad publication ID is rejected before even probing or launching.
        ns=argparse.Namespace(run_dir=str(temp/'invalid-id'),development_canary=False)
        with patch.object(h,'isolated_probe',side_effect=AssertionError('probe reached before ID validation')):
            try:h.run(ns)
            except ValueError as error:
                if 'basename before training' not in str(error):raise
            else:raise ValueError('Invalid run basename accepted')
        print(json.dumps(dict(status='PASS',fresh_local_report_commit=True,idempotent_same_commit=True,
            ordinary_fixture_publication_rejected_without_mutation=True,formal_fixture_export_rejected=True,
            discovery_profile_mismatch_rejected_at_second_audit=True,audit_double_is_live_evidence=False,
            bad_basename_rejected_before_probe=True,
            training_calls=0,external_GitHub_writes=0,source_run_complete_sha256=p.digest(root/'canary/RUN_COMPLETE'))))


if __name__=='__main__':
    main()
