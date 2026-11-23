"""Shared report-only Git delivery; never starts or resumes training."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def require(ok,message):
    if not ok:raise ValueError(message)


def git_bytes(p,*args):
    return subprocess.check_output(['git',*args],cwd=p.ROOT)


def ancestor(p,older,newer):
    return subprocess.run(['git','merge-base','--is-ancestor',older,newer],cwd=p.ROOT,
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def clean_merge_tree(p,left,right):
    """Compute a trivial three-way tree without touching the working index.

    Content conflicts require explicit review; report-only merges are disjoint.
    The temporary index also works with Git versions before merge-tree --write-tree.
    """
    bases=p.git('merge-base','--all',left,right).splitlines()
    require(len(bases)==1,'Ambiguous merge base requires review')
    with tempfile.TemporaryDirectory(prefix='q-rmea-merge-') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        subprocess.check_output(['git','read-tree','-i','-m',bases[0],left,right],cwd=p.ROOT,env=env)
        return subprocess.check_output(['git','write-tree'],cwd=p.ROOT,env=env).decode().strip()


def origin_gate(p):
    origin=p.git('remote','get-url','origin')
    allowed={p.REPOSITORY_ORIGIN}
    if p.REPOSITORY_ORIGIN=='https://github.com/tivilou/Q-Attention.git':
        allowed.add('git@github.com:tivilou/Q-Attention.git')
    require(origin in allowed,'Unexpected origin')
    require(p.git('remote','get-url','--push','--all','origin').splitlines()==[origin],
            'Unexpected push destination')
    if origin=='git@github.com:tivilou/Q-Attention.git':
        literal=origin.encode('ascii')
        require(git_bytes(p,'remote','get-url','origin')==literal+b'\n'
                and git_bytes(p,'remote','get-url','--push','--all','origin')==literal+b'\n',
                'SSH effective URL requires the exact literal repository')
        raw=git_bytes(p,'config','--null','--get-regexp',r'^remote\.origin\.(url|pushurl)$')
        require(raw.endswith(b'\0'),'Invalid configured origin framing')
        rows=[row.split(b'\n',1) for row in raw[:-1].split(b'\0')]
        require(all(len(row)==2 for row in rows),'Invalid configured origin')
        fetch=[value for name,value in rows if name==b'remote.origin.url']
        push=[value for name,value in rows if name==b'remote.origin.pushurl']
        require(fetch==[literal] and (not push or push==[literal]),
                'SSH origin requires the literal repository URL without aliases')
    return origin


def branch_gate(p,allowed):
    require(p.git('branch','--show-current')=='1.1','Report delivery requires branch 1.1')
    origin_gate(p)
    for marker in ('MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD'):
        path=Path(p.git('rev-parse','--git-path',marker))
        if not path.is_absolute():path=Path(p.ROOT)/path
        require(not path.exists(),'Finish existing Git operation before report publication')
    for ref in ('origin/main','origin/1.1'):
        require(ancestor(p,ref,'HEAD'),'Merge '+ref+' before delivery; do not retrain')
    rows=git_bytes(p,'status','--porcelain=v1','-z','--untracked-files=all').decode().split('\0')
    rows=[row for row in rows if row]
    require(all(row[3:] in allowed for row in rows),'Changes outside this report must be preserved separately')
    require(set(p.git('diff','--cached','--name-only').splitlines())<=allowed,'Staged files outside report')


def validate_report(p,target):
    expected=set(p.REPORT_FILES)
    require(all(Path(n).name==n and not n.startswith('.') for n in expected),'Invalid report whitelist')
    require(target.is_dir() and not target.is_symlink()
            and {f.name for f in target.iterdir()}==expected,'Report inventory differs')
    require(all(f.is_file() and not f.is_symlink() and f.stat().st_size<=2*1024**2
                for f in target.iterdir()),'Only compact regular report files may enter Git')
    receipt=p.read(target/'receipt.json');manifest=p.read(target/'manifest.json')
    require(receipt.get('completed') is True and receipt.get('experiment')==p.EXPERIMENT
            and receipt.get('profile')=='discovery' and receipt.get('whitelist')==sorted(expected),
            'Report receipt is not the complete declared discovery')
    require(set(receipt['files'])==expected-{'receipt.json'}
            and all(p.digest(target/n)==h for n,h in receipt['files'].items()),'Report receipt hashes differ')
    require(manifest['config']==p.config('discovery')
            and manifest['source_sha256']==p.identity(p.sources())
            and manifest['source_hashes']==p.sources()
            and receipt['source_sha256']==manifest['source_sha256']
            and receipt['git_revision']==manifest['git_revision'],'Report provenance differs')
    baseline=manifest['git_revision']
    require(re.fullmatch(r'[0-9a-f]{40}',baseline) is not None,'Full training commit identity required')
    for name,h in manifest['source_hashes'].items():
        require(hashlib.sha256(git_bytes(p,'show',baseline+':'+name)).hexdigest()==h,
                'Training commit/source differs: '+name)
    return manifest


def verify_report_commit(p,commit,expected,hashes,baseline,publish_head):
    parents=p.git('rev-list','--parents','-n','1',commit).split()
    require(len(parents)==2 and parents[0]==commit,'Report must be a single-parent commit')
    require(ancestor(p,baseline,parents[1]) and ancestor(p,commit,publish_head),
            'Report/run ancestry differs')
    changed=set(p.git('diff','--name-only',parents[1],commit).splitlines())
    added=set(p.git('diff','--name-only','--diff-filter=A',parents[1],commit).splitlines())
    require(changed==added==expected,'Commit must add exactly the approved seven files')
    for name,h in hashes.items():
        require(hashlib.sha256(git_bytes(p,'show',commit+':'+name)).hexdigest()==h,
                'Committed report differs: '+name)
        require(hashlib.sha256(git_bytes(p,'show',publish_head+':'+name)).hexdigest()==h,
                'Report changed after its publication commit')
    for pending in p.git('rev-list',publish_head,'--not','origin/main','origin/1.1').splitlines():
        if pending==commit:continue
        parents=p.git('rev-list','--parents','-n','1',pending).split()
        require(len(parents)==3 and any(ancestor(p,parent,ref)
                for parent in parents[1:] for ref in ('origin/main','origin/1.1')),
                'Pending history contains work beyond the report and upstream merges')
        require(clean_merge_tree(p,*parents[1:])==p.git('rev-parse',pending+'^{tree}'),
                'Pending merge differs from its independently computed clean tree')


def publish_report(p,runtime,output,run_root,*,push=True):
    """Export once, commit only this report, and retry pushes without retraining."""
    output=Path(output).resolve();run_root=Path(run_root).resolve();root=Path(p.ROOT).resolve()
    require(output!=run_root and output.is_relative_to(run_root) and not run_root.is_relative_to(root)
            and not output.is_relative_to(root),
            'Private completed run must stay outside the checkout')
    require(re.fullmatch(r'\d{8}T\d{6}Z_seed13',run_root.name) is not None,'Startup timestamp_seed13 required')
    require(re.fullmatch(r'[a-z0-9_]+',p.REPORT_SUBDIRECTORY) is not None,'Invalid report subdirectory')
    parent=root/'reports'/p.REPORT_SUBDIRECTORY;target=parent/run_root.name
    require(not any(f.is_symlink() for f in (root/'reports',parent,target)),'Symlinked report destination')
    relative=target.relative_to(root).as_posix();expected={relative+'/'+n for n in p.REPORT_FILES}
    p.review_gate(action='export')
    lock=Path(p.git('rev-parse','--git-path','q-rmea-publish.lock'))
    if not lock.is_absolute():lock=root/lock
    with p.lock(lock):
        # Refresh refs only; never merge, reset, stash or force-push user work.
        origin_gate(p)
        p.git('fetch','--no-tags','origin','+refs/heads/main:refs/remotes/origin/main',
              '+refs/heads/1.1:refs/remotes/origin/1.1')
        branch_gate(p,expected)
        runtime.export(output,target)
        manifest=validate_report(p,target);baseline=manifest['git_revision']
        require(ancestor(p,baseline,'HEAD'),'Training revision is not an ancestor of this checkout')
        hashes={name:p.digest(root/name) for name in expected}
        tracked=set(p.git('ls-tree','-r','--name-only','HEAD','--',*sorted(expected)).splitlines())
        require(not tracked or tracked==expected,'Partial pre-existing tracked report')
        commit=None
        if not tracked:
            base=p.git('rev-parse','HEAD')
            p.git('add','--',*sorted(expected))
            require(set(p.git('diff','--cached','--name-only').splitlines())==expected,'Unexpected staged changes')
            require(set(p.git('diff','--cached','--name-only','--diff-filter=A').splitlines())==expected,
                    'Only new report additions may be committed')
            require(all(hashlib.sha256(git_bytes(p,'show',':'+n)).hexdigest()==h
                        for n,h in hashes.items()),'Staged report differs from approved bytes')
            p.git('diff','--cached','--check')
            require(p.git('rev-parse','HEAD')==base,'HEAD moved before report commit')
            p.git('commit','-m','reports: '+p.EXPERIMENT+' '+run_root.name)
            commit=p.git('rev-parse','HEAD')
            require(p.git('rev-list','--parents','-n','1',commit).split()==[commit,base],
                    'HEAD moved during report commit')
        else:
            # Handles a crash after commit but before the private status write.
            record=run_root/'report-publication.json'
            if record.is_file():
                previous=p.read(record)
                require(previous.get('schema')=='q-rmea.report-publication.v1'
                        and previous.get('experiment')==p.EXPERIMENT
                        and previous.get('report')==relative and previous.get('files')==hashes
                        and previous.get('training_revision')==baseline,'Publication retry identity differs')
                commit=previous['report_commit']
            else:
                added_commits=set(p.git('log','--diff-filter=A','--format=%H','--',relative).splitlines())
                require(len(added_commits)==1,'Ambiguous report creation history')
                commit=added_commits.pop()
        publish_head=p.git('rev-parse','HEAD')
        verify_report_commit(p,commit,expected,hashes,baseline,publish_head)
        state=dict(schema='q-rmea.report-publication.v1',experiment=p.EXPERIMENT,
            report=relative,report_commit=commit,files=hashes,training_revision=baseline,
            status='committed_push_pending',training_repeated=False)
        status=run_root/'report-publication.json'
        p.atomic_json(status,state)
        branch_gate(p,set())
        validate_report(p,target)
        require(p.git('rev-parse','HEAD')==publish_head,'HEAD moved during publication checks')
        if not push:return state
        # A remote advance causes a normal non-fast-forward failure. The commit
        # stays intact; after an ordinary report-only merge this method reuses it.
        p.git('push','--no-follow-tags','origin',publish_head+':refs/heads/1.1')
        remote=p.git('ls-remote','origin','refs/heads/1.1').split()
        require(len(remote)==2 and remote[0]==publish_head,'Remote push identity differs')
        state['status']='pushed';state['pushed_head']=remote[0]
        p.atomic_json(status,state)
        return state


def main(argv=None):
    """Report-only retry entry; the normal handoff calls the same function."""
    import argparse
    import json
    from types import SimpleNamespace
    from .rmea_experiments import build
    from .rmea_token_transport_recovery import run_lock
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Completed private discovery directory')
    parser.add_argument('--run-root',type=Path,required=True,help='External startup timestamp_seed13 directory')
    args=parser.parse_args(argv)
    study=build('fair-reference');values=dict(vars(study.p))
    values.update(REPOSITORY_ORIGIN='https://github.com/tivilou/Q-Attention.git',
                  REPORT_SUBDIRECTORY='q_rmea_fair_reference',lock=run_lock)
    print(json.dumps(publish_report(SimpleNamespace(**values),study.runtime,args.output,args.run_root)))


if __name__=='__main__':main()
