"""Branch and publication boundaries; real launch gated by exact-source review."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from q_attention.experiments import srm_handoff as h,srm_protocol as p,srm_runner as runner


@pytest.fixture
def repository(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    def git(*argv):return subprocess.check_output(['git',*argv],cwd=root,text=True).strip()
    git('init','-b','1.1');git('config','user.name','test');git('config','user.email','test@example.invalid')
    (root/'code').write_text('unit fixture');git('add','code');git('commit','-m','fixture')
    for ref in ('origin/main','origin/1.1'):git('update-ref','refs/remotes/'+ref,'HEAD')
    monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(h,'source_contract',lambda:{})
    return root,git


def test_branch_gate_clean_and_staged_report(repository):
    root,git=repository;h.branch_gate()
    report=root/'reports/q_srm_screen/20261025T000000Z_seed13';report.mkdir(parents=True)
    (report/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='clean'):h.branch_gate()
    h.branch_gate('reports/q_srm_screen/20261025T000000Z_seed13')
    git('add','reports');h.branch_gate('reports/q_srm_screen/20261025T000000Z_seed13')
    (root/'private.log').write_text('unit fixture')
    with pytest.raises(ValueError,match='private.log'):h.branch_gate('reports/q_srm_screen/20261025T000000Z_seed13')


def test_branch_gate_stale_remote_and_main(repository):
    root,git=repository
    (root/'code').write_text('future');git('commit','-am','future')
    future=git('rev-parse','HEAD');git('reset','--hard','HEAD~1')
    git('update-ref','refs/remotes/origin/1.1',future)
    with pytest.raises(ValueError,match='origin/1.1'):h.branch_gate()
    git('switch','-c','main')
    with pytest.raises(ValueError,match='branch 1.1'):h.branch_gate()


def test_report_path_rejects_escape_and_symlink(repository,tmp_path):
    root,_=repository;run=tmp_path/'20261025T000000Z_seed13'
    assert h.resolve_report(run)==root/'reports/q_srm_screen'/run.name
    with pytest.raises(ValueError):h.resolve_report(tmp_path/'new')
    with pytest.raises(ValueError):h.resolve_report(run,str(tmp_path/'outside'))
    (root/'reports').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinked'):h.resolve_report(run)


def test_release_requires_pass_and_exact_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);monkeypatch.setattr(p,'validate_review',lambda:None)
    monkeypatch.setattr(h,'source_contract',lambda:{'unit':'hash'})
    file=tmp_path/'docs/validation/q_srm_handoff_review.json';file.parent.mkdir(parents=True)
    data=dict(review_scope='both',verdict='PASS_WITH_RISK',must_fix=[],source_sha256=p.identity(h.source_contract()))
    p.atomic_json(file,data)
    with pytest.raises(ValueError,match='PASS'):h.release_review()
    data['verdict']='PASS';p.atomic_json(file,data);h.release_review()
    data['source_sha256']='stale';p.atomic_json(file,data)
    with pytest.raises(ValueError):h.release_review()


def test_cpu_only_for_synthetic():
    env,selection=h.gpu_environment('cpu',True)
    assert env['CUDA_VISIBLE_DEVICES']=='' and selection['execution_profile']=='low_memory_serial'
    with pytest.raises(ValueError,match='canary-only'):h.gpu_environment('cpu',False)


def test_auto_gpu_respects_visible_mask(monkeypatch):
    import torch
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda i:((9+i)*1024**3,20*1024**3))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'unit GPU')
    env,selection=h.gpu_environment('auto',False)
    assert env['CUDA_VISIBLE_DEVICES']=='GPU-b' and selection['selected_visible_index']==1
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda i:(7*1024**3,20*1024**3))
    with pytest.raises(ValueError,match='8GiB'):h.gpu_environment('auto',False)


def test_utility_separate_from_mismatch_attribution():
    cfg=p.config('screen');rows=[]
    for i,seed in enumerate(cfg['seeds']):
        for mode in p.MODES:
            value=.1+.02*i+(.01 if mode in ('quantum','key_mismatch') else 0)
            rows.append(dict(seed=seed,mode=mode,test={'metrics':{'macro_f1':value}}))
    result=runner.aggregate(rows,cfg)
    assert result['utility_gate_passed'] and result['mismatch_parity']
    assert result['classification']=='promote_to_formal_preparation'
    assert result['quantum_attribution']=='not established'
    assert len(result['relative_deltas'])==5


@pytest.fixture
def publishing_repository(repository,tmp_path,monkeypatch):
    root,git=repository
    remote=tmp_path/'origin.git'
    subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
    git('remote','add','origin',str(remote))
    git('push','origin','HEAD:main','HEAD:1.1')
    baseline=git('rev-parse','HEAD')
    raw=tmp_path/'20261025T000000Z_seed13';raw.mkdir()
    p.atomic_json(raw/'handoff.json',dict(canary_only=True))
    target=h.resolve_report(raw);target.mkdir(parents=True)
    for name in h.REPORT_FILES-{'run_summary.json'}:
        p.atomic_json(target/name,dict(fingerprint='fixture'))
    p.atomic_json(target/'run_summary.json',dict(schema_version='q-srm.handoff-report.v1',
        data={'train':{'source_count':1}},fingerprint='fixture',git_revision=baseline,
        stage='preflight',run_id=raw.name,sha256={n:p.sha(target/n) for n in h.REPORT_FILES-{'run_summary.json'}}))
    # Git integration tests substitute only immutable-source reconstruction;
    # the real launcher/exporter are exercised by the separate shell acceptance.
    def export(*args):
        h.branch_gate(target.relative_to(root).as_posix());h.validate_report(target)
        return target
    monkeypatch.setattr(h,'export_report',export)
    return root,git,remote,raw,target,baseline


def test_publish_push_retry_and_idempotence(publishing_repository,monkeypatch):
    root,git,remote,raw,target,baseline=publishing_repository
    before=p.sha(raw/'handoff.json')
    with pytest.raises(RuntimeError,match='injected report push failure'):
        h.publish_report(raw,inject_failure=True)
    commit=git('rev-parse','HEAD')
    assert commit!=baseline and git('status','--porcelain')==''
    assert git('ls-remote','origin','refs/heads/1.1').split()[0]==baseline
    monkeypatch.setattr(h,'run',lambda *_:pytest.fail('export retry must not train'))
    for _ in range(2):
        assert h.main(['export','--run-dir',str(raw)])==0
        assert git('rev-parse','HEAD')==commit
        assert git('ls-remote','origin','refs/heads/1.1').split()[0]==commit
    assert p.sha(raw/'handoff.json')==before
    assert len(git('diff','--name-only',baseline,commit).splitlines())==7


def test_publish_real_remote_rejection_retries_same_commit(publishing_repository):
    root,git,remote,raw,_,baseline=publishing_repository
    hook=remote/'hooks/pre-receive'
    hook.write_text('#!/bin/sh\nexit 1\n');hook.chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError):h.publish_report(raw)
    commit=git('rev-parse','HEAD');assert commit!=baseline
    hook.unlink();h.publish_report(raw)
    assert git('rev-parse','HEAD')==commit
    assert git('ls-remote','origin','refs/heads/1.1').split()[0]==commit


@pytest.mark.parametrize('staged',[False,True])
def test_publish_refuses_unrelated_dirty_files(publishing_repository,staged):
    root,git,_,raw,_,baseline=publishing_repository
    (root/'private.pt').write_bytes(b'not public')
    if staged:git('add','private.pt')
    with pytest.raises(ValueError,match='private.pt'):h.publish_report(raw)
    assert git('rev-parse','HEAD')==baseline


def test_publish_refuses_unrelated_commit_after_run(publishing_repository):
    root,git,_,raw,_,baseline=publishing_repository
    (root/'code').write_text('unrelated');git('commit','-am','unrelated')
    with pytest.raises(ValueError,match='exactly the seven'):h.publish_report(raw)
    assert git('ls-remote','origin','refs/heads/1.1').split()[0]==baseline


def test_publish_refreshes_remote_ancestry(publishing_repository,tmp_path):
    root,git,remote,raw,_,baseline=publishing_repository
    other=tmp_path/'other'
    subprocess.run(['git','clone','-b','1.1',str(remote),str(other)],check=True,capture_output=True)
    def run(*args):return subprocess.check_output(['git',*args],cwd=other,text=True)
    run('config','user.name','test');run('config','user.email','test@example.invalid')
    (other/'code').write_text('remote update');run('commit','-am','remote update');run('push','origin','1.1')
    with pytest.raises(ValueError,match='origin/1.1'):h.publish_report(raw)
    assert git('rev-parse','HEAD')==baseline


def test_publish_rejects_commit_hook_extra_payload(publishing_repository):
    root,git,remote,raw,_,baseline=publishing_repository
    hook=root/'.git/hooks/pre-commit'
    hook.write_text('#!/bin/sh\nprintf private > private.pt\ngit add private.pt\n');hook.chmod(0o755)
    with pytest.raises(ValueError,match='exactly the seven'):h.publish_report(raw)
    assert git('ls-remote','origin','refs/heads/1.1').split()[0]==baseline


@pytest.mark.skipif(os.environ.get('QSRM_HANDOFF_LIFECYCLE')!='1',reason='requires reviewed clean branch1.1 fixture checkout')
def test_real_shell_entry_export_failure_retry(tmp_path):
    # Intended only in an isolated clean fixture clone, never canonical collaborator branch.
    root=tmp_path/'20261025T000000Z_seed13'
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=p.ROOT,text=True).strip()
    assert Path(remote).is_absolute() and Path(remote).is_dir(),'synthetic publication requires an isolated local bare remote'
    env=dict(os.environ,PYTHON_BIN=sys.executable,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run(['bash','scripts/run_q_srm_handoff.sh','--canary-only','--gpu','cpu','--run-dir',str(root),'--inject-export-failure'],cwd=p.ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'injected report copy failure' in result.stderr,result.stdout+'\n'+result.stderr
    target=p.ROOT/'reports/q_srm_screen'/root.name
    assert not target.exists() and not list(target.parent.glob('.'+root.name+'-*'))
    before={f.relative_to(root).as_posix():p.sha(f) for f in root.rglob('*') if f.is_file()}
    for _ in range(2):
        result=subprocess.run(['bash','scripts/export_q_srm_handoff.sh','--run-dir',str(root)],cwd=p.ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    assert before=={f.relative_to(root).as_posix():p.sha(f) for f in root.rglob('*') if f.is_file()}
    assert {f.name for f in target.iterdir()}==h.REPORT_FILES
    record=h.validate_report(target)
    assert record['stage']=='preflight' and record['data'] and record['formal'] is False
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.ROOT,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/1.1'],cwd=p.ROOT,text=True).split()[0]==commit
    report_identity=p.sha(target/'run_summary.json')
    # Allowlist and stored identity reject a private payload in the final directory.
    (target/'private.pt').write_bytes(b'unit private payload')
    with pytest.raises(ValueError,match='allowlist'):h.validate_report(target)
    (target/'private.pt').unlink()
    p.atomic_json(tmp_path/'acceptance.json',dict(status='passed',source_run=str(root),exporter_source_sha256=p.identity(h.source_contract()),
        failure_reason='injected report copy failure',retry_count=2,immutable_source=True,final_report_identity=report_identity,
        conditions=5,traces=50,files=7,automatic_publication='passed_local_bare_remote',cuda='not_available_on_owner_host'))
