"""Coordinator fault injection; passing fixtures are not neural runtime evidence."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.retacred import runtime_preflight as r
from baselines.retacred.common import atomic_json,digest,select_models

ROOT = Path(__file__).resolve().parents[1]


def fake_runtime(job,directory):
    directory.mkdir(parents=True)
    checks = {key:'passed' for key in r.REQUIRED_CHECKS[job['model']]}
    checks['traces'] = []
    for phase in ('train','test'):
        name = phase+('-group' if job['model'] == 'metaentail' else '')
        tensor = directory/(name+'.pt')
        tensor.write_bytes(b'synthetic-transport-fixture-not-a-Torch-tensor')
        metadata = {'status':'complete_diagnostic_scope','phase':phase,'tensor_artifact':tensor.name,
                    'sha256':digest(tensor),'size_bytes':tensor.stat().st_size,
                    'tensors':{'unit_fixture':{'value_storage':'complete_numeric_tensor','numel':1,'shape':[1]}}}
        atomic_json(tensor.with_suffix('.json'),metadata)
        checks['traces'].append({'json':name+'.json','pt':tensor.name,'sha256':digest(tensor)})
    checkpoint = directory/'post-update1.pt'
    checkpoint.write_bytes(b'synthetic-checkpoint-transport-fixture')
    checks['checkpoint'] = {'path':checkpoint.name,'sha256':digest(checkpoint)}
    value = {'schema':job['model']+'-runtime-gate.v1','mode':job['mode'],'device':job['device'],
             'status':'tiny_runtime_verified' if job['mode'] == 'tiny' else 'full_foundation_runtime_verified',
             'formal_training':False,'formal_comparison_admitted':False,'backend_imported':True,
             'weights_read':job['mode'] == 'full','pending':['actual_neural_runtime_not_exercised_by_this_unit_fixture'],
             'checks':checks}
    atomic_json(directory/'runtime-evidence.json',value)
    return value


def fixtures(monkeypatch):
    calls = []
    monkeypatch.setattr(r,'require_runtime_memory',lambda minimum:64*1024**3)
    monkeypatch.setattr(r.luke_sources,'prepare',lambda path,**kwargs:(path,{'files':{},'network_downloads':0}))
    monkeypatch.setattr(r,'ensure_assets',lambda name,root,offline:(root/name,{'unit_fixture':True}))
    def run(argv,**kwargs):
        name = next(key for key,value in r.MODULES.items() if value in argv)
        job = {'model':name,'mode':argv[argv.index('--mode')+1],'device':argv[argv.index('--device')+1]}
        directory = Path(argv[argv.index('--output-dir')+1])
        calls.append((argv,kwargs))
        fake_runtime(job,directory)
        return SimpleNamespace(returncode=0,stdout='unit fixture stdout\n',stderr='')
    monkeypatch.setattr(r.subprocess,'run',run)
    return calls


def command(output,*args):
    return ['--device','cpu','--offline','--output-dir',str(output),*args]


def test_four_tasks_serial_assets_sources_and_complete_transport_validation(tmp_path,monkeypatch):
    calls = fixtures(monkeypatch)
    output = tmp_path/'valid'
    assert r.main(command(output)) == 0
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'diagnostic_scope_verified' and result['formal_training'] is False
    assert len(calls) == 4
    assert [(job['model'],job['mode']) for job in result['jobs']] == [
        ('luke','tiny'),('luke','full'),('metaentail','tiny'),('metaentail','full')]
    assert all(job['runtime_invoked'] and job['command']['exit_status'] == 0 for job in result['jobs'])
    assert r.checked_manifest(output)[1] == result
    assert len(select_models('ready')) == 5
    with pytest.raises(ValueError):
        select_models('metaentail')
    assert not (output/'exchange_receipt.json').exists()


def test_low_memory_all_pending_without_source_weights_backend_or_network(tmp_path,monkeypatch):
    def gate(minimum):
        raise RuntimeError('fixture headroom insufficient for '+str(minimum))
    monkeypatch.setattr(r,'require_runtime_memory',gate)
    monkeypatch.setattr(r.luke_sources,'prepare',lambda *a,**k:pytest.fail('source acquisition before resource gate'))
    monkeypatch.setattr(r,'ensure_assets',lambda *a,**k:pytest.fail('assets before resource gate'))
    monkeypatch.setattr(r,'invoke',lambda *a,**k:pytest.fail('runtime before resource gate'))
    output = tmp_path/'pending'
    assert r.main(command(output)) == 2
    result = json.loads((output/'result.json').read_text())
    assert all(job['status'] == 'pending_memory_gate' and not job['runtime_invoked'] for job in result['jobs'])
    assert not (output/'logs').exists()
    assert all('full_foundation_runtime_verified' not in line for line in (output/'summary.md').read_text().splitlines())
    r.checked_manifest(output)


def test_missing_luke_source_skips_only_its_full_dependency(tmp_path,monkeypatch):
    calls = fixtures(monkeypatch)
    def missing(*args,**kwargs):
        raise FileNotFoundError('fixture offline source missing')
    monkeypatch.setattr(r.luke_sources,'prepare',missing)
    output = tmp_path/'missing'
    assert r.main(command(output)) == 2
    jobs = json.loads((output/'result.json').read_text())['jobs']
    assert [job['status'] for job in jobs] == ['pending_source_cache','pending_tiny_dependency',
                                              'tiny_runtime_verified','full_foundation_runtime_verified']
    assert len(calls) == 2


@pytest.mark.parametrize('kind',['checks','phase','hash','summary_only','checkpoint','allocation','schema'])
def test_claimed_success_with_missing_or_bad_evidence_rejected(tmp_path,kind):
    job = {'model':'metaentail','mode':'tiny','device':'cpu'}
    directory = tmp_path/'child'
    value = fake_runtime(job,directory)
    if kind == 'checks':
        value['checks'].pop(r.REQUIRED_CHECKS['metaentail'][0])
    elif kind == 'hash':
        (directory/'train-group.pt').write_bytes(b'changed')
    elif kind == 'checkpoint':
        (directory/'post-update1.pt').unlink()
    elif kind == 'allocation':
        value['weights_read'] = True
    elif kind == 'schema':
        value['schema'] = 'other'
    else:
        path = directory/'test-group.json'
        metadata = json.loads(path.read_text())
        if kind == 'phase':
            metadata['phase'] = 'train'
        else:
            metadata['tensors']['unit_fixture']['value_storage'] = 'shape_only'
        atomic_json(path,metadata)
    atomic_json(directory/'runtime-evidence.json',value)
    with pytest.raises((ValueError,FileNotFoundError)):
        r.verify_runtime(directory,job,0)


def test_child_failure_continues_other_family_and_returns_failure(tmp_path,monkeypatch):
    calls = fixtures(monkeypatch)
    original = r.invoke
    def failure(job,*args):
        if job['model'] == 'luke':
            raise RuntimeError('fixture child failure')
        return original(job,*args)
    monkeypatch.setattr(r,'invoke',failure)
    output = tmp_path/'failed'
    assert r.main(command(output)) == 1
    jobs = json.loads((output/'result.json').read_text())['jobs']
    assert [job['status'] for job in jobs] == ['failed','pending_tiny_dependency',
                                              'tiny_runtime_verified','full_foundation_runtime_verified']
    assert len(calls) == 2


def test_timeout_preserves_redacted_logs(tmp_path,monkeypatch):
    fixtures(monkeypatch)
    monkeypatch.setenv('PROJECT_EXCHANGE_TOKEN','fixture-secret')
    def timeout(argv,**kwargs):
        raise subprocess.TimeoutExpired(argv,1,output=b'fixture-secret stdout',stderr=b'fixture-secret stderr')
    monkeypatch.setattr(r.subprocess,'run',timeout)
    output = tmp_path/'timeout'
    assert r.main(command(output,'--models','metaentail','--mode','tiny')) == 1
    assert 'fixture-secret' not in (output/'logs/metaentail-tiny.stdout.log').read_text()
    assert '[redacted]' in (output/'logs/metaentail-tiny.stderr.log').read_text()
    job = json.loads((output/'result.json').read_text())['jobs'][0]
    assert job['command']['exit_status'] == 124 and job['status'] == 'failed'


def fake_uploader(monkeypatch, *, bad=False):
    calls = []
    def upload_declared_files(**kwargs):
        calls.append(kwargs)
        files = [{'path':name,**r.file_record(path),'uploaded':True} for path,name in kwargs['files']]
        if bad:
            files.pop()
        return {'status':'uploaded','target_dir':kwargs['target_dir'],'files':files}
    monkeypatch.setattr(r,'uploader',lambda:SimpleNamespace(upload_declared_files=upload_declared_files))
    monkeypatch.setenv('PROJECT_EXCHANGE_URL','https://fixture.example')
    monkeypatch.setenv('PROJECT_EXCHANGE_TOKEN','fixture-private-token')
    return calls


def test_upload_only_retries_transport_without_reexecuting_diagnostics(tmp_path,monkeypatch):
    calls = fixtures(monkeypatch)
    monkeypatch.delenv('PROJECT_EXCHANGE_URL',raising=False)
    monkeypatch.delenv('PROJECT_EXCHANGE_TOKEN',raising=False)
    output = tmp_path/'retry'
    assert r.main(command(output,'--models','metaentail','--mode','tiny','--upload')) == 1
    assert len(calls) == 1 and json.loads((output/'exchange_pending.json').read_text())['status'] == 'failed'
    original = digest(output/'artifact_manifest.json')
    uploaded = fake_uploader(monkeypatch)
    assert r.main(['--upload-only',str(output)]) == 0
    assert len(calls) == 1 and len(uploaded) == 1
    assert digest(output/'artifact_manifest.json') == original
    receipt = json.loads((output/'exchange_receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert 'fixture-private-token' not in json.dumps(receipt)
    assert 'fixture.example' not in json.dumps(receipt)


def test_incomplete_receipt_not_success_and_local_results_retained(tmp_path,monkeypatch):
    fixtures(monkeypatch)
    fake_uploader(monkeypatch,bad=True)
    output = tmp_path/'bad-receipt'
    assert r.main(command(output,'--models','metaentail','--mode','tiny','--upload')) == 1
    assert (output/'result.json').is_file() and not (output/'exchange_receipt.json').exists()
    assert json.loads((output/'exchange_pending.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('kind',['changed','extra','symlink'])
def test_upload_refuses_changed_extra_or_linked_evidence(tmp_path,monkeypatch,kind):
    fixtures(monkeypatch)
    output = tmp_path/'guard'
    assert r.main(command(output,'--models','metaentail','--mode','tiny')) == 0
    uploaded = fake_uploader(monkeypatch)
    if kind == 'changed':
        (output/'metaentail/tiny/train-group.pt').write_bytes(b'changed')
    elif kind == 'extra':
        (output/'unrelated.json').write_text('{}')
    else:
        original = output/'metaentail/tiny/train-group.pt'
        stored = tmp_path/'original.pt'
        original.rename(stored)
        original.symlink_to(stored)
    assert r.main(['--upload-only',str(output)]) == 1
    assert not uploaded


def test_large_evidence_shards_reconstruct_reuse_and_detect_damage(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'SHARD_BYTES',8)
    original = tmp_path/'large.pt'
    original.write_bytes(b'0123456789abcdef0123456789')
    atomic_json(tmp_path/'artifact_manifest.json',{'unit_fixture':True})
    manifest = {'files':{'large.pt':r.file_record(original)}}
    first = r.delivery_files(tmp_path,manifest)
    assert original not in [path for path,name in first]
    shards = [(path,name) for path,name in first if '.part000' in name]
    assert len(shards) == 4 and b''.join(path.read_bytes() for path,name in shards) == original.read_bytes()
    assert r.delivery_files(tmp_path,manifest) == first
    shards[0][0].write_bytes(b'changed')
    with pytest.raises(ValueError,match='shard bytes'):
        r.delivery_files(tmp_path,manifest)


def test_auto_respects_hidden_gpus_and_probes_without_importing_backend(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**k:pytest.fail('disabled GPU probed'))
    assert r.choose_device('auto')[0] == 'cpu'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    monkeypatch.setattr(r.subprocess,'run',lambda argv,**k:SimpleNamespace(returncode=0,stdout='0\n1\n2\n',stderr='',args=argv))
    assert r.choose_device('auto')[0] == 'cuda'


@pytest.mark.parametrize('name',['all','luke,luke','unknown',''])
def test_invalid_models_no_output(tmp_path,name):
    output = tmp_path/'invalid'
    assert r.main(command(output,'--models',name)) == 1
    assert not output.exists()


def test_existing_and_symlinked_output_preserved(tmp_path):
    output = tmp_path/'existing'
    output.mkdir()
    stored = output/'keep'
    stored.write_bytes(b'original')
    assert r.main(command(output)) == 1 and stored.read_bytes() == b'original'
    alias = tmp_path/'alias'
    alias.symlink_to(output,target_is_directory=True)
    assert r.main(command(alias)) == 1 and stored.read_bytes() == b'original'


def test_offline_missing_assets_no_transport(tmp_path,monkeypatch):
    monkeypatch.setattr(r.mnli_assets,'fetch',lambda *a,**k:pytest.fail('offline assets fetched'))
    with pytest.raises(FileNotFoundError):
        r.ensure_assets('metaentail',tmp_path,True)


def test_describe_portable_shell_moved_checkout_and_no_backend(tmp_path):
    script = ROOT/'scripts/check_retacred_reference_runtime.sh'
    env = dict(os.environ,PYTHON_BIN=sys.executable,PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run(['bash',str(script),'--describe'],cwd=tmp_path,env=env,
                          text=True,capture_output=True,timeout=20)
    assert done.returncode == 0,done.stderr
    assert len(json.loads(done.stdout)['jobs']) == 4
    moved = tmp_path/'moved'
    shutil.copytree(ROOT/'baselines',moved/'baselines',ignore=shutil.ignore_patterns('__pycache__'))
    (moved/'scripts').mkdir()
    shutil.copyfile(script,moved/'scripts'/script.name)
    done = subprocess.run(['bash',str(moved/'scripts'/script.name),'--describe'],cwd=tmp_path,env=env,
                          text=True,capture_output=True,timeout=20)
    assert done.returncode == 0,done.stderr
    audit = r'''
import sys
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','tensorflow','jax'): raise AssertionError('backend')
 if event.startswith('socket.'): raise AssertionError('network')
sys.addaudithook(guard)
from baselines.retacred.runtime_preflight import main
raise SystemExit(main(['--describe','--output-dir',sys.argv[1]]))
'''
    output = tmp_path/'must-not-exist'
    done = subprocess.run([sys.executable,'-c',audit,str(output)],cwd=ROOT,env=env,text=True,
                          capture_output=True,timeout=20)
    assert done.returncode == 0,done.stderr
    assert not output.exists()


def test_no_git_mutation_or_formal_runner_imports():
    tree = ast.parse(Path(r.__file__).read_text())
    assert not any(isinstance(node,ast.ImportFrom) and node.module in ('train','orchestrate','report')
                   for node in ast.walk(tree))
    source = Path(r.__file__).read_text()
    assert all(value not in source for value in ('git push','git commit','git add','killpg','kill('))
