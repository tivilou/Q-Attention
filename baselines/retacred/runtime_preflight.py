"""Serial, bounded LUKE/MNLI diagnostics with private evidence delivery.

This entry never admits a method, launches formal training or changes Git.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .assets import CA, SHARD_BYTES, reject_symlinks, shard_file, uploader
from .common import ROOT, PACKAGE, atomic_json, digest
from .luke import lock as luke_lock, require_runtime_memory
from . import luke_assets, luke_sources, mnli_assets

MODULES = {'luke':'baselines.retacred.luke_runtime_fidelity',
           'metaentail':'baselines.retacred.metaentail_runtime_fidelity'}
REQUIRED_CHECKS = {
    'luke':('fixed_input_forward_loss', 'gold_input_isolation',
        'word_entity_attention_head_gradients', 'dropout_trace_transparency',
        'nonzero_diagnostic_update', 'diagnostic_scheduler_recovery',
        'exact_post_update_resume', 'actual_numeric_train_test_trace_roundtrip'),
    'metaentail':('native_forward_and_pair_CE', 'gold_input_isolation',
        'nonzero_embedding_attention_NLI_gradients', 'train_dropout_trace_transparency',
        'diagnostic_nonzero_update_and_scheduler', 'changed_resume_contract_rejected_before_mutation',
        'exact_post_update_pair_cursor_resume', 'test_forward_trace_transparency',
        'actual_complete_train_test_group_tensor_roundtrip'),
}
SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


class Pending(RuntimeError):
    def __init__(self, status, reason):
        self.status = status
        super().__init__(reason)


def relative_name(name):
    if not isinstance(name, str) or not name or any(not SAFE.fullmatch(part) for part in name.split('/')):
        raise ValueError('unsafe evidence path')
    return name


def redacted(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    value = str(value or '')
    for key in ('PROJECT_EXCHANGE_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        secret = os.environ.get(key)
        if secret:
            value = value.replace(secret, '[redacted]')
    return value


def models(raw):
    selected = raw.split(',')
    if not selected or len(set(selected)) != len(selected) or any(name not in MODULES for name in selected):
        raise ValueError('diagnostic families must be luke,metaentail or one of them')
    return selected


def describe(selected, mode, device):
    modes = ('tiny','full') if mode == 'both' else (mode,)
    jobs = []
    for name in selected:
        module = importlib.import_module(MODULES[name])
        for stage in modes:
            descriptor = module.description(stage, 'cpu' if device == 'auto' else device)
            jobs.append({'model':name, 'mode':stage, 'module':MODULES[name],
                'minimum_host_headroom_bytes':descriptor['minimum_host_headroom_bytes'],
                'optimizer_claim':descriptor['optimizer_claim'], 'pending':descriptor['pending']})
    return {'schema':'reference-runtime-preflight.v1', 'formal_training':False,
            'formal_comparison_admitted':False, 'device_request':device,
            'execution':'one_diagnostic_process_at_a_time', 'jobs':jobs}


def choose_device(requested):
    if requested != 'auto':
        return requested, {'selection':'explicit', 'device':requested}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() in ('','-1'):
        return 'cpu', {'selection':'CUDA_VISIBLE_DEVICES_disables_GPU', 'device':'cpu'}
    try:
        done = subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        device = 'cuda' if done.returncode == 0 and done.stdout.strip() else 'cpu'
        return device, {'selection':'nvidia-smi_probe', 'device':device,
            'argv':done.args, 'exit_status':done.returncode,
            'stdout':redacted(done.stdout), 'stderr':redacted(done.stderr)}
    except (OSError, subprocess.TimeoutExpired) as error:
        return 'cpu', {'selection':'nvidia-smi_unavailable', 'device':'cpu', 'error_type':type(error).__name__}


def ensure_assets(name, asset_root, offline):
    client = luke_assets if name == 'luke' else mnli_assets
    cache = asset_root / ('luke' if name == 'luke' else 'mnli') / client.REVISION
    checkpoint = cache / 'model'
    reject_symlinks(checkpoint)
    if offline:
        if name == 'luke':
            spec = client.validate_lock(json.loads(client.LOCK_PATH.read_text()))
            hashes = {}
            for row in spec['files']:
                path = checkpoint / row['name']
                if not path.is_file():
                    raise FileNotFoundError('offline LUKE asset missing: ' + row['name'])
                actual = client.verify_original(path, row['name'])
                if actual != row['sha256']:
                    raise ValueError('offline LUKE original SHA mismatch')
                hashes[row['name']] = actual
            evidence = {'files':hashes, 'weights_locally_verified':True, 'neural_runtime_verified':False}
        else:
            if any(not (checkpoint/name).is_file() for name in client.originals()):
                raise FileNotFoundError('offline MNLI assets incomplete')
            evidence = client.verify_checkpoint(checkpoint)
    else:
        missing = (any(not (checkpoint/row['name']).exists() for row in
                       json.loads(client.LOCK_PATH.read_text())['files']) if name == 'luke'
                   else any(not (checkpoint/item).exists() for item in client.originals()))
        if missing and (not os.environ.get('PROJECT_EXCHANGE_URL') or not os.environ.get('PROJECT_EXCHANGE_TOKEN')):
            raise Pending('pending_assets_configuration', 'missing assets require PROJECT_EXCHANGE_URL and PROJECT_EXCHANGE_TOKEN')
        prepared = client.fetch(cache_dir=cache)
        if prepared != checkpoint:
            raise ValueError('asset client returned an unexpected checkpoint path')
        evidence = json.loads((cache/'fetch-verification.json').read_text())
    return checkpoint, evidence


def verify_runtime(directory, job, exit_status):
    path = directory/'runtime-evidence.json'
    reject_symlinks(path)
    value = json.loads(path.read_text())
    status = value.get('status')
    expected = 'tiny_runtime_verified' if job['mode'] == 'tiny' else 'full_foundation_runtime_verified'
    if exit_status == 2 and status == 'pending_memory_gate':
        if value.get('backend_imported') is not False or value.get('weights_read') is not False:
            raise ValueError('memory-pending child claimed model allocation')
        return status, value
    if exit_status != 0 or status != expected:
        raise ValueError('diagnostic child exit/status did not pass: ' + str(status))
    if (value.get('schema') != job['model']+'-runtime-gate.v1'
            or value.get('mode') != job['mode'] or value.get('device') != job['device']
            or value.get('formal_training') is not False or value.get('formal_comparison_admitted') is not False
            or value.get('backend_imported') is not True
            or value.get('weights_read') is not (job['mode'] == 'full')):
        raise ValueError('diagnostic child scope/allocation mismatch')
    checks = value.get('checks', {})
    if any(checks.get(key) != 'passed' for key in REQUIRED_CHECKS[job['model']]):
        raise ValueError('required runtime checks missing or incomplete')
    traces = checks.get('traces', [])
    if len(traces) != 2:
        raise ValueError('both selected train and test traces required')
    phases = set()
    for trace in traces:
        metadata = directory/relative_name(trace['json'])
        tensor = directory/relative_name(trace['pt'])
        reject_symlinks(metadata)
        reject_symlinks(tensor)
        fields = json.loads(metadata.read_text())
        if (fields.get('status') != 'complete_diagnostic_scope' or fields.get('phase') not in ('train','test')
                or tensor.stat().st_size <= 0 or digest(tensor) != trace['sha256']
                or fields.get('sha256') != trace['sha256'] or fields.get('tensor_artifact') != tensor.name
                or fields.get('size_bytes') != tensor.stat().st_size):
            raise ValueError('trace metadata/numeric artifact identity mismatch')
        if (not fields.get('tensors') or any(row.get('value_storage') != 'complete_numeric_tensor'
                or row.get('numel',0) <= 0 for row in fields['tensors'].values())):
            raise ValueError('shape-only or empty trace is not complete')
        phases.add(fields['phase'])
    if phases != {'train','test'}:
        raise ValueError('training/testing trace phases differ')
    checkpoint = checks.get('checkpoint', {})
    target = directory/relative_name(checkpoint.get('file', checkpoint.get('path','')))
    reject_symlinks(target)
    if not target.is_file() or target.stat().st_size <= 0 or digest(target) != checkpoint.get('sha256'):
        raise ValueError('post-update checkpoint identity mismatch')
    return status, value


def invoke(job, directory, source, checkpoint, timeout, logs):
    argv = [sys.executable,'-u','-m',MODULES[job['model']], '--mode',job['mode'],
            '--device',job['device'], '--output-dir',str(directory)]
    if job['model'] == 'luke':
        argv += ['--source-cache',str(source)]
    if job['mode'] == 'full':
        argv += ['--checkpoint',str(checkpoint)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0')
    if job['device'] == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        done = subprocess.run(argv, cwd=ROOT, env=env, text=True, errors='replace', capture_output=True, timeout=timeout)
        code, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = 124, error.stdout, error.stderr
    logs.mkdir(parents=True, exist_ok=True)
    prefix = job['model']+'-'+job['mode']
    (logs/(prefix+'.stdout.log')).write_text(redacted(stdout), encoding='utf-8')
    (logs/(prefix+'.stderr.log')).write_text(redacted(stderr), encoding='utf-8')
    job['command'] = {'argv':argv, 'cwd':str(ROOT), 'exit_status':code,
        'environment_overrides':{key:env[key] for key in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},
        'stdout_log':'logs/'+prefix+'.stdout.log', 'stderr_log':'logs/'+prefix+'.stderr.log'}
    if code == 124:
        raise RuntimeError('bounded diagnostic timed out; own child terminated, evidence retained')
    status, evidence = verify_runtime(directory,job,code)
    job['runtime_evidence_sha256'] = digest(directory/'runtime-evidence.json')
    job['remaining_fidelity_gates'] = evidence['pending']
    return status


def outcome(jobs):
    if any(job['status'] == 'failed' for job in jobs):
        return 'failed', 1
    if any(job['status'].startswith('pending_') for job in jobs):
        return 'pending', 2
    return 'diagnostic_scope_verified', 0


def write_summary(run, result):
    atomic_json(run/'result.json',result)
    lines = ['# Re-TACRED bounded runtime preflight', '',
        'Diagnostic evidence only; no formal training, comparison admission or paper-score certification.', '',
        '| family | mode | device | status |', '| --- | --- | --- | --- |']
    for job in result['jobs']:
        lines.append('| '+ ' | '.join(str(job[key]) for key in ('model','mode','device','status'))+' |')
    lines += ['', 'Overall: '+result['status']+'; exit '+str(result['exit_status']),
              '', 'Private artifacts remain under this run directory; do not add them to Git.']
    (run/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(args, run, result):
    device, selection = choose_device(args.device)
    result['device_selection'] = selection
    for job in result['jobs']:
        job.update(device=device,status='pending_not_started',runtime_invoked=False)
    previous = {}
    for index, job in enumerate(result['jobs']):
        job['device'] = device
        directory = run/job['model']/job['mode']
        job['directory'] = directory.relative_to(run).as_posix()
        job['runtime_invoked'] = False
        print(f"[runtime {index+1}/{len(result['jobs'])}] {job['model']} {job['mode']} / {device}", flush=True)
        try:
            try:
                job['host_headroom_bytes'] = require_runtime_memory(job['minimum_host_headroom_bytes'])
            except RuntimeError as error:
                raise Pending('pending_memory_gate',str(error)) from error
            if job['mode'] == 'full' and previous.get(job['model'],'tiny_runtime_verified') != 'tiny_runtime_verified':
                raise Pending('pending_tiny_dependency','selected tiny diagnostic did not pass; full run not launched')
            source = None
            if job['model'] == 'luke':
                try:
                    source, evidence = luke_sources.prepare(args.luke_source_cache,offline=args.offline)
                except FileNotFoundError as error:
                    raise Pending('pending_source_cache',str(error)) from error
                atomic_json(run/'luke-source-verification.json',evidence)
            checkpoint = None
            if job['mode'] == 'full':
                try:
                    checkpoint, evidence = ensure_assets(job['model'], args.asset_root, args.offline)
                except FileNotFoundError as error:
                    raise Pending('pending_assets_cache',str(error)) from error
                atomic_json(run/(job['model']+'-asset-verification.json'),evidence)
            job['runtime_invoked'] = True
            job['status'] = invoke(job,directory,source,checkpoint,args.timeout_seconds,run/'logs')
        except Pending as error:
            job.update(status=error.status,reason=redacted(error))
            atomic_json(directory/'runtime-evidence.json', {'schema':'reference-runtime-parent-gate.v1',
                'status':job['status'], 'reason':job['reason'], 'runtime_invoked':False,
                'backend_imported':False, 'weights_read':False})
        except Exception as error:
            job.update(status='failed',error_type=type(error).__name__,reason=redacted(error))
        previous[job['model']] = job['status']
        print('  '+job['status']+((': '+job['reason']) if 'reason' in job else ''), flush=True)
        result['status'],result['exit_status'] = outcome(result['jobs'][:index+1])
        write_summary(run,result)
    return result


def file_record(path):
    reject_symlinks(path)
    if not path.is_file():
        raise ValueError('declared evidence is not a regular file')
    return {'sha256':digest(path), 'size_bytes':path.stat().st_size}


def inventory(run):
    files = {}
    top = ('result.json','summary.md','luke-source-verification.json',
           'luke-asset-verification.json','metaentail-asset-verification.json')
    for name in top:
        if (run/name).exists():
            files[name] = file_record(run/name)
    for family in MODULES:
        for mode in ('tiny','full'):
            directory = run/family/mode
            expected = ('runtime-evidence.json','post-update1.pt','train.pt','train.json','test.pt','test.json')
            if family == 'metaentail':
                expected = ('runtime-evidence.json','post-update1.pt','train-group.pt','train-group.json','test-group.pt','test-group.json')
            if directory.exists():
                reject_symlinks(directory)
                if any(path.name not in expected for path in directory.iterdir()):
                    raise ValueError('unexpected child evidence file')
                for path in directory.iterdir():
                    files[path.relative_to(run).as_posix()] = file_record(path)
    logs = run/'logs'
    if logs.exists():
        for path in logs.iterdir():
            if path.name not in {name+'-'+mode+'.'+kind+'.log' for name in MODULES
                                 for mode in ('tiny','full') for kind in ('stdout','stderr')}:
                raise ValueError('unexpected runtime log')
            files[path.relative_to(run).as_posix()] = file_record(path)
    return files


def build_manifest(run):
    relative_name(run.name)
    manifest = {'schema':'reference-runtime-artifacts.v1',
        'target_dir':'q-attention/retacred-reference-runtime/'+run.name,
        'files':inventory(run)}
    atomic_json(run/'artifact_manifest.json',manifest)
    return manifest


def checked_manifest(run):
    reject_symlinks(run)
    manifest = json.loads((run/'artifact_manifest.json').read_text())
    if (manifest.get('schema') != 'reference-runtime-artifacts.v1'
            or manifest.get('target_dir') != 'q-attention/retacred-reference-runtime/'+relative_name(run.name)
            or manifest.get('files') != inventory(run)):
        raise ValueError('saved evidence inventory changed; upload rejected')
    result = json.loads((run/'result.json').read_text())
    if result.get('schema') != 'reference-runtime-preflight.v1' or result.get('formal_training') is not False:
        raise ValueError('runtime result scope mismatch')
    allowed = {'artifact_manifest.json','exchange_pending.json','exchange_receipt.json','exchange-parts'}
    allowed |= {Path(name).parts[0] for name in manifest['files']}
    if any(path.name not in allowed for path in run.iterdir()):
        raise ValueError('undeclared file in runtime run directory')
    return manifest,result


def delivery_files(run, manifest):
    declared = []
    for name, identity in manifest['files'].items():
        path = run/relative_name(name)
        if identity['size_bytes'] <= SHARD_BYTES:
            declared.append((path,name))
            continue
        parts_dir = run/'exchange-parts'/name
        reject_symlinks(parts_dir)
        metadata = parts_dir/'parts.json'
        if metadata.exists():
            spec = json.loads(metadata.read_text())
            if spec.get('original') != {'name':name,**identity}:
                raise ValueError('existing shard manifest differs; preserved')
        else:
            if parts_dir.exists() and any(parts_dir.iterdir()):
                raise ValueError('incomplete old shard cache preserved; use a new run directory')
            parts = shard_file(path,parts_dir,shard_bytes=SHARD_BYTES)
            spec = {'schema':'reference-runtime-shards.v1', 'original':{'name':name,**identity}, 'parts':parts}
            atomic_json(metadata,spec)
        import hashlib
        total, combined = 0,hashlib.sha256()
        for index,part in enumerate(spec['parts']):
            if part['name'] != path.name+f'.part{index:04d}':
                raise ValueError('shard order changed')
            target = parts_dir/relative_name(part['name'])
            if file_record(target) != {'sha256':part['sha256'],'size_bytes':part['size_bytes']}:
                raise ValueError('shard bytes changed')
            with target.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):
                    total += len(block)
                    combined.update(block)
            declared.append((target,name+'.parts/'+part['name']))
        if total != identity['size_bytes'] or combined.hexdigest() != identity['sha256']:
            raise ValueError('shards do not reconstruct original evidence')
        declared.append((metadata,name+'.parts.json'))
    declared.append((run/'artifact_manifest.json','artifact_manifest.json'))
    return declared


def upload(run):
    manifest,result = checked_manifest(run)
    declared = delivery_files(run,manifest)
    pending = {'schema':'reference-runtime-delivery.v1','status':'pending',
        'target_dir':manifest['target_dir'],
        'files':{name:file_record(path) for path,name in declared}}
    atomic_json(run/'exchange_pending.json',pending)
    try:
        url = luke_assets.validate_url(os.environ.get('PROJECT_EXCHANGE_URL',''))
        token = os.environ.get('PROJECT_EXCHANGE_TOKEN','')
        if not token:
            raise ValueError('PROJECT_EXCHANGE_TOKEN required for private delivery')
        receipt = uploader().upload_declared_files(base_url=url,token=token,
            target_dir=manifest['target_dir'],files=declared,ca_file=CA)
        received = {row['path']:{'sha256':row['sha256'],'size_bytes':row['size_bytes']}
                    for row in receipt.get('files',[]) if row.get('uploaded') is True}
        if (receipt.get('status') != 'uploaded' or receipt.get('target_dir') != manifest['target_dir']
                or len(receipt.get('files',[])) != len(pending['files']) or received != pending['files']):
            raise ValueError('private delivery receipt incomplete or mismatched')
        atomic_json(run/'exchange_receipt.json',{**pending,'status':'complete'})
        atomic_json(run/'exchange_pending.json',{**pending,'status':'complete'})
    except Exception as error:
        atomic_json(run/'exchange_pending.json',{**pending,'status':'failed','error_type':type(error).__name__})
        raise
    return result['exit_status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',default='luke,metaentail')
    parser.add_argument('--mode',choices=('tiny','full','both'),default='both')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    parser.add_argument('--asset-root',type=Path,default=ROOT/'.external/assets')
    parser.add_argument('--luke-source-cache',type=Path,default=ROOT/luke_lock()['source']['cache_dir'])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--upload',action='store_true')
    parser.add_argument('--upload-only',type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 7200:
        parser.error('timeout must be between1 and7200 seconds')
    if args.upload_only and (args.describe or args.output_dir or args.upload):
        parser.error('--upload-only is exclusive with --describe/--output-dir/--upload')
    try:
        if args.upload_only:
            return upload(args.upload_only.absolute())
        result = describe(models(args.models),args.mode,args.device)
        if args.describe:
            print(json.dumps(result,indent=2))
            return 0
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        run = (args.output_dir or ROOT/'runs/retacred_reference_runtime'/run_id).absolute()
        reject_symlinks(run)
        relative_name(run.name)
        if run.exists():
            raise FileExistsError('preflight output must be a new directory; existing evidence preserved')
        run.mkdir(parents=True)
        result.update(status='running',exit_status=2,source_sha256={name:digest(PACKAGE/name) for name in
            ('runtime_preflight.py','luke_sources.py','luke_runtime_fidelity.py','metaentail_runtime_fidelity.py',
             'luke_foundation.lock.json','luke_assets.lock.json','metaentail_assets.lock.json')})
        args.asset_root = args.asset_root.absolute()
        args.luke_source_cache = args.luke_source_cache.absolute()
        reject_symlinks(args.asset_root)
        reject_symlinks(args.luke_source_cache)
        execute(args,run,result)
        build_manifest(run)
        print('[runtime] '+result['status']+'; private result: '+str(run/'summary.md'),flush=True)
        if args.upload:
            upload(run)
            print('[runtime] private delivery complete',flush=True)
        return result['exit_status']
    except Exception as error:
        print('[runtime] failed: '+type(error).__name__+': '+redacted(error),file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
