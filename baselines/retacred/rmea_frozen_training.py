"""Shared frozen-feature lifecycle for the existing scattering and protected-context studies.

Study-specific heads, measurements, replay, unit checks and report semantics are
explicit runtime callbacks. Checkpoint/resume and parent/child fencing stay here.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[2]
INPUT_RUN=Path('/hy-tmp/Q-Attention/worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13/discovery')

def require(ok,message):
    if not ok:raise ValueError(message)


class FrozenFeatureTraining:
    def __init__(self,protocol,runtime):
        if runtime.protocol is not protocol:
            raise ValueError('Runtime/protocol mismatch')
        self.p=protocol
        self.runtime=runtime

    def unit(self,out):
        return self.runtime.unit(out)

    def replay(self,output,condition,split):
        self.prepared(output)
        return self.runtime.replay(output,condition,split)

    def execution_environment(self):
        import platform, numpy, torch
        return dict(python=platform.python_version(), implementation=platform.python_implementation(), numpy=numpy.__version__, torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(), cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))

    def metadata_temporary(self, f, root):
        """Interrupted atomic metadata writes are non-authoritative and kept as evidence."""
        import re
        rel = f.relative_to(root).as_posix()
        names = ('run-identity.json', '.preparing.json', 'handoff.json', 'discovery/manifest.json', 'recovery-inputs/input-recovery.json')
        return not f.is_symlink() and f.is_file() and (f.stat().st_size <= 2 * 1024 ** 2) and any((re.fullmatch(re.escape(n) + '\\.[a-z0-9_]{8}', rel) for n in names))

    def fresh_attempt(self, folder):
        require(folder.is_dir(), 'Existing scheduler output directory required')
        names = {f.name for f in folder.iterdir() if f.name != '.run.lock' and (not self.metadata_temporary(f, folder))}
        if not names:
            return
        require(names == {'progress.json'}, 'Preserve existing attempt output')
        row = self.p.read(folder / 'progress.json')
        expected = Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name
        require(row.get('schema') == 'research.progress.v1' and row.get('attempt') == expected and (row.get('phase') == 'experiment') and (type(row.get('current')) is int) and (row['current'] == 0), 'Existing progress is not an initial scheduler record')

    def sync_directory(self, directory):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def verify_installed_pin(self, folder, root, profile):
        phase = self.p.read(folder / 'restore-install.json')
        pin = self.p.read(root / 'restore-pin.json')
        require(phase['status'] == 'complete' and phase['pin_sha256'] == self.p.digest(root / 'restore-pin.json') and (phase['run_id'] == root.name) and (phase['profile'] == profile) and (phase['source_sha256'] == self.p.identity(self.p.sources())), 'Pinned installation is incomplete or different')
        for row in pin['acknowledged_generations']:
            directory = root / 'cos-channel' / row['request']['generation']
            require(self.p.read(directory / 'request.json') == row['request'] and self.p.read(directory / 'ack.json') == row['ack'] and (self.p.digest(directory / 'request.json') == row['ack']['request_sha256']), 'Installed pinned prefix differs; use --restore-pin')
        pointer = self.p.read(root / 'cos-channel/cloud-latest.json')
        require(pointer['done'] >= pin['done'], 'Installed cloud pointer precedes pin')
        if pointer['generation'] == pin['generation']:
            require(pointer['request_sha256'] == pin['acknowledged_generations'][-1]['ack']['request_sha256'], 'Pinned cloud pointer differs')

    def durable_inputs(self, root):
        """One preparation/restore barrier, including linked or extracted inputs."""
        for f in root.rglob('*'):
            require(not f.is_symlink(), 'Regular recovery inputs required')
            if f.is_file():
                with f.open('rb') as handle:
                    os.fsync(handle.fileno())
        for d in sorted((f for f in root.rglob('*') if f.is_dir()), key=lambda f: len(f.parts), reverse=True):
            self.sync_directory(d)
        self.sync_directory(root)
        self.sync_directory(root.parent)

    def restore_identity(self, folder, root, profile, *, inject_failure=False):
        """Same identity admission for the formal pinned restore and its tiny canary."""
        from baselines.retacred.rmea_token_transport_parallel import pin_receipts, install_pinned_workers
        output = root / 'discovery'
        m = self.prepared(output)
        cfg = self.p.config(profile)
        handoff = self.p.read(root / 'handoff.json')
        pin = self.p.read(root / 'restore-pin.json')
        require(m['config'] == cfg and handoff['run_id'] == root.name and (handoff['profile'] == profile) and (handoff['config'] == cfg) and (handoff['source_sha256'] == self.p.identity(self.p.sources())), 'Restored run/source/profile differs')
        total = len(self.p.MODES) * cfg['epochs'] * ((cfg['counts']['train'] + 31) // 32)
        latest = output / 'checkpoints/latest.pt'
        best = output / 'checkpoints/best.pt'
        receipts = pin_receipts(pin, run_id=root.name, identity=dict(config=self.p.identity(cfg), source=self.p.identity(self.p.sources())), total=total, sha256=self.p.digest(latest), latest_bytes=latest.stat().st_size, best_sha256=self.p.digest(best), best_bytes=best.stat().st_size)
        intent = dict(run_id=root.name, profile=profile, source_sha256=self.p.identity(self.p.sources()), config_sha256=self.p.identity(cfg))
        phase = folder / 'restore-install.json'
        installation = dict(intent, pin_sha256=self.p.digest(root / 'restore-pin.json'), status='pending')
        if phase.exists():
            require(all((self.p.read(phase).get(k) == value for (k, value) in installation.items() if k != 'status')), 'Different interrupted restore installation')
        else:
            self.p.atomic_json(phase, installation)
        for (index, row) in enumerate(receipts):
            directory = root / 'cos-channel' / row['request']['generation']
            for (name, key) in (('request.json', 'request'), ('ack.json', 'ack')):
                f = directory / name
                if f.exists():
                    require(self.p.read(f) == row[key], 'Different interrupted COS prefix')
                else:
                    self.p.atomic_json(f, row[key])
            require(self.p.digest(directory / 'request.json') == row['ack']['request_sha256'], 'Installed request bytes differ')
            self.sync_directory(directory)
            self.sync_directory(directory.parent)
            self.sync_directory(root)
            if inject_failure and index == 0:
                raise OSError('Injected partially installed pinned receipt interruption')
        last = receipts[-1]
        for f in (latest, best):
            fd = os.open(f, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.sync_directory(latest.parent)
        self.sync_directory(output)
        self.p.atomic_json(root / 'cos-channel/cloud-latest.json', dict(generation=last['request']['generation'], done=last['request']['done'], index_commit=last['ack']['index_commit'], request_sha256=last['ack']['request_sha256']))
        self.sync_directory(root / 'cos-channel')
        self.sync_directory(root)
        install_pinned_workers(output, cfg, self.p)
        self.durable_inputs(root)
        self.p.atomic_json(phase, dict(installation, status='complete'))
        self.sync_directory(folder)
        f = folder / 'run-identity.json'
        if f.exists():
            require(self.p.read(f) == intent, 'Existing outer run identity differs')
        else:
            self.p.atomic_json(f, intent)
        self.sync_directory(folder)
        self.verify_installed_pin(folder, root, profile)
        return intent

    def prepared(self, output):
        """Authenticate non-checkpoint inputs again before local or pinned continuation."""
        output = Path(output)
        if hasattr(self.runtime, 'validate_inputs'):
            self.runtime.validate_inputs(output.parent)
        root = output.parent
        m = self.p.read(output / 'manifest.json')
        cfg = m['config']
        profile = cfg['profile']
        require(cfg == self.p.config(profile), 'Frozen configuration differs')
        require(m['execution_environment'] == self.execution_environment(), 'Pinned execution environment differs')
        source = root / 'recovery-inputs/source-manifest.json'
        original = root / 'recovery-inputs/original-manifest.json'
        require(self.p.digest(original) == self.p.BASE_MANIFEST and self.p.digest(source) == (self.p.BASE_MANIFEST if profile == 'discovery' else self.p.CANARY_MANIFEST), 'Pinned source manifest differs')
        names = ['metadata.json'] + [s + '-' + k + '.npy' for s in cfg['counts'] for k in ('h', 'roles', 'offsets', 'targets')]
        cache = {n: self.p.read(source)['cache'][n] for n in names}
        require(m['cache'] == cache and m['source_sha256'] == self.p.identity(self.p.sources()) and (m['source_hashes'] == self.p.sources()) and (m['git_revision'] == self.p.git('rev-parse', 'HEAD')) and (m['source_input_manifest'] == self.p.BASE_MANIFEST), 'Training source/cache provenance differs')
        require(self.p.digest(root / 'recovery-inputs/labels.json') == self.p.LABELS_SHA, 'Pinned labels differ')
        summary = ROOT / 'reports/q_rmea_token_transport/20261121T034306Z_seed13/summary.json'
        require(self.p.digest(summary) == self.p.BASE_SUMMARY, 'Pinned public reference differs')
        old = self.p.read(summary)
        expected = dict(summary_sha256=self.p.BASE_SUMMARY, result=old['results'][self.p.REFERENCE_METHOD], initial_state_sha256=old['common_initial_state_sha256'], order_sha256=old['common_epoch_order_sha256'], comparison=self.p.REFERENCE_DESCRIPTION)
        require(m['reference'] == expected, 'Historical reference metrics differ')
        ref = root / 'recovery-inputs/input-recovery.json'
        d = self.p.read(ref)
        dep = d['dependency']
        raw = (json.dumps(dep['request'], ensure_ascii=False, indent=2) + '\n').encode()
        require(hashlib.sha256(raw).hexdigest() == self.p.INPUT_REQUEST_SHA and dep['ack']['request_sha256'] == self.p.INPUT_REQUEST_SHA and (self.p.identity(dep['ack']) == self.p.INPUT_ACK_ID) and (set(dep['request']['files']) == set(dep['ack']['objects'])) and (d['cache'] == cache) and (d['source_manifest_sha256'] == self.p.BASE_MANIFEST) and (m['input_recovery_reference_sha256'] == self.p.digest(ref)), 'Immutable input dependency differs')
        require(set((x.name for x in (output / 'cache').iterdir())) == set(names), 'Minimal cache inventory differs')
        if hasattr(self.runtime, 'validate_training_cache'):
            self.runtime.validate_training_cache(output, cache)
        return m

    def initialize(self, root, profile, inputs):
        """Link immutable cache bytes once; never rewrite a historical input/run."""
        root = Path(root).resolve()
        inputs = Path(inputs).resolve()
        cfg = self.p.config(profile)
        require(not (root.is_relative_to(inputs) or inputs.is_relative_to(root) or root.is_relative_to(ROOT)), 'Independent run root required')
        output = root / 'discovery'
        if not (output / 'manifest.json').exists():
            marker = root / '.preparing.json'
            intent = dict(run_id=root.name, profile=profile, source_sha256=self.p.identity(self.p.sources()), config_sha256=self.p.identity(cfg))
            if marker.exists():
                require(self.p.read(marker) == intent, 'Interrupted preparation identity differs')
            else:
                require(not root.exists() or all((f.name == '.lock' or self.metadata_temporary(f, root) for f in root.iterdir())), 'Preparation identity missing')
                self.p.atomic_json(marker, intent)
                self.sync_directory(root)
                self.sync_directory(root.parent)
        require(self.p.digest(inputs / 'manifest.json') == self.p.BASE_MANIFEST, 'Original discovery manifest differs')
        summary = inputs / 'summary.json'
        if not summary.exists():
            summary = ROOT / 'reports/q_rmea_token_transport/20261121T034306Z_seed13/summary.json'
        require(self.p.digest(summary) == self.p.BASE_SUMMARY, 'Pinned baseline summary differs')
        old = self.p.read(summary)
        source = inputs if profile == 'discovery' else inputs.parent / 'canary'
        require(self.p.digest(source / 'manifest.json') == (self.p.BASE_MANIFEST if profile == 'discovery' else self.p.CANARY_MANIFEST), 'Fixture identity differs')
        original = self.p.read(source / 'manifest.json')
        names = ['metadata.json'] + [s + '-' + k + '.npy' for s in cfg['counts'] for k in ('h', 'roles', 'offsets', 'targets')]
        cache = {}
        for n in names:
            f = source / 'cache' / n
            ref = original['cache'][n]
            require(not f.is_symlink() and f.is_file() and (f.stat().st_size == ref['bytes']) and (self.p.digest(f) == ref['sha256']), 'Protected cache differs: ' + n)
            cache[n] = ref
        labels = inputs.parent / 'recovery-inputs/labels.json'
        require(self.p.digest(labels) == self.p.LABELS_SHA, 'Train-derived labels differ')
        req = inputs.parent / 'cos-channel/000-preparation/request.json'
        ack = req.parent / 'ack.json'
        (request, receipt) = (self.p.read(req), self.p.read(ack))
        require(receipt['request_sha256'] == self.p.digest(req) == self.p.INPUT_REQUEST_SHA and self.p.identity(receipt) == self.p.INPUT_ACK_ID and receipt['index_commit'] and (set(receipt['objects']) == set(request['files'])), 'Original input recovery receipt differs')
        for (n, row) in request['files'].items():
            obj = receipt['objects'][n]
            require(obj.get('version_id') and obj['sha256'] == row['sha256'] and (obj['bytes'] == row['bytes']), 'Original pinned input object differs')
        output = root / 'discovery'
        if (output / 'manifest.json').exists():
            self.prepared(output)
            return output
        allowed = {'handoff.json', '.lock', '.preparing.json'} | {'discovery/cache/' + n for n in names} | {'recovery-inputs/' + n for n in ('labels.json', 'input-recovery.json', 'source-manifest.json', 'original-manifest.json')}
        allowed |= set(getattr(self.p, 'EXTRA_RECOVERY_INPUTS', ()))
        require(not root.exists() or all((f.relative_to(root).as_posix() in allowed or self.metadata_temporary(f, root) for f in root.rglob('*') if f.is_file())), 'Preparation has training or unknown state')
        (output / 'cache').mkdir(parents=True, exist_ok=True)
        (root / 'recovery-inputs').mkdir(exist_ok=True)

        def link(src, dest):
            if dest.exists():
                require(not dest.is_symlink() and self.p.digest(dest) == self.p.digest(src), 'Interrupted preparation bytes differ')
            else:
                os.link(src, dest)
        for n in names:
            link(source / 'cache' / n, output / 'cache' / n)
        link(labels, root / 'recovery-inputs/labels.json')
        link(source / 'manifest.json', root / 'recovery-inputs/source-manifest.json')
        link(inputs / 'manifest.json', root / 'recovery-inputs/original-manifest.json')
        self.p.atomic_json(root / 'recovery-inputs/input-recovery.json', dict(schema=self.p.SCHEMA_PREFIX+'.input-reference.v1', source_manifest_sha256=self.p.BASE_MANIFEST, source_manifest_canary=self.p.CANARY_MANIFEST, baseline_summary=self.p.BASE_SUMMARY, dependency=dict(request=request, ack=receipt), cache=cache, restore='Download pinned original preparation parts to managed storage, verify all bytes, use existing restore_preparation; install only named cache files by identity. Restore new minimal preparation and pinned latest/best; run explicit resume. No raw/model data to owner-PC.'))
        self.p.atomic_json(root / 'handoff.json', dict(schema=self.p.SCHEMA_PREFIX+'.owner-run.v1', run_id=root.name, profile=profile, source_sha256=self.p.identity(self.p.sources()), config=cfg, environment='Existing pinned qattn Python/torch environment; no full environment image', recovery='Full-state local latest/best, global10percent COS; immutable original input reference'))
        if hasattr(self.runtime, 'initialize_inputs'):
            self.runtime.initialize_inputs(root, inputs)
        self.durable_inputs(root)
        self.p.atomic_json(output / 'manifest.json', dict(schema=self.p.SCHEMA_PREFIX+'.manifest.v1', config=cfg, config_sha256=self.p.identity(cfg), execution_environment=self.execution_environment(), git_revision=self.p.git('rev-parse', 'HEAD'), source_sha256=self.p.identity(self.p.sources()), source_hashes=self.p.sources(), cache=cache, source_input_manifest=self.p.BASE_MANIFEST, canary_input_manifest=self.p.CANARY_MANIFEST if profile == 'canary' else None, input_recovery_reference_sha256=self.p.digest(root / 'recovery-inputs/input-recovery.json'), reference=dict(summary_sha256=self.p.BASE_SUMMARY, result=old['results'][self.p.REFERENCE_METHOD], initial_state_sha256=old['common_initial_state_sha256'], order_sha256=old['common_epoch_order_sha256'], comparison=self.p.REFERENCE_DESCRIPTION)))
        self.sync_directory(output)
        self.prepared(output)
        return output

    def worker(self, args):
        require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) == 0, 'Parent-death fencing failed')
        require(os.getppid() == args.parent_pid, 'Parent identity changed')
        import torch
        from baselines.retacred.rmea_token_transport_parallel import condition_recovery
        from baselines.retacred.rmea_token_transport_recovery import run_lock
        from baselines.retacred.rmea_token_transport_study import train
        require(torch.cuda.device_count() == 1 and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'One allocated GPU required')
        output = args.output.resolve()
        manifest = self.p.read(output / 'manifest.json')
        cfg = manifest['config']
        require(cfg == self.p.config(cfg['profile']), 'Frozen config differs')
        require(manifest['execution_environment'] == self.execution_environment(), 'Worker execution environment differs')
        folder = output / '.workers' / args.condition
        with run_lock(folder / '.lock'):
            assignment = self.p.read(folder / 'assignment.json')
            require(assignment == dict(token=args.assignment_token, parent_pid=args.parent_pid, condition=args.condition, device=os.environ['CUDA_VISIBLE_DEVICES'], source=self.p.identity(self.p.sources())), 'Active parent assignment differs')
            require(not (output / 'RUN_COMPLETE').exists() and (not (folder / 'WORKER_COMPLETE').exists()), 'Preserve sealed run/condition')
            require(args.resume or not (folder / 'latest.pt').exists(), 'Explicit worker resume required')
            recovery = condition_recovery(output, cfg, args.condition, protocol=self.p, resume=args.resume)

            def pause(*_):
                recovery.stop = True
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, pause)
            result = train(cfg, output, self.p.read(output.parent / 'recovery-inputs/labels.json'), recovery=recovery, modes=(args.condition,), finalize=False, runtime=self.runtime)
            self.p.atomic_json(folder / 'result.json', result)
            (folder / 'RUN_PAUSED').unlink(missing_ok=True)
            self.sync_directory(output / self.p.ANALYSIS_DIRECTORY)
            self.sync_directory(output)
            self.sync_directory(folder)
            self.sync_directory(folder.parent)
            self.p.atomic_json(folder / 'WORKER_COMPLETE', dict(result_sha256=self.p.digest(folder / 'result.json'), terminal_sha256=self.p.digest(folder / 'latest.pt'), checkpoint_sha256=self.p.digest(output / 'checkpoints' / (args.condition + '-final.pt')), source=self.p.identity(self.p.sources()), condition=args.condition, **self.runtime.worker_receipt(output,args.condition)))
            self.sync_directory(folder)

    def parallel(self, output, devices, *, resume=False, pause_after=0, restore_pin=False):
        from baselines.retacred.rmea_token_transport_parallel import parallel_train
        m = self.prepared(output)
        cfg = m['config']
        require(cfg == self.p.config(cfg['profile']) and m['source_sha256'] == self.p.identity(self.p.sources()), 'Run config/source differs')
        parallel_train(cfg, output, self.p.read(output.parent / 'recovery-inputs/labels.json'), devices, cos=False, resume=resume, pause_after=pause_after, restore_pin=restore_pin, runtime=self.runtime)

    def require_unit(self, folder):
        require(folder is not None, 'Explicit completed unit output required')
        a = self.p.read(folder / 'unit-validation.json')
        b = self.p.read(folder / 'RUN_COMPLETE')
        require(a.get('gate') is True and a['source_sha256'] == self.p.identity(self.p.sources()) and (a['contract_sha256'] == self.p.identity(self.p.contract())) and (b['artifacts']['unit-validation.json'] == self.p.digest(folder / 'unit-validation.json')), 'Unit identity/success differs')

    def cloud_canary(self, target, devices, output):
        """Publish one paused aggregate, fetch pinned bytes to a fresh directory, resume."""
        import math, shutil, urllib.request, urllib.parse
        import torch
        from baselines.retacred.rmea_token_transport_recovery import Recovery, restore_preparation
        r = Recovery(target, self.p.config('canary'), enabled=True, protocol=self.p)
        r.state = torch.load(r.latest, map_location='cpu', weights_only=True)
        done = r.state['done']
        require(math.ceil(r.total / 10) <= done < r.total, 'Nonterminal committed COS canary required')
        self.runtime.prepare(r, self.p.read(r.parent / 'recovery-inputs/labels.json'), None)
        directory = r.channel / '010-progress'
        directory.mkdir()
        for f in (r.latest, r.best):
            os.link(f, directory / f.name)
        r._publish('010-progress', [directory / 'latest.pt', directory / 'best.pt'], done)
        receipts = []
        downloads = output / 'pinned-downloads'
        downloads.mkdir()
        for name in ('000-preparation', '010-progress'):
            request = self.p.read(r.channel / name / 'request.json')
            ack = self.p.read(r.channel / name / 'ack.json')
            receipts.append(dict(request=request, ack=ack))
            for (rel, row) in request['files'].items():
                obj = ack['objects'][rel]
                url = ack['restore_urls'][rel]
                u = urllib.parse.urlsplit(url)
                require(u.scheme == 'https' and urllib.parse.parse_qs(u.query).get('versionId') == [obj['version_id']], 'Version-pinned HTTPS restore required')
                f = downloads / Path(rel).name
                require(not f.exists(), 'Unique download role required')
                size = 0
                h = hashlib.sha256()
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=90) as src, f.open('xb') as dst:
                    while True:
                        b = src.read(1024 ** 2)
                        if not b:
                            break
                        size += len(b)
                        require(size <= row['bytes'], 'Restore exceeds sealed size')
                        h.update(b)
                        dst.write(b)
                require(size == row['bytes'] and h.hexdigest() == row['sha256'], 'Pinned restored bytes differ')
        restored = output / 'restored' / r.parent.name
        restore_preparation(sorted(downloads.glob('*.tarpart')), restored)
        dest = restored / 'discovery'
        (dest / 'checkpoints').mkdir()
        for n in ('latest.pt', 'best.pt'):
            shutil.copyfile(downloads / n, dest / 'checkpoints' / n)
        obj = receipts[-1]['ack']['objects']['cos-channel/010-progress/latest.pt']
        self.p.atomic_json(restored / 'restore-pin.json', dict(run_id=r.parent.name, identity=r.identity, total=r.total, done=done, generation='010-progress', sha256=obj['sha256'], version_id=obj['version_id'], acknowledged_generations=receipts))
        from baselines.retacred.rmea_token_transport_recovery import run_lock
        with run_lock(restored.parent / '.run.lock'):
            try:
                self.restore_identity(restored.parent, restored, 'canary', inject_failure=True)
            except OSError:
                pass
            else:
                raise ValueError('Pinned install interruption not injected')
            require(not (restored.parent / 'run-identity.json').exists() and self.p.read(restored.parent / 'restore-install.json')['status'] == 'pending', 'Interrupted restore admitted too early')
            self.restore_identity(restored.parent, restored, 'canary')
        cloud = Recovery(dest, self.p.config('canary'), enabled=True, protocol=self.p)
        before = self.p.digest(restored / 'cos-channel/000-preparation/request.json')
        self.runtime.prepare(cloud, self.p.read(restored / 'recovery-inputs/labels.json'), None)
        cloud.drain_pending()
        cloud.verify_cloud_history()
        require(self.p.digest(restored / 'cos-channel/000-preparation/request.json') == before, 'Restored preparation was repackaged')
        with run_lock(restored.parent / '.run.lock'):
            self.verify_installed_pin(restored.parent, restored, 'canary')
            self.parallel(dest, devices, resume=True)
        return (dest, dict(generation='010-progress', done=done, total=r.total, objects=sum((len(x['request']['files']) for x in receipts)), bytes=sum((x['bytes'] for v in receipts for x in v['request']['files'].values())), interrupted_pin_install_retry=True, enabled_preparation_reused=True, scope='Live COS VersionId/hash download, interrupted prefix installation retry, enabled preparation reuse and fresh-directory local continuation on the same managed node; not a different-provider drill'))

    def canary(self, args):
        """Serial/parallel pause-resume uses real tiny cache; no full-data training."""
        import torch
        from baselines.retacred.orchestrate import gpu_ids
        from baselines.retacred.rmea_token_transport_parallel import equal_state
        from experiments.q_rmea_query_alignment import fresh_output
        self.require_unit(args.unit_output)
        fresh_output(args.output)
        require(bool(os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()), 'Explicit scheduler GPU allocation required')
        devices = gpu_ids('auto')
        require(len(devices) == 3, 'Three-GPU group required')
        suffix = args.output.parent.name
        serial = self.initialize(args.output / ('serial-' + suffix), 'canary', args.inputs)
        self.parallel(serial, devices[:1])
        target = self.initialize(args.output / ('canary-' + suffix), 'canary', args.inputs)
        try:
            self.parallel(target, devices, pause_after=1)
        except SystemExit as exc:
            require(exc.code == 75, 'Expected safe pause')
        else:
            raise ValueError('Pause fault did not execute')
        paused = []
        for kind in self.p.MODES:
            x = torch.load(target / '.workers' / kind / 'latest.pt', map_location='cpu', weights_only=True)
            paused.append(x['done'])
            require(not x['done'] or x['optimizer'] is not None or x['completed'], 'Paused state must preserve optimizer or completed work')
        require(0 < sum(paused) < 9 and any((0 < n < 3 for n in paused)), 'Pause must exercise actual optimizer continuation')
        (restored, cloud) = self.cloud_canary(target, devices, args.output)
        self.parallel(target, devices, resume=True)
        for kind in self.p.MODES:
            for part in ('latest.pt', 'best.pt'):
                a = torch.load(serial / '.workers' / kind / part, map_location='cpu', weights_only=True)
                b = torch.load(target / '.workers' / kind / part, map_location='cpu', weights_only=True)
                keys = ('active', 'optimizer', 'completed', 'best', 'cursor', 'done', 'initial_sha', 'order_sha') if part == 'latest.pt' else None
                require(equal_state({k: a[k] for k in keys}, {k: b[k] for k in keys}) if keys else equal_state(a, b), 'Serial/parallel resumed scientific state differs')
                c = torch.load(restored / '.workers' / kind / part, map_location='cpu', weights_only=True)
                require(equal_state({k: a[k] for k in keys}, {k: c[k] for k in keys}) if keys else equal_state(a, c), 'Pinned COS resumed scientific state differs')
        replays = self.runtime.replay_all(target)
        public = args.output / 'canary-report'
        before = {n: self.p.digest(target / n) for n in ('manifest.json', 'summary.json', 'RUN_COMPLETE')}
        try:
            self.runtime.export(target, public, inject_failure=True)
        except OSError:
            pass
        else:
            raise ValueError('Exporter fault was not injected')
        require(not public.exists() and (not list(public.parent.glob('.' + public.name + '-*'))), 'Failed export residue')
        export = self.runtime.export(target, public)
        require(self.runtime.export(target, public) == export, 'Existing installed report retry differs')
        require(before == {n: self.p.digest(target / n) for n in before}, 'Export retried with mutated run')
        self.p.atomic_json(args.output / 'canary-validation.json', dict(gate=True, source_sha256=self.p.identity(self.p.sources()), contract_sha256=self.p.identity(self.p.contract()), serial_parallel_resume_equal=True, replays=replays, export_retry=True, report=export, cos_live_pinned_restore=True, cloud_restore=cloud, input_dependency_versions_verified=False, scope='Local and live COS continuation passed; original full-input dependency verification is separate'))
        self.p.atomic_json(args.output / 'RUN_COMPLETE', dict(artifacts={'canary-validation.json': self.p.digest(args.output / 'canary-validation.json')}))

    def run_location(self, args):
        import re
        run_id = args.run_id or self.p.EXPERIMENT+'-seed13-' + args.output.parent.name
        require(re.fullmatch(re.escape(self.p.EXPERIMENT)+r'-seed13-[A-Za-z0-9_-]+', run_id) is not None, 'Safe unique run ID required')
        return args.output / run_id / 'discovery'

    def deliver(self, args):
        output = self.run_location(args)
        self.prepared(output)
        report = self.runtime.export(output, args.output / (self.p.EXPERIMENT+'-safe-report'))
        record = dict(run=str(output), report=report, source_sha256=self.p.identity(self.p.sources()))
        final = args.output / 'run-complete.json'
        if final.exists():
            require(self.p.read(final) == record, 'Existing delivery identity differs')
        else:
            self.p.atomic_json(final, record)
        return record

    def preserve_admission(self, root, args):
        """Keep only the compact already-verified gates needed for replacement-node CLI."""
        sources = {'unit/unit-validation.json': args.unit_output / 'unit-validation.json', 'unit/RUN_COMPLETE': args.unit_output / 'RUN_COMPLETE'}
        sources.update({'canary/' + n: args.admission.parent / n for n in ('admission.json', 'dependency-verification.json', 'canary-validation.json', 'RUN_COMPLETE')})
        for (name, source) in sources.items():
            target = root / 'recovery-inputs/admission' / name
            data = source.read_bytes()
            require(len(data) < 2 * 1024 ** 2, 'Compact admission required')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.read_bytes() == data, 'Protected admission differs')
            else:
                temp = target.with_name('.' + target.name + '.installing')
                with temp.open('wb') as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
        self.durable_inputs(root / 'recovery-inputs')

    def run(self, args):
        from baselines.retacred.orchestrate import gpu_ids
        from baselines.retacred.rmea_token_transport_parallel import parallel_train
        from baselines.retacred.rmea_token_transport_recovery import run_lock
        from experiments.q_rmea_query_alignment import fresh_output
        self.require_unit(args.unit_output)
        require(args.admission is not None, 'Exact canary/COS admission required')
        require(args.admission_sha256 == self.p.digest(args.admission), 'Owner-approved admission digest differs')
        a = self.p.read(args.admission)
        validation = args.admission.parent / 'canary-validation.json'
        seal = args.admission.parent / 'RUN_COMPLETE'
        require(self.p.read(seal) == dict(artifacts={'canary-validation.json': self.p.digest(validation)}) and a.get('canary_validation_sha256') == self.p.digest(validation) and (a.get('canary_seal_sha256') == self.p.digest(seal)), 'Completed canary admission seal differs')
        c = self.p.read(validation)
        require(all((a.get(k) == c.get(k) for k in ('source_sha256', 'contract_sha256', 'serial_parallel_resume_equal', 'export_retry', 'cos_live_pinned_restore'))), 'Canary gate differs from admission')
        require(a.get('source_sha256') == self.p.identity(self.p.sources()) and a.get('contract_sha256') == self.p.identity(self.p.contract()) and (a.get('serial_parallel_resume_equal') is True) and (a.get('export_retry') is True) and (a.get('cos_live_pinned_restore') is True) and (a.get('input_dependency_versions_verified') is True) and (a.get('input_request_sha256') == self.p.INPUT_REQUEST_SHA) and (a.get('input_acknowledgement_sha256') == self.p.INPUT_ACK_ID), 'Formal training admission incomplete')
        depfile = args.admission.parent / 'dependency-verification.json'
        dep = self.p.read(depfile)
        require(a.get('dependency_verification_sha256') == self.p.digest(depfile) and dep.get('verified') is True and (dep.get('request_sha256') == self.p.INPUT_REQUEST_SHA) and (dep.get('acknowledgement_sha256') == self.p.INPUT_ACK_ID) and (dep.get('objects') == 13) and (dep.get('total_pinned_bytes') == 6947440640) and (dep.get('payload_bytes_read') == 0), 'Input dependency acceptance record differs')
        require(bool(os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()), 'Explicit scheduler GPU allocation required')
        devices = gpu_ids('auto')
        require(len(devices) == 3, 'Three allocated GPUs required')
        output = self.run_location(args)
        root = output.parent
        identity = args.output / 'run-identity.json'
        intent = dict(run_id=root.name, profile='discovery', source_sha256=self.p.identity(self.p.sources()), config_sha256=self.p.identity(self.p.config('discovery')))
        if args.resume and (not args.restore_pin):
            require(identity.is_file() and self.p.read(identity) == intent, 'Resume requires an existing identified run')
        elif args.resume:
            require(root.is_dir(), 'Pinned restore requires an existing restored run')
        with run_lock(args.output / '.run.lock'):
            if args.resume and args.restore_pin:
                require(self.restore_identity(args.output, root, 'discovery') == intent, 'Pinned identity differs')
            elif args.resume:
                require(self.p.read(identity) == intent, 'Run identity changed')
                if (args.output / 'restore-install.json').exists():
                    self.verify_installed_pin(args.output, root, 'discovery')
            else:
                self.fresh_attempt(args.output)
                self.p.atomic_json(identity, intent)
                self.sync_directory(args.output)
            root.mkdir(exist_ok=True)
            if not (output / 'manifest.json').exists():
                self.initialize(root, 'discovery', args.inputs)
            require(self.prepared(output)['config']['profile'] == 'discovery', 'Formal run requires discovery profile')
            self.preserve_admission(root, args)
            if (output / 'RUN_COMPLETE').exists():
                self.deliver(args)
                return
            parallel_train(self.p.config('discovery'), output, self.p.read(root / 'recovery-inputs/labels.json'), devices, cos=True, resume=args.resume, restore_pin=False, runtime=self.runtime)
            self.deliver(args)

    def main(self):
        a = argparse.ArgumentParser(description=__doc__)
        a.add_argument('action', choices=('unit', 'canary', 'run', 'worker', 'replay', 'export'))
        a.add_argument('--output', type=Path, required=True)
        a.add_argument('--inputs', type=Path, default=INPUT_RUN)
        a.add_argument('--unit-output', type=Path)
        a.add_argument('--admission', type=Path)
        a.add_argument('--admission-sha256')
        a.add_argument('--condition', choices=getattr(self.p, 'TRACE_MODES', self.p.MODES))
        a.add_argument('--split', choices=('train', 'valid', 'test'))
        a.add_argument('--run-id')
        a.add_argument('--parent-pid', type=int)
        a.add_argument('--assignment-token')
        a.add_argument('--resume', action='store_true')
        a.add_argument('--restore-pin', action='store_true')
        args = a.parse_args()
        require(not args.restore_pin or (args.action == 'run' and args.resume), '--restore-pin requires run --resume')
        self.p.review_gate(action=args.action)
        if args.action == 'worker':
            self.worker(args)
        elif args.action == 'unit':
            self.unit(args.output)
        elif args.action == 'canary':
            self.canary(args)
        elif args.action == 'replay':
            print(json.dumps(self.replay(args.output, args.condition, args.split)))
        elif args.action == 'export':
            from baselines.retacred.rmea_token_transport_recovery import run_lock
            with run_lock(args.output / '.run.lock'):
                print(json.dumps(self.deliver(args)))
        else:
            self.run(args)
