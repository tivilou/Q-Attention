#!/usr/bin/env python3
"""Capture bounded CPU development tests; publish sealed packages from the owner PC.

Trusted, synchronous test recipes only. No shell, background daemon, scheduler,
training resume, or automatic collaborator experiment submission.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import qpersist as q
from cos_store import CosStore, CosError


class Error(Exception):
    pass


def clean_env():
    # Never give the test or Git subprocess cloud credentials from the publisher.
    blocked = ('SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL', 'API_KEY', 'PRIVATE_KEY',
               'SESSION', 'AUTH', 'BEARER')
    result = {k: v for k, v in os.environ.items()
              if not any(word in k.upper() for word in blocked)
              and not k.upper().startswith(('TENCENTCLOUD_', 'AWS_', 'ALIPAN_'))}
    result.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    return result


def git(repo, *args):
    proc = subprocess.run(['git', '-C', str(repo), *args], env=clean_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    if proc.returncode:
        raise Error('Git operation failed; inspect repository privately')
    return proc.stdout.decode('utf-8').strip()


def ident(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', value):
        raise Error('Invalid project identity')
    return value


def recipe_load(path):
    data = q.json_file(path)
    if set(data) != {'schema', 'project_id', 'kind', 'timeout_seconds', 'command',
                     'include', 'outputs', 'environment_file'} or data['schema'] != 'research.dev-test.recipe.v1':
        raise Error('Invalid recipe schema')
    ident(data['project_id'])
    if data['kind'] not in {'unit', 'smoke', 'diagnostic', 'preflight', 'audit'}:
        raise Error('Development/test kind required')
    if type(data['timeout_seconds']) is not int or not 1 <= data['timeout_seconds'] <= 900:
        raise Error('Timeout must be 1..900 seconds')
    command = data['command']
    if not isinstance(command, list) or not command or command[0] != '{python}' or not all(isinstance(x, str) and x and '\x00' not in x for x in command):
        raise Error('Explicit Python argv required')
    for key in ('include', 'outputs'):
        if not isinstance(data[key], list) or not all(isinstance(x, str) for x in data[key]) or len(set(data[key])) != len(data[key]):
            raise Error('Explicit unique file paths required')
        for name in data[key]:
            q.safe_relative(name)
    q.safe_relative(data['environment_file'])
    if not data['include'] or data['environment_file'] not in data['include']:
        raise Error('Include the environment definition')
    return data


def capture(repo, recipe_path, out):
    if os.name != 'posix':
        raise Error('Test capture requires Linux process groups; publication supports Windows')
    recipe = recipe_load(recipe_path)
    repo = Path(repo).resolve(strict=True)
    out = Path(out).absolute()
    if out == repo or repo in out.parents:
        raise Error('Attempt directory must be outside the source checkout')
    if git(repo, 'status', '--porcelain'):
        raise Error('Commit selected code first; source checkout must be clean')
    commit = git(repo, 'rev-parse', 'HEAD')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise Error('Full Git commit required')
    # Only explicitly selected regular tracked files, not whole directories/home.
    for name in recipe['include']:
        q.regular(repo, name)
        if git(repo, 'ls-files', '--error-unmatch', '--', name) != name:
            raise Error('Selected input must be one tracked file')
    out.mkdir(parents=True, exist_ok=False)
    payload = out / 'payload'
    (payload / 'test/output').mkdir(parents=True)
    for name in recipe['include']:
        target = payload / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy exact committed blobs, independently of later working-tree writes.
        proc = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{name}'],
                              env=clean_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if proc.returncode:
            raise Error('Cannot read committed input')
        q.write_new(target, proc.stdout)
        target.chmod(0o700 if (repo / name).stat().st_mode & 0o111 else 0o600)
        # Reject a clean-looking checkout with filters/skip-worktree differences.
        if q.digest(target) != q.digest(repo / name):
            raise Error('Working input differs from committed bytes')
    q.write_new(payload / 'test/recipe.json', q.encoded(recipe))
    q.write_new(payload / 'test/python-environment.json', q.encoded(dict(
        python_version=sys.version.split()[0],
        packages=sorted((dict(name=d.metadata['Name'], version=d.version)
                         for d in importlib.metadata.distributions()), key=lambda x: (x['name'] or '', x['version'])))))
    output = payload / 'test/output'
    command = [x.replace('{python}', sys.executable).replace('{output}', str(output))
               for x in recipe['command']]
    env = clean_env()
    env['RESEARCH_OUTPUT_DIR'] = str(output)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    clock = time.monotonic()
    timed_out = False
    with open(payload / 'test/stdout.txt', 'xb') as stdout, open(payload / 'test/stderr.txt', 'xb') as stderr:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=stdout,
                                stderr=stderr, start_new_session=True)
        try:
            try:
                code = proc.wait(timeout=recipe['timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait(timeout=10)
        finally:
            # Close any ordinary background descendants before snapshotting.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if git(repo, 'rev-parse', 'HEAD') != commit or git(repo, 'status', '--porcelain'):
        raise Error('Test modified the source checkout; attempt remains unsealed')
    status = 'timed_out' if timed_out else ('passed' if code == 0 else 'failed')
    for name in recipe['outputs']:
        q.regular(output, name)
    info = dict(schema='research.dev-test.capture.v1', attempt=uuid.uuid4().hex,
                project_id=recipe['project_id'], kind=recipe['kind'], code_commit=commit,
                environment_sha256=q.digest(payload / 'code' / recipe['environment_file']),
                recipe_sha256=q.digest(payload / 'test/recipe.json'), test_status=status,
                returncode=code, started_utc=started, duration_seconds=round(time.monotonic() - clock, 3),
                python_version=sys.version.split()[0], cpu_only=True)
    q.write_new(payload / 'capture.json', q.encoded(info))
    metadata = dict(kind='closed_run', code_commit=commit,
                    environment_sha256=info['environment_sha256'], dataset_sha256={})
    includes = ['capture.json', 'test/recipe.json', 'test/python-environment.json', 'test/stdout.txt', 'test/stderr.txt']
    includes += ['code/' + x for x in recipe['include']]
    includes += ['test/output/' + x for x in recipe['outputs']]
    q.inventory(payload, includes, metadata, out / 'inventory.json', True)
    package = q.pack(payload, out / 'inventory.json', out / 'package')
    sealed = dict(schema='research.dev-test.sealed.v1', capture=info, **package)
    q.write_new(out / 'SEALED.json', q.encoded(sealed))
    return sealed


@contextmanager
def lock(path):
    path = Path(path)
    with open(path, 'a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Error('Another publisher holds this lock') from None
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if q.json_file(path) != value:
            raise Error('Immutable metadata conflict')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix='.dev-archive-', dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(q.encoded(value))
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise Error('Metadata target appeared concurrently')
            os.rename(temporary, path)
            q.sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def index_prepare(repo, relative):
    if git(repo, 'symbolic-ref', '--short', 'HEAD') != 'main':
        raise Error('Use a dedicated private index main checkout')
    changed = git(repo, 'diff', '--name-only').splitlines() + git(repo, 'diff', '--cached', '--name-only').splitlines()
    if any(name != relative for name in changed):
        raise Error('Index has unrelated uncommitted tracked changes')
    untracked = git(repo, 'ls-files', '--others', '--exclude-standard').splitlines()
    if any(name != relative for name in untracked):
        raise Error('Index has unrelated untracked files')
    git(repo, 'fetch', 'origin', 'main')
    ahead = git(repo, 'rev-list', 'origin/main..HEAD').splitlines()
    if ahead:
        # Recover exactly our single pending commit after a failed push/crash.
        if len(ahead) != 1 or git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines() != [relative]:
            raise Error('Index has unrelated unpublished history')
        # Never rebase or overwrite concurrently published metadata automatically.
        if git(repo, 'merge-base', 'HEAD', 'origin/main') != git(repo, 'rev-parse', 'origin/main'):
            raise Error('Concurrent index update; reconcile the pending commit privately')
    elif git(repo, 'rev-parse', 'HEAD') != git(repo, 'rev-parse', 'origin/main'):
        git(repo, 'merge', '--ff-only', 'origin/main')


def commit_index(repo, relative, record):
    path = repo / relative
    immutable_json(path, record)
    git(repo, 'add', '--', relative)
    staged = git(repo, 'diff', '--cached', '--name-only').splitlines()
    if staged:
        if staged != [relative]:
            raise Error('Refuse unrelated staged index changes')
        git(repo, 'commit', '-m', 'Record development-test archive ' + record['attempt'])
    git(repo, 'push', 'origin', 'HEAD:refs/heads/main')
    git(repo, 'fetch', 'origin', 'main')
    head = git(repo, 'rev-parse', 'HEAD')
    if git(repo, 'merge-base', head, 'origin/main') != head:
        raise Error('Remote index does not contain the archive commit')
    return head


def seal_check(attempt):
    sealed = q.json_file(attempt / 'SEALED.json')
    manifest = q.json_file(attempt / 'package/manifest.json')
    q.validate_manifest(manifest)
    info = sealed['capture']
    fields = {'schema', 'attempt', 'project_id', 'kind', 'code_commit', 'environment_sha256',
              'recipe_sha256', 'test_status', 'returncode', 'started_utc', 'duration_seconds',
              'python_version', 'cpu_only'}
    if (sealed.get('schema') != 'research.dev-test.sealed.v1'
            or set(info) != fields
            or info.get('schema') != 'research.dev-test.capture.v1'
            or not q.SNAP.fullmatch(info['attempt'])
            or info['kind'] not in {'unit', 'smoke', 'diagnostic', 'preflight', 'audit'}
            or info['test_status'] not in {'passed', 'failed', 'timed_out'}
            or info['cpu_only'] is not True
            or type(info['returncode']) is not int
            or (info['test_status'] == 'passed') != (info['returncode'] == 0)
            or sealed['snapshot'] != manifest['snapshot']
            or sealed['manifest_sha256'] != q.digest(attempt / 'package/manifest.json')
            or info['code_commit'] != manifest['metadata']['code_commit']
            or info['environment_sha256'] != manifest['metadata']['environment_sha256']):
        raise Error('Invalid seal or package identity')
    ident(info['project_id'])
    for chunk in manifest['chunks']:
        local = q.regular(attempt / 'package', chunk['name'])
        if q.digest(local) != chunk['sha256'] or local.stat().st_size != chunk['size']:
            raise Error('Corrupt sealed package')
    return sealed


def publish(attempt, index, store_factory=CosStore):
    attempt, index = Path(attempt).resolve(strict=True), Path(index).resolve(strict=True)
    sealed = seal_check(attempt)
    info = sealed['capture']
    relative = 'projects/' + info['project_id'] + '/snapshots/dev-' + info['attempt'] + '.json'
    # Dedicated trusted checkout, private visibility verified by the operator.
    git_dir = Path(git(index, 'rev-parse', '--absolute-git-dir'))
    with lock(attempt / '.publish.lock'), lock(git_dir / 'dev-archive.lock'):
        index_prepare(index, relative)
        config = q.json_file(index / 'projects' / info['project_id'] / 'deployment.json')
        storage = q.json_file(index / 'storage.json')
        if config.get('project_id') != info['project_id'] or config.get('storage_namespace') != 'projects/' + info['project_id']:
            raise Error('Index project namespace mismatch')
        if config.get('cos_bucket') != storage.get('bucket') or config.get('cos_region') != storage.get('region'):
            raise Error('Index storage location mismatch')
        location = (config['cos_bucket'], config['cos_region'], 'projects/' + info['project_id'] + '/dev-tests/')
        receipt_path = attempt / 'publish-receipt.json'
        if receipt_path.exists():
            receipt = q.json_file(receipt_path)
            if (receipt.get('snapshot') != sealed['snapshot']
                    or receipt.get('manifest_sha256') != sealed['manifest_sha256']
                    or receipt.get('status') != 'remote_readback_verified'):
                raise Error('Existing publish receipt mismatch')
        else:
            temporary_receipt = attempt / ('.publish-receipt-' + uuid.uuid4().hex + '.json')
            try:
                receipt = q.publish(attempt / 'package', store_factory(*location), temporary_receipt)
                immutable_json(receipt_path, receipt)
            finally:
                temporary_receipt.unlink(missing_ok=True)
        # Fresh pinned reader, independent of the publisher's in-memory state.
        pinned = store_factory(*location, receipt=receipt)
        with tempfile.TemporaryDirectory(prefix='dev-archive-restore-') as tmp:
            restored = Path(tmp) / 'restored'
            restored_receipt = q.restore(pinned, sealed['snapshot'], sealed['manifest_sha256'], restored)
            if q.json_file(restored / 'capture.json') != info:
                raise Error('Sealed test identity differs from restored payload')
        record = dict(schema='research.dev-archive.index.v1', project_id=info['project_id'],
                      attempt=info['attempt'], capture=info, publish_receipt=receipt,
                      independent_restore=restored_receipt,
                      scope='Selected closed owner development-test materials only')
        index_commit = commit_index(index, relative, record)
        complete = dict(schema='research.dev-archive.complete.v1', status='archived',
                        test_status=info['test_status'], snapshot=sealed['snapshot'],
                        manifest_sha256=sealed['manifest_sha256'], index_path=relative,
                        index_commit=index_commit)
        # A later fast-forward may change checkout HEAD; preserve the first receipt.
        existing = attempt / 'ARCHIVED.json'
        if existing.exists():
            previous = q.json_file(existing)
            if {k: v for k, v in previous.items() if k != 'index_commit'} != {k: v for k, v in complete.items() if k != 'index_commit'}:
                raise Error('Archive completion conflict')
            return previous
        immutable_json(existing, complete)
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    cap = sub.add_parser('capture', help='Linux worker: test and seal; no cloud credentials')
    cap.add_argument('--repo', required=True)
    cap.add_argument('--recipe', required=True)
    cap.add_argument('--out', required=True)
    pub = sub.add_parser('publish', help='Owner: COS, independent restore, private Git index')
    pub.add_argument('--attempt', required=True)
    pub.add_argument('--index', required=True)
    args = parser.parse_args()
    try:
        if args.operation == 'capture':
            result = capture(args.repo, args.recipe, args.out)
            print(q.encoded(result).decode(), end='')
            return 0 if result['capture']['test_status'] == 'passed' else 2
        result = publish(args.attempt, args.index)
        print(q.encoded(result).decode(), end='')
        return 0 if result['test_status'] == 'passed' else 2
    except (Error, q.Error, CosError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print('dev_archive: incomplete; preserve attempt and retry publication after checking private diagnostics.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
