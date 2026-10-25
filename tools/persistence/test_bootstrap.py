"""Exercise bundle recovery without network or dependency downloads."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux' or sys.version_info[:2] != (3, 10), reason='Linux Python 3.10 bootstrap')


@pytest.fixture
def bundle_case(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Bootstrap Test')
    git('config', 'user.email', 'bootstrap@example.invalid')
    (source / 'payload.txt').write_text('verified code\n')
    git('add', 'payload.txt')
    git('commit', '-qm', 'fixture')
    commit = git('rev-parse', 'HEAD')
    bundle = tmp_path / 'code.bundle'
    git('bundle', 'create', str(bundle), '--all')
    # Stop at the environment step. Real Git operations and hashes are exercised;
    # the independent deployment drill installs the real pinned environment.
    python_probe = tmp_path / 'python-probe'
    python_probe.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exec "' + sys.executable + '" "$@"; fi\nexit 79\n')
    python_probe.chmod(0o700)
    git_bin = shutil.which('git')
    wrappers = tmp_path / 'bin'
    wrappers.mkdir()
    git_probe = wrappers / 'git'
    git_probe.write_text('#!/bin/sh\nif [ "$1" = "clone" ]; then exit 88; fi\nexec "' + git_bin + '" "$@"\n')
    git_probe.chmod(0o700)
    scripts = Path(__file__).parent
    env = dict(os.environ, PYTHON_BIN=str(python_probe),
               EXPECTED_ENV_SHA256=hashlib.sha256((scripts / 'requirements-linux-py310.lock').read_bytes()).hexdigest(),
               EXPECTED_BUNDLE_SHA256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
               PATH=str(wrappers) + os.pathsep + os.environ['PATH'])
    destination = tmp_path / 'deployment'
    def run(revision=commit):
        return subprocess.run(['bash', str(scripts / 'bootstrap.sh'), revision, str(destination), str(bundle)], env=env, capture_output=True, text=True, timeout=20)
    return bundle, commit, destination, run


def test_complete_bundle_reaches_environment_without_clone(bundle_case):
    bundle, commit, destination, run = bundle_case
    result = run()
    assert result.returncode == 79, result.stderr
    assert subprocess.check_output(['git', '-C', str(destination / 'code'), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert (destination / 'code/payload.txt').read_text() == 'verified code\n'
    assert (destination / 'DEPLOYMENT_INCOMPLETE').is_file()
    assert not (destination / 'ENVIRONMENT_INSTALLED').exists()


def test_corrupt_bundle_rejected_before_destination(bundle_case):
    bundle, commit, destination, run = bundle_case
    with bundle.open('ab') as out:
        out.write(b'corruption')
    result = run()
    assert result.returncode == 2
    assert 'Bundle hash mismatch' in result.stderr
    assert not destination.exists()


def test_absent_commit_cannot_certify_bundle_recovery(bundle_case):
    bundle, commit, destination, run = bundle_case
    result = run('0' * 40)
    assert result.returncode != 0
    assert (destination / 'DEPLOYMENT_INCOMPLETE').is_file()
    assert not (destination / 'ENVIRONMENT_INSTALLED').exists()
