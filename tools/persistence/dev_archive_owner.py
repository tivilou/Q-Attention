#!/usr/bin/env python3
"""Owner-only one-command capture -> pull -> COS -> private-index adapter.

Uses an explicitly selected serverctl development binding and strict SSH.
Profile paths/credentials stay private and outside all payloads/repositories.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import uuid

import dev_archive as d
import qpersist as q


def run(argv, timeout=240):
    result = subprocess.run(argv, env=d.clean_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode not in (0, 2):
        raise d.Error('Owner transport failed; preserve the attempt, do not rerun tests blindly')
    return result


def resolve(profile):
    executable = shutil.which('serverctl')
    if not executable:
        raise d.Error('serverctl required on the owner control machine')
    argv = [executable, 'resolve', profile['project_root'], '--role', 'development', '--json']
    if os.name == 'nt':
        # PowerShell literals prevent shell interpolation of paths.
        quote = lambda s: "'" + s.replace("'", "''") + "'"
        script = '& ' + ' '.join(quote(x) for x in argv)
        argv = ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script]
    rows = json.loads(run(argv, 30).stdout)
    selected = [r for r in rows if r['id'] == profile['server_id']]
    if len(selected) != 1 or selected[0].get('auth') != 'key' or not selected[0].get('host_key_fingerprint', '').startswith('SHA256:'):
        raise d.Error('Explicit authorized verified key-auth server required')
    return selected[0]


def flow(profile, retry=None):
    # A retry operates only on the existing sealed package, never the worker.
    if retry:
        attempt = Path(retry).resolve(strict=True)
        d.seal_check(attempt)
    else:
        server = resolve(profile)
        name = uuid.uuid4().hex
        attempt = Path(profile['local_root']).resolve() / name
        attempt.mkdir(parents=True, exist_ok=False)
        remote = profile['remote_root'].rstrip('/') + '/' + name
        recipe = profile['repo'].rstrip('/') + '/' + profile['recipe']
        address = server['user'] + '@' + server['host']
        options = ['-i', server['identity_file'], '-o', 'BatchMode=yes',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15']
        argv = [profile['python'], profile['repo'].rstrip('/') + '/tools/persistence/dev_archive.py',
                'capture', '--repo', profile['repo'], '--recipe', recipe, '--out', remote]
        q.write_new(attempt / 'transport.json', q.encoded(dict(server_id=server['id'],
                    remote_attempt=remote, state='requested; execution may be uncertain after connection loss')))
        response = run(['ssh', '-n', *options, '-p', str(server['port']), address, shlex.join(argv)], 960)
        sealed = json.loads(response.stdout)
        # New destination, not a recursive sync/delete of an existing directory.
        run(['scp', '-r', *options, '-P', str(server['port']),
             address + ':' + remote + '/package', str(attempt)], 240)
        q.write_new(attempt / 'SEALED.json', q.encoded(sealed))
        d.seal_check(attempt)
    # Load cloud credentials only after worker operations; never serialize values.
    secret = q.json_file(Path(profile['credential_file']).resolve(strict=True))
    previous = {k: os.environ.get(k) for k in ('TENCENTCLOUD_SECRET_ID', 'TENCENTCLOUD_SECRET_KEY', 'TENCENTCLOUD_SESSION_TOKEN')}
    try:
        os.environ['TENCENTCLOUD_SECRET_ID'] = secret['SecretId']
        os.environ['TENCENTCLOUD_SECRET_KEY'] = secret['SecretKey']
        os.environ.pop('TENCENTCLOUD_SESSION_TOKEN', None)
        if secret.get('Token'):
            os.environ['TENCENTCLOUD_SESSION_TOKEN'] = secret['Token']
        return d.publish(attempt, profile['index'])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='Private owner JSON; paths/references only, no key values')
    parser.add_argument('--retry', help='Existing local sealed attempt; skips SSH/test execution')
    args = parser.parse_args()
    try:
        result = flow(q.json_file(args.profile), args.retry)
        print(q.encoded(result).decode(), end='')
        return 0 if result['test_status'] == 'passed' else 2
    except Exception as exc:
        # Do not echo credential paths, provider messages or signed URLs.
        print('dev_archive_owner: incomplete (' + type(exc).__name__ + '); preserve local and remote attempts.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
