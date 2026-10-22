#!/usr/bin/env python3
"""Read-only HTTPS/authentication and optional cached Q-CFRA receipt check."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def check(run=None):
    spec = importlib.util.spec_from_file_location('project_exchange', ROOT/'scripts/collab/lib/exchange_upload.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    base = os.environ.get('PROJECT_EXCHANGE_URL', '').rstrip('/')
    token = os.environ.get('PROJECT_EXCHANGE_TOKEN', '')
    if helper.parse_base_url(base)[0] != 'https' or not token:
        raise ValueError('HTTPS exchange configuration required')
    context = helper.ssl_context_for('https', helper.DEFAULT_CA_FILE)

    def open_route(route, method='GET'):
        request = Request(base+route, headers=helper._auth_headers(token), method=method)
        return urlopen(request, context=context, timeout=30)

    with open_route('/healthz') as response:
        health = json.load(response)
    if health.get('status') != 'ok' or health.get('max_bytes', 0) < 64*1024**2:
        raise ValueError('exchange health or part-size limit differs')
    with open_route('/api/files?path=q-attention') as response:
        if response.status != 200:
            raise ValueError('authenticated project namespace unavailable')
    count = 0
    if run is not None:
        if not Path(run).is_dir():
            raise ValueError('existing run directory required')
        manifest = json.loads((Path(run)/'run_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 'cfra-run.v1':
            raise ValueError('existing Q-CFRA run manifest required')
        receipt_path = Path(run)/'export_state/exchange.json'
        if not receipt_path.exists() and (Path(run)/'EXPORT_COMPLETE.json').exists():
            raise ValueError('completed export has no private exchange receipt')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            target = 'q-attention/retacred_q_cfra_formal_single_seed/'+manifest['run_id']
            if receipt['target_dir'] != target or receipt['fingerprint'] != manifest['fingerprint']:
                raise ValueError('cached receipt identity differs')
            for part in receipt['parts']:
                if part.get('verified') is not True:
                    raise ValueError('cached part is unverified')
                remote = helper.safe_relative(target+'/'+part['name'], label='cached part')
                with open_route('/upload/'+quote(remote, safe='/._-'), 'HEAD') as response:
                    if (int(response.headers['Content-Length']) != part['size_bytes'] or
                        response.headers.get('ETag') != '"sha256:'+part['sha256']+'"'):
                        raise ValueError('cached part is absent or changed on current exchange')
                count += 1
    return dict(status='exchange_preflight_passed', https_verified=True,
                authenticated_namespace=True, cached_parts_verified=count, read_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='existing Q-CFRA run; verify cached parts on current exchange')
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.run_dir)))
    except Exception as exc:
        detail = 'HTTP'+str(exc.code) if isinstance(exc, HTTPError) else type(exc).__name__
        print('exchange_preflight_failed ('+detail+'): check current private environment, repository CA and cached evidence; '
              'preserve the original run and receipts, report the blocker, and do not restart training.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
