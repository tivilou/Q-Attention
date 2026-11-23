"""One CPU witness solve for the already frozen and collected decision set."""
import argparse
import hashlib
import json
import signal
from pathlib import Path
import numpy as np
from witness import solve, fixtures


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixtures-only', action='store_true')
    args = p.parse_args(); signal.alarm(600); args.output.mkdir(parents=True, exist_ok=True)
    if args.fixtures_only:
        result = fixtures()
        positive, _ = solve(np.array([(4 / 3) * np.outer(np.eye(8)[0] + np.eye(8)[7],
                        np.eye(8)[0] + np.eye(8)[7]) / 2 - np.eye(8)]), np.array([1.]))
        assert positive['checked_delta'] > 1 / 3 - 1e-4
        result['solver_ghz_checked_delta'] = positive['checked_delta']
        write(args.output / 'fixtures.json', result); print(json.dumps(result)); return
    assert not (args.output / 'summary.json').exists(), 'No repeated scientific solve'
    collection = json.loads((args.output / 'collection.json').read_bytes())
    if collection['status'] == 'under_supported':
        result = dict(status='under_supported', promotion_gate=False)
    else:
        pin = hashlib.sha256((args.output / 'operators.npz').read_bytes()).hexdigest()
        assert pin == collection['operators_sha256']
        with np.load(args.output / 'operators.npz', allow_pickle=False) as f:
            result, certificate = solve(f['operators'], f['weights'])
        if certificate is not None:
            np.savez(args.output / 'certificate.private.npz', **certificate)
        result['promotion_gate'] = result.get('checked_delta', -1) >= .01
    result.update(collection=collection, L5='NOT_ACCEPTED',
        claim_scope='Conditional exposed VALID numerical necessity against trace-one biseparable substitutions for one native relation, in the original fixed tensor encoding. Not net accuracy/Macro loss, inherited L4, hardware advantage, or exact real-arithmetic certification.')
    write(args.output / 'summary.json', result)
    write(args.output / 'RUN_COMPLETE', dict(summary_sha256=hashlib.sha256((args.output / 'summary.json').read_bytes()).hexdigest()))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
