"""Real parent-child private-pack canary; marker fault, exact cleanup, retry."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output.resolve()
    assert not root.exists() or root.is_dir() and not any(root.iterdir())
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('extract.py').resolve()
    logs = []
    with tempfile.TemporaryDirectory(prefix='ownership-canary-', dir=root) as folder:
        sandbox = Path(folder)
        data, protocol, target = [sandbox / n for n in ['train.jsonl', 'protocol.json', 'result']]
        rows = []
        for tokens in [['a', 'b'], ['A', 'b'], ['a', 'β']]:
            for sub, obj in [([0, 1], [1, 2]), ([1, 2], [0, 1])]:
                rows.append(dict(tokens=tokens, subject=sub, object=obj, label='DO_NOT_EXPORT'))
        rows += [dict(rows[0], label='CONFLICT_IGNORED'), dict(rows[0])]
        raw = b''.join((json.dumps(r) + '\n').encode() for r in rows)
        data.write_bytes(raw)
        protocol.write_bytes((json.dumps(dict(schema='qattn.evidence-ownership.extractor.v1', expected_sha256=hashlib.sha256(raw).hexdigest(), expected_rows=8, expected_eligible_groups=3, group_budget=3)) + '\n').encode())
        pinned = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data, protocol, source]}
        args = ['--input', str(data), '--protocol', str(protocol), '--output', str(target)]
        inject = '''import pathlib,runpy,sys
original=pathlib.Path.write_bytes
def write(self,data):
 if self.name=='RUN_COMPLETE':raise OSError('injected completion marker failure')
 return original(self,data)
pathlib.Path.write_bytes=write
source=sys.argv[1];sys.argv=[source]+sys.argv[2:];runpy.run_path(source,run_name='__main__')
'''
        def call(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf8', timeout=20)
            logs.append(dict(command=cmd, stdout=r.stdout, stderr=r.stderr, exit_status=r.returncode))
            return r
        target.mkdir()
        fail = call([sys.executable, '-B', '-c', inject, str(source), *args])
        assert fail.returncode != 0 and 'injected completion marker failure' in fail.stderr
        assert set(p.name for p in target.iterdir()) == {'private-pack.json', 'compact-report.json'}
        first = (target / 'private-pack.json').read_bytes()
        assert target.parent == sandbox and all(not p.is_symlink() and p.is_file() for p in target.iterdir())
        for name in ['private-pack.json', 'compact-report.json']:
            (target / name).unlink()
        target.rmdir()
        assert not target.exists()
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in pinned.items())
        target.mkdir()
        ok = call([sys.executable, '-B', str(source), *args])
        assert ok.returncode == 0 and set(p.name for p in target.iterdir()) == {'private-pack.json', 'compact-report.json', 'RUN_COMPLETE'}
        assert (target / 'private-pack.json').read_bytes() == first
        report = json.loads((target / 'compact-report.json').read_bytes())
        assert report['counts']['rows'] == 8 and report['counts']['eligible_groups'] == report['counts']['selected_groups'] == 3
        assert report['private_pack_mode'] == '0o600' and report['output_directory_mode'] == '0o700'
        assert b'DO_NOT_EXPORT' not in first and b'CONFLICT_IGNORED' not in first
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in pinned.items())
    command_results = [dict(exit_status=r['exit_status'], stdout_sha256=hashlib.sha256(r['stdout'].encode()).hexdigest(), stderr_sha256=hashlib.sha256(r['stderr'].encode()).hexdigest()) for r in logs]
    result = dict(schema='qattn.evidence-ownership.lifecycle-canary.v1', status='PASS', injected_failure_observed=True, partial_output_not_complete=True, partial_output_cleaned=True, retry_same_immutable_source_inputs=True, private_pack_retry_byteexact=True, permissions_verified=True, fixture_counts=report['counts'], real_data_read=False, reader_launches=0, training=0, command_results=command_results)
    import runpy
    runpy.run_path(str(source))['validate_canary_report'](result)
    (root / 'compact-report.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    (root / 'RUN_COMPLETE').write_bytes(b'private-pack lifecycle canary complete\n')
    print(json.dumps(dict(status='PASS', injected_failure_and_retry=True, real_data_read=False)))


if __name__ == '__main__':
    main()
