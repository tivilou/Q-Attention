"""Restore preserved originals into a derived control directory, never the live node."""
from pathlib import Path
import argparse
import hashlib
import json

r = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--target-dir', type=Path, default=r / 'modified')
args = ap.parse_args()
target = args.target_dir.resolve()
assert target in ((r / 'modified').resolve(), (r / 'rollback-check').resolve())
assert target.is_dir() and not target.is_symlink()
record = json.loads((r / 'prepared.json').read_bytes())
for name, pin in record['files'].items():
    data = (r / 'before' / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin['original_sha256']
    path = target / name
    assert not path.is_symlink()
    path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['original_sha256']
print(json.dumps(dict(original_control_copies_restored=True, live_node_touched=False)))
