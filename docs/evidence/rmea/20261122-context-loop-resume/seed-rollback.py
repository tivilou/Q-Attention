"""Revert only the exact fresh-run seeding correction in a selected checkout."""
import argparse
import hashlib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, type=Path)
a = p.parse_args()
target = a.root.resolve() / 'baselines/retacred/rmea_token_transport_study.py'
before = '9cadb54b6088cf32e01c7bb08b375442ad68febe5d06628ac173647c4e0d0e85'
after = '14d796f4970fddd7f218ef637b9a41a6e0a871dac978025b30f13713b18b6e7b'
data = target.read_bytes()
assert hashlib.sha256(data).hexdigest() == after, 'Selected source has changed; preserve it'
original = (Path(__file__).parent / 'seed-before.py').read_bytes()
assert hashlib.sha256(original).hexdigest() == before and before != after
temporary = target.with_name(target.name + '.seed-rollback.tmp')
assert not temporary.exists()
temporary.write_bytes(original)
temporary.replace(target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == before
print('PASS: exact pre-seeding source restored')
