"""Reconstruct the declared source fingerprint from Git blobs, no checkout."""
import argparse, hashlib, json, subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--pins', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
pins = json.loads(args.pins.read_text(encoding='utf8'))
actual = {}
for name, expected in pins['source_hashes'].items():
    raw = subprocess.check_output(['git', '-C', str(args.repo), 'show',
                                   pins['git_revision'] + ':' + name], timeout=45)
    actual[name] = hashlib.sha256(raw).hexdigest()
    assert actual[name] == expected, name
fingerprint = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert fingerprint == pins['discovery_manifest_source_sha256']
assert len(actual) == pins['source_files'] == 31
record = dict(status='PASS', declared_git_revision=pins['git_revision'],
              source_files=len(actual), aggregate_sha256=fingerprint,
              manifest_schema=pins['manifest_schema'], no_checkout=True,
              source_contract=pins['source_contract'])
args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
print(json.dumps(record))
