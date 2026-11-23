"""Remove this exact added diagnostic version from an explicitly supplied checkout."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();identities={'experiments/diagnostics/rmea_normalized_support/oracle_probe.py': '9596b547c7cd7208ffff926bd6c3678afc810b95b9de85337e2dff46a73e54a6', 'experiments/diagnostics/rmea_normalized_support/explicit_probe.py': '7ac81ab79820c508e19e1d0c10d1b1743ea035234119d33201b06839283515f2', 'experiments/diagnostics/rmea_normalized_support/moment_slice.py': '31c97861cba7bc425486af81b797701550721288835417ffd678d65ca501a250'}
paths=[]
for n,h in identities.items():
 f=(r/n).resolve();assert f.is_relative_to(r) and f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h
 paths.append(f)
for f in paths:f.unlink();assert not f.exists()
print('Exact three diagnostics removed; existing source preserved.')
