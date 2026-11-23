"""Remove only this exact added diagnostic from an explicitly supplied checkout."""
import argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();f=(r/'experiments/diagnostics/normalized_effect_rank/verify_witness.py').resolve();assert f.is_relative_to(r)
assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()=='3d8ddda7f8d323b81eb134160fa417256351e6e99d3ba5164813d3499ecf0e2c'
f.unlink();assert not f.exists();print('Exact diagnostic removed; existing source files preserved.')
