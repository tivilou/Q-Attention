"""Remove this exact added prototype and diagnostic from an explicit checkout."""
import argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();files={'experiments/diagnostics/rmea_relative_return/relative_return.py': '8d55b9047081a22754be6ea34ec06d77c04b868670cd25cb3ada3e124c6782dc', 'experiments/diagnostics/rmea_relative_return/probe.py': '5d047f7cf34fcff6b7b7d53bcd4942a06aab8d3e2a001a4edc098dcd1ab0cd2f'};paths=[]
for n,h in files.items():
 f=(r/n).resolve();assert f.is_relative_to(r) and f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h;paths.append(f)
for f in paths:f.unlink();assert not f.exists()
print('Exact two relative-return files removed; original source retained.')
