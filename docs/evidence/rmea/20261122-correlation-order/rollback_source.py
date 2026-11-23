"""Remove only this exact added diagnostic, retaining all original source."""
import argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();files={'experiments/diagnostics/rmea_correlation_order/correlation_order.py': 'bd13b9462b616bc7a1bd5410be35f20550b390fbb61fdc959dc8bc0ef4502c90', 'experiments/diagnostics/rmea_correlation_order/probe.py': 'ff907a1a77d5490872c9e646ca78ca1cda7bb7e5026b06f447ff6ad599137162'}
paths=[]
for n,h in files.items():
 f=(r/n).resolve();assert f.is_relative_to(r) and f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h;paths.append(f)
for f in paths:f.unlink();assert not f.exists()
print('Exact two channel diagnostic files removed; original core preserved.')
