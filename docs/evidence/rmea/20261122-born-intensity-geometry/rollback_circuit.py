from pathlib import Path
import argparse,hashlib,json
a=argparse.ArgumentParser();a.add_argument('--root',required=True,type=Path);v=a.parse_args();p=v.root/'experiments/q_rmea_intensity_circuit_probe.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2121c299c67ee6bb08e135b3fa3cc0ee3e96a811ee337b28fe47db668ee53a7d'
p.unlink();assert not p.exists();print(json.dumps(dict(verified=True,removed_added_file=1)))
