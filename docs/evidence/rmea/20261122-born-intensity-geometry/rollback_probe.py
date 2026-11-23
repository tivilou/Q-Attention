from pathlib import Path
import argparse,hashlib,json
a=argparse.ArgumentParser();a.add_argument('--root',required=True,type=Path);v=a.parse_args();p=v.root/'experiments/q_rmea_intensity_fiber_probe.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a5bdb48d60034e4ceb3008c5b3d3fe121ea4a8994ef1fa840f6545bc6b20d3c4'
p.unlink();assert not p.exists();print(json.dumps(dict(verified=True,removed_added_file=1)))
