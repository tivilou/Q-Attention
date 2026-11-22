"""Remove only this new prototype's exact source files; never touch core/data."""
import argparse,hashlib,json
from pathlib import Path
EXPECTED={'experiments/diagnostics/rmea_normalized_mass/normalized_mass.py': 'f3434aec98df5aab547e6ba3f3ef03970fa8a091296890b4364d5fc7c8c0f20c', 'experiments/diagnostics/rmea_normalized_mass/probe.py': 'ab71d2766e3ca273da62a9eda07f16e39161dc3f9540ba26497ee080d62edd8e'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve(strict=True);targets=[]
 for rel,h in EXPECTED.items():
  target=root/rel
  assert target.resolve().is_relative_to(root) and not any(x.is_symlink() for x in [target,*target.parents] if x.is_relative_to(root))
  if target.exists():assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==h;targets.append(target)
 for target in targets:target.unlink()
 print(json.dumps(dict(removed=len(targets),core_and_data_untouched=True)))
if __name__=='__main__':main()
