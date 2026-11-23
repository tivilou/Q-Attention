"""Remove only the exact new relative-return plugin; preserve original native head."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve(strict=True)
native=root/'baselines/retacred/rmea.py';target=root/'baselines/retacred/rmea_relative_return.py'
if target.is_symlink() or native.is_symlink():raise ValueError('Regular source required')
if hashlib.sha256(native.read_bytes()).hexdigest()!='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0':raise ValueError('Native source differs')
if target.exists():
 if hashlib.sha256(target.read_bytes()).hexdigest()!='39728d236fef33b6f682eff0f722bef3c72939b0e198e375f5e1aa7a2089abfc':raise ValueError('Preserve changed plugin')
 target.unlink()
print(json.dumps(dict(plugin_absent=not target.exists(),native_preserved=True)))
