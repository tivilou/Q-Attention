"""Read-only bounded metadata/checkpoint hash probe; no Torch or task inference."""
from pathlib import Path
import ast,hashlib,io,json,struct,sys,time,zipfile
spec=json.load(sys.stdin);start=time.monotonic();out=[];errors=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   b=f.read(4*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
def header(f):
 assert f.read(6)==b'\x93NUMPY'
 version=tuple(f.read(2));assert version in ((1,0),(2,0),(3,0))
 size=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
 assert size<65536
 j=ast.literal_eval(f.read(size).decode('latin1' if version!=(3,0) else 'utf8'))
 return {k:j[k] for k in ('descr','fortran_order','shape')}
for name,expected in spec['files'].items():
 assert time.monotonic()-start<spec['probe_budget']['max_seconds']
 p=Path(name);row=dict(path=name,exists=p.is_file(),symlink=p.is_symlink(),expected_sha256=expected)
 if row['exists'] and not row['symlink']:
  row.update(bytes=p.stat().st_size,sha256=digest(p));row['match']=row['sha256']==expected
 else:row['match']=False
 if not row['match']:errors.append(name)
 out.append(row)
archives={}
for group in ('distinct-token-evidence-20261125','connected-value-pooling-20261125'):
 p=Path('/hy-tmp/Q-Attention/control')/group/'output/diagnostic.private.npz'
 if p.is_file():
  with zipfile.ZipFile(p) as z:
   assert len(z.infolist())<128
   archives[group]={n:header(z.open(n)) for n in z.namelist() if n.endswith('.npy')}
inputs={}
for name,expected in spec['inputs'].items():
 p=Path(name);row=dict(path=name,exists=p.is_file(),expected_sha256=expected,contents_hashed_in_probe=name in spec['files'])
 if p.is_file():
  row['bytes']=p.stat().st_size
  if p.suffix=='.npy':
   with p.open('rb') as f:row['header']=header(f)
 inputs[name]=row
c=json.loads(Path(spec['cohort']).read_bytes())
fit,assess=c['fit'],c['assessment'];assert len(fit)==8192 and len(assess)==4096 and not set(fit)&set(assess)
result=dict(status='PASS_EXISTING_IDENTITY_PROBE' if not errors and all(x['exists'] for x in inputs.values()) else 'TECHNICAL_NONADMISSION',
 checked_files=out,errors=errors,archives=archives,inputs=inputs,
 cohort=dict(fit=len(fit),assessment=len(assess),sha256=digest(Path(spec['cohort'])),metadata_sha256=c['metadata_sha256'],target_sha256=c['target_sha256']),
 large_input_full_hash_pending=True,checkpoint_decodes=0,new_task_forwards=0,optimizer_updates=0,seconds=time.monotonic()-start)
print(json.dumps(result))
