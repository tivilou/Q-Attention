"""Allocated, read-only source/input/reference binding before clone updates."""
from pathlib import Path
import hashlib,json,time
r=Path(__file__).resolve().parent;root=Path('/hy-tmp/Q-Attention');history=root/'control'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((r/'design.json').read_bytes());checked={}
def check(path,h):
 assert sha(path)==h,str(path)
 checked[str(path)]=h
for n,h in cfg['source_hashes'].items():check(root/cfg['source']/n,h)
for folder,pins in cfg['reference_pins'].items():
 for n,h in pins.items():check(history/folder/('output/summary.json' if n=='summary.json' else n),h)
 d=json.loads((history/folder/'design.json').read_bytes())
 for k in ('seed','updates','batch_size','microbatch_size','lr','weight_decay','betas','optimizer_eps','clip_norm','cohort_sha256','training_dtype','dense_batches'):
  assert d[k]==cfg[k],(folder,k)
for key,row in cfg['references'].items():
 s=json.loads((history/row['source']/'output/summary.json').read_bytes())['results'][key.rsplit('/',1)[-1]]
 for k in ('initial_sha256','order_sha256','updates','parameters','all40_macro_f1','class_support','confusion_matrix','final_sha256'):
  assert s[k]==row[k],(key,k)
for n,h in {
 'train-h.npy':'b69b1bfb97f20932b26dcea8b13a788a12ec98ec43a68de1801960ad5c526635',
 'train-roles.npy':'b62e4e23836c8b5b9b930e5b9d993f0890a729bfdadbcee02b08c23cae0c3d73',
 'train-offsets.npy':'a1ef178eae3b9c16b4cfa20dc66af15a1dd0238bb9fc2fffd55b7a6328c25601',
 'train-targets.npy':'6ce409b586a421995f7d70906f062a7e91fe9d35a04be5359a2f58a3485ee604',
 'metadata.json':'9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e'}.items():
 # Stream the large immutable arrays rather than materializing several GB.
 p=root/'inputs/native-baseline/discovery/cache'/n
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 assert digest.hexdigest()==h,n
 checked[str(p)]=h
check(root/'inputs/native-baseline/discovery/manifest.json',cfg['native_manifest_sha256'])
cohort=history/'staged-bottleneck-design-20261125/cohort.private.json'
check(cohort,cfg['cohort_sha256']);(r/'cohort.private.json').write_bytes(cohort.read_bytes())
import numpy as np
c=json.loads(cohort.read_bytes());rng=np.random.default_rng(cfg['seed']);batches=[]
while len(batches)<cfg['updates']:
 order=rng.permutation(np.asarray(c['fit'],dtype=np.int64));batches.extend(order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size']))
order_sha=hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in batches[:cfg['updates']])).hexdigest()
assert order_sha==cfg['order_sha256']
out=dict(status='PASS_EXACT_INPUT_SOURCE_REFERENCE_BINDING',files=checked,order_sha256=order_sha,observed_at=time.time(),
 cohort_sha256=cfg['cohort_sha256'],task_forward=0,updates=0,reference_fits=0)
(r/'input-binding.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],files=len(checked),order_sha256=order_sha)))
