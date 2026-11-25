from pathlib import Path
import sys,json,hashlib,time,importlib.metadata as md
v=json.load(sys.stdin);root=Path(v['payload_root']);base=root/v['completed_relative'];cache=base/'cache'
environment={k:md.version(k) for k in ['torch','numpy']}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={}
files={str(base/'manifest.json'):v['completed_manifest_sha256'],str(base/v['checkpoint_relative']):v['checkpoint_sha256'],v['cohort_path']:v['cohort_sha256'],str(cache/'metadata.json'):v['cohort_metadata_sha256']}
files.update({str(cache/n):h for n,h in v['train_cache'].items()})
files.update({str(root/v['source_relative']/n):h for n,h in v['source_hashes'].items()})
for n,h in files.items():
 p=Path(n);assert p.resolve().is_relative_to(root.resolve())
 checks[n]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'sha256':digest(p) if p.is_file() else None,'expected':h}
cohort=json.loads(Path(v['cohort_path']).read_bytes())
out={'status':'PASS_EXACT_INPUTS' if all(x['sha256']==x['expected'] for x in checks.values()) else 'FAIL_EXACT_INPUTS','checks':checks,'cohort_fit':len(cohort['fit']),'cohort_assessment':len(cohort['assessment']),'cohort_group_overlap':cohort['group_overlap'],'target_pin_matches':cohort['target_sha256']==v['train_cache']['train-targets.npy'],'metadata_pin_matches':cohort['metadata_sha256']==v['cohort_metadata_sha256'],'environment':environment,'checked_at':time.time(),'native_forwards':0,'decoded_task_rows':0}
print(json.dumps(out))
