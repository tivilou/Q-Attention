from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention/.codex-artifacts')
prior=Path('D:/CodexHome/task-artifacts/qattn-context-aliasing-source-exclusion-20261126')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
write=lambda n,x:(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
h=prior/'hypothesis.json';assert sha(h)=='c8c6aa152eeeb891d1902a9cee2f4012441daee21ec8db5ad12921d3f2ce438c'
models={};files={};local=[]
for group,modes in [('distinct-token-evidence-20261125',('quantum_distinct','dephased_distinct','gaussian_distinct')),('connected-value-pooling-20261125',('gaussian_covariance',))]:
 p=root/group;cfg=read(p/'design.json');summary=read(p/'summary.json')
 assert summary['status']=='completed' and summary['assessment_items']==4096 and summary['train_items']==8192
 remote='/hy-tmp/Q-Attention/control/'+group
 for mode in modes:
  row=summary['results'][mode]
  models[mode]=dict(checkpoint=remote+'/output/'+mode+'/final.pt',checkpoint_sha256=row['final_sha256'],
   parameters=row['parameters'],updates=row['updates'],all40_macro_f1=row['all40_macro_f1'],
   initial_sha256=row['initial_sha256'],order_sha256=row['order_sha256'],plugin=remote+'/plugin.py',source_group=group)
  files[models[mode]['checkpoint']]=row['final_sha256']
 files[remote+'/output/summary.json']=sha(p/'summary.json')
 files[remote+'/output/diagnostic.private.npz']=summary['private_diagnostic_sha256']
 files[remote+'/design.json']=sha(p/'design.json')
 for name,value in cfg['code_hashes'].items():
  assert sha(p/name)==value
  files[remote+'/'+name]=value
  if name.endswith('.py'):ast.parse((p/name).read_text('utf8'))
 for name,value in cfg['source_hashes'].items():files['/hy-tmp/Q-Attention/'+cfg['source']+'/'+name]=value
 local.extend(dict(path=str(p/name),sha256=sha(p/name)) for name in ('design.json','summary.json','plugin.py'))
cohort='/hy-tmp/Q-Attention/control/staged-bottleneck-design-20261125/cohort.private.json'
files[cohort]='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0'
cache='/hy-tmp/Q-Attention/inputs/native-baseline/discovery/cache'
inputs={cache+'/'+n:h for n,h in {
 'train-h.npy':'b69b1bfb97f20932b26dcea8b13a788a12ec98ec43a68de1801960ad5c526635',
 'train-roles.npy':'b62e4e23836c8b5b9b930e5b9d993f0890a729bfdadbcee02b08c23cae0c3d73',
 'train-offsets.npy':'a1ef178eae3b9c16b4cfa20dc66af15a1dd0238bb9fc2fffd55b7a6328c25601',
 'train-targets.npy':'6ce409b586a421995f7d70906f062a7e91fe9d35a04be5359a2f58a3485ee604',
 'metadata.json':'9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e'}.items()}
manifest='/hy-tmp/Q-Attention/inputs/native-baseline/discovery/manifest.json'
files[manifest]='dfb347aef35657c5463486e1043aca0617ccc11d3aca1eea851a59af15ec5f49'
for p,v in inputs.items():
 if not p.endswith(('train-h.npy','train-roles.npy')):files[p]=v
out=dict(status='FROZEN_READONLY_IDENTITY_PROBE',created_at=datetime.now(timezone.utc).isoformat(),
 prediction=dict(path=str(h),sha256=sha(h)),models=models,files=files,inputs=inputs,cohort=cohort,manifest=manifest,source='/hy-tmp/Q-Attention/'+cfg['source'],
 local_pins=local,read_only_probe='Hash small files and checkpoint bytes, inspect npz/npy headers and cohort mapping only. Large h/roles contents await allocated diagnostic full-hash gate. No checkpoint decoding, labels/statistics/new task forward.',
 probe_budget=dict(max_seconds=60,GPU=0,updates=0,capacity=0),
 scientific_hypothesis_unchanged=True,training_admitted=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
write('identity-spec.json',out)
print(json.dumps(dict(status=out['status'],checkpoints=len(models),small_file_pins=len(files),cache_pins=len(inputs))))
