"""Separately admitted train-only feature reconstruction, with a NEW identity.

No heads, training, validation/test members, cloud calls or automatic replay.
The parent stays free of scientific imports and audits in a separate process.
"""
import argparse
import hashlib
import heapq
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from marked_words import marked_words

HERE = Path(__file__).resolve().parent
if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ:
    raise RuntimeError('Non-optimized Python and absent PYTHONOPTIMIZE required in every process')
for name,value in {'USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false',
                   'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():
    os.environ[name]=value


def plain_path(path):
    path = path.absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise RuntimeError('Symlink input/output ancestry rejected')
    return path.resolve(strict=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('xb') as handle:
        handle.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def git(*args):
    return subprocess.check_output(['git','-C',str(HERE),*args], stderr=subprocess.DEVNULL).decode().strip()


def contract():
    return json.loads((HERE/'contract.json').read_bytes())


def sources():
    return {name:sha(HERE/name) for name in ['job.py','marked_words.py','contract.json']}


def select(path, cfg):
    path=plain_path(path)
    assert path.is_file() and not path.is_symlink() and sha(path)==cfg['input_train_sha256']
    observed = 0; labels = set()
    def rows():
        nonlocal observed
        with path.open(encoding='utf8') as handle:
            for line in handle:
                row=json.loads(line); observed+=1; labels.add(row['label'])
                rank=hashlib.sha256(('rmea-readiness-v1:train:'+str(row['metadata']['id'])).encode()).hexdigest()
                yield rank,row
    chosen=[row for _,row in heapq.nsmallest(cfg['train_count'], rows(), key=lambda x:x[0])]
    assert observed==cfg['full_train_count'] and sorted(labels)==sorted(cfg['labels'])
    ids=[hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest() for row in chosen]
    assert len(set(ids))==cfg['train_count']
    assert hashlib.sha256(''.join(ids).encode()).hexdigest()==cfg['train_ids_sha256']
    return chosen,ids


def environment(cfg):
    actual={name:importlib.metadata.version(name) for name in cfg['environment']}
    assert actual==cfg['environment'] and list(sys.version_info[:2])==cfg['python_minor']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Scheduler must hide GPUs'
    return actual


def assets(path, cfg):
    path=plain_path(path)
    assert path.is_dir() and not path.is_symlink()
    assert {p.name for p in path.iterdir()}=={r['name'] for r in cfg['assets']}
    for row in cfg['assets']:
        item=path/row['name']
        assert item.is_file() and not item.is_symlink()
        assert item.stat().st_size==row['size_bytes'] and sha(item)==row['sha256']


def preflight_gate(args, cfg):
    assert args.preflight is not None and args.preflight_sha256 is not None
    assert args.preflight_acceptance is not None and args.preflight_acceptance_sha256 is not None
    receipt=plain_path(args.preflight_acceptance)
    assert sha(receipt)==args.preflight_acceptance_sha256
    accepted=json.loads(receipt.read_bytes())
    assert accepted['schema']=='qattn.cache-preflight.owner-acceptance.v1'
    assert accepted['status']=='ACCEPTED_PREFLIGHT_ONLY' and accepted['task_state']=='complete'
    assert accepted['returncode']==0 and accepted['quiescent'] is True and accepted['attempt_count']==1
    assert accepted['sources']==sources() and accepted['revision']==git('rev-parse','HEAD')
    assert accepted['environment']==environment(cfg) and accepted['train_source_sha256']==cfg['input_train_sha256']
    assert accepted['asset_hashes']=={r['name']:r['sha256'] for r in cfg['assets']}
    root=plain_path(args.preflight)
    assert accepted['preflight_root']==str(root) and accepted['closed_sha256']==args.preflight_sha256
    assert not root.is_symlink() and sha(root/'RUN_COMPLETE')==args.preflight_sha256
    closed=json.loads((root/'RUN_COMPLETE').read_bytes())
    assert closed['schema']=='qattn.train-only-cache.closed.v2' and closed['action']=='preflight'
    assert set(closed['files'])=={'features.npz','cache-manifest.json','cache-validation.json'}
    for name,digest in closed['files'].items(): assert sha(root/name)==digest
    m=json.loads((root/'cache-manifest.json').read_bytes())
    v=json.loads((root/'cache-validation.json').read_bytes())
    assert v['status']=='PASS_SCHEMA_PROVENANCE' and m['counts']=={'train':cfg['preflight_count']}
    assert m['sources']==sources() and m['revision']==git('rev-parse','HEAD')
    assert m['environment']==environment(cfg)
    assert m['train_source_sha256']==cfg['input_train_sha256'] and m['asset_revision']==cfg['asset_revision']
    assert m['asset_hashes']=={r['name']:r['sha256'] for r in cfg['assets']}
    assert m['peak_rss_kib']<cfg['full_gate']['max_peak_rss_kib']
    projected=m['timing']['setup_seconds']+m['timing']['encode_seconds']*cfg['train_count']/cfg['preflight_count']
    assert projected<=cfg['full_gate']['elapsed_extrapolation_seconds']
    return {'preflight_closed_sha256':args.preflight_sha256,'projected_seconds_heuristic':projected,
            'owner_acceptance_sha256':args.preflight_acceptance_sha256}


def roles_for(row, np):
    n=len(row['tokens']);s,o=row['subject'],row['object']
    assert 0<=s[0]<s[1]<=n and 0<=o[0]<o[1]<=n
    assert set(range(*s)).isdisjoint(range(*o))
    p=np.arange(n,dtype=np.float32)
    return np.stack([np.clip((p-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((p-(o[0]+o[1]-1)/2)/32,-1,1),
                     ((p>=s[0])&(p<s[1])).astype(np.float32),((p>=o[0])&(p<o[1])).astype(np.float32)],axis=-1)


def produce(args):
    cfg=contract();env=environment(cfg);assets(args.assets,cfg)
    assert not git('status','--porcelain=v1'), 'Exact clean source required'
    selected,ids=select(args.train_jsonl,cfg)
    count=cfg['preflight_count'] if args.phase=='preflight' else cfg['train_count']
    selected=selected[:count];ids=ids[:count]
    out=plain_path(args.output)
    assert not any(out.iterdir()), 'Fresh empty output required'
    assert not out.is_relative_to(Path(git('rev-parse','--show-toplevel')))
    started=time.monotonic()
    import numpy as np
    import torch
    from transformers import RobertaModel,RobertaTokenizerFast
    torch.set_num_threads(cfg['threads']);torch.manual_seed(13);np.random.seed(13)
    torch.use_deterministic_algorithms(True)
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(args.assets,local_files_only=True,add_pooling_layer=False,
        torch_dtype=torch.float32,attn_implementation='eager').to('cpu').eval()
    encoder.requires_grad_(False);setup=time.monotonic()-started;encode_seconds=0.
    features=[];roles=[];targets=[];offsets=[0]
    for start in range(0,count,cfg['batch_size']):
        batch=selected[start:start+cfg['batch_size']];marked=[marked_words(row) for row in batch]
        tokens=tokenizer([words for words,_ in marked],is_split_into_words=True,padding=True,truncation=False,return_tensors='pt')
        assert tokens['input_ids'].shape[1]<=512
        before=time.monotonic()
        with torch.inference_mode():
            hidden=encoder(**tokens).last_hidden_state.float()
            for i,row in enumerate(batch):
                mapping=tokens.word_ids(i);original=marked[i][1];word=[]
                for j in range(len(row['tokens'])):
                    pieces=[k for k,w in enumerate(mapping) if w is not None and original[w]==j]
                    assert pieces, 'Original token missing'
                    word.append(hidden[i,pieces].mean(0))
                h=torch.stack(word).half().numpy().copy();features.append(h)
                offsets.append(offsets[-1]+len(h));roles.append(roles_for(row,np));targets.append(cfg['labels'].index(row['label']))
        encode_seconds+=time.monotonic()-before
        if start%256==0:print(json.dumps({'stage':'extract','train_entries':min(start+cfg['batch_size'],count)}),flush=True)
    del encoder,hidden
    arrays={'train_features':np.concatenate(features),'train_roles':np.concatenate(roles),
            'train_offsets':np.asarray(offsets,dtype=np.int64),'train_targets':np.asarray(targets,dtype=np.int64),
            'train_ids':np.asarray(ids,dtype='S64')}
    assert sum(a.nbytes for a in arrays.values())<cfg['resources']['rebuild']['disk_mb']*1024**2-1024**2
    np.savez(out/'features.npz',**arrays)
    save(out/'cache-manifest.json',{'schema':'qattn.train-only-cache-reconstruction.result.v2','action':args.phase,
         'revision':git('rev-parse','HEAD'),'sources':sources(),'environment':env,'counts':{'train':count},
         'train_source_sha256':cfg['input_train_sha256'],'train_ids_sha256':hashlib.sha256(''.join(ids).encode()).hexdigest(),
         'labels':cfg['labels'],'asset_revision':cfg['asset_revision'],'asset_hashes':{r['name']:r['sha256'] for r in cfg['assets']},
         'threads':cfg['threads'],'batch_size':cfg['batch_size'],'float_profile':cfg['float_profile'],
         'python_execution':{'optimize':sys.flags.optimize,'PYTHONOPTIMIZE_present':'PYTHONOPTIMIZE' in os.environ},
         'timing':{'setup_seconds':setup,'encode_seconds':encode_seconds,'elapsed_seconds':time.monotonic()-started},
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'features':{'sha256':sha(out/'features.npz'),'bytes':(out/'features.npz').stat().st_size},
         'old_cache_byte_or_numeric_equivalence':False,'head_training':False,'validation_test_members_read':False,'cos_requests':0})


def audit(args):
    import numpy as np
    cfg=contract();selected,ids=select(args.train_jsonl,cfg)
    n=cfg['preflight_count'] if args.phase=='preflight' else cfg['train_count'];selected=selected[:n];ids=ids[:n]
    out=args.output;manifest=json.loads((out/'cache-manifest.json').read_bytes())
    assert manifest['sources']==sources() and manifest['revision']==git('rev-parse','HEAD')
    assert manifest['environment']==environment(cfg) and manifest['train_source_sha256']==cfg['input_train_sha256']
    assert manifest['asset_revision']==cfg['asset_revision'] and manifest['asset_hashes']=={r['name']:r['sha256'] for r in cfg['assets']}
    assert manifest['labels']==cfg['labels'] and manifest['threads']==cfg['threads'] and manifest['batch_size']==cfg['batch_size']
    assert manifest['float_profile']==cfg['float_profile'] and not manifest['old_cache_byte_or_numeric_equivalence']
    assert manifest['python_execution']=={'optimize':0,'PYTHONOPTIMIZE_present':False}
    assert manifest['counts']=={'train':n} and manifest['train_ids_sha256']==hashlib.sha256(''.join(ids).encode()).hexdigest()
    assert manifest['features']=={'sha256':sha(out/'features.npz'),'bytes':(out/'features.npz').stat().st_size}
    with np.load(out/'features.npz',allow_pickle=False) as a:
        assert set(a.files)=={'train_'+k for k in ['features','roles','offsets','targets','ids']}
        h,r,off,y,stored=[a['train_'+k] for k in ['features','roles','offsets','targets','ids']]
        assert h.dtype==np.float16 and r.dtype==np.float32 and off.dtype==y.dtype==np.int64 and stored.dtype==np.dtype('S64')
        assert np.array_equal(off,np.cumsum([0]+[len(row['tokens']) for row in selected]))
        assert h.shape==(int(off[-1]),1024) and r.shape==(int(off[-1]),4) and y.shape==stored.shape==(n,)
        assert np.isfinite(h).all() and np.isfinite(r).all()
        assert stored.astype(str).tolist()==ids and y.tolist()==[cfg['labels'].index(row['label']) for row in selected]
        assert np.array_equal(r,np.concatenate([roles_for(row,np) for row in selected]))
    save(out/'cache-validation.json',{'status':'PASS_SCHEMA_PROVENANCE','count':n,'manifest_sha256':sha(out/'cache-manifest.json'),
         'features_sha256':sha(out/'features.npz'),'scope':'Frozen inputs/schema only; old-cache numeric equivalence unproven.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['preflight','rebuild','_produce','_audit'],required=True)
    p.add_argument('--phase',choices=['preflight','rebuild']);p.add_argument('--train-jsonl',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',type=Path);p.add_argument('--preflight-sha256')
    p.add_argument('--preflight-acceptance',type=Path);p.add_argument('--preflight-acceptance-sha256');args=p.parse_args()
    if args.action in ['_produce','_audit']:
        assert args.phase is not None
        return produce(args) if args.action=='_produce' else audit(args)
    assert args.phase is None and (not args.output.exists() or not any(args.output.iterdir()))
    if any(p.is_symlink() for p in [args.output.absolute(),*args.output.absolute().parents]):
        raise RuntimeError('Symlink output ancestry rejected')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='rebuild':preflight_gate(args,contract())
    environment(contract());assets(args.assets,contract())
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    flags=['--phase',args.action,'--train-jsonl',str(args.train_jsonl),'--assets',str(args.assets),'--output',str(args.output)]
    for action in ['_produce','_audit']:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--action',action,*flags],check=True)
    save(args.output/'RUN_COMPLETE',{'schema':'qattn.train-only-cache.closed.v2','action':args.action,
         'files':{n:sha(args.output/n) for n in ['features.npz','cache-manifest.json','cache-validation.json']}})
    print(json.dumps({'status':'PASS','action':args.action,'head_training':False,'cos_requests':0}))


if __name__=='__main__':main()
