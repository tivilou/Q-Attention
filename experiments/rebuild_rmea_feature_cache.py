"""Extract pinned train/valid frozen features; no head training or test access."""
import argparse
import hashlib
import heapq
import importlib.metadata
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baselines.retacred.common import atomic_json,digest
from scripts.audit_rmea_inputs import EXPECTED
from scripts.audit_rmea_tokenization import marked_words

LIMITS={'train':2048,'valid':1024}
AUDIT_SHA='c21d23017140d89a3632853d0f33d9e76c5cb7d574d4a9108b63d1ed23bf0996'
TOKEN_SHA='4d25f24c525e7f706d829a5b5c90750a86d984404cb4fa8251813d7274e2e44a'

def environment():
    expected={'torch':'2.5.1+cpu','numpy':'1.26.4','transformers':'4.44.2','tokenizers':'0.19.1'}
    actual={name:importlib.metadata.version(name) for name in expected}
    if actual!=expected or sys.version_info[:2]!=(3,10):raise ValueError('Pinned CPU runtime differs')
    return dict(python=sys.version.split()[0],**actual)

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def choose(data,split,n):
    def rows():
        with (data/(split+'.jsonl')).open(encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                rank=hashlib.sha256(('rmea-readiness-v1:'+split+':'+str(row['metadata']['id'])).encode()).hexdigest()
                yield rank,row
    selected=[row for _,row in heapq.nsmallest(n,rows(),key=lambda pair:pair[0])]
    if len(selected)!=n:raise ValueError('Insufficient fixed samples')
    return selected

def dependencies(args):
    if git('status','--porcelain'):raise ValueError('Clean source required')
    if digest(args.input_audit)!=AUDIT_SHA or digest(args.tokenization_audit)!=TOKEN_SHA:raise ValueError('Inspection identity differs')
    inspection=json.loads(args.input_audit.read_text());token=json.loads(args.tokenization_audit.read_text())
    if not inspection['dataset_identity_matches_reference']:raise ValueError('Data identity not verified')
    if any(r['over_encoder_limit512'] or r['missing_entity_rows'] or r['missing_word_rows'] for r in token['splits'].values()):raise ValueError('Token coverage incomplete')
    for split in LIMITS:
        if digest(args.data_dir/(split+'.jsonl'))!=EXPECTED[split][1]:raise ValueError('Source JSONL differs')
    lock=json.loads((ROOT/'baselines/retacred/assets.lock.json').read_text())
    if args.assets.is_symlink() or {p.name for p in args.assets.iterdir()}!={r['name'] for r in lock['files']}:raise ValueError('Exact loader-visible asset inventory required')
    for row in lock['files']:
        path=args.assets/row['name']
        if path.is_symlink() or path.stat().st_size!=row['size_bytes'] or digest(path)!=row['sha256']:raise ValueError('Encoder asset differs')
    environment()
    return inspection,lock

def produce(args):
    inspection,lock=dependencies(args)
    out=args.run_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Fresh private output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('CPU recipe must hide GPUs')
    limits={s:4 for s in LIMITS} if args.action=='_preflight' else LIMITS
    selected={s:choose(args.data_dir,s,n) for s,n in limits.items()}
    labels=inspection['train_derived_labels']
    if len(labels)!=40 or len(set(labels))!=40:raise ValueError('Forty unique train-derived labels required')
    label_to_id={name:i for i,name in enumerate(labels)}
    out.mkdir(parents=True);started=time.monotonic()
    import numpy as np
    import torch
    from transformers import RobertaModel,RobertaTokenizerFast
    import_seconds=time.monotonic()-started
    torch.set_num_threads(4);torch.manual_seed(13);np.random.seed(13);torch.use_deterministic_algorithms(True)
    device=torch.device('cpu')
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(args.assets,local_files_only=True,add_pooling_layer=False,
        torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
    model_load_seconds=time.monotonic()-started-import_seconds
    print(json.dumps(dict(stage='encoder_loaded',import_seconds=import_seconds,model_load_seconds=model_load_seconds,encoder_dtype='float32')),flush=True)
    encoder.requires_grad_(False);arrays={};ids_hashes={}
    encode_seconds=0.
    for split,rows in selected.items():
        features=[];offsets=[0];role_features=[];targets=[];sample_ids=[]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4];marked=[marked_words(row) for row in batch]
            encoded=tokenizer([words for words,_ in marked],is_split_into_words=True,padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>512:raise ValueError('Unexpected length; truncation forbidden')
            with torch.inference_mode():
                before_encode=time.monotonic()
                hidden=encoder(**{k:v.to(device) for k,v in encoded.items()}).last_hidden_state.float()
                encode_seconds+=time.monotonic()-before_encode
                for i,row in enumerate(batch):
                    original=marked[i][1];mapping=encoded.word_ids(i);word_features=[]
                    for j in range(len(row['tokens'])):
                        pieces=[k for k,w in enumerate(mapping) if w is not None and original[w]==j]
                        if not pieces:raise ValueError('Original word disappeared')
                        word_features.append(hidden[i,pieces].mean(0))
                    words=torch.stack(word_features).half().cpu().numpy().copy()
                    features.append(words);offsets.append(offsets[-1]+len(words))
                    positions=np.arange(len(words),dtype=np.float32);subj,obj=row['subject'],row['object']
                    roles=np.stack([np.clip((positions-(subj[0]+subj[1]-1)/2)/32,-1,1),np.clip((positions-(obj[0]+obj[1]-1)/2)/32,-1,1),((positions>=subj[0])&(positions<subj[1])).astype(np.float32),((positions>=obj[0])&(positions<obj[1])).astype(np.float32)],axis=-1)
                    role_features.append(roles);targets.append(label_to_id[row['label']]);sample_ids.append(hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest())
            if start%256==0:print(json.dumps(dict(stage='extract',split=split,records=min(start+4,len(rows)),encode_seconds=encode_seconds)),flush=True)
        arrays.update({split+'_features':np.concatenate(features),split+'_roles':np.concatenate(role_features),split+'_offsets':np.array(offsets,dtype=np.int64),split+'_targets':np.array(targets,dtype=np.int64),split+'_ids':np.array(sample_ids,dtype='S64')})
        ids_hashes[split]=hashlib.sha256(''.join(sample_ids).encode()).hexdigest()
    del encoder,hidden
    np.savez(out/'features.npz',**arrays)
    manifest=dict(schema='qattn.rmea.rebuilt-features.v1',profile='preflight' if args.action=='_preflight' else 'rebuild',counts=limits,
        source_revision=git('rev-parse','HEAD'),producer_sha256=digest(Path(__file__)),source_data={s:EXPECTED[s][1] for s in limits},
        input_audit_sha256=AUDIT_SHA,tokenization_audit_sha256=TOKEN_SHA,encoder_revision=lock['revision'],encoder_assets={r['name']:r['sha256'] for r in lock['files']},
        labels=labels,sample_rule='SHA256(rmea-readiness-v1:split:original_id), ascending',sample_id_hashes=ids_hashes,
        dtype='float32 CPU encoder and word-piece mean; saved features float16; roles float32',device='cpu',threads=4,batch_size=4,attention='eager',
        environment=environment(),elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing=dict(import_seconds=import_seconds,model_load_seconds=model_load_seconds,encode_seconds=encode_seconds,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime),
        features=dict(sha256=digest(out/'features.npz'),bytes=(out/'features.npz').stat().st_size),
        test_loaded=False,head_training=False,formal_runs=0,original_cache_bitwise_equivalence=False,
        boundary='New CPU cache identity. Same pinned inputs/sample rule; no claim of numeric or byte equivalence to historical GPU cache.')
    atomic_json(out/'cache-manifest.json',manifest)
    atomic_json(out/'RUN_COMPLETE',dict(schema='qattn.rebuilt-cache.closed.v1',files={n:digest(out/n) for n in ('features.npz','cache-manifest.json')}))

def audit(args):
    import numpy as np
    out=args.run_dir.resolve();manifest=json.loads((out/'cache-manifest.json').read_text());closed=json.loads((out/'RUN_COMPLETE').read_text())
    if closed['schema']!='qattn.rebuilt-cache.closed.v1' or set(closed['files'])!={'features.npz','cache-manifest.json'}:raise ValueError('Closure differs')
    if any(digest(out/n)!=h for n,h in closed['files'].items()):raise ValueError('Closed data changed')
    inspection,lock=dependencies(args)
    if manifest['schema']!='qattn.rmea.rebuilt-features.v1' or manifest['environment']!=environment():raise ValueError('Schema/runtime identity differs')
    if manifest['source_revision']!=git('rev-parse','HEAD') or manifest['producer_sha256']!=digest(Path(__file__)):raise ValueError('Source differs')
    limits={s:4 for s in LIMITS} if manifest['profile']=='preflight' else LIMITS
    if manifest['profile'] not in ('preflight','rebuild') or manifest['counts']!=limits or manifest['test_loaded'] is not False or manifest['head_training'] is not False or manifest['formal_runs']!=0:raise ValueError('Workload differs')
    fixed=dict(source_data={s:EXPECTED[s][1] for s in limits},input_audit_sha256=AUDIT_SHA,tokenization_audit_sha256=TOKEN_SHA,
        sample_rule='SHA256(rmea-readiness-v1:split:original_id), ascending',dtype='float32 CPU encoder and word-piece mean; saved features float16; roles float32',
        device='cpu',threads=4,batch_size=4,attention='eager',original_cache_bitwise_equivalence=False)
    if any(manifest[k]!=v for k,v in fixed.items()) or set(manifest['sample_id_hashes'])!=set(limits):raise ValueError('Frozen preprocessing metadata differs')
    if manifest['labels']!=inspection['train_derived_labels'] or manifest['encoder_revision']!=lock['revision'] or manifest['encoder_assets']!={r['name']:r['sha256'] for r in lock['files']}:raise ValueError('Provenance differs')
    if manifest['features']!=dict(sha256=digest(out/'features.npz'),bytes=(out/'features.npz').stat().st_size):raise ValueError('Cache reference differs')
    with np.load(out/'features.npz',allow_pickle=False) as arrays:
        if set(arrays.files)!={s+'_'+k for s in limits for k in ('features','roles','offsets','targets','ids')}:raise ValueError('Cache inventory differs')
        for split,n in limits.items():
            rows=choose(args.data_dir,split,n);ids=[hashlib.sha256(str(r['metadata']['id']).encode()).hexdigest() for r in rows]
            h,roles,offsets,y,stored=[arrays[split+'_'+k] for k in ('features','roles','offsets','targets','ids')]
            expected_offsets=np.cumsum([0]+[len(r['tokens']) for r in rows])
            if h.dtype!=np.float16 or roles.dtype!=np.float32 or offsets.dtype!=np.int64 or y.dtype!=np.int64 or stored.dtype!=np.dtype('S64'):raise ValueError('Dtype differs')
            if h.shape!=(expected_offsets[-1],1024) or roles.shape!=(len(h),4) or y.shape!=(n,) or stored.shape!=(n,) or not np.array_equal(offsets,expected_offsets):raise ValueError('Shapes/offsets differ')
            if not np.isfinite(h).all() or not np.isfinite(roles).all():raise ValueError('Nonfinite cache')
            if stored.astype(str).tolist()!=ids or y.tolist()!=[manifest['labels'].index(r['label']) for r in rows]:raise ValueError('Sample/target binding differs')
            if manifest['sample_id_hashes'][split]!=hashlib.sha256(''.join(ids).encode()).hexdigest():raise ValueError('Sample identity differs')
            expected_roles=[]
            for row in rows:
                p=np.arange(len(row['tokens']),dtype=np.float32);s,o=row['subject'],row['object']
                expected_roles.append(np.stack([np.clip((p-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((p-(o[0]+o[1]-1)/2)/32,-1,1),((p>=s[0])&(p<s[1])).astype(np.float32),((p>=o[0])&(p<o[1])).astype(np.float32)],axis=-1))
            if not np.array_equal(roles,np.concatenate(expected_roles)):raise ValueError('Role binding differs')
    atomic_json(out/'cache-validation.json',dict(status='passed',profile=manifest['profile'],counts=limits,manifest_sha256=digest(out/'cache-manifest.json'),features_sha256=digest(out/'features.npz'),source_revision=git('rev-parse','HEAD'),test_loaded=False,head_training=False,scope='Input/schema/provenance audit; historical GPU numeric equivalence unverified'))
    print(json.dumps(dict(status='passed',profile=manifest['profile'],counts=limits)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preflight','rebuild','_preflight','_rebuild','audit'])
    for name in ('data-dir','assets','input-audit','tokenization-audit','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.action in ('preflight','rebuild'):
        argv=[sys.executable,'-B',str(Path(__file__).resolve())]
        flags=sum(([ '--'+k.replace('_','-'),str(getattr(a,k))] for k in ('data_dir','assets','input_audit','tokenization_audit','run_dir')),[])
        subprocess.run(argv+['_'+a.action]+flags,check=True)
        subprocess.run(argv+['audit']+flags,check=True)
    elif a.action=='audit':audit(a)
    else:produce(a)
if __name__=='__main__':main()
