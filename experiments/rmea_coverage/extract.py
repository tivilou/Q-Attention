"""313 train-only feature additions; no fitted head or test reader."""
import argparse,collections,gc,hashlib,importlib.metadata,json,os,resource,subprocess,sys,time
from pathlib import Path
from coverage_data import load_base,load_overlay,select,sid,rank,roles_for,sha,array_sha,validate_arrays,BASE_SHA,MANIFEST_SHA,TRAIN_SHA

def save(p,x):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,separators=(',',':'))+'\n',encoding='utf8');os.replace(tmp,p)
def environment():
    expected={'torch':'2.5.1+cpu','numpy':'1.26.4','transformers':'4.44.2','tokenizers':'0.19.1'}
    actual={n:importlib.metadata.version(n) for n in expected}
    assert actual==expected and sys.version_info[:2]==(3,10)
    return dict(python=sys.version.split()[0],**actual)

def main():
    for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='4'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import numpy as np
    repo=Path(__file__).resolve().parents[2]
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
    assert not git('status','--porcelain=v1') and os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    env=environment();out=args.output.resolve();assert not out.is_relative_to(repo)
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    spec=json.loads((Path(__file__).parent/'contract.json').read_text());assert spec['new_records']==313 and spec['cap']==16
    from unit_checks import run_checks
    save(out/'unit.json',run_checks());started=time.monotonic()
    d,m=load_base(args.base);labels=m['labels'];assert labels==spec['labels']
    base_array_hashes={n:array_sha(a) for n,a in d.items()}
    support=np.bincount(d['train_targets'],minlength=40).tolist();assert support==spec['original_train_support']
    old_ids=d['train_ids'].astype(str).tolist();old_counts=dict(zip(labels,support))
    raw=args.data_dir/'train.jsonl';assert not raw.is_symlink() and sha(raw)==TRAIN_SHA
    def rows():
        with raw.open(encoding='utf8') as f:
            for line in f:yield json.loads(line)
    selected,old_rows,census=select(rows(),old_ids,labels,old_counts,cap=16)
    assert census==spec['official_train_counts'] and sum(census.values())==58465 and len(selected)==313
    assert not {sid(r) for r in selected}&set(d['valid_ids'].astype(str))
    for i,key in enumerate(old_ids):
        row=old_rows[key];lo,hi=map(int,d['train_offsets'][i:i+2])
        assert hi-lo==len(row['tokens']) and d['train_targets'][i]==labels.index(row['label'])
        assert np.array_equal(d['train_roles'][lo:hi],roles_for(row))
    lockfile=Path(__file__).parent/'assets.lock.json';assert sha(lockfile)==spec['assets_lock_sha256']
    lock=json.loads(lockfile.read_text())
    assert args.assets.is_dir() and not args.assets.is_symlink() and {p.name for p in args.assets.iterdir()}=={r['name'] for r in lock['files']}
    for r in lock['files']:
        path=args.assets/r['name'];assert not path.is_symlink() and path.stat().st_size==r['size_bytes'] and sha(path)==r['sha256']
    assert m['encoder_revision']==lock['revision'] and m['encoder_assets']=={r['name']:r['sha256'] for r in lock['files']}
    import torch
    from transformers import RobertaModel,RobertaTokenizerFast
    from marking import marked_words
    torch.set_num_threads(4);torch.manual_seed(13);np.random.seed(13);torch.use_deterministic_algorithms(True)
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(args.assets,local_files_only=True,add_pooling_layer=False,torch_dtype=torch.float32,attn_implementation='eager').to('cpu').eval()
    encoder.requires_grad_(False);encode_seconds=0.
    def encode(batch):
        nonlocal encode_seconds
        marked=[marked_words(r) for r in batch]
        encoded=tokenizer([w for w,_ in marked],is_split_into_words=True,padding=True,truncation=False,return_tensors='pt')
        assert encoded['input_ids'].shape[1]<=512
        with torch.inference_mode():
            before=time.monotonic();hidden=encoder(**encoded).last_hidden_state.float();encode_seconds+=time.monotonic()-before
            result=[]
            for i,row in enumerate(batch):
                mapping=encoded.word_ids(i);original=marked[i][1];words=[]
                for j in range(len(row['tokens'])):
                    pieces=[k for k,w in enumerate(mapping) if w is not None and original[w]==j];assert pieces
                    words.append(hidden[i,pieces].mean(0))
                result.append(torch.stack(words).half().cpu().numpy().copy())
        return result
    # Same first batch as the original CPU producer, no anchor labels in encoder.
    anchors=encode([old_rows[k] for k in old_ids[:4]]);anchor_diff=[];anchor_equal=True
    for i,a in enumerate(anchors):
        lo,hi=map(int,d['train_offsets'][i:i+2]);old=d['train_features'][lo:hi]
        assert a.shape==old.shape and np.allclose(a.astype(np.float32),old.astype(np.float32),atol=.002,rtol=.002)
        anchor_diff.append((a.astype(np.float32)-old.astype(np.float32)).ravel());anchor_equal&=np.array_equal(a,old)
    delta=np.concatenate(anchor_diff);parity=dict(records=4,max_abs=float(np.abs(delta).max()),rmse=float(np.sqrt(np.square(delta).mean())),bitwise_equal=bool(anchor_equal),atol=.002,rtol=.002)
    print(json.dumps(dict(stage='anchor_parity',**parity)),flush=True)
    features=[]
    for start in range(0,313,4):
        features.extend(encode(selected[start:start+4]))
        if start%64==0:print(json.dumps(dict(stage='encode_topup',records=min(start+4,313),encode_seconds=encode_seconds)),flush=True)
    del encoder;gc.collect()
    extra=dict(features=np.concatenate(features),roles=np.concatenate([roles_for(r) for r in selected]),
        offsets=np.cumsum([0]+[len(r['tokens']) for r in selected],dtype=np.int64),targets=np.array([labels.index(r['label']) for r in selected],dtype=np.int64),ids=np.array([sid(r) for r in selected],dtype='S64'))
    validate_arrays(extra,selected,labels)
    np.savez(out/'topup-features.npz',**extra)
    with np.load(out/'topup-features.npz',allow_pickle=False) as f:reopened={n:f[n] for n in f.files}
    validate_arrays(reopened,selected,labels)
    assert {n:array_sha(a) for n,a in extra.items()}=={n:array_sha(a) for n,a in reopened.items()}
    assert sha(args.base/'features.npz')==BASE_SHA and {n:array_sha(a) for n,a in d.items()}==base_array_hashes
    added=np.bincount(extra['targets'],minlength=40).tolist();combined=[a+b for a,b in zip(support,added)]
    assert all(n>=min(16,census[k]) for k,n in zip(labels,combined)) and sum(combined)==2361
    selection=dict(schema='qattn.coverage.private-selection.v1',rule='Keep original2048; per training label add min16-cap available records by ascending SHA256(rmea-coverage-topup-v1:original_id), then IDhash; order labelID then rank.',items=[dict(id_hash=sid(r),label_id=labels.index(r['label']),rank=rank(r)) for r in selected])
    save(out/'selection.json',selection)
    manifest=dict(schema='qattn.coverage.overlay.v1',source_revision=git('rev-parse','HEAD'),producer_sha256=sha(Path(__file__)),contract_sha256=sha(Path(__file__).parent/'contract.json'),base_features_sha256=BASE_SHA,base_manifest_sha256=MANIFEST_SHA,
        base_array_sha256=base_array_hashes,extra_array_sha256={n:array_sha(a) for n,a in extra.items()},source_train_sha256=TRAIN_SHA,encoder_revision=lock['revision'],encoder_assets=m['encoder_assets'],environment=env,
        labels=labels,original_train_support=support,added_train_support=added,combined_train_support=combined,new_records=313,combined_train_records=2361,unchanged_valid_records=1024,
        selection_sha256=sha(out/'selection.json'),topup_features_sha256=sha(out/'topup-features.npz'),selected_id_sequence_sha256=hashlib.sha256(b''.join(extra['ids'])).hexdigest(),dtype=m['dtype'],device='cpu',threads=4,batch_size=4,
        test_loaded=False,head_training=False,formal_runs=0,anchor_parity=parity,base_reencoded=False,new_encoder_records=313,parity_encoder_records=4)
    save(out/'overlay-manifest.json',manifest)
    validation=dict(status='PASS',new_records=313,base_unchanged=True,word_role_target_id_binding=True,private_selection_sha256=manifest['selection_sha256'],anchor_parity=parity,consumer_roundtrip_required=True)
    save(out/'validation.json',validation)
    summary=dict(status='PASS',schema='qattn.coverage.summary.v1',source_revision=manifest['source_revision'],contract_sha256=manifest['contract_sha256'],producer_sha256=manifest['producer_sha256'],labels=labels,
        original_train_support=support,added_train_support=added,combined_train_support=combined,new_records=313,combined_train_records=2361,unchanged_valid_records=1024,all_classes_present=all(n>0 for n in combined),minimum_train_support=min(combined),
        cap=16,official_availability_cap=True,topup_bytes=(out/'topup-features.npz').stat().st_size,base_reused_bytes=(args.base/'features.npz').stat().st_size,base_features_sha256=BASE_SHA,base_manifest_sha256=MANIFEST_SHA,
        overlay_manifest_sha256=sha(out/'overlay-manifest.json'),topup_features_sha256=manifest['topup_features_sha256'],anchor_parity=parity,environment=env,elapsed_seconds=time.monotonic()-started,encode_seconds=encode_seconds,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        test_loaded=False,head_training=False,formal_runs=0,cos_bytes=0,no_source_spectrum_retry=True,consumer_roundtrip_required=True,
        limitation='Exploratory train-only coverage repair. No new utility, class-balance guarantee, model result or evidence-tier promotion. Other dependencies remain externally pinned; no cloud recovery certification.')
    save(out/'summary.json',summary)
    save(out/'RUN_COMPLETE',dict(schema='qattn.coverage.closed.v1',source_revision=manifest['source_revision'],files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    merged,_=load_overlay(args.base,out,sha(out/'RUN_COMPLETE'))
    assert all(np.array_equal(merged['valid_'+k],d['valid_'+k]) for k in ['features','roles','offsets','targets','ids'])
    assert all(np.array_equal(merged['train_'+k][:len(d['train_'+k])],d['train_'+k]) for k in ['features','roles','targets','ids'])
    assert np.array_equal(merged['train_offsets'][:2049],d['train_offsets'])
    print(json.dumps(dict(status='COMPLETE',consumer_roundtrip='PASS',new_records=313,combined_train=2361,topup_bytes=summary['topup_bytes'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
