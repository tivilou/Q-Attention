"""Bounded common-feature baseline check; never trains on the full dataset.

No test data is loaded. Quantum mechanisms and formal training are deliberately
absent. All three baselines use identical frozen word features and data order.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.retacred.common import atomic_json, digest, metrics
from scripts.audit_rmea_inputs import EXPECTED
from scripts.audit_rmea_tokenization import marked_words

AUDIT_SHA='c21d23017140d89a3632853d0f33d9e76c5cb7d574d4a9108b63d1ed23bf0996'
TOKEN_AUDIT_SHA='4d25f24c525e7f706d829a5b5c90750a86d984404cb4fa8251813d7274e2e44a'
SEED=13
LIMITS={'train':2048,'valid':1024}
MODES=('uniform','relation_softmax','real_overlap')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def finish(run_dir,output):
    """Export only closed artifacts, without importing Torch or rerunning work."""
    if output.resolve().is_relative_to(ROOT) or run_dir.resolve().is_relative_to(ROOT):
        raise ValueError('Keep all private results outside source checkout')
    summary=json.loads((run_dir/'summary-private.json').read_text())
    if summary['revision']!=git('rev-parse','HEAD') or git('status','--porcelain'):
        raise ValueError('Export source differs or is dirty')
    for name,row in summary['artifacts'].items():
        path=run_dir/name
        if name not in ('features.npz','heads.pt') or path.is_symlink() or digest(path)!=row['sha256']:
            raise ValueError('Closed artifact missing or changed')
    if (output/'result.json').exists():raise FileExistsError('Preserve existing result')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'result.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--input-audit',type=Path)
    parser.add_argument('--tokenization-audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--export-only',type=Path,metavar='CLOSED_RUN')
    parser.add_argument('--fail-before-result',action='store_true')
    args=parser.parse_args()
    if args.export_only:
        finish(args.export_only,args.output);return
    if git('status','--porcelain'):raise ValueError('Clean exact source is required')
    revision=git('rev-parse','HEAD')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Requires one queue-assigned GPU')
    if not all((args.data_dir,args.assets,args.input_audit,args.tokenization_audit)):
        parser.error('Data/assets and both audited inspection paths are required')
    if digest(args.input_audit)!=AUDIT_SHA or digest(args.tokenization_audit)!=TOKEN_AUDIT_SHA:
        raise ValueError('Readiness inspection identity differs')
    inspection=json.loads(args.input_audit.read_text())
    token_inspection=json.loads(args.tokenization_audit.read_text())
    if not inspection['dataset_identity_matches_reference']:raise ValueError('Data identity not verified')
    if any(row['over_encoder_limit512'] or row['missing_entity_rows'] or row['missing_word_rows']
           for row in token_inspection['splits'].values()):raise ValueError('Token coverage incomplete')
    for split in LIMITS:
        if digest(args.data_dir/(split+'.jsonl'))!=EXPECTED[split][1]:raise ValueError('Data changed')
    asset_lock=json.loads((ROOT/'baselines/retacred/assets.lock.json').read_text())
    for row in asset_lock['files']:
        file=args.assets/row['name']
        if file.is_symlink() or file.stat().st_size!=row['size_bytes'] or digest(file)!=row['sha256']:
            raise ValueError('Pinned encoder asset mismatch')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Select a fresh output directory')
    if args.output.resolve().is_relative_to(ROOT):raise ValueError('Keep private outputs outside checkout')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.frozen_feature_head import FrozenFeatureHead as Head
    from transformers import RobertaModel,RobertaTokenizerFast
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0')
    if torch.cuda.device_count()!=1:raise ValueError('Expected exactly one assigned GPU')
    started=time.monotonic()
    labels=inspection['train_derived_labels'];assert len(labels)==40
    label_to_id={name:i for i,name in enumerate(labels)}
    def choose(split,limit):
        def rows():
            with (args.data_dir/(split+'.jsonl')).open(encoding='utf8') as handle:
                for line in handle:
                    row=json.loads(line)
                    rank=hashlib.sha256(('rmea-readiness-v1:'+split+':'+str(row['metadata']['id'])).encode()).hexdigest()
                    yield rank,row
        selected=heapq.nsmallest(limit,rows(),key=lambda pair:pair[0])
        return [row for _,row in selected]
    selected={split:choose(split,limit) for split,limit in LIMITS.items()}
    tokenizer=RobertaTokenizerFast.from_pretrained(args.assets,add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(args.assets,local_files_only=True,add_pooling_layer=False,
                                         torch_dtype=torch.float16).to(device).eval()
    encoder.requires_grad_(False)
    cached={};arrays={};record_ids={}
    for split,rows in selected.items():
        features=[];offsets=[0];role_features=[];targets=[];sample_ids=[]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            marked=[marked_words(row) for row in batch]
            encoded=tokenizer([words for words,_ in marked],is_split_into_words=True,padding=True,
                              truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>512:raise ValueError('Unexpected length; no truncation allowed')
            with torch.inference_mode():
                hidden=encoder(**{k:v.to(device) for k,v in encoded.items()}).last_hidden_state.float()
                for i,row in enumerate(batch):
                    original=marked[i][1];mapping=encoded.word_ids(i)
                    word_features=[]
                    for j in range(len(row['tokens'])):
                        pieces=[k for k,w in enumerate(mapping) if w is not None and original[w]==j]
                        if not pieces:raise ValueError('Original word disappeared')
                        word_features.append(hidden[i,pieces].mean(0))
                    # NumPy copies remove inference-tensor provenance before head autograd.
                    words=torch.stack(word_features).half().cpu().numpy().copy()
                    features.append(words);offsets.append(offsets[-1]+len(words))
                    positions=np.arange(len(words),dtype=np.float32)
                    subj,obj=row['subject'],row['object']
                    roles=np.stack([np.clip((positions-(subj[0]+subj[1]-1)/2)/32,-1,1),
                                    np.clip((positions-(obj[0]+obj[1]-1)/2)/32,-1,1),
                                    ((positions>=subj[0])&(positions<subj[1])).astype(np.float32),
                                    ((positions>=obj[0])&(positions<obj[1])).astype(np.float32)],axis=-1)
                    role_features.append(roles);targets.append(label_to_id[row['label']])
                    sample_ids.append(hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest())
            if start%256==0:print(json.dumps({'stage':'extract','split':split,'records':min(start+4,len(rows))}),flush=True)
        flat=np.concatenate(features);roles=np.concatenate(role_features);off=np.array(offsets,dtype=np.int64)
        arrays.update({split+'_features':flat,split+'_roles':roles,split+'_offsets':off,
                       split+'_targets':np.array(targets,dtype=np.int64),split+'_ids':np.array(sample_ids,dtype='S64')})
        cached[split]=(flat,roles,off,targets)
        record_ids[split]=hashlib.sha256(''.join(sample_ids).encode()).hexdigest()
    del encoder,hidden
    torch.cuda.empty_cache()
    np.savez(args.output/'features.npz',**arrays)
    del arrays,selected

    def batch(split,indices):
        flat,roles,offsets,targets=cached[split]
        lengths=[offsets[i+1]-offsets[i] for i in indices];maxlen=max(lengths)
        h=np.zeros((len(indices),maxlen,1024),np.float32);r=np.zeros((len(indices),maxlen,4),np.float32)
        mask=np.zeros((len(indices),maxlen),bool)
        for j,(i,length) in enumerate(zip(indices,lengths)):
            h[j,:length]=flat[offsets[i]:offsets[i+1]];r[j,:length]=roles[offsets[i]:offsets[i+1]];mask[j,:length]=True
        return (torch.from_numpy(h).to(device),torch.from_numpy(r).to(device),torch.from_numpy(mask).to(device),
                torch.tensor([targets[i] for i in indices],device=device))

    def evaluate(head):
        head.eval();gold=[];pred=[];loss=0.
        with torch.no_grad():
            for start in range(0,LIMITS['valid'],128):
                h,r,mask,y=batch('valid',list(range(start,min(start+128,LIMITS['valid']))))
                logits,weights=head(h,r,mask)
                if not torch.isfinite(logits).all() or not torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-5):
                    raise ValueError('Invalid logits or attention normalization')
                if weights.masked_select(~mask[:,None,:].expand_as(weights)).abs().sum()!=0:raise ValueError('Padding receives attention')
                loss+=F.cross_entropy(logits,y,reduction='sum').item();gold+=y.tolist();pred+=logits.argmax(-1).tolist()
        return metrics(gold,pred,labels,loss/len(gold))

    results={};checkpoints={};common_init={}
    for mode in MODES:
        torch.manual_seed(SEED);head=Head(mode).to(device)
        init=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in [head.value.weight,head.value.bias,head.decoder,head.bias])).hexdigest()
        common_init[mode]=init
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
        initial=evaluate(head);epochs=[];best=None;best_state=None;gradient_norms={}
        for epoch in range(2):
            head.train();order=torch.randperm(LIMITS['train'],generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
            for start in range(0,len(order),128):
                h,r,mask,y=batch('train',order[start:start+128]);optimizer.zero_grad(set_to_none=True)
                logits,_=head(h,r,mask);loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                if epoch==0 and start==0:
                    gradient_norms={name:float(p.grad.norm()) for name,p in head.named_parameters() if p.grad is not None}
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            report=evaluate(head);epochs.append(report)
            if best is None or report['macro_f1']>best['macro_f1']:
                best=report;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            print(json.dumps({'stage':'head','mode':mode,'epoch':epoch+1,'valid_macro_f1':report['macro_f1']}),flush=True)
        results[mode]=dict(initial=initial,epochs=epochs,selected=best,gradient_norms_first_update=gradient_norms,
                           parameters=sum(p.numel() for p in head.parameters()),updates=32)
        checkpoints[mode]=best_state
        del optimizer,head
    if len(set(common_init.values()))!=1:raise ValueError('Common value/decoder initialization differs')
    torch.save(checkpoints,args.output/'heads.pt')
    for split in LIMITS:
        if digest(args.data_dir/(split+'.jsonl'))!=EXPECTED[split][1]:raise ValueError('Data changed during pilot')
    majority=labels.index(max(inspection['splits']['train']['label_counts'],key=inspection['splits']['train']['label_counts'].get))
    majority_report=metrics(cached['valid'][3],[majority]*LIMITS['valid'],labels,None)
    summary=dict(schema='qattn.rmea.baseline-readiness.v1',scope='technical_only',revision=revision,
                 counts=LIMITS,seed=SEED,epochs=2,test_loaded=False,full_data_training=False,
                 labels=labels,results=results,majority=majority_report,common_value_decoder_init=common_init,
                 sample_id_hashes=record_ids,elapsed_seconds=time.monotonic()-started,
                 encoder_revision=asset_lock['revision'],encoder_sha256=next(r['sha256'] for r in asset_lock['files'] if r['name']=='pytorch_model.bin'),
                 input_audit_sha256=AUDIT_SHA,tokenization_audit_sha256=TOKEN_AUDIT_SHA,
                 environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()),
                 artifacts={name:dict(sha256=digest(args.output/name),bytes=(args.output/name).stat().st_size) for name in ('features.npz','heads.pt')},
                 limitations=['One small pilot, not formal utility, novelty or L2/L3 evidence.',
                              'B0/B1/B2 are task-matched original heads, not reproductions of named papers.',
                              'Parameter counts are reported; no effective-capacity equality is claimed.',
                              'Official document/sentence/input overlaps remain disclosed in the input inspection.',
                              'Per-sample full mechanism traces are not certified by this qualification pilot.'])
    atomic_json(args.output/'summary-private.json',summary)
    if args.fail_before_result:raise RuntimeError('Injected export failure; retry with --export-only, never retrain')
    finish(args.output,args.output)


if __name__=='__main__':main()
