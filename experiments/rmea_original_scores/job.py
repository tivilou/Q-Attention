"""CPU-only frozen-checkpoint replay, never a training or threshold experiment."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import numpy as np
from common import complete,guard,read,safe,sha
from core import count_batch,finish

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def torch_setup(contract):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    assert torch.__version__==contract['environment']['torch'] and np.__version__==contract['environment']['numpy']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name,digest in contract['legacy_sources'].items():assert sha(ROOT/name)==digest
    sys.path.insert(0,str(HERE.parent/'rmea_context_geometry'))
    from legacy.rmea import RMEAHead
    return torch,RMEAHead


def unit(contract):
    torch,Head=torch_setup(contract)
    scores=np.full((5,40),-9,np.float32);scores[:,0]=4
    scores[0,1]=3;scores[1,2]=2;scores[1,3]=3
    scores[2,3]=2;scores[2,4]=2;scores[3,4]=4
    gold=np.array([1,2,3,4,0],np.int64)
    result=finish(count_batch(scores,gold))
    assert result['eligible']==4 and result['strict_gold_winner']==2
    assert result['criterion']=='FAIL_DEFINED' and result['nonnull_maximum_ties']==1
    assert result['null_positive_maximum_ties']==1
    assert finish(count_batch(scores[4:],gold[4:]))['criterion']=='UNDEFINED_EMPTY'
    bad=scores.copy();bad[0,0]=np.nan
    try:count_batch(bad,gold)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite score accepted')
    torch.manual_seed(20261118)
    with torch.inference_mode():
        for mode in contract['heads']:
            head=Head(mode).eval().requires_grad_(False)
            h=torch.randn(2,5,1024);roles=torch.randn(2,5,4);mask=torch.ones((2,5),dtype=torch.bool)
            z=head(h,roles,mask)[0]
            assert z.dtype==torch.float32 and z.shape==(2,40) and torch.isfinite(z).all()
    # A real tar spans several chunks. A second extraction reuses sealed files;
    # changed headers and changed payload each stop instead of becoming accepted.
    import restore
    with tempfile.TemporaryDirectory(prefix='rmea-original-unit-') as temporary:
        root=Path(temporary);payload=b'fixture-data'*900;archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w',format=tarfile.PAX_FORMAT) as tar:
            info=tarfile.TarInfo('cache/fixture.bin');info.size=len(payload);info.mode=0o600;info.mtime=0
            tar.addfile(info,io.BytesIO(payload))
        raw=archive.getvalue();chunk_size=4096
        for index,start in enumerate(range(0,len(raw),chunk_size)):
            path=root/f'chunks/seed29/{index}.bin';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw[start:start+chunk_size])
        entry=dict(stage='seed29',source_name='cache/fixture.bin',destination='shared/cache/fixture.bin',
            header_offset=0,data_offset=512,header_bytes=512,header_sha256=hashlib.sha256(raw[:512]).hexdigest(),
            bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        plan=dict(files=[entry],chunk_bytes=chunk_size)
        first=restore.extract(root,plan);assert restore.extract(root,plan)==first
        altered=dict(entry,header_sha256='0'*64)
        try:restore.extract(root,dict(files=[altered],chunk_bytes=chunk_size))
        except AssertionError:pass
        else:raise AssertionError('Altered tar header accepted')
        altered=dict(entry,sha256='0'*64)
        try:restore.extract(root,dict(files=[altered],chunk_bytes=chunk_size))
        except AssertionError:pass
        else:raise AssertionError('Altered payload accepted')
    return dict(status='PASS_UNIT',checks=['strict_winner_and_ties','undefined_empty','nonfinite_rejected',
        'three_native_fp32_head_shapes','multi_chunk_tar_selected_extract','sealed_extract_reuse','corrupt_header_and_payload_rejected'],
        synthetic_only=True,cos_requests=0,training=False)


def pinned_inputs(args,contract):
    assert sha(args.plan)==contract['restore_plan_sha256']
    assert sha(args.expected)==contract['expected_report_sha256']
    plan=read(args.plan);expected=read(args.expected);acceptance=read(args.inputs/'restore-acceptance.json')
    assert acceptance['status']=='PASS_SELECTED_FILE_RESTORE' and acceptance['plan_sha256']==contract['restore_plan_sha256']
    assert acceptance['all_tar_headers_verified'] and acceptance['all_inner_hashes_verified']
    assert set(acceptance['files'])=={x['destination'] for x in plan['files']}
    for entry in plan['files']:
        path=safe(args.inputs,'assets/'+entry['destination'])
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    assert expected['seeds']==contract['seeds'] and expected['heads']==contract['heads']
    assert expected['labels'][0]=='no_relation' and len(expected['labels'])==40
    cache=safe(args.inputs,'assets/shared/cache')
    metadata=read(cache/'metadata.json')
    data={}
    for split,n in contract['counts'].items():
        a={key:np.load(cache/f'{split}-{key}.npy',mmap_mode='r',allow_pickle=False) for key in ['h','roles','offsets','targets']}
        assert len(a['targets'])==n and a['offsets'].shape==(n+1,)
        assert a['offsets'][0]==0 and a['offsets'][-1]==len(a['h'])
        assert (np.diff(a['offsets'])>0).all() and a['h'].dtype==np.float16 and a['h'].shape[1]==1024
        assert a['roles'].dtype==np.float32 and a['roles'].shape==(len(a['h']),4)
        assert a['targets'].dtype==np.int64 and a['targets'].min()>=0 and a['targets'].max()<40
        assert len(metadata[split])==n and len({x['id_hash'] for x in metadata[split]})==n
        data[split]=a
    return data,expected


def anchors(torch,head,root,seed,mode,contract):
    errors=[]
    with torch.inference_mode():
        for split in ['valid','test']:
            for index in [0,1]:
                path=safe(root,f'assets/seed{seed}/trace/{mode}-selected-{split}-{index}.npz')
                with np.load(path,allow_pickle=False) as saved:
                    h,roles,mask=[torch.from_numpy(saved[k].copy()) for k in ['h','roles','mask']]
                    original=saved['logits'].copy()
                z=head(h,roles,mask)[0].numpy()
                assert z.dtype==original.dtype==np.float32 and np.isfinite(z).all()
                assert np.allclose(z,original,rtol=contract['anchor_rtol'],atol=contract['anchor_atol'])
                assert np.array_equal(z.argmax(-1),original.argmax(-1))
                errors.append(float(abs(z-original).max()))
    return errors


def batch(torch,data,start,size):
    indices=range(start,min(start+size,len(data['targets'])))
    lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices]
    n=max(lengths);h=np.zeros((len(lengths),n,1024),np.float32)
    roles=np.zeros((len(lengths),n,4),np.float32);mask=np.zeros((len(lengths),n),bool)
    for j,(i,length) in enumerate(zip(indices,lengths)):
        lo,hi=map(int,data['offsets'][i:i+2]);h[j,:length]=data['h'][lo:hi];roles[j,:length]=data['roles'][lo:hi];mask[j,:length]=True
    return torch.from_numpy(h),torch.from_numpy(roles),torch.from_numpy(mask),np.array(data['targets'][list(indices)],np.int64)


def replay(args,contract,preflight=False):
    torch,Head=torch_setup(contract);data,expected=pinned_inputs(args,contract)
    cells=[];anchor_errors={};started=time.monotonic()
    with torch.inference_mode():
        for seed in contract['seeds']:
            for mode in contract['heads']:
                head=Head(mode).eval().requires_grad_(False)
                path=safe(args.inputs,f'assets/seed{seed}/checkpoints/{mode}-selected.pt')
                payload=torch.load(path,map_location='cpu',weights_only=True)
                assert all(v.dtype==torch.float32 for v in payload.values())
                head.load_state_dict(payload,strict=True)
                anchor_errors[f'{seed}:{mode}']=anchors(torch,head,args.inputs,seed,mode,contract)
                if preflight:continue
                for split,d in data.items():
                    totals=None
                    for start in range(0,len(d['targets']),contract['batch_size']):
                        h,roles,mask,gold=batch(torch,d,start,contract['batch_size'])
                        z=head(h,roles,mask)[0].numpy()
                        current=count_batch(z,gold)
                        if totals is None:totals=current
                        else:
                            for key,value in current.items():
                                totals[key]=(np.array(totals[key])+np.array(value)).tolist() if key=='confusion' else totals[key]+value
                    result=finish(totals);original=expected['cells'][f'{seed}:{mode}:{split}']
                    bridge=(result['confusion']==original['confusion'] and result['total']==original['items'])
                    result.update(seed=seed,head=mode,split=split,original_confusion_exact=bridge)
                    if not bridge:result.update(criterion='INVALID_REPLAY_BRIDGE',proportion=None)
                    cells.append(result)
                    print(__import__('json').dumps(dict(stage='replay_cell',seed=seed,head=mode,split=split,
                        eligible=result['eligible'],strict_gold_winner=result['strict_gold_winner'],bridge=bridge)),flush=True)
    if preflight:
        return dict(status='PASS_ORIGINAL_INPUT_PREFLIGHT',anchors=anchor_errors,selected_heads=15,
            original_anchors=60,model_training=False,fullsplit_replay=False,cos_requests=0)
    assert len(cells)==30
    invalid=any(c['criterion']=='INVALID_REPLAY_BRIDGE' for c in cells)
    criterion='INVALID' if invalid else ('FAIL_DEFINED_CELL' if any(c['criterion']=='FAIL_DEFINED' for c in cells) else
        ('UNASSESSABLE_EMPTY' if any(c['criterion']=='UNDEFINED_EMPTY' for c in cells) else 'ALL30_DEFINED_PASS'))
    return dict(schema='qattn.original-scores.diagnostic.v1',status='INVALID_REPLAY_BRIDGE' if invalid else 'PASS_DESCRIPTIVE_REPLAY',
        cells=cells,allcell_criterion=criterion,anchors=anchor_errors,elapsed_seconds=time.monotonic()-started,
        precision='native_float32_cpu',batch_size=contract['batch_size'],original_count_confusion_bridge='Exact for every validcell',
        original_bitwise_gpu_score_equivalence=False,model_training=False,new_method=False,cos_requests=0,
        limits=['Retrospective finite-sample description, not statistical population rejection or a gain guarantee.',
                'CPU replay of frozen original heads; anchor tolerance and count/confusion bridge do not establish every original GPU score bitwise.',
                'Exactfloat32 ties/strictwinner applied without threshold search; accepted L4 unchanged.'],candidate='NONE',L5='NOT_ACCEPTED')


def main():
    guard();parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['unit','preflight','diagnose'],required=True)
    for name in ['inputs','plan','expected']:parser.add_argument('--'+name,type=Path)
    parser.add_argument('--preflight-acceptance',type=Path)
    parser.add_argument('--preflight-acceptance-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();contract=read(HERE/'contract.json')
    if args.action=='diagnose':
        assert args.preflight_acceptance and args.preflight_acceptance_sha256
        assert sha(args.preflight_acceptance)==args.preflight_acceptance_sha256
        accepted=read(args.preflight_acceptance)
        assert accepted['status']=='ACCEPTED_ORIGINAL_INPUT_PREFLIGHT_ONLY'
        assert accepted['contract_sha256']==sha(HERE/'contract.json')
        assert accepted['task_state']=='complete' and accepted['quiescent'] and accepted['returncode']==0
        assert accepted['report']['selected_heads']==15 and accepted['report']['original_anchors']==60
        assert accepted['report']['status']=='PASS_ORIGINAL_INPUT_PREFLIGHT'
        assert accepted['owner_binding_exact_restored'] and accepted['worker_binding_exact_restored']
    result=unit(contract) if args.action=='unit' else replay(args,contract,args.action=='preflight')
    result.update(action=args.action,contract_sha256=sha(HERE/'contract.json'),
        environment=contract['environment'],python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ))
    complete(args.output,args.action,result)
    print(__import__('json').dumps(dict(status=result['status'],action=args.action,training=False)))
    if result['status']=='INVALID_REPLAY_BRIDGE':raise SystemExit(2)


if __name__=='__main__':main()
