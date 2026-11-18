"""Managed synthetic recovery/projection tests; real GET is separately gated."""
import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import subprocess
import tarfile
import tempfile
import time
import tokenize
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import complete,guard,read,save,sha,safe
from experiments.rmea_ecqp_recovery import restore


def must_fail(function,errors=(AssertionError,ValueError,RuntimeError,FileNotFoundError)):
    try:function()
    except errors:return
    raise AssertionError('Invalid fixture unexpectedly accepted')


def recovery_unit(folder):
    import numpy as np
    checks=[]
    # The tiny fixture tests the actual extractor with small H, original-size
    # sidecar counts and deliberately cross-chunk tar/NPY/value boundaries.
    side=folder/'sidecars';side.mkdir()
    offsets=np.zeros(58466,dtype='<i8');offsets[1]=1;offsets[2]=3;offsets[3:]=8
    roles=np.zeros((8,4),dtype='<f4');roles[0,2]=roles[1,3]=1
    targets=np.zeros(58465,dtype='<i8')
    for n,a in [('train-offsets.npy',offsets),('train-roles.npy',roles),('train-targets.npy',targets)]:np.save(side/n,a,allow_pickle=False)
    save(side/'metadata.json',dict(train=[dict(id_hash='a',words=1),dict(id_hash='b',words=2)]))
    h=np.arange(8*1024,dtype=np.float32).reshape(8,1024).astype('<f2')/128
    buf=io.BytesIO();np.save(buf,h,allow_pickle=False);payload=buf.getvalue()
    member=tarfile.TarInfo('cache/train-h.npy');member.size=len(payload);member.mode=0o600;member.mtime=0
    header=member.tobuf(format=tarfile.PAX_FORMAT,encoding='utf8',errors='surrogateescape')
    archive=header+payload+b'\0'*8192;chunk_size=1024
    root=folder/'recovery';root.mkdir()
    for i,start in enumerate(range(0,len(archive),chunk_size)):
        target=safe(root,f'chunks/seed29/{i}.bin');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive[start:start+chunk_size])
    plan=dict(source_stage='seed29',chunk_bytes=chunk_size,original_H=dict(header_offset=0,data_offset=512,bytes=len(payload),
        header_sha256=hashlib.sha256(header).hexdigest(),sha256=hashlib.sha256(payload).hexdigest()),
        npy_header_maximum_bytes=4096,expected_shape_first_dimension=8,prefix_examples=2,prefix_tokens=3,
        trusted_sidecars={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in side.iterdir()},
        prefix_role_values_sha256=hashlib.sha256(roles[:3].tobytes()).hexdigest(),
        prefix_offsets_values_sha256=hashlib.sha256(offsets[:3].tobytes()).hexdigest())
    got=restore.extract(root,plan,side)
    a=np.load(root/'assets/train-h.npy',allow_pickle=False)
    assert np.array_equal(a,h[:3]) and got['prefix_values_sha256']==hashlib.sha256(h[:3].tobytes()).hexdigest()
    assert not got['full_original_H_hash_recomputed'] and got['full_original_H_sha256']==hashlib.sha256(payload).hexdigest()
    assert set(got['files'])=={'train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json'}
    assert np.array_equal(np.load(root/'assets/train-offsets.npy'),offsets[:3])
    checks.append('actual_multichunk_NPY_prefix_values_sidecar_hashes_and_distinct_prefix_identity')
    must_fail(lambda:restore.extract(root,plan,side));checks.append('unsealed_or_existing_output_is_not_overwritten')
    bad=copy.deepcopy(plan);bad['original_H']['header_sha256']='0'*64
    must_fail(lambda:restore.parse_header(root,bad))
    bad=copy.deepcopy(plan);bad['expected_shape_first_dimension']=9
    must_fail(lambda:restore.parse_header(root,bad))
    bad=copy.deepcopy(plan);bad['prefix_tokens']=9
    must_fail(lambda:restore.parse_header(root,bad))
    checks.append('wrong_tar_hash_NPY_shape_and_out_of_range_prefix_rejected')
    first=root/'chunks/seed29/0.bin';original=first.read_bytes();broken=bytearray(original);broken[512+10]=255;first.write_bytes(broken)
    # NumPy's malformed header tokenizer can raise TokenError directly, before
    # its usual ValueError conversion. Both explicitly mean rejected input.
    must_fail(lambda:restore.parse_header(root,plan),errors=(ValueError,tokenize.TokenError));first.write_bytes(original)
    middle=root/'chunks/seed29/2.bin';old=middle.read_bytes();middle.write_bytes(b'')
    must_fail(lambda:restore.read_range(root,'seed29',2048,10,chunk_size));middle.write_bytes(old)
    checks.append('malformed_NPY_header_and_truncated_chunk_rejected')
    # Exercise the real download function with a fake opener, never the network.
    raw=b'fixture-pinned-object';entry=dict(id='seed29:0',stage='seed29',index=0,host='fixture.invalid',key='test.bin',
        version='fixture-v1',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    transport=dict(_sha256='fixtureplan',chunk_bytes=64,maximum_get_requests=1,maximum_payload_bytes=len(raw),chunks=[entry])
    url='https://fixture.invalid/test.bin?versionId=fixture-v1'
    class Response(io.BytesIO):
        status=200
        headers={'Content-Length':str(len(raw)),'x-cos-version-id':'fixture-v1'}
    class Opener:
        calls=0
        def open(self,*args,**kwargs):self.calls+=1;return Response(raw)
    opener=Opener();net=folder/'transport';net.mkdir()
    with patch.object(restore.urllib.request,'build_opener',return_value=opener):
        restore.download(net,transport,{'seed29:0':url});restore.download(net,transport,{'seed29:0':url})
    assert opener.calls==1
    failed=folder/'bad-version';failed.mkdir()
    class WrongResponse(Response):headers={'Content-Length':str(len(raw)),'x-cos-version-id':'wrong'}
    class WrongOpener(Opener):
        def open(self,*args,**kwargs):self.calls+=1;return WrongResponse(raw)
    wrong=WrongOpener()
    with patch.object(restore.urllib.request,'build_opener',return_value=wrong):
        must_fail(lambda:restore.download(failed,transport,{'seed29:0':url}))
        must_fail(lambda:restore.download(failed,transport,{'seed29:0':url}))
    assert wrong.calls==1 and read(failed/'network-state.json')['chunks']['seed29:0']['status']=='FAILED_PRESERVE'
    checks.append('pinned_transport_reuse_zero_extra_GET_and_wrong_version_one_attempt_no_retry')
    wrongurl=folder/'bad-url';wrongurl.mkdir()
    with patch.object(restore.urllib.request,'build_opener',return_value=Opener()):
        must_fail(lambda:restore.download(wrongurl,transport,{'seed29:0':'https://other.invalid/test.bin?versionId=fixture-v1'}))
    checks.append('foreign_signed_host_rejected_before_GET')
    return checks


def projection_unit():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from experiments.rmea_ecqp.head import ECQPHead
    from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
    from experiments.rmea_ecqp.job import unit as legacy_unit
    # Reproduce the previous recursive-loading gap from immutable Git bytes,
    # inside the managed synthetic job, without modifying the live source.
    previous=subprocess.check_output(['git','-C',str(ROOT),'show',
        'a7ea12ba06b681e50c41585a6c8d36964604993d:experiments/rmea_ecqp/head.py'])
    assert hashlib.sha256(previous).hexdigest()=='60d8c25ccb0d93b208c815ad8264a280b63ef7d61d7e7a7115930f644bf62c5f'
    namespace={'__name__':'ecqp_previous_fixture'}
    exec(compile(previous,'immutable_previous_ecqp_head','exec'),namespace)
    prior=nn.Module();prior.add_module('head',namespace['ECQPHead']())
    prior_state={n:v.clone() for n,v in prior.state_dict().items()};prior_state['head.projection'][0,0]*=-1
    prior.load_state_dict(prior_state,strict=False)
    assert prior.head.projection[0,0]==prior_state['head.projection'][0,0]
    old=read(ROOT/'experiments/rmea_ecqp/contract.json')
    legacy=legacy_unit(old)
    assert len(legacy['checks'])==11
    torch.manual_seed(20261118)
    h=torch.randn(2,4,1024,dtype=torch.float64)*.1
    roles=torch.zeros(2,4,4,dtype=torch.float64);roles[:,0,2]=1;roles[:,1,3]=1;mask=torch.ones(2,4,dtype=torch.bool)
    native=RMEAHead('quantum').double();head=ECQPHead().double();head.load_native(native)
    y=torch.tensor([1,2]);F.cross_entropy(native(h,roles,mask)[0],y).backward();F.cross_entropy(head(h,roles,mask)[0],y).backward()
    grad_errors={n:float((p.grad-dict(head.named_parameters())[n].grad).abs().max()) for n,p in native.named_parameters()}
    assert max(grad_errors.values())==0
    wrapper=nn.Module();wrapper.add_module('head',head)
    buffer=io.BytesIO();torch.save(wrapper.state_dict(),buffer);buffer.seek(0);state=torch.load(buffer,weights_only=True)
    restored=nn.Module();restored.add_module('head',ECQPHead().double());restored.load_state_dict(state)
    torch.testing.assert_close(restored.head(h,roles,mask)[0],head(h,roles,mask)[0],rtol=0,atol=0)
    rejected=0
    for strict in [True,False]:
        for kind in ['missing','signflip','shape','NaN']:
            bad={n:v.clone() for n,v in state.items()}
            if kind=='missing':del bad['head.projection']
            elif kind=='signflip':bad['head.projection'][0,0]*=-1
            elif kind=='shape':bad['head.projection']=bad['head.projection'][:1]
            else:bad['head.projection'][0,0]=torch.nan
            target=nn.Module();target.add_module('head',ECQPHead().double())
            must_fail(lambda:target.load_state_dict(bad,strict=strict));rejected+=1
    assert rejected==8
    return dict(legacy_regression=legacy,shared_gradient_max_abs=max(grad_errors.values()),recursive_invalid_states_rejected=rejected,
        baseline_behavior='Pinned previous implementation accepts substituted P through outer-module recursive strictFalse loading.',
        modified_behavior='All8 recursive invalid-state combinations rejected; valid save/load forward and shared gradients exact.',
        checks=['zero_B_all_shared_native_gradients_exact','actual_torch_save_load_outer_module_forward_exact',
            'recursive_missing_sign_substitution_bad_shape_NaN_rejected_with_strict_true_and_false'])


def main():
    guard();parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['unit'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    c=read(Path(__file__).with_name('contract.json'))
    for name,digest in c['dependencies'].items():assert sha(ROOT/name)==digest
    assert {n:importlib.metadata.version(n) for n in c['environment']}==c['environment']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ecqp-recovery-unit-') as folder:
        recovery=recovery_unit(Path(folder));projection=projection_unit()
    result=dict(status='PASS_ECQP_RECOVERY_AND_RECURSIVE_PROJECTION_UNIT',recovery_checks=recovery,projection=projection,
        contract_sha256=sha(Path(__file__).with_name('contract.json')),elapsed_seconds=time.monotonic()-started,
        environment={n:importlib.metadata.version(n) for n in c['environment']},
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ),
        task_training=False,real_data_reads=0,COS_requests=0,utility='NOT_RUN',L5='NOT_ACCEPTED')
    complete(args.output,args.action,result)
    print(json.dumps(dict(status=result['status'],recovery_checks=len(recovery),projection_checks=3,legacy_checks=11)))


if __name__=='__main__':main()
