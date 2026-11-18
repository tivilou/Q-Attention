"""Single-pass, version-pinned real prefix recovery through the shared scheduler."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import urllib.parse

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.rmea_original_scores.common import complete,guard,read,safe,sha


def private_file(path):
    path=Path(path).absolute()
    assert path.is_file() and all(not p.is_symlink() for p in [path,*path.parents])
    assert path.stat().st_mode & 0o077==0, 'Private input must be owner-only'
    return path


def main():
    guard();p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--urls',type=Path,required=True)
    p.add_argument('--sidecars',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cpath=Path(__file__).with_name('real_contract.json');c=read(cpath)
    assert c['schema']=='qattn.ecqp.real-prefix-recovery.v1' and c['allowed_actions']==['recover']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert {n:importlib.metadata.version(n) for n in c['environment']}==c['environment']
    for n,h in c['dependencies'].items():assert sha(ROOT/n)==h
    plan_path=private_file(a.plan);urls_path=private_file(a.urls)
    assert sha(plan_path)==c['plan_sha256']
    plan=read(plan_path)
    assert plan['maximum_payload_bytes']==c['maximum_payload_bytes']==402653184
    assert plan['maximum_get_requests']==c['maximum_get_requests']==6
    assert plan['attempts_per_chunk']==c['max_attempts']==1 and c['cloud_writes']==0
    root=a.root.absolute();output=a.output.absolute();sidecars=a.sidecars.absolute()
    for path in [root,output,sidecars]:assert all(not p.is_symlink() for p in [path,*path.parents])
    assert root!=output and not root.is_relative_to(output) and not output.is_relative_to(root)
    assert not root.exists(), 'Fresh attempt only; inspect a previous recovery instead of replaying'
    assert not output.exists() or not list(output.iterdir())
    urls=read(urls_path);assert set(urls)=={row['id'] for row in plan['chunks']} and len(urls)==6
    for row in plan['chunks']:
        u=urllib.parse.urlsplit(urls[row['id']])
        assert u.scheme=='https' and u.hostname==row['host'] and u.port in (None,443)
        assert not u.username and not u.password and not u.fragment
        assert urllib.parse.unquote(u.path)=='/'+row['key']
        assert urllib.parse.parse_qs(u.query).get('versionId')==[row['version']]
    from experiments.rmea_ecqp_recovery import restore
    started=time.monotonic()
    receipt=restore.run(plan_path,c['plan_sha256'],urls_path,sidecars,root)
    assert receipt['status']=='PASS_TRAIN_PREFIX_RECOVERY' and receipt['reproducible_chunks_removed']
    assert receipt['COS_GET_requests']==receipt['new_GET_requests']==6 and receipt['payload_bytes']==402653184
    assert receipt['automatic_retries']==0 and not receipt['full_original_H_hash_recomputed']
    assert receipt['sidecar_original_hashes_verified'] and not receipt['task_training'] and receipt['model_forwards']==0
    assert not safe(root,'recovery.lock').exists()
    assert not any(safe(root,'chunks').rglob('*.bin'))
    assert set(receipt['files'])=={'train-h.npy','train-offsets.npy','train-roles.npy','train-targets.npy','metadata.json'}
    for name,row in receipt['files'].items():
        path=safe(root,'assets/'+name)
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    report=dict(status='PASS_REAL_PINNED_TRAIN_PREFIX_RECOVERY',contract_sha256=sha(cpath),
        plan_sha256=c['plan_sha256'],prefix_receipt_sha256=sha(root/'prefix-acceptance.json'),
        files=receipt['files'],header=receipt['header'],prefix_values_sha256=receipt['prefix_values_sha256'],
        full_original_H_hash_recomputed=False,sidecar_original_hashes_verified=True,
        prefix_examples=4096,prefix_tokens=151571,hidden_size=1024,
        COS_GET_requests=6,new_GET_requests=6,payload_bytes=402653184,automatic_retries=0,cloud_writes=0,
        reproducible_chunks_removed=True,long_lived_credentials_on_worker=False,elapsed_seconds=time.monotonic()-started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ),
        environment=c['environment'],real_data_reads=True,model_forwards=0,task_training=False,utility='NOT_RUN',L5='NOT_ACCEPTED')
    complete(output,'recover',report)
    print(json.dumps(dict(status=report['status'],GET_requests=6,payload_bytes=402653184,
        prefix_examples=4096,prefix_tokens=151571,model_forwards=0,task_training=False)),flush=True)


if __name__=='__main__':main()
