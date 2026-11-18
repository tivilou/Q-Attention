"""Managed synthetic regression comparing historical and corrected readers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
DTYPES=('<f4','<f8','<i4','<u8','>i8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(root,expected_hash,baseline,folder):
    sys.path.insert(0,str(root))
    assert digest(root/'experiments/rmea_ecqp_screen/runner.py')==expected_hash
    import numpy as np
    import torch
    from unittest.mock import patch
    from experiments.rmea_ecqp_screen import runner
    assert Path(runner.__file__).resolve()==(root/'experiments/rmea_ecqp_screen/runner.py').resolve()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder.mkdir();assets=folder/'assets';assets.mkdir()
    lengths=np.full(4096,37,dtype='<i8');lengths[:19]+=1
    offsets=np.concatenate([np.zeros(1,dtype='<i8'),lengths.cumsum()])
    assert offsets.dtype==np.dtype('<i8') and offsets[-1]==151571
    h=np.lib.format.open_memmap(assets/'train-h.npy',mode='w+',dtype='<f2',shape=(151571,1024))
    roles=np.zeros((151571,4),dtype='<f4')
    for i in range(4096):
        lo=int(offsets[i]);h[lo,0]=(i%11)/16;h[lo+1,1]=(i%7)/16
        roles[lo,2]=1;roles[lo+1,3]=1
    h.flush();del h
    for n,a in [('train-offsets.npy',offsets),('train-roles.npy',roles),('train-targets.npy',np.arange(4096,dtype='<i8')%40)]:
        np.save(assets/n,a,allow_pickle=False)
    metadata=[dict(id_hash=f'synthetic-{i}',words=int(lengths[i])) for i in range(4096)]
    (assets/'metadata.json').write_text(json.dumps(dict(train=metadata)),encoding='utf8')
    cohort=[dict(id_hash=r['id_hash'],source_index=i,group=f'synthetic-group-{i}',split='fit' if i<3124 else 'eval') for i,r in enumerate(metadata)]
    donors={}
    for split in ['fit','eval']:
        rows=[r for r in cohort if r['split']==split]
        donors[split]=runner.cohort_derangement([r['id_hash'] for r in rows],[r['group'] for r in rows],split)
    cp=folder/'cohort.json';dp=folder/'donors.json';receipt=folder/'prefix-acceptance.json'
    cp.write_text(json.dumps(dict(rows=cohort)),encoding='utf8');dp.write_text(json.dumps(donors),encoding='utf8')
    plan_hash=hashlib.sha256(b'synthetic-dtype-test-no-real-plan').hexdigest()
    table={p.name:dict(bytes=p.stat().st_size,sha256=runner.digest(p)) for p in assets.iterdir()}
    def load(dtype):
        array=offsets.astype(dtype);assert np.array_equal(array,offsets)
        np.save(assets/'train-offsets.npy',array,allow_pickle=False)
        p=assets/'train-offsets.npy';table[p.name]=dict(bytes=p.stat().st_size,sha256=runner.digest(p))
        receipt.write_text(json.dumps(dict(status='PASS_TRAIN_PREFIX_RECOVERY',prefix_examples=4096,
            full_original_H_hash_recomputed=False,sidecar_original_hashes_verified=True,plan_sha256=plan_hash,files=table)),encoding='utf8')
        return runner.read_training_prefix(assets,receipt,cp,runner.digest(cp),dp,runner.digest(dp),
            expected_receipt_sha256=runner.digest(receipt),expected_plan_sha256=plan_hash)
    valid=load('<i8');summary_hash=hashlib.sha256(valid['summaries'].tobytes()).hexdigest();del valid
    rows=[]
    for dtype in DTYPES:
        with patch.object(runner.ECQPHead,'checked_inputs',wraps=runner.ECQPHead.checked_inputs) as pool:
            try:
                data=load(dtype)
            except AssertionError:
                accepted=False
            else:
                accepted=True;assert hashlib.sha256(data['summaries'].tobytes()).hexdigest()==summary_hash;del data
            assert accepted==baseline,(dtype,accepted,baseline)
            assert pool.call_count==(4096 if baseline else 0)
            rows.append(dict(dtype=dtype,numerical_offsets_identical=True,accepted=accepted,entity_pool_calls=pool.call_count))
    valid=load('<i8');assert hashlib.sha256(valid['summaries'].tobytes()).hexdigest()==summary_hash
    print(json.dumps(dict(status='BASELINE_WRONG_DTYPES_ACCEPTED' if baseline else 'MODIFIED_WRONG_DTYPES_REJECTED',
        runner_sha256=expected_hash,valid_int64_before_and_after=True,valid_summary_sha256=summary_hash,cases=rows)),flush=True)


def main():
    assert not sys.flags.optimize and 'PYTHONOPTIMIZE' not in os.environ
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--root',type=Path)
    p.add_argument('--expected-hash');p.add_argument('--baseline',action='store_true');p.add_argument('--folder',type=Path)
    p.add_argument('--baseline-root',type=Path);p.add_argument('--baseline-revision');p.add_argument('--baseline-hash')
    p.add_argument('--modified-hash');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.child:
        child(a.root.resolve(strict=True),a.expected_hash,a.baseline,a.folder);return
    assert a.output and a.baseline_root and a.baseline_revision and a.baseline_hash and a.modified_hash
    base=a.baseline_root.resolve(strict=True)
    assert subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD']).decode().strip()==a.baseline_revision
    assert not subprocess.check_output(['git','-C',str(base),'status','--porcelain=v1']).strip()
    assert digest(base/'experiments/rmea_ecqp_screen/runner.py')==a.baseline_hash
    started=time.monotonic();results={};commands=[]
    with tempfile.TemporaryDirectory(prefix='ecqp-offset-dtype-') as directory:
        for name,root,expected in [('baseline',base,a.baseline_hash),('modified',ROOT,a.modified_hash)]:
            argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--root',str(root),'--expected-hash',expected,'--folder',str(Path(directory)/name)]
            if name=='baseline':argv.append('--baseline')
            r=subprocess.run(argv,capture_output=True,timeout=240)
            commands.append(dict(command=argv,stdout=r.stdout.decode(),stderr=r.stderr.decode(),exit_status=r.returncode))
            assert r.returncode==0,commands[-1];results[name]=json.loads(r.stdout)
    assert results['baseline']['valid_summary_sha256']==results['modified']['valid_summary_sha256']
    assert len(results['baseline']['cases'])==len(results['modified']['cases'])==5
    sys.path.insert(0,str(ROOT))
    from experiments.rmea_original_scores.common import complete
    report=dict(status='PASS_OFFSET_DTYPE_DIFFERENTIAL_SYNTHETIC_UNIT',results=results,commands=commands,
        elapsed_seconds=time.monotonic()-started,wrong_dtypes=5,valid_int64_outputs_identical=True,
        real_data_reads=0,COS_requests=0,task_training=False,utility='NOT_RUN',L5='NOT_ACCEPTED')
    complete(a.output,'unit',report)
    print(json.dumps({k:report[k] for k in ['status','elapsed_seconds','wrong_dtypes','valid_int64_outputs_identical','L5']}))


if __name__=='__main__':main()
