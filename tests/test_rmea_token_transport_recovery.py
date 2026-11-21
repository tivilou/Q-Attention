"""Focused full-state, cadence and exact live-parent interruption acceptance."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

# Set before importing Torch/CUDA, including the fresh-process restore fixture.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_token_transport_protocol as p
from baselines.retacred.rmea_token_transport_recovery import Recovery,atomic_torch,restore_rng


def equal(a,b):
    import torch
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def local_fixture(root,cloud=False):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cfg=p.config('canary');cfg['counts']=dict(train=96,valid=8,test=8)
    output=root/'discovery';(output/'cache').mkdir(parents=True)
    (output/'cache/fixture').write_bytes(b'immutable fixture input')
    recovery=Recovery(output,cfg,enabled=cloud)
    torch.manual_seed(73);random.seed(73);np.random.seed(73)
    head=torch.nn.Linear(3,2).double().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001)
    x=torch.tensor([[.1,.2,.3],[.7,.4,.9]],device='cuda',dtype=torch.float64)
    def step(h,opt):
        opt.zero_grad(set_to_none=True);loss=(h(x)**2).sum();loss.backward();opt.step()
    for _ in range(2):step(head,optimizer)
    atomic_torch(recovery.best,{'fixture':{'state':copy.deepcopy(head.cpu().state_dict())}});head.cuda()
    saved=recovery.save(mode=0,epoch=0,next_start=64,history=[],loss_sum=7.,head=head,optimizer=optimizer,
        completed={},best={'fixture':{'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}},initial_sha='initial',order_sha='order')
    saved_rng=copy.deepcopy(saved['rng'])
    expected_random=(random.random(),np.random.rand(),torch.rand(1),torch.rand(1,device='cuda').cpu())
    step(head,optimizer);expected_head=copy.deepcopy(head.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict())
    if cloud:
        directory=recovery.channel/'010-progress';directory.mkdir(parents=True)
        for f in (recovery.latest,recovery.best):os.link(f,directory/f.name)
        recovery._publish('010-progress',[directory/'latest.pt',directory/'best.pt'],2)
        ack=p.read(directory/'ack.json');rel='cos-channel/010-progress/latest.pt'
        target=root/'restored-latest.pt';h=hashlib.sha256()
        with urllib.request.urlopen(ack['restore_urls'][rel],timeout=90) as src,target.open('xb') as dst:
            for data in iter(lambda:src.read(1024**2),b''):h.update(data);dst.write(data)
        assert h.hexdigest()==ack['objects'][rel]['sha256']
        checkpoint=torch.load(target,map_location='cpu',weights_only=True)
    else:
        checkpoint=Recovery(output,cfg,resume=True).state
    if cloud:
        out=root/'fresh-process-restored.pt'
        subprocess.run([sys.executable,'-B',__file__,'--restore-fixture',str(target),'--output',str(out)],check=True,cwd=p.ROOT)
        restored_result=torch.load(out,map_location='cpu',weights_only=True)
        equal(tuple(v.cpu() if torch.is_tensor(v) else v for v in expected_random),restored_result['random'])
        equal({k:v.cpu() for k,v in expected_head.items()},restored_result['head'])
        from baselines.retacred.rmea_token_transport_recovery import cpu_tree
        equal(cpu_tree(expected_optimizer),restored_result['optimizer'])
    restored=torch.nn.Linear(3,2).double().cuda();restored.load_state_dict(checkpoint['active'])
    opt=torch.optim.AdamW(restored.parameters(),lr=.001);opt.load_state_dict(checkpoint['optimizer'])
    restore_rng(checkpoint['rng'])
    actual_random=(random.random(),np.random.rand(),torch.rand(1),torch.rand(1,device='cuda').cpu())
    equal(expected_random,actual_random);step(restored,opt)
    equal(expected_head,restored.state_dict());equal(expected_optimizer,opt.state_dict())
    bad=copy.deepcopy(checkpoint);bad['cursor']['next_start']=63
    try:recovery._validate_cursor(bad)
    except ValueError:pass
    else:raise AssertionError('Invalid cursor accepted')
    return dict(local_optimizer_bitwise=True,python_numpy_torch_cuda_rng_equal=True,
        cos_pinned_restore_optimizer_bitwise=cloud,fixture_total_updates=recovery.total)


def preparation_fixture(root):
    from baselines.retacred.rmea_token_transport_recovery import cache_inventory
    output=root/'discovery';(output/'cache').mkdir(parents=True)
    (output/'cache/fixture').write_bytes(b'closed preparation')
    cfg=p.config('canary')
    p.atomic_json(output/'manifest.json',dict(cache=cache_inventory(output/'cache'),config=cfg,source_sha256=p.identity(p.sources())))
    recovered=Recovery(output,cfg,resume=True);assert recovered.state is None
    atomic_torch(recovered.best,{'contradictory':'completed validation'})
    try:Recovery(output,cfg,resume=True)
    except ValueError:pass
    else:raise AssertionError('Missing checkpoint with contradictory progress accepted')
    return dict(zero_update_preparation_retry=True,missing_latest_with_progress_rejected=True)


def lock_fixture(root):
    import signal,time
    from baselines.retacred.rmea_token_transport_recovery import run_lock
    root.mkdir(parents=True);lock=root/'.run.lock';ready=root/'ready'
    code='from pathlib import Path;from baselines.retacred.rmea_token_transport_recovery import run_lock;import sys,time;\nwith run_lock(Path(sys.argv[1])):Path(sys.argv[2]).write_text("ready");time.sleep(60)'
    child=subprocess.Popen([sys.executable,'-B','-c',code,str(lock),str(ready)],cwd=p.ROOT)
    try:
        deadline=time.monotonic()+15
        while not ready.exists():
            if child.poll() is not None or time.monotonic()>deadline:raise AssertionError('Lock fixture startup failed')
            time.sleep(.05)
        child.kill();child.wait(timeout=10)
        with run_lock(lock):pass
    finally:
        if child.poll() is None:child.kill();child.wait()
    return dict(sigkill_lock_released=True)


def cadence_fixture(root):
    cfg=p.config('discovery');r=Recovery(root/'discovery',cfg)
    assert r.total==29248
    calls=[];r.enabled=True;r.channel.mkdir(parents=True)
    atomic_torch(r.latest,{'fixture':1});atomic_torch(r.best,{'fixture':2})
    def published(name,files,done):
        calls.append((name,done));p.atomic_json(r.channel/name/'ack.json',{'fixture':True})
    r._publish=published
    for done in range(1,29249):
        r.state={'done':done};r.milestone()
    r.milestone(final=True)
    assert [row[1] for row in calls]==[2925,5850,8775,11700,14624,17549,20474,23399,26324,29248],calls
    assert len(calls)==10
    # Orphaned sealed generation must be drained at its saved cursor before any update.
    orphan=root/'orphan-run'/'discovery';(orphan/'cache').mkdir(parents=True)
    a=Recovery(orphan,cfg,enabled=True);a.state={'done':2925}
    g=a.channel/'010-progress';g.mkdir(parents=True)
    atomic_torch(g/'latest.pt',dict(schema='fixture',done=2925,best={},cursor=dict(mode=0,epoch=1,next_start=35104,history=[{}]),completed={},total=29248))
    drained=[]
    a._publish=lambda name,files,done:drained.append(done)
    a.drain_pending();assert drained==[2925]
    return dict(total=29248,cloud_generations=10,thresholds=[row[1] for row in calls])


def parent_fixture(root,args):
    base=root/'uninterrupted';interrupted=root/'interrupted'
    common=[sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py','run','--development-canary','--canary-only','--execution','serial',
        '--gpu','auto','--data-dir',args.data_dir,'--assets',args.assets]
    def run(argv,ok=True):
        result=subprocess.run(argv,cwd=p.ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if (result.returncode==0)!=ok:
            print(result.stdout[-3000:]);print(result.stderr[-3000:]);raise AssertionError(('Child status differs',result.returncode,argv))
        return result.returncode
    run(common+['--run-dir',str(base)])
    run(common+['--run-dir',str(interrupted),'--pause-after','2'],ok=False)
    assert not (interrupted/'canary/RUN_COMPLETE').exists()
    import torch
    cp=torch.load(interrupted/'canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
    assert cp['done']==2 and cp['cursor']['next_start']==16 and len(cp['completed'])==1
    run(common+['--run-dir',str(interrupted),'--resume','--inject-export-failure'],ok=False)
    assert (interrupted/'canary/RUN_COMPLETE').exists()
    target=interrupted.parent/(interrupted.name+'-safe-report');assert not target.exists()
    before=p.digest(interrupted/'canary/RUN_COMPLETE')
    run([sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py','export','--development-canary','--run-dir',str(interrupted)])
    assert p.digest(interrupted/'canary/RUN_COMPLETE')==before
    from baselines.retacred.rmea_token_transport_study import state_identity
    for kind in p.MODES:
        first=torch.load(base/'canary/checkpoints'/(kind+'-final.pt'),map_location='cpu',weights_only=True)
        second=torch.load(interrupted/'canary/checkpoints'/(kind+'-final.pt'),map_location='cpu',weights_only=True)
        assert state_identity(first)==state_identity(second)
    a=p.read(base/'canary/summary.json');b=p.read(interrupted/'canary/summary.json')
    for kind in p.MODES:
        for key in ('valid','test','history','updates','final_state_sha256'):
            assert a['results'][kind][key]==b['results'][kind][key],(kind,key)
    boundary=root/'boundary'
    run(common+['--run-dir',str(boundary),'--pause-after','-1'],ok=False)
    saved=torch.load(boundary/'canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
    assert saved['done']==1 and saved['cursor']['mode']==1 and saved['active'] is None
    run(common+['--run-dir',str(boundary),'--resume'])
    for kind in p.MODES:
        first=torch.load(base/'canary/checkpoints'/(kind+'-final.pt'),map_location='cpu',weights_only=True)
        second=torch.load(boundary/'canary/checkpoints'/(kind+'-final.pt'),map_location='cpu',weights_only=True)
        assert state_identity(first)==state_identity(second)
    return dict(fresh_head_boundary_bitwise=True,real_parent_four_heads_bitwise=True,budget_not_repeated=True,export_failure_cleanup=True,export_only_retry_immutable=True)


def restore_fixture(checkpoint,output):
    import torch
    import numpy as np
    from baselines.retacred.rmea_token_transport_recovery import cpu_tree
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    value=torch.load(checkpoint,map_location='cpu',weights_only=True)
    head=torch.nn.Linear(3,2).double().cuda();head.load_state_dict(value['active'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001);optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rng'])
    randoms=(random.random(),np.random.rand(),torch.rand(1),torch.rand(1,device='cuda').cpu())
    x=torch.tensor([[.1,.2,.3],[.7,.4,.9]],device='cuda',dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True);loss=(head(x)**2).sum();loss.backward();optimizer.step()
    atomic_torch(output,dict(random=randoms,head=cpu_tree(head.state_dict()),optimizer=cpu_tree(optimizer.state_dict())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restore-fixture',type=Path)
    parser.add_argument('--data-dir');parser.add_argument('--assets');parser.add_argument('--cloud',action='store_true')
    args=parser.parse_args()
    if args.restore_fixture:restore_fixture(args.restore_fixture,args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    result={'local':local_fixture(args.output/('recovery-'+args.output.parent.name),args.cloud),'cadence':cadence_fixture(args.output/'cadence'),'lock':lock_fixture(args.output/'lock'),'preparation':preparation_fixture(args.output/'preparation')}
    if args.data_dir:result['parent']=parent_fixture(args.output/'parent',args)
    p.atomic_json(args.output/'recovery-validation.json',result)
    (args.output/'RUN_COMPLETE').write_text('focused recovery checks passed\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
