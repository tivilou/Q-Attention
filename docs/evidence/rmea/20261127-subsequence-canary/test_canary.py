"""No remote calls, task rows, optimizer updates or CUDA."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,subprocess,sys,tempfile
import native_canary as c

def rejects(fn,err=AssertionError):
    try:fn()
    except err:return
    raise AssertionError('Bad fixture admitted')

def main():
    with tempfile.TemporaryDirectory(prefix='qattn-canary-guard-') as d:
        root=Path(d).resolve()
        a=SimpleNamespace(output=root/'output',progress=root/'progress.json',attempt='single-test')
        fields=dict(allocation_id='id',generation=1,project_id='q-attention',host_id='fixture',run_id='single',runner_attempt_id=a.attempt)
        c.save(root/'permit.json',fields)
        v=dict(allocation=dict(**fields,permit_path=str(root/'permit.json'),permit_sha256=c.sha(root/'permit.json'),gpu_uuids=['GPU-example']),argv=[sys.executable,'-B',str(Path(c.__file__).resolve())])
        c.save(root/'launch.json',v)
        old=os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES']='GPU-example'
        try:
            c.guard_parent(a,root)
            rejects(lambda:c.guard_parent(SimpleNamespace(output=root/'different',progress=a.progress,attempt=a.attempt),root))
            rejects(lambda:c.guard_parent(SimpleNamespace(output=a.output,progress=a.progress,attempt='wrong'),root))
            v['allocation']['gpu_uuids'].append('GPU-wrong');c.save(root/'launch.json',v)
            rejects(lambda:c.guard_parent(a,root))
            v['allocation']['gpu_uuids'].pop();c.save(root/'launch.json',v)
            a.output.mkdir();rejects(lambda:c.guard_parent(a,root));a.output.rmdir()
            c.save(root/'permit.json',{**fields,'run_id':'wrong'});rejects(lambda:c.guard_parent(a,root))
        finally:
            if old is None:del os.environ['CUDA_VISIBLE_DEVICES']
            else:os.environ['CUDA_VISIBLE_DEVICES']=old
    x={k:dict(update_seconds=1.5) for k in c.TAGS}
    ev={k:.25 for k in (*c.TAGS,'path_D','terminal_D')}
    p=c.forecast(x,{},ev,0.,0.)
    assert p['per_arm_base_seconds'][c.TAGS[0]]==980 and p['per_arm_base_seconds'][c.TAGS[1]]==972
    assert p['two_gpu_allocated_compute_gpu_seconds']>=p['single_gpu_compute_seconds']
    assert c.forecast(x,{},dict(list(ev.items())[:-1]),0.,0.) is None
    r=subprocess.run([sys.executable,'-B','-O',str(Path(c.__file__).resolve()),'--help'],capture_output=True,text=True)
    assert r.returncode and 'Unoptimized Python required' in r.stderr
    with tempfile.TemporaryDirectory(prefix='qattn-canary-journal-') as tmp:
        path=Path(tmp)/'partial.json';record={}
        save=lambda:c.save(path,record)
        def native_like(f,x,y):
            for a in range(0,len(y),4):f(x[a:a+4])
            return 1.0
        calls=[]
        def fail_early(x):
            calls.append(len(x))
            if len(calls)==2:raise ValueError('after a committed four-row microbatch')
        y=list(range(128))
        rejects(lambda:c.tracked_backward(native_like,fail_early,y,y,record,save),ValueError)
        assert c.load(path)==dict(training_rows_seen=4,training_rows_possible=8,backward_in_flight=True,full_backward_complete=False)
        def late_gradient_failure():
            assert c.tracked_backward(native_like,lambda x:None,y,y,record,save)==1.0
            raise ValueError('immediately after full backward, before validation')
        rejects(late_gradient_failure,ValueError)
        assert c.load(path)==dict(training_rows_seen=128,training_rows_possible=128,backward_in_flight=False,full_backward_complete=True)
    print(json.dumps(dict(status='PASS_ACTUAL_CANARY_PARENT_GUARD_AND_COST_FIXTURES',
          negatives=['alternate_output','wrong_attempt','extra_gpu','existing_output','changed_permit','incomplete_cost','python_optimize','mid_backward_4_confirmed_8_possible','post_backward_128_confirmed'],
          canary_sha256=c.sha(c.__file__),test_sha256=c.sha(__file__),real_model_forwards=0,real_optimizer_updates=0,allocations=0)))

if __name__=='__main__':main()
