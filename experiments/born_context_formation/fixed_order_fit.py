"""Shared bounded fits for deterministic task adapters and fixed sample orders.

The task owns model/data/loss/optimizer and scientific decisions. This module
owns committed updates, local best/latest/final, bounded lanes and progress.
No network, allocation, hypothesis selection or automatic retry.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
import hashlib,json,os,random,signal,subprocess,threading,time


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def bounded(call,seconds):
    if seconds<=0:raise TimeoutError('Active compute budget exhausted')
    timer=threading.Timer(seconds,lambda:os.kill(os.getpid(),signal.SIGALRM));timer.daemon=True;timer.start()
    started=time.monotonic()
    try:
        result=call();elapsed=time.monotonic()-started
        if elapsed>=seconds:raise TimeoutError('Active compute budget exhausted')
        return result,elapsed
    finally:timer.cancel()


def require_running(stopped):
    if stopped.is_set():raise RuntimeError('Coupled campaign stopped at batch boundary')


def owned_child(spec,*,seconds,deadline,record):
    """Main-thread child ownership, including campaign alarm and timeout evidence."""
    started=time.monotonic();child=None;out='';err='';failure=None;code=None
    alarm_signal=getattr(signal,'SIGALRM',None)
    previous=signal.getsignal(alarm_signal) if alarm_signal is not None else None
    alarm_observed=[False]
    # communicate owns the absolute deadline. An alarm during cleanup must
    # never interrupt termination, reap or the evidence commit.
    def alarm(signum,frame):alarm_observed[0]=True
    if alarm_signal is not None:signal.signal(alarm_signal,alarm)
    try:
        remaining=min(seconds,deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Campaign deadline before child start')
        child=subprocess.Popen(spec['argv'],cwd=spec['cwd'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        remaining=min(started+seconds,deadline)-time.monotonic()
        if remaining<=0:raise TimeoutError('Child deadline exhausted during launch')
        out,err=child.communicate(timeout=remaining);code=child.returncode
    except BaseException as error:
        failure=repr(error)
        if child is not None:
            def stop(hard=False):
                try:
                    if os.name=='posix':os.killpg(child.pid,signal.SIGKILL if hard else signal.SIGTERM)
                    elif child.poll() is None:child.kill() if hard else child.terminate()
                except ProcessLookupError:pass
            # The leader may already have exited while descendants hold pipes.
            stop()
            try:out,err=child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stop(hard=True)
                try:out,err=child.communicate(timeout=5)
                except subprocess.TimeoutExpired as drain_error:
                    def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
                    out,err=decoded(drain_error.output),decoded(drain_error.stderr)
                    child.stdout.close();child.stderr.close();child.wait(timeout=2)
                    failure+='; bounded pipe drain exhausted'
            code=child.returncode
    finally:
        if failure is None and time.monotonic()>=deadline:failure='Campaign deadline exhausted'
        value=dict(command=spec,stdout=out,stderr=err,exit_code=code,failure=failure,
            elapsed_seconds=time.monotonic()-started,child_pid=child.pid if child else None,
            child_reaped=child is None or child.poll() is not None,alarm_observed=alarm_observed[0])
        try:atomic_json(record,value)
        finally:
            if alarm_signal is not None:signal.signal(alarm_signal,previous)
    return value


def lanes(jobs,devices,call,progress):
    """Same ready queue on one or two allocated devices; coupled fail-fast."""
    if not devices or len(set(devices))!=len(devices):raise ValueError('Distinct allocated devices required')
    ready=Queue();stopped=threading.Event();results={};lock=threading.Lock()
    for job in jobs:ready.put(job)
    def lane(device):
        while not stopped.is_set():
            try:job=ready.get_nowait()
            except Empty:return
            progress(job,'running',device=device)
            try:result=call(job,device,stopped)
            except BaseException:
                stopped.set();progress(job,'failed',device=device);raise
            with lock:results[job]=result
            progress(job,'complete',device=device)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(lane,d) for d in devices]
        failures=[]
        for f in futures:
            try:f.result()
            except BaseException as error:stopped.set();failures.append(error)
        if failures:raise failures[0]
    if set(results)!=set(jobs):raise ValueError('Coupled queue incomplete')
    return results


def rng_state(device):
    import numpy as np
    import torch
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
        torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))


def restore_rng(state,device):
    import numpy as np
    import torch
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state(state['torch_cuda'].cpu(),device)


def fit(model,task,batches,folder,*,mode,device,contract_sha,order_sha,progress,stopped,
        atomic_torch,cpu_tree,state_identity,checkpoint_every=32,resume=False,pause_after=0):
    import torch
    folder=Path(folder);folder.mkdir(exist_ok=resume)
    initial=state_identity(model.state_dict());base=lambda:state_identity({n:p for n,p in model.named_parameters() if not p.requires_grad})
    original_base=base();device=torch.device(device)
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats(device);model.to(device).train();opt=task.optimizer(model)
        trainable=[p for p in model.parameters() if p.requires_grad]
        done=0;history=[];best_loss=float('inf');best_state=None;spent=0.
        if resume:
            state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
            assert all(state[k]==v for k,v in dict(mode=mode,contract_sha=contract_sha,order_sha=order_sha,
                initial_sha=initial,base_sha=original_base,total_updates=len(batches)).items())
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],device)
            done=state['next_update'];history=state['losses'];best_loss=state['best_loss'];best_state=state['best_state'];spent=state['active_seconds']
            assert done==len(history) and 0<=done<len(batches)
        started=time.monotonic();first_gradients=state.get('first_gradients') if resume else None
        def save():
            value=dict(schema='fixed-order-fit.checkpoint.v1',mode=mode,contract_sha=contract_sha,order_sha=order_sha,
                initial_sha=initial,base_sha=original_base,total_updates=len(batches),next_update=done,
                model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_state(device),
                losses=history,best_loss=best_loss,best_state=best_state,first_gradients=first_gradients,
                active_seconds=spent+time.monotonic()-started)
            atomic_torch(folder/'latest.pt',value)
        for update in range(done,len(batches)):
            if stopped.is_set():save();require_running(stopped)
            inputs,y=task.batch(batches[update],device);opt.zero_grad(set_to_none=True)
            loss=task.backward(model,inputs,y);task.check_gradients(model)
            if update==0:first_gradients={n:dict(norm=float(p.grad.norm()),nonzero=int(torch.count_nonzero(p.grad)),elements=p.numel()) for n,p in model.named_parameters() if p.requires_grad}
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);opt.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite updated parameter')
            done=update+1;history.append(float(loss));improved=float(loss)<best_loss
            if improved:
                best_loss=float(loss);best_state=cpu_tree(model.state_dict())
                atomic_torch(folder/'best.pt',dict(model=best_state,update=done,loss=best_loss,contract_sha=contract_sha,mode=mode))
            if done%checkpoint_every==0 or improved or done==len(batches) or pause_after and done==pause_after:save()
            progress(mode,'training',step=done,loss=float(loss),device=str(device))
            if pause_after and done==pause_after:
                atomic_json(folder/'RUN_PAUSED',dict(next_update=done,contract_sha=contract_sha))
                model.cpu();return dict(paused=True,next_update=done)
        assert base()==original_base
        state=cpu_tree(model.state_dict());atomic_torch(folder/'final.pt',state)
        final_sha=hashlib.sha256((folder/'final.pt').read_bytes()).hexdigest()
        result=dict(mode=mode,updates=done,sample_exposures=sum(len(x) for x in batches),initial_sha=initial,
            base_sha=original_base,base_unchanged=True,order_sha=order_sha,contract_sha=contract_sha,
            final_sha256=final_sha,final_state_sha256=state_identity(state),first_gradient_inventory=first_gradients,
            losses=history,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            active_seconds=spent+time.monotonic()-started,finished_at=time.time(),
            registered_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in trainable))
        atomic_json(folder/'fit.json',result);model.cpu();return result
