"""One original-environment FIT128 qualification; no selection/qualification access."""
import argparse,hashlib,itertools,json,math,os,signal,sys,time
from pathlib import Path

if sys.flags.optimize:raise RuntimeError('Unoptimized Python required')
TAGS=('phase_presence_q','phase_presence_block_d','phase_presence_real16','gaussian_covariance')
CAPS=dict(wall=900,arm=120,aggregate=600,bytes=4294967296)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def save(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.pending')
    with q.open('w',encoding='utf-8') as f:
        f.write(json.dumps(v,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)
    if os.name=='posix':
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def tracked_backward(backward,forward,inputs,targets,record,commit,synchronize):
    # The next invocation alone does not prove device completion.
    n=len(targets);submitted=0
    def invoke(*x):
        nonlocal submitted
        size=len(x[0]);assert 0<size<=4 and submitted+size<=n
        if submitted:
            synchronize()
        record.update(training_rows_seen=submitted,training_rows_possible=submitted+size,
                      backward_in_flight=True,full_backward_complete=False)
        commit()
        submitted+=size
        return forward(*x)
    loss=backward(invoke,inputs,targets)
    assert submitted==n
    synchronize()
    record.update(training_rows_seen=n,training_rows_possible=n,backward_in_flight=False,full_backward_complete=True)
    commit()
    return loss

def probe_details(model,example):
    import torch
    inp=tuple(x.detach().clone() for x in example)
    # Two padding slots including an interior hole. Preserve all real words.
    h,r,m=inp
    def hole(x,fill):
        shape=list(x.shape);shape[1]=1
        blank=torch.full(shape,fill,device=x.device,dtype=x.dtype)
        return torch.cat((x[:,:1],blank,x[:,1:],blank),1)
    hp,rp,mp=hole(h,float('nan')),hole(r,float('nan')),hole(m,False)
    hp.requires_grad_(True);rp.requires_grad_(True)
    out=model(hp,rp,mp,details=True)
    assert torch.isfinite(out['logits']).all() and torch.isfinite(alpha_of(out)).all()
    assert torch.count_nonzero(alpha_of(out).masked_select(~mp[:,None,:])).item()==0
    grad=torch.autograd.grad(out['logits'].sum(),(hp,rp))
    for g in grad:assert torch.isfinite(g).all() and torch.count_nonzero(g.masked_select(~mp[:,:,None])).item()==0
    with torch.no_grad():plain=model(*inp,details=True)
    keep=[0,*range(2,mp.shape[1]-1)]
    errors=(float((plain['logits']-out['logits']).abs().max()),float((alpha_of(plain)-alpha_of(out)[:,:,keep]).abs().max()))
    return plain,errors

def alpha_of(out):
    return out['alpha'] if 'alpha' in out else out['attention'].transpose(1,2)

def independent_q_reference(model,example):
    """Evaluate native CPU float64 score masses from explicit effect matrices."""
    import numpy as np, torch
    from baselines.retacred.rmea import circuit
    h,r,mask=example
    h,r=h.cpu().double()[0,mask[0].cpu()],r.cpu().double()[0,mask[0].cpu()]
    words=circuit((math.pi*torch.tanh(model.prepare(torch.cat((h,r),1)))).reshape(len(h),2,3,2)).numpy()
    query=circuit(math.pi*torch.tanh(model.effect)).numpy()
    P=np.diag([1,0,1,0,0,1,0,1]);I=np.eye(8)
    eff=[np.outer(c,c.conj()) for c in query];rhos=[np.outer(c,c.conj()) for c in words]
    gated=lambda x:P@x@P+(I-P)@x@(I-P)
    positive=np.asarray([[np.trace(gated(m)@a).real for m in eff[1:]] for a in rhos])
    v=model.value(h).numpy();w=model.decoder.numpy();b=model.bias.numpy();result={}
    for name,E in [('coherent',eff[0]),('block_D',gated(eff[0]))]:
        present=np.asarray([.25+.5*np.trace(E@a).real for a in rhos])
        raw=np.column_stack((present,1-present,positive))+1e-6
        alpha=raw/raw.sum(axis=0,keepdims=True);pooled=alpha.T@v
        u=(pooled[2:]*w[1:]).sum(axis=1)+b[1:];g=(pooled[0]-pooled[1])@w[0]+b[0]
        z=np.r_[0.,g+u-np.logaddexp.reduce(u)]
        result[name]=(torch.from_numpy(z[None,:]),torch.from_numpy(alpha.T[None,:,:]))
    return result

def centered_error(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double()
    return float(((a-a.mean(-1,keepdim=True))-(b-b.mean(-1,keepdim=True))).abs().max())

def c_health(masses,attention,mask,eps):
    n=mask.sum(-1);use=n>3
    mass=(masses*mask[:,None,:]).sum((1,2))/(masses.shape[1]*n)
    tv=(attention-mask[:,None,:]/n[:,None,None]).abs().sum(-1).mean(-1)/2
    return mass[use].detach().cpu()/eps,tv[use].detach().cpu()

def forecast(rows,profile,evals,task_collation_seconds,checkpoint_probe_seconds):
    """Finite screen cost projection, not an empirical speedup or certified upper bound."""
    if set(rows)!=set(TAGS) or set(evals)!=set((*TAGS,'block_D')):return None
    # Each logical assessment pass has 8 selection and 16 qualification batches.
    arm={k:640*(rows[k]['update_seconds']+task_collation_seconds+checkpoint_probe_seconds)+48*evals[k] for k in TAGS}
    arm[TAGS[0]]+=16*evals['block_D']
    # Both installed GPUs are chargeable even when a lane is idle; compute
    # the two-lane fixed roster and keep one-card projection as a comparator.
    lanes=[0.,0.]
    for v in arm.values():lanes[min(range(2),key=lanes.__getitem__)]+=v
    sec1=sum(arm.values())*1.1;sec2=max(lanes)*1.1
    return dict(per_arm_base_seconds=arm,single_gpu_compute_seconds=sec1,two_gpu_compute_seconds=sec2,
                two_gpu_allocated_compute_gpu_seconds=2*sec2,uncertainty_multiplier=1.10,
                fit_updates_per_arm=640,regular_score_batches_per_arm=48,fixed_Q_batches_each=16,
                cost_scope='FIT128 timed batch rates only. Parent startup/queue/release, CPU bootstrap and I/O, full-prefix parent deadline and observed length distribution must still be charged; never an automatic PASS.')

def run(args,root,cfg):
    import copy,numpy as np,torch
    import goal_aligned_runner as runr
    EPS=1e-6
    from rmea_plugin_registry import resolve_plugin
    assert torch.__version__=='2.4.1+cu121'
    declared=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    assert len(declared)==1 and declared[0] and torch.cuda.device_count()==1
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    b,cohort,_,order,_=runr.preflight(root,cfg['stage'])
    sys.path[:0]=[str(root),b['borrowed_directory'],b['source']]
    import sequence_task as task
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_matched_routers import initialize_head
    from baselines.retacred.rmea_token_transport_study import state_identity
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree
    from baselines.retacred.rmea_task_runtime import chunks
    assert sha(task.__file__)==runr.BORROWED['sequence_task.py']
    assert sha(Path(b['source'])/'baselines/retacred/rmea_task_runtime.py')==cfg['runtime_sha256']
    plug=resolve_plugin('phase_presence');assert Path(plug.__file__).resolve()==root/'rmea_phase_presence.py'
    models,initials=runr.checked_models(plug,torch,RMEAHead,initialize_head,state_identity)
    assert initials==cfg['initial_sha256']
    idx=order[0];assert len(idx)==128 and hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==cfg['first128_sha256']
    assert set(idx)<=set(cohort['indices']['fit']) and not set(idx)&(set(cohort['indices']['qualification'])|set(cohort['indices']['selection']))
    data=task.FrozenWords(b['cache'],b['cohort_path'])
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(CAPS['bytes']/torch.cuda.get_device_properties(device).total_memory,device)
    tick=time.monotonic();inp,y=data.batch(idx,device);torch.cuda.synchronize()
    collation=time.monotonic()-tick;assert y.shape==(128,)
    lens=inp[2].sum(-1).cpu().numpy()
    # Outcome-independent lowest-position shortest example in the first microbatch.
    pos=int(np.argmin(lens[:4]));n=int(lens[pos])
    example=tuple(x[pos:pos+1,:n].contiguous() for x in inp)
    assert len(list(chunks(inp)))==32
    rows={};evaluations={};cp=[];start=time.monotonic();stop=None
    for arm in TAGS:
        model=models[arm];assert state_identity(model.state_dict())==initials[arm]
        torch.cuda.reset_peak_memory_stats(device)
        ref=copy.deepcopy(model).cpu().double().eval()
        model=model.to(device).train()
        save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='preflight_'+arm,current=sum(r['updates'] for r in rows.values()),total=4,unit='technical_condition',heartbeat_at=time.time(),observed_at=time.time()))
        plain,padding=probe_details(model,example)
        with torch.no_grad():
            cpu=tuple(x.cpu().double() if x.dtype!=torch.bool else x.cpu() for x in example)
            details=ref(*cpu,details=True)
            parity_logit=centered_error(plain['logits'],details['logits'])
            parity_alpha=float((alpha_of(plain).cpu()-alpha_of(details)).abs().max())
            assert parity_logit<=5e-4 and parity_alpha<=5e-5 and padding[0]<=5e-4 and padding[1]<=5e-5
            comparisons={}
            if arm==TAGS[0]:
                refs=independent_q_reference(ref,example)
                for name,flags in [('coherent',{}),('block_D',dict(dephase=True))]:
                    q=model(*example,details=True,**flags)
                    expected,alpha=refs[name]
                    comparisons[name]=dict(centered_logit_error=centered_error(q['logits'],expected),alpha_error=float((alpha_of(q).cpu()-alpha).abs().max()))
                    assert comparisons[name]['centered_logit_error']<=5e-4 and comparisons[name]['alpha_error']<=5e-5
                assert torch.equal(plain['positive_scores'],q['positive_scores'])
            if arm==TAGS[-1]:
                from legacy_moments import make_head
                old=make_head('gaussian_covariance').double()
                state={('head.'+k if k!='moment_decoder' else k):v for k,v in ref.state_dict().items()}
                old.load_state_dict(state,strict=True)
                aa=old.components(*cpu)[0];zz=old(*cpu)
                comparisons['legacy_Gcov']=dict(centered_logit_error=centered_error(plain['logits'],zz),alpha_error=float((alpha_of(plain).cpu()-aa).abs().max()))
                assert comparisons['legacy_Gcov']['centered_logit_error']<=5e-4 and comparisons['legacy_Gcov']['alpha_error']<=5e-5
        opt=task.optimizer(model);opt.zero_grad(set_to_none=True)
        record=dict(updates=0,optimizer_step_uncertain=False,training_rows_seen=0,training_rows_possible=0,
                    backward_in_flight=False,full_backward_complete=False,centered_logit_error=parity_logit,alpha_error=parity_alpha,
                    padding_error=padding,initial_sha256=initials[arm],probe_comparisons=comparisons)
        rows[arm]=record
        # Journal consumption before any update capable work; failure never authorizes replay.
        save(args.output/'partial-rows.json',rows)
        mass=[];tv=[]
        def training_forward(*x):
            if arm==TAGS[2]:
                d=model(*x,details=True)
                with torch.no_grad():
                    h,r,m=x
                    _,ident=model.masses(h.masked_fill(~m[...,None],0).float(),r.masked_fill(~m[...,None],0).float(),False)
                    a,b=c_health(ident.transpose(1,2),alpha_of(d)[:,2:].detach(),m,EPS)
                mass.append(a);tv.append(b)
                return d['logits']
            return model(*x)
        torch.cuda.synchronize();t=time.monotonic()
        commit=lambda:save(args.output/'partial-rows.json',rows)
        loss=tracked_backward(task.backward,training_forward,inp,y,record,commit,
                              lambda:torch.cuda.synchronize(device))
        task.check_gradients(model)
        assert record['training_rows_seen']==128 and math.isfinite(float(loss))
        gn={k:float(torch.linalg.vector_norm(p.grad)) for k,p in model.named_parameters()}
        assert all(math.isfinite(v) for v in gn.values())
        if arm==TAGS[2]:
            mm=torch.cat(mass);tt=torch.cat(tv)
            op=math.sqrt(sum(v*v for k,v in gn.items() if k.startswith('prepare.')))
            terminal=gn['effect']
            numerator=math.sqrt(sum(v*v for k,v in gn.items() if k not in plug.COMMON))
            denom=math.sqrt(sum(v*v for v in gn.values()))
            c=dict(eligible=int(mm.numel()),fraction_mass_at_least_10eps=float((mm>=10).double().mean()) if len(mm) else 0.,
                   median_tv=float(np.median(tt.numpy())) if len(tt) else 0.,preparer_gradient_norm=op,query_gradient_norm=terminal,
                   route_gradient_fraction=numerator/denom if denom else 0.)
            record['classical_health']=c
            if not (c['eligible']>=16 and c['fraction_mass_at_least_10eps']>=.8 and c['median_tv']>=.001 and op>1e-10 and terminal>1e-10 and c['route_gradient_fraction']>=1e-8):
                record['gate']='C_PREFLIGHT_DEGENERATE';stop='C_PREFLIGHT_DEGENERATE'
                save(args.output/'partial-rows.json',rows)
                break
        before={k:p.detach().clone() for k,p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        record.update(update_attempted=True,optimizer_step_uncertain=True);commit()
        opt.step();torch.cuda.synchronize();seconds=time.monotonic()-t
        after=state_identity(model.state_dict())
        record.update(updates=1,optimizer_step_uncertain=False,update_seconds=seconds,output_sha256=after,gradients=gn)
        commit()
        assert after!=initials[arm] and all(torch.isfinite(p).all() for p in model.parameters())
        changed={k for k,p in model.named_parameters() if not torch.equal(p,before[k])}
        assert set(plug.COMMON)&changed and changed-set(plug.COMMON)
        assert torch.cuda.max_memory_allocated(device)<=CAPS['bytes']
        if seconds>CAPS['arm'] or sum(r.get('update_seconds',0) for r in rows.values())>CAPS['aggregate']:
            stop='TECHNICAL_COST_GATE';break
        checkpoint=args.output/(arm+'.discard-only.private.pt')
        t=time.monotonic()
        atomic_torch(checkpoint,dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict())))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state_identity(saved['model'])==after and len(saved['optimizer']['state'])==len(list(model.parameters()))
        cp.append(time.monotonic()-t);record['discard_checkpoint_verified']=True
        model.eval()
        methods={arm:model}
        if arm==TAGS[0]:methods.update(block_D=model.same_state_block_dephased_logits)
        with torch.no_grad():
            for key,fn in methods.items():
                torch.cuda.synchronize();t=time.monotonic();logits=task.score_batch(fn,inp);torch.cuda.synchronize()
                evaluations[key]=time.monotonic()-t
                assert logits.shape==(128,40) and torch.isfinite(logits).all()
        record['peak_bytes']=int(torch.cuda.max_memory_allocated(device));assert record['peak_bytes']<=CAPS['bytes']
        save(args.output/'partial-rows.json',rows)
        model.cpu();del before,opt,saved,plain,details,ref
    projection=forecast(rows,cfg.get('profile'),evaluations,collation,max(cp,default=0.)) if stop is None else None
    status='PASS_NATIVE_TECHNICAL_ONLY' if stop is None else 'DEFER_EXACT_CANARY'
    return dict(status=status,stop=stop,attempt=args.attempt,profile_sha256=runr.PROFILE,spec_sha256=args.spec_sha256,
         cohort_sha256=runr.COHORT,order_sha256=b['order_sha256'],first128_sha256=cfg['first128_sha256'],
         torch=torch.__version__,one_gpu=True,elapsed_seconds=time.monotonic()-start,results=rows,
         updates=sum(v['updates'] for v in rows.values()),training_exposures=sum(v['training_rows_seen'] for v in rows.values()),
         assessment_rows=0,valid_test_read=False,eval_seconds_fit_batch_only=evaluations,projection=projection,no_warmstart=True)

def guard_parent(a,root):
    assert not a.output.exists()
    assert a.output.resolve()==root/'output' and a.progress.resolve()==root/'progress.json'
    launch=load(root/'launch.json');alloc=launch['allocation'];assert alloc['runner_attempt_id']==a.attempt and len(alloc['gpu_uuids'])==1
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(alloc['gpu_uuids'])
    assert sha(alloc['permit_path'])==alloc['permit_sha256']
    permit=load(alloc['permit_path'])
    for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):assert alloc[k]==permit[k]
    assert launch['argv'][0]==sys.executable and Path(launch['argv'][2]).resolve()==Path(__file__).resolve()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('spec','progress','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('attempt','spec-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    assert sha(a.spec)==a.spec_sha256
    c=load(a.spec);assert c['scope']=='ONE_FIT128_CLONE_PER_ARM' and c['attempt']==a.attempt and c['caps']==CAPS
    assert c['prior_gpu_seconds']==343.895 and c['remaining_gpu_seconds']==3256.105
    signal.alarm(CAPS['wall'])
    guard_parent(a,root)
    assert sha(__file__)==c['canary_sha256'] and sha(root/'legacy_moments.py')==c['legacy_sha256']
    a.output.mkdir()
    summary=run(a,root,c)
    save(a.output/'summary.json',summary)
    save(a.output/'RUN_COMPLETE',dict(summary_sha256=sha(a.output/'summary.json'),partial_sha256=sha(a.output/'partial-rows.json'),spec_sha256=a.spec_sha256,attempt=a.attempt))
    save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='terminal_'+summary['status'],current=summary['updates'],total=4,unit='technical_condition',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps({k:summary[k] for k in ('status','stop','attempt','updates','training_exposures','elapsed_seconds')}))

if __name__=='__main__':main()
