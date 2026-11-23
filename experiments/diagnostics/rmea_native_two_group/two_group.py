"""Three native effect directions, one checkpoint, zero parameter updates."""
import argparse,hashlib,importlib.util,json,math,signal,sys,time
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    q=p.with_suffix('.pending');q.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());q.replace(p)


def select(g0,gp,ordinary,zero=1e-12):
    """Calibration-only Euclidean directions. Degeneracy is explicit."""
    g0,gp,ordinary=map(np.asarray,(g0,gp,ordinary))
    assert g0.shape==gp.shape==ordinary.shape and all(np.isfinite(g).all() for g in (g0,gp,ordinary))
    delta=g0-gp;den=float(np.vdot(delta,delta));scale=max(1.,float(np.vdot(g0,g0)),float(np.vdot(gp,gp)))
    lam=None if den<=zero**2*scale else float(np.clip(-np.vdot(gp,delta)/den,0.,1.))
    common=np.zeros_like(g0) if lam is None else lam*g0+(1-lam)*gp
    candidates=dict(frequency=ordinary,balanced=(g0+gp)/2,common=common)
    directions={};info={}
    for name,g in candidates.items():
        norm=float(np.linalg.norm(g));valid=norm>zero and (name!='common' or lam is not None)
        directions[name]=-g/norm if valid else np.zeros_like(g)
        info[name]=dict(valid=bool(valid),norm=norm,
            calibration_group_derivatives=[float(np.vdot(g0,directions[name])),float(np.vdot(gp,directions[name]))] if valid else None)
    if info['common']['valid']:
        assert max(info['common']['calibration_group_derivatives'])<=-info['common']['norm']+1e-9*max(1.,info['common']['norm'])
    return directions,dict(lambda_class0=lam,denominator=den,directions=info)


def fixtures():
    a=np.array([1.,0.]);b=np.array([0.,2.]);d,s=select(a,b,(a+b)/2)
    assert abs(s['lambda_class0']-.8)<1e-12
    assert max(s['directions']['common']['calibration_group_derivatives'])<max(s['directions']['balanced']['calibration_group_derivatives'])
    assert not select(a,-a,a)[1]['directions']['common']['valid']
    assert not select(a,a,a)[1]['directions']['common']['valid']
    assert not select(np.zeros(2),a,a)[1]['directions']['common']['valid']
    return dict(status='PASS',two_objective_closed_form=True,opposite_equal_and_zero_degeneracies_unresolved=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):
        ap.add_argument('--'+n,required=True,type=Path)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    allocation=json.loads(a.allocation.read_bytes())['allocation']
    assert allocation['runner_attempt_id']==a.attempt and len(allocation['gpu_uuids'])==1
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(a.source/n)==h
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    assert digest(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    prior=a.reference/'output/diagnostic.private.npz';assert digest(prior)==cfg['split_reference_sha256']
    parent=Path(cfg['completed_direction_control']);cached=parent/'output/diagnostic.private.npz'
    assert digest(cached)==cfg['completed_direction_sha256']
    assert digest(parent/'transfer.py')==cfg['unchanged_helper_sha256']
    spec=importlib.util.spec_from_file_location('accepted_native_tables',parent/'transfer.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    a.output.mkdir(exist_ok=True);assert not (a.output/'started.json').exists()
    write(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    signal.alarm(cfg['timeout_seconds']);started=time.time();check=fixtures()
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(a.source));from baselines.retacred.rmea import RMEAHead,circuit,EPS
    assert EPS==1e-6
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.device_count()==1
    model=RMEAHead('quantum').double();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    model.to('cuda');model.eval()
    for p in model.parameters():p.requires_grad_(False)
    model.effect.requires_grad_(True);initial=model.effect.detach().clone()
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['effect'] and model.effect.numel()==480
    old=dict(np.load(cached,allow_pickle=False));prior_arrays=dict(np.load(prior,allow_pickle=False))
    y=old['labels'];cal=old['calibration'];groups=old['groups'];prior_z=old['logits']
    assert y.shape==(8192,) and int(cal.sum())==4192
    assert all(np.array_equal(old[k],prior_arrays[k]) for k in ('labels','calibration','groups'))
    assert not set(groups[cal])&set(groups[~cal])
    fit0=np.flatnonzero(cal&(y==0));n0=len(fit0);np_=int((cal&(y!=0)).sum())
    assess0=int((~cal&(y==0)).sum());assessp=int((~cal&(y!=0)).sum())
    assert [n0,np_,assess0,assessp]==cfg['group_counts']
    cache=a.inputs/'discovery/cache'
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    assert np.array_equal(data['targets'][:8192],y)

    def batch(ids):
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
        h=np.zeros((len(ids),max(lengths),1024),np.float64);r=np.zeros(h.shape[:2]+(4,),np.float64);m=np.zeros(h.shape[:2],bool)
        for j,i in enumerate(ids):
            lo,hi=map(int,data['offsets'][i:i+2]);n=hi-lo
            h[j,:n]=data['h'][lo:hi];r[j,:n]=data['roles'][lo:hi];m[j,:n]=True
        return tuple(torch.from_numpy(x).to('cuda') for x in (h,r,m)),torch.tensor(y[ids],device='cuda',dtype=torch.long)

    def progress(phase,current,total):
        write(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,current=current,
            total=total,unit='example',observed_at=time.time(),heartbeat_at=time.time()))

    # Reuse the exact previous calibration mean gradient; only its class0
    # component needs a new backward pass. Group-positive gradient follows
    # from the row-count identity, without recomputing the completed run.
    g0=torch.zeros_like(model.effect)
    for lo in range(0,n0,cfg['microbatch_size']):
        ids=fit0[lo:lo+cfg['microbatch_size']];inputs,target=batch(ids);z,_=model(*inputs)
        g0+=torch.autograd.grad(F.cross_entropy(z,target,reduction='sum')/n0,model.effect)[0]
        if lo%128==0:progress('calibration_class0_gradient',min(lo+len(ids),n0),n0)
    g0=g0.cpu().numpy();gm=old['gcal'];gp=(len(np.flatnonzero(cal))*gm-n0*g0)/np_
    directions,selection=select(g0,gp,gm,cfg['zero_gradient_tolerance'])
    assert np.max(np.abs(directions['frequency']-old['direction']))<1e-12
    directions_array=np.stack([directions[k] for k in cfg['modes']])
    np.save(a.output/'directions.private.npy',directions_array,allow_pickle=False)
    frozen=dict(chosen_at=time.time(),selection=selection,group_counts=[n0,np_,assess0,assessp],
        directions_sha256=digest(a.output/'directions.private.npy'),rule='Calibration only; original/equal-group/minimum-norm convex gradient; all480 native effect coordinates')
    write(a.output/'directions-frozen.json',frozen)
    phi=circuit(math.pi*torch.tanh(initial));dphis={};endpoints={};phi_fd=0.
    for name in ('balanced','common'):
        if not selection['directions'][name]['valid']:continue
        direction=torch.from_numpy(directions[name]).to('cuda')
        p,dp=torch.autograd.functional.jvp(lambda theta:circuit(math.pi*torch.tanh(theta)),initial,direction)
        assert torch.max(torch.abs(p-phi))<1e-12
        h=cfg['finite_difference_step'];plus=circuit(math.pi*torch.tanh(initial+h*direction));minus=circuit(math.pi*torch.tanh(initial-h*direction))
        error=float((dp-(plus-minus)/(2*h)).abs().max());assert error<cfg['finite_difference_tolerance'];phi_fd=max(phi_fd,error)
        dphis[name]=dp;endpoints[name]=(plus,minus)
    ga0=torch.zeros_like(model.effect);derivatives={k:[] for k in dphis};native_error=0.;fd_error=0.
    assessment_started=time.time();assert assessment_started>=frozen['chosen_at']
    for lo in range(0,len(y),cfg['microbatch_size']):
        ids=np.arange(lo,min(lo+cfg['microbatch_size'],len(y)));inputs,target=batch(ids)
        detail=model(*inputs,details=True);z=detail['logits'];psi=detail['psi'].detach();u=detail['utilities'].detach();mask=inputs[2]
        zz=z.detach().cpu().numpy();error=float(np.max(np.abs(zz-prior_z[ids])));assert error<1e-8
        native_error=max(native_error,error);assert np.array_equal(zz.argmax(-1),prior_z[ids].argmax(-1))
        assert torch.max(torch.abs(detail['phi'].detach()-phi))<1e-12
        take=np.flatnonzero((~cal[ids])&(y[ids]==0))
        if len(take):
            index=torch.tensor(take,device=z.device);loss=F.cross_entropy(z[index],target[index],reduction='sum')/assess0
            ga0+=torch.autograd.grad(loss,model.effect)[0]
        q=(detail['probability'].detach()+EPS)*mask[:,None];normalizer=q.sum(-1,keepdim=True)
        pooled=(q*u).sum(-1,keepdim=True)/normalizer;amp=torch.einsum('rd,btd->brt',phi.conj(),psi)
        for name,dp in dphis.items():
            amp_dot=torch.einsum('rd,btd->brt',dp.conj(),psi)
            p_dot=2*(amp.conj()*amp_dot).real*mask[:,None]
            dz=(p_dot*(u-pooled)).sum(-1)/normalizer.squeeze(-1);assert torch.isfinite(dz).all()
            if lo<cfg['fixture_rows']:
                def direct(e):
                    raw=(torch.einsum('rd,btd->brt',e.conj(),psi).abs().square()+EPS)*mask[:,None]
                    return (raw*u).sum(-1)/raw.sum(-1)+model.bias
                plus,minus=endpoints[name];fd=(direct(plus)-direct(minus))/(2*cfg['finite_difference_step'])
                error=float((fd-dz).abs().max());assert error<cfg['finite_difference_tolerance'];fd_error=max(fd_error,error)
            derivatives[name].append(dz.cpu().numpy())
        if lo%128==0 or lo+len(ids)==len(y):progress('two_group_direction_transfer',lo+len(ids),len(y))
    ga0=ga0.cpu().numpy();ga=old['gassessment'];gap=(4000*ga-assess0*ga0)/assessp
    derivatives={k:np.concatenate(v) for k,v in derivatives.items()};derivatives['frequency']=old['logit_derivative']
    rows={};identity_error=0.;arrays=[]
    for name in cfg['modes']:
        if not selection['directions'][name]['valid']:
            rows[name]=dict(status='unresolved',reason='Degenerate/zero calibration direction; no substitute')
            arrays.append(np.zeros_like(prior_z));continue
        dz=derivatives[name];arrays.append(dz)
        table,ce,margin=helper.tables(prior_z,dz,y,cal,cfg['derivative_tolerance'])
        row=dict(status='valid',groups={})
        for partition,gradients,keep in (('calibration',(g0,gp),cal),('assessment',(ga0,gap),~cal)):
            group_rows=[]
            for label_zero,g in zip((True,False),gradients):
                m=keep&((y==0) if label_zero else (y!=0));v=float(ce[m].mean());inner=float(np.vdot(g,directions[name]))
                identity_error=max(identity_error,abs(v-inner));assert abs(v-inner)<1e-9
                group_rows.append(v)
            row['groups'][partition]=dict(class0=group_rows[0],positive=group_rows[1],worst=max(group_rows))
        row['tables']=table;rows[name]=row
    tau=cfg['derivative_tolerance']*max(1.,float(np.linalg.norm(ga0)),float(np.linalg.norm(gap)))
    if rows['balanced']['status']!='valid' or rows['common']['status']!='valid':gate='UNRESOLVED_CLOSE_EXACT_PROFILE';gap_value=None
    else:
        common=rows['common']['groups']['assessment']['worst'];balanced=rows['balanced']['groups']['assessment']['worst'];gap_value=balanced-common
        gate='PASS_LOCAL_DIRECTION_ONLY' if common < -tau and gap_value>tau else 'FAIL_CLOSE_EXACT_PROFILE'
    assert torch.equal(model.effect.detach(),initial) and digest(a.checkpoint)==cfg['checkpoint_sha256']
    assert digest(a.output/'directions.private.npy')==frozen['directions_sha256']
    private=a.output/'diagnostic.private.npz'
    np.savez_compressed(private,logits=prior_z,logit_derivative=np.stack(arrays),labels=y,calibration=cal,groups=groups,
        gcal_zero=g0,gcal_positive=gp,gassessment_zero=ga0,gassessment_positive=gap,directions=directions_array)
    check.update(phi_jvp_fd_error=phi_fd,first8_logit_fd_error=fd_error,all_group_autograd_jvp_error=identity_error,
        allrow_prior_error=native_error,weights_unchanged=True)
    summary=dict(profile=cfg['profile'],gate=gate,results=rows,selection=selection,assessment_threshold=tau,
        balanced_minus_common_worst=gap_value,seconds=time.time()-started,optimizer_updates=0,
        directions_frozen=frozen,assessment_started_at=assessment_started,design_sha256=digest(a.design),
        diagnostic_sha256=digest(private),checkpoint_sha256=cfg['checkpoint_sha256'],claim_scope=cfg['claim_scope'])
    write(a.output/'summary.json',summary);write(a.output/'fixtures.json',check)
    write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in ('summary.json','fixtures.json','directions-frozen.json','directions.private.npy','diagnostic.private.npz')}))
    print(json.dumps(dict(gate=gate,seconds=summary['seconds'],assessment={k:v.get('groups',{}).get('assessment') for k,v in rows.items()})))


if __name__=='__main__':main()
