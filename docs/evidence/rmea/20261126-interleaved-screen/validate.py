"""Affected numerical checks; independent dense NumPy oracle, no dataset reads."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,time
import numpy as np
import torch
r=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source'))
ap.add_argument('--history',type=Path,default=Path('E:/Claude.Code/Paper.Writing/Q-Attention/.codex-artifacts'))
ap.add_argument('--cache',type=Path)
args=ap.parse_args();source=args.source;history=args.history
sys.path.insert(0,str(source));sys.path.insert(0,str(r))
import plugin
import adapter
oracle_path=r/'interleaved_math.py' if (r/'interleaved_math.py').exists() else r.parent/'interleaved_math.py'
spec=importlib.util.spec_from_file_location('dense_oracle',oracle_path)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
from baselines.retacred.rmea import RMEAHead,circuit


def identity(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().numpy()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def real_matrix(a):
    u=np.eye(8)
    for row in a:
        gate=np.array([[1.]])
        for y in row:
            c,s=np.cos(y/2),np.sin(y/2)
            gate=np.kron(gate,np.array([[c,-s],[s,c]]))
        u=gate@u
        for control,target in ((0,1),(1,2),(2,0)):
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            u=u[perm]
    return u


def run():
    review=json.loads((r/'review.json').read_bytes())
    assert review['response']['verdict']=='PASS_WITH_RISK', 'Inspect mechanism review before executing changed code'
    selection=json.loads((r/'selection.json').read_bytes())
    assert selection['packet_sha256']==review['packet_sha256']
    packet=json.loads((r/'review-packet.private.json').read_bytes())
    for entry in packet['files']:
        if entry['path'].replace('\\','/').rsplit('/',1)[-1] == 'plugin.py':
            assert hashlib.sha256((r/'plugin.py').read_bytes()).hexdigest()==entry['sha256']
    adoption=json.loads((r/'review-adoption.json').read_bytes())
    assert hashlib.sha256((r/'adapter.py').read_bytes()).hexdigest()==adoption['adapter_after_sha256']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();j=np.arange(1,13);errors={};trace=[]
    w=np.asarray([1.2*np.sin((i+1)*j*.173).reshape(2,3,2) for i in range(32)])
    q=np.asarray([.9*np.cos((i+2)*j*.137).reshape(2,3,2) for i in range(40)])
    reference={kind:np.empty((1,40,32)) for kind in plugin.MODES}
    for t in range(32):
        for rel in range(40):
            p=oracle.probabilities(w[t],q[rel])
            reference['quantum_middle'][0,rel,t]=p['mid']
            reference['dephased_middle'][0,rel,t]=p['dephased']
            x=w[t].reshape(12);z=q[rel].reshape(4,3)
            v=real_matrix(x[6:].reshape(2,3))@real_matrix(z).T@real_matrix(x[:6].reshape(2,3))
            reference['real8_middle'][0,rel,t]=v[0,0]**2
    for dtype in (torch.float64,torch.float32):
        for kind in plugin.MODES:
            word=torch.tensor(w[None],dtype=dtype,requires_grad=True)
            relation=torch.tensor(q,dtype=dtype,requires_grad=True)
            p=plugin.probability(word,relation,kind)
            error=float(np.max(np.abs(p.detach().numpy()-reference[kind])))
            tolerance=1e-11 if dtype==torch.float64 else 2e-6
            assert error<tolerance,(dtype,kind,error)
            errors[str(dtype)+'/'+kind]=error
            weights=torch.linspace(-.7,.8,p.numel(),dtype=dtype).reshape_as(p)
            (p*weights).sum().backward()
            assert torch.isfinite(word.grad).all() and torch.isfinite(relation.grad).all()
            if kind!='real8_middle':
                assert torch.count_nonzero(word.grad[...,1,:,1])==0
            if kind=='dephased_middle':
                assert torch.count_nonzero(word.grad[...,0,:,1])==0
                assert torch.count_nonzero(relation.grad[:,1,:,1])==0
            if dtype==torch.float64:
                for wi in ((0,0,0,0,0),(0,0,1,0,1)):
                    a=w.copy();b=w.copy();idx=wi[1:]
                    a[idx]+=1e-6;b[idx]-=1e-6
                    def f(v):
                        if kind=='real8_middle':
                            vals=[]
                            for z in q:
                                x=v[0].reshape(12)
                                mat=real_matrix(x[6:].reshape(2,3))@real_matrix(z.reshape(4,3)).T@real_matrix(x[:6].reshape(2,3))
                                vals.append(mat[0,0]**2)
                        else:
                            vals=[oracle.probabilities(v[0],z)['mid' if kind=='quantum_middle' else 'dephased'] for z in q]
                        return float(np.dot(vals,weights.detach().numpy()[0,:,0]))
                    fd=(f(a)-f(b))/2e-6;ad=float(word.grad[wi])
                    assert abs(fd-ad)<2e-8,(kind,wi,fd,ad)
                    trace.append(dict(kind=kind,coordinate=wi,analytical=ad,numerical=fd,error=abs(fd-ad)))
    # Complex amplitude restores the analytically dropped terminal phase.
    wc=torch.tensor(w[None],dtype=torch.float64);qc=torch.tensor(q,dtype=torch.float64)
    zero=torch.zeros((1,32,8),dtype=torch.complex128);zero[...,0]=1
    b=plugin.layer(wc[...,0,:,:],zero)
    bra=plugin.terminal_bra(wc[...,1,:,0]).to(torch.complex128)
    query=plugin.layer_matrix(qc[:,1])@plugin.layer_matrix(qc[:,0])
    amplitude=torch.einsum('bti,rij,btj->brt',bra,query.conj().transpose(-1,-2),b)
    amplitude=amplitude*torch.exp(-.5j*wc[...,1,:,1].sum(-1))[:,None]
    phase_error=0.;density_error=0.
    for t in range(32):
        w1,w2=[oracle.block(x) for x in w[t]]
        rho=np.outer(w1[:,0],w1[:,0].conj());rho=np.diag(np.diag(rho))
        for rel in range(40):
            qdag=oracle.circuit(q[rel]).conj().T
            phase_error=max(phase_error,abs(complex(amplitude[0,rel,t])-(w2@qdag@w1)[0,0]))
            out=qdag@rho@qdag.conj().T;out=np.diag(np.diag(out));out=w2@out@w2.conj().T
            density_error=max(density_error,abs(out[0,0].real-reference['dephased_middle'][0,rel,t]))
    assert phase_error<1e-11 and density_error<1e-11
    operator_result=dict(status='PASS_INDEPENDENT_OPERATOR_AND_ANGLE_GRADIENT_CHECKS',
        maximum_errors=errors,angle_gradient_checks=trace,complex_amplitude_restored_phase_max_error=phase_error,
        explicit_density_dephase_max_error=density_error,torch_version=torch.__version__,native_initial_identity_still_required=True)
    (r/'operator-validation.json').write_bytes((json.dumps(operator_result,indent=2)+'\n').encode())
    torch.manual_seed(13)
    initial={k:v.clone() for k,v in RMEAHead('quantum').float().state_dict().items()}
    models=plugin.initialize(initial,13)
    cfg=json.loads((r/'design.json').read_bytes())
    ids={k:identity(m.state_dict()) for k,m in models.items()}
    assert set(ids.values())=={cfg['expected_initial_sha256']},ids
    assert all(sum(p.numel() for p in m.parameters())==81028 for m in models.values())
    # Reconstruct old reviewed initializers, including their actual key prefixes.
    reconstructed={}
    for folder,arms in [('context-spectral-flow-20261125',['quantum','dephased']),
                        ('staged-bottleneck-design-20261125',['real12','gaussian_base']),
                        ('connected-value-pooling-20261125',['gaussian_covariance','gaussian_mean_product'])]:
        path=history/folder/'plugin.py'
        spec=importlib.util.spec_from_file_location('old_'+folder.replace('-','_'),path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        oldmodels=module.initialize(initial,13)
        for arm in arms:
            key=folder+'/'+arm;old=oldmodels[arm]
            reconstructed[key]=identity(old.state_dict())
            assert reconstructed[key]==cfg['references'][key]['initial_sha256']
            assert all(torch.equal(old.head.state_dict()[k],initial[k]) for k in plugin.SHARED)
    for folder,arm in [('semantic-connected-evidence-20261125','quantum_shared_mean_product'),
                       ('distinct-token-evidence-20261125','gaussian_distinct')]:
        directory=history/folder
        sys.path.insert(0,str(directory))
        spec=importlib.util.spec_from_file_location('old_'+folder.replace('-','_'),directory/'plugin.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=module.initialize(initial,13)[arm];key=folder+'/'+arm
        reconstructed[key]=identity(old.state_dict())
        assert reconstructed[key]==cfg['references'][key]['initial_sha256'],(key,reconstructed[key])
        assert all(torch.equal(old.head.state_dict()[k],initial[k]) for k in plugin.SHARED)
        sys.path.pop(0)
    # Artificial feature rows test the real head API; no dataset access/efficacy.
    h=torch.linspace(-.2,.3,8*3*1024).reshape(8,3,1024)
    roles=torch.zeros(8,3,4);mask=torch.ones(8,3,dtype=torch.bool);mask[0,2]=False
    fixture=adapter.fixtures(models,(h,roles,mask),torch.arange(8))
    real_fixture=None
    if args.cache:
        cohort=json.loads((r/'cohort.private.json').read_bytes());ids8=cohort['fit'][:8]
        data={k:np.load(args.cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids8]
        h=np.zeros((8,max(lengths),1024),np.float32);roles=np.zeros((8,max(lengths),4),np.float32)
        mask=np.zeros((8,max(lengths)),bool)
        for i,ix in enumerate(ids8):
            lo,hi=map(int,data['offsets'][ix:ix+2]);h[i,:hi-lo]=data['h'][lo:hi];roles[i,:hi-lo]=data['roles'][lo:hi];mask[i,:hi-lo]=True
        inputs=tuple(torch.from_numpy(v) for v in (h,roles,mask))
        labels=torch.from_numpy(np.asarray(data['targets'][ids8],dtype=np.int64))
        real_fixture=adapter.fixtures(models,inputs,labels)
        assert all(identity(model.state_dict())==cfg['expected_initial_sha256'] for model in models.values())
        real_fixture.update(device='cpu',designated_first8_fit_index_sha256=hashlib.sha256(np.asarray(ids8,dtype='<i8').tobytes()).hexdigest(),
                            initialized_models_retained=True)
    result=dict(status='PASS_IMPLEMENTATION_NUMERICAL_AND_GRADIENT_CHECKS',
        maximum_errors=errors,angle_gradient_checks=trace,initial_identities=ids,
        complex_amplitude_restored_phase_max_error=phase_error,explicit_density_dephase_max_error=density_error,
        reconstructed_reference_initial_identities=reconstructed,artificial_head_fixtures=fixture,
        real_head_fixtures=real_fixture,torch_version=torch.__version__,
        seconds=time.perf_counter()-started,real_task_rows=8 if args.cache else 0,task_fits=0,GPU=0,
        limitations='Technical clone checks only, not efficacy, independent task confirmation, effective-capacity equality or L3/L4-T. Exact remote input binding required before screen.')
    (r/'validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(status=result['status'],max_FP64=max(v for k,v in errors.items() if '64' in k),
         max_FP32=max(v for k,v in errors.items() if '32' in k),reference_initializers=len(reconstructed),seconds=result['seconds'])))


if __name__=='__main__':run()
