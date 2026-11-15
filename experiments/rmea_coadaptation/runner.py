"""A source-faithful synthetic counterexample, not a task-utility experiment."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run']);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    src=Path(__file__).resolve().parent;repo=src.parents[1];cfg=json.loads((src/'contract.json').read_bytes());started=time.monotonic()
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],stderr=subprocess.DEVNULL).decode().strip()
    assert not git('status','--porcelain=v1') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for n,h in cfg['model_source_hashes'].items():assert sha(repo/n)==h
    import torch
    from torch.nn import functional as F
    assert torch.__version__=='2.5.1+cpu';torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False)
    sys.path.insert(0,str(repo));from experiments.rmea_allocation.legacy.rmea import RMEAHead,EPS
    assert EPS==cfg['epsilon']==1e-6
    out=args.output.resolve();assert not out.is_relative_to(repo) and not out.is_symlink();out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    h=torch.zeros((2,4,1024),dtype=torch.float64);h[:,0,0]=1;h[:,1,1]=1
    roles=torch.zeros((2,4,4),dtype=torch.float64);position=torch.arange(4,dtype=torch.float64)
    spans=[dict(subject=[2,3],object=[3,4]),dict(subject=[3,4],object=[2,3])]
    for i,row in enumerate(spans):
        for channel,key in enumerate(['subject','object']):
            lo,hi=row[key];roles[i,:,channel]=torch.clamp((position-(lo+hi-1)/2)/32,-1,1);roles[i,:,channel+2]=((position>=lo)&(position<hi)).double()
    mask=torch.ones((2,4),dtype=torch.bool);labels=torch.tensor([0,1]);k=math.atanh(.5);Z=2+4*EPS
    heads=[]
    for mode in ['A','B']:
        head=RMEAHead('quantum').double().eval()
        with torch.no_grad():
            for p in head.parameters():p.zero_()
            assert head.value.weight.shape==(64,1024) and head.decoder.shape==(40,64)
            head.prepare.bias[0]=k
            head.prepare.weight[1,0]=-k;head.prepare.weight[1,1]=k
            head.prepare.weight[9,1024]=32*k;head.prepare.weight[9,1025]=-32*k
            head.effect[:,0,0,0]=k
            if mode=='B':head.effect[:,0,0,1]=k;head.effect[:,1,1,1]=k
            head.value.weight[0,0]=Z;head.value.weight[0,1]=-Z
            head.decoder[0,0]=1 if mode=='A' else -1;head.decoder[1,0]=-head.decoder[0,0]
            head.bias[2:]=-100
        heads.append(head)
    with torch.no_grad():a,b=[head(h,roles,mask,details=True) for head in heads]
    atol=cfg['absolute_tolerance']
    def close(x,y):
        y=torch.as_tensor(y,dtype=x.dtype);err=float((x-y).abs().max());assert err<atol;return err
    desired=torch.zeros_like(a['logits']);desired[:,2:]=-100;desired[0,:2]=torch.tensor([1.,-1.]);desired[1,:2]=torch.tensor([-1.,1.])
    probabilities=torch.tensor([[1.,0.,.5,.5],[0.,1.,.5,.5]],dtype=torch.float64)[:,None,:].expand(-1,40,-1)
    errors={'A_probability':close(a['probability'],probabilities),'B_probability':close(b['probability'],probabilities[:,:,[1,0,2,3]]),'A_logits':close(a['logits'],desired),'B_logits':close(b['logits'],desired)}
    expected_alpha=(probabilities+EPS)/Z;errors['A_alpha']=close(a['alpha'],expected_alpha);errors['B_alpha']=close(b['alpha'],expected_alpha[:,:,[1,0,2,3]])
    plus=torch.zeros(8,dtype=torch.complex128);plus[0]=plus[2]=2**-.5
    minus=plus.clone();minus[2]*=-1
    def projector(v):return v[..., :,None]*v.conj()[...,None,:]
    errors['A_effect']=close(projector(a['phi']),projector(plus));errors['B_effect']=close(projector(b['phi']),projector(minus))
    for i in range(2):
        for token in range(4):
            expected=plus if (i+token)%2==0 else minus
            if token>=2:expected=plus.clone();expected[2]*=(1j if i==0 else -1j)
            errors[f'psi_{i}_{token}']=close(projector(a['psi'][i,token]),projector(expected))
    swaps=[(b['alpha']*a['utilities']).sum(-1)+heads[0].bias,(a['alpha']*b['utilities']).sum(-1)+heads[1].bias]
    flipped=desired.clone();flipped[:,:2]*=-1
    for i,z in enumerate(swaps):errors[f'swap_{i}']=close(z,flipped)
    recovered=[]
    for i,(donor,recipient) in enumerate([(b,a),(a,b)]):
        utility=recipient['utilities'].clone();utility[:,:2]*=-1
        z=(donor['alpha']*utility).sum(-1)+heads[i].bias;recovered.append(z);errors[f'redecoded_{i}']=close(z,desired)
    own_ce=[F.cross_entropy(x['logits'],labels,reduction='none') for x in [a,b]]
    swap_ce=[F.cross_entropy(x,labels,reduction='none') for x in swaps]
    assert all(torch.equal(x['logits'].argmax(-1),labels) for x in [a,b])
    assert all(torch.equal(z.argmax(-1),1-labels) for z in swaps)
    for i,(before,after) in enumerate(zip(own_ce,swap_ce)):errors[f'ce_damage_{i}']=close(after-before,torch.full((2,),2.,dtype=torch.float64))
    assert torch.equal(h[0],h[1]) and torch.equal(h.mean(1)[0],h.mean(1)[1])
    values=dict(schema='qattn.synthetic-circuit-counterexample.numeric.v1',input_scope='Synthetic frozen-feature boundary only; no encoder/text/dataset access.',nonzero_h=[dict(token=0,coordinate=0,value=1),dict(token=1,coordinate=1,value=1)],spans=spans,roles=roles.tolist(),epsilon=EPS,normalizer=Z,kappa=k,A_probability=a['probability'][:,0].tolist(),B_probability=b['probability'][:,0].tolist(),A_alpha=a['alpha'][:,0].tolist(),B_alpha=b['alpha'][:,0].tolist(),A_active_logits=a['logits'][:,:2].tolist(),B_active_logits=b['logits'][:,:2].tolist(),swapped_active_logits=[z[:,:2].tolist() for z in swaps],redecoded_active_logits=[z[:,:2].tolist() for z in recovered],dummy_logits=-100,own_ce=[z.tolist() for z in own_ce],swap_ce=[z.tolist() for z in swap_ce],uniform_pooled_equal=True)
    save(out/'numeric-fixture.json',values)
    validation=dict(status='PASS',source_hashes=cfg['model_source_hashes'],parameter_value_width=64,relations=40,effect_scope='Shared pure effects realized by the original finite depth2 circuit;40 independent binary effects.',role_scope='Actual four-role recipe from valid non-overlapping ordered spans; full linear1028->12 preparation weights explicitly set.',maximum_absolute_error=max(errors.values()),errors=errors,finite_parameters=all(bool(torch.isfinite(p).all()) for head in heads for p in head.parameters()),source_module='experiments.rmea_allocation.legacy.rmea',new_model_forward_calls=2,real_data_forward_calls=0,training_steps=0)
    assert validation['finite_parameters'];save(out/'source-validation.json',validation)
    summary=dict(status='PASS',schema='qattn.coadaptation.synthetic.v1',source_revision=git('rev-parse','HEAD'),contract_sha256=sha(src/'contract.json'),cases=2,own_correct=[2,2],swap_correct=[0,0],redecoded_correct=[2,2],ce_damage_per_case=2.,maximum_absolute_error=max(errors.values()),synthetic_head_forward_calls=2,real_data_forward_calls=0,training_runs=0,formal_runs=0,new_cos_reads=0,test_loaded=False,l5_accepted=False,elapsed_seconds=time.monotonic()-started,interpretation='A source-faithful pure-effect counterexample to interpreting mutual attention-swap harm as lost task information. A known coadaptation/representation-equivalence construction, not new utility, observed natural-language behavior or L5 novelty.')
    save(out/'summary.json',summary);files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()};save(out/'RUN_COMPLETE',dict(status='COMPLETE',files=files,source_revision=summary['source_revision']))
    print(json.dumps(dict(status='COMPLETE',cases=2,training_runs=0,formal_runs=0,synthetic_forward_calls=2)))
if __name__=='__main__':main()
