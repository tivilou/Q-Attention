"""Fixed full-TRAIN initial/selected query learning census; no fitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,signal,sys,threading,time
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def save(p,v):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def metrics(y,z):
    pred=z.argmax(-1);cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
    zz=z.astype(np.float64);zz-=zz.max(-1,keepdims=True)
    return dict(confusion_matrix=cm.tolist(),all40_macro_f1=float(f.mean()),
        ce=float((np.log(np.exp(zz).sum(-1))-zz[np.arange(len(y)),y]).mean()),correct=int(np.trace(cm)))

def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    source=a.root/cfg['source'];completed=a.root/cfg['completed'];cache=completed/'cache'
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h,n
    for n,h in cfg['source_hashes'].items():assert digest(source/n)==h,n
    assert digest(completed/'manifest.json')==cfg['manifest_sha256']
    for n,v in cfg['train_cache'].items():assert digest(cache/n)==v['sha256'],n
    assert digest(a.root/cfg['labels'])==cfg['labels_sha256']
    labels=json.loads((a.root/cfg['labels']).read_bytes());assert len(labels)==40
    sys.path.insert(0,str(source));import torch
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_matched_routers import initialize_head
    from baselines.retacred.rmea_token_transport_study import state_identity
    from gradients import query_gradient,query_differential,fixtures,aggregate
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and not a.output.exists();a.output.mkdir(parents=True)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    y=np.array(data['targets']);assert y.shape==(58465,) and y.min()==0 and y.max()==39
    assert data['offsets'].shape==(58466,) and data['offsets'][0]==0
    assert data['offsets'][-1]==len(data['h'])==len(data['roles'])
    torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict();models={}
    for mode in cfg['conditions']:
        models[mode]={stage:initialize_head(mode,initial,dict(seed=13,training_dtype='float32')) for stage in ('initial','selected')}
        selected=completed/cfg['selected'][mode]['path'];assert digest(selected)==cfg['selected'][mode]['sha256']
        state=torch.load(selected,map_location='cpu',weights_only=True)
        assert state_identity(state)==cfg['selected'][mode]['state_sha256']
        models[mode]['selected'].load_state_dict(state,strict=True)
    assert state_identity(models['quantum']['initial'].state_dict())==cfg['initial_quantum_state_sha256']
    begin=time.monotonic();signal.alarm(cfg['timeout_seconds']);cursors={m+'_'+s:0 for m in cfg['conditions'] for s in ('initial','selected')};lock=threading.Lock()
    def progress(key,current):
        with lock:
            cursors[key]=current
            save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='query_learning_'+key,current=sum(cursors.values()),
                total=4*len(y),unit='example',children={k:dict(current=v,total=len(y)) for k,v in cursors.items()},heartbeat_at=time.time(),observed_at=time.time()))
    def batch(lo,hi,device):
        lengths=np.diff(data['offsets'][lo:hi+1]);h=np.zeros((hi-lo,int(lengths.max()),1024),np.float32)
        roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
        for j,i in enumerate(range(lo,hi)):
            left,right=map(int,data['offsets'][i:i+2]);size=right-left
            h[j,:size]=data['h'][left:right];roles[j,:size]=data['roles'][left:right];mask[j,:size]=True
        return tuple(torch.from_numpy(v).to(device) for v in (h,roles,mask)),torch.from_numpy(y[lo:hi]).long().to(device)
    def worker(lane,mode):
        torch.cuda.set_device(lane);device=torch.device('cuda',lane);reports={};checks={}
        for stage in ('initial','selected'):
            start=time.monotonic();key=mode+'_'+stage;model=models[mode][stage].to(device).eval()
            identity=state_identity({k:v.cpu() for k,v in model.state_dict().items()})
            first,fy=batch(0,3,device);checks[stage]=fixtures(model,mode,first,fy)
            dq=query_differential(model,mode);g=np.empty((len(y),480),np.float64);z=np.empty((len(y),40),np.float32)
            for lo in range(0,len(y),128):
                assert time.monotonic()-start<cfg['state_timeout_seconds'],'Fixed state compute budget exhausted'
                hi=min(lo+128,len(y));inputs,targets=batch(lo,hi,device)
                gradient,logits,_=query_gradient(model,mode,inputs,targets,dq=dq)
                g[lo:hi]=gradient.cpu().numpy();z[lo:hi]=logits.cpu().numpy()
                if lo%2048==0:progress(key,hi)
            observed=metrics(y,z);correct=z.argmax(-1)==y
            if stage=='selected':
                reference=cfg['prior_train_metrics'][mode]
                assert observed['confusion_matrix']==reference['confusion_matrix']
                assert max(abs(observed[k]-reference[k]) for k in ('ce','all40_macro_f1'))<1e-5
            statistics=aggregate(g,y,correct,labels)
            after=state_identity({k:v.cpu() for k,v in model.state_dict().items()});assert identity==after
            result=dict(mode=mode,stage=stage,metrics=observed,statistics=statistics,model_state_sha256=identity,
                state_unchanged=True,examples=len(y),query_coordinates=480,seconds=time.monotonic()-start)
            assert result['seconds']<cfg['state_timeout_seconds']
            save(a.output/(key+'.json'),result)
            reports[stage]={k:result[k] for k in ('model_state_sha256','state_unchanged','seconds','examples','query_coordinates')}
            reports[stage].update(ce=observed['ce'],all40_macro_f1=observed['all40_macro_f1'],correct=observed['correct'],
                coherence=statistics['all']['coherence'],wrong_coherence=statistics['wrong'].get('coherence'),
                class_means_opposed_to_sample_weighted=statistics['class_means_opposed_to_sample_weighted'],
                class_means_opposed_to_equal_class=statistics['class_means_opposed_to_equal_class'],
                per_example_alignment_to_sample_mean=statistics['sample_weighted_mean_alignment'])
            model.cpu();del g,z,gradient,logits,dq;progress(key,len(y))
        return mode,reports,checks
    results={};checks={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,i,m) for i,m in enumerate(cfg['conditions'])]
        for job in jobs:
            mode,record,check=job.result();results[mode]=record;checks[mode]=check
    elapsed=time.monotonic()-begin;assert elapsed<cfg['timeout_seconds'];signal.alarm(0)
    save(a.output/'fixtures.json',dict(status='PASS',checks=checks,scope='New query chain vs first3 real autograd at both precisions; no lifecycle matrix.'))
    report=dict(profile=cfg['profile'],results=results,seconds=elapsed,train_examples=58465,states=4,
        query_coordinates=480,optimizer_updates=0,new_valid=0,new_test=0,cos_writes=0,per_example_binary_outputs=0,
        code_hashes=cfg['code_hashes'],claim='Local coordinate-dependent TRAIN gradient cancellation only; not cause, common-descent impossibility, improved method, independent utility or newT.',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
    save(a.output/'summary.json',report)
    names=['summary.json','fixtures.json']+[m+'_'+s+'.json' for m in cfg['conditions'] for s in ('initial','selected')]
    save(a.output/'RUN_COMPLETE',{n:digest(a.output/n) for n in names});progress('real12_selected',len(y))
    print(json.dumps(report))

if __name__=='__main__':main()
