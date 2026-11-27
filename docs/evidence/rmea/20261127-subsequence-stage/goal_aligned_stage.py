"""Task/stage semantics for frozen profile v1; no model, optimizer or scheduler."""
from pathlib import Path
import hashlib,json,os
import numpy as np

CHECKPOINTS=(160,320,480,640)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def row_weights(targets,train_counts,supports):
    y=np.asarray(targets,dtype=np.int64);a=np.asarray(train_counts,dtype=np.float64);b=np.asarray(supports,dtype=np.float64)
    assert a.shape==b.shape==(40,) and np.all(a>0) and np.all(b>0)
    assert np.array_equal(np.bincount(y,minlength=40),b)
    return (a/a.sum()/b)[y]

def confusion(y,p,weights):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64);w=np.asarray(weights,dtype=np.float64)
    assert y.shape==p.shape==w.shape and y.ndim==1 and np.isfinite(w).all() and np.all(w>=0)
    assert np.all((y>=0)&(y<40)&(p>=0)&(p<40))
    return np.bincount(40*y+p,weights=w,minlength=1600).reshape(40,40)

def macro(cm):
    cm=np.asarray(cm,dtype=np.float64);assert cm.shape==(40,40) and np.isfinite(cm).all() and np.all(cm>=0)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den>0).mean())

def weighted_ce(logits,y,weights):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(y,dtype=np.int64);w=np.asarray(weights,dtype=np.float64)
    assert z.shape==(len(y),40) and w.shape==y.shape and np.isfinite(z).all() and abs(w.sum()-1)<1e-12
    z=z-z.max(-1,keepdims=True)
    return float(np.dot(w,np.log(np.exp(z).sum(-1))-z[np.arange(len(y)),y]))

def evaluate(logits,y,weights):
    z=np.asarray(logits);assert z.shape==(len(y),40) and np.isfinite(z).all()
    cm=confusion(y,z.argmax(-1),weights)
    return dict(weighted_macro_f1=macro(cm),weighted_ce=weighted_ce(z,y,weights),weighted_confusion=cm.tolist())

def choose(receipts):
    assert receipts and len({x['step'] for x in receipts})==len(receipts)
    assert all(x['step'] in CHECKPOINTS and np.isfinite(x['weighted_macro_f1']) for x in receipts)
    return min(receipts,key=lambda x:(-x['weighted_macro_f1'],x['step']))

def stable(curves,q,controls):
    return all(curves[q][step]>=curves[c][step] for step in (480,640) for c in controls)

def grouped_bootstrap(y,predictions,weights,groups,q,controls,repetitions=500,seed=20261127):
    y=np.asarray(y,dtype=np.int64);w=np.asarray(weights,dtype=np.float64);g=np.asarray(groups)
    assert len(y)==len(w)==len(g) and repetitions==500 and seed==20261127
    assert len(controls)==3 and q not in controls
    unique,inverse=np.unique(g,return_inverse=True);G=len(unique);rng=np.random.default_rng(seed)
    preds={k:np.asarray(predictions[k],dtype=np.int64) for k in [q,*controls]}
    assert all(p.shape==y.shape for p in preds.values()) and G>0
    point={c:macro(confusion(y,preds[q],w))-macro(confusion(y,preds[c],w)) for c in controls}
    minimum=[];absent=np.zeros(40,dtype=np.int64)
    for _ in range(repetitions):
        multiplicity=np.bincount(rng.integers(G,size=G),minlength=G)[inverse]
        ww=w*multiplicity;qs=macro(confusion(y,preds[q],ww))
        minimum.append(min(qs-macro(confusion(y,preds[c],ww)) for c in controls))
        absent+=np.bincount(y,weights=multiplicity,minlength=40)==0
    return dict(point_deltas=point,minimum_delta_quantiles_05_95=np.quantile(minimum,[.05,.95]).tolist(),
                sparse_label_absence_frequency=(absent/repetitions).tolist(),groups=G,repetitions=repetitions,seed=seed,
                scope='Exposed development group uncertainty only, not independent confirmation or seed/scale uncertainty')

def validate_roles(indices,groups):
    roles=('fit','selection','qualification');sets={k:set(map(int,indices[k])) for k in roles}
    assert {k:len(indices[k]) for k in roles}==dict(fit=4096,selection=1024,qualification=2048)
    assert all(len(sets[k])==len(indices[k]) for k in roles)
    for i,a in enumerate(roles):
        for b in roles[i+1:]:
            assert not sets[a]&sets[b],'row overlap'
            assert not {groups[x] for x in sets[a]}&{groups[x] for x in sets[b]},'sentence group overlap'

def batches(indices):
    a=np.asarray(indices,dtype=np.int64);assert len(a)==4096 and len(set(a))==4096
    rng=np.random.default_rng(13);result=[]
    for _ in range(20):
        order=rng.permutation(a);result.extend(order[j:j+128] for j in range(0,4096,128))
    assert len(result)==640
    return result,hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in result)).hexdigest()

class SelectionJournal:
    """Exactly one selection pass per checkpoint. Incomplete BEGIN fails closed.

    This is a selection-stage delta, not a replacement of core latest/best/resume.
    Native checkpoint bytes and prediction arrays remain in private run outputs.
    """
    def __init__(self,folder,mode,contract_sha):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=True,parents=True)
        self.mode=mode;self.contract_sha=contract_sha
    def begin(self,step,checkpoint_sha):
        assert step in CHECKPOINTS
        path=self.folder/f'selection-{step}.begin.json'
        value=dict(mode=self.mode,step=step,contract_sha=self.contract_sha,checkpoint_sha=checkpoint_sha)
        with path.open('x',encoding='utf-8') as f:json.dump(value,f)
        return value
    def commit(self,value,metric,prediction_sha):
        step=value['step'];assert json.loads((self.folder/f'selection-{step}.begin.json').read_bytes())==value
        assert abs(sum(map(sum,metric['weighted_confusion']))-1)<1e-12
        receipt={**value,**metric,'prediction_sha':prediction_sha}
        path=self.folder/f'selection-{step}.json';temporary=path.with_suffix('.pending')
        assert not path.exists() and not temporary.exists()
        temporary.write_text(json.dumps(receipt,allow_nan=False)+'\n');os.replace(temporary,path)
        return receipt
    def receipts(self,require_complete=False):
        result=[]
        for step in CHECKPOINTS:
            b=self.folder/f'selection-{step}.begin.json';p=self.folder/f'selection-{step}.json'
            assert b.exists()==p.exists(),'Incomplete selection pass: defer; no hidden repeat within four-pass cap'
            if p.exists():
                v=json.loads(p.read_bytes());begin=json.loads(b.read_bytes())
                assert all(v[k]==value for k,value in begin.items()) and v['mode']==self.mode and v['contract_sha']==self.contract_sha
                result.append(v)
        if require_complete:assert len(result)==4
        return result

def attach_selection_progress(base_progress,*,models,score_selection,folders,contract_sha,atomic_torch,cpu_tree,state_identity):
    """Use the accepted core's committed-update progress hook; no optimizer delta."""
    def progress(mode,state,**values):
        step=values.get('step')
        if state=='training' and step in CHECKPOINTS:
            model=models[mode];folder=Path(folders[mode]);journal=SelectionJournal(folder,mode,contract_sha)
            journal.receipts()  # never skip an incomplete pass from a prior failure
            snapshot=folder/f'selection-{step}.private.pt'
            assert not snapshot.exists()
            before=state_identity(model.state_dict());atomic_torch(snapshot,cpu_tree(model.state_dict()))
            begin=journal.begin(step,sha(snapshot));training=model.training
            try:
                model.eval();metric,prediction_sha=score_selection(model,mode,step)
            finally:model.train(training)
            assert state_identity(model.state_dict())==before,'Evaluation mutated state'
            journal.commit(begin,metric,prediction_sha)
        base_progress(mode,state,**values)
    return progress
