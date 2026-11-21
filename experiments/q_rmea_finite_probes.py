"""Synthetic preparation/value-port audit on one protected native checkpoint."""
import argparse
from fractions import Fraction as F
import hashlib,io,itertools,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import q_rmea_output_matching as m
p=m.p
NAME='experiments/q_rmea_finite_probes.py'
FILES=(NAME,'experiments/q_rmea_output_matching.py','experiments/q_rmea_query_alignment.py')
REVIEW=Path('docs/evidence/rmea/20261121-task-factorization/finite-probes-implementation-review.json')
PROPOSAL=Path('reports/q_rmea_centered_output/20261121_valid96/next-design.json')
CHECKPOINT=Path('/hy-tmp/Q-Attention/worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13/discovery/checkpoints/native-final.pt')
SHA='2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51'
TOL=1e-10

def design():
    return dict(schema='q-rmea.finite-probes.design.v1',proposal_sha256=p.digest(ROOT/PROPOSAL),checkpoint_sha256=SHA,
        port_shape=[76,1024],modular_prime=2147483647,
        frame='Lexicographic4^3 first-layer(RY,RZ)=(0,0),(pi/2,0),(pi/2,pi/2),(-pi/2,0); secondlayerzero',
        selection='One scipy QR pivoting40x64 scores: first9columns; pivotQR of selectedtranspose: first9rows. No retries',
        decoder='Fixed first2valuecoordinates; all40projectedrowsnonzero; exact determinantrows0,1 nonzero',
        certificate='mpmath.iv dps40 idealrealcircuit with exactbinary64 checkpoint/epsilon. Neumann ||I-C*Sminor||infinity upper<1; C is binary64 inverse',
        probes='T9,zero baseline+16singletonvalueprobes; fixedroleszero; minimum-norm h solve, actualnativefullheadCPUforward',
        gates=dict(port_modular_rank=76,inverse_residual=TOL,decoder_singular_min=1e-8,neumann_upper=1.,probe_residual=TOL),
        launch='One CPU process,1CPU/4GiB/256MiB/180s,emptyCUDA,exactcleanreviewedsource,maxattempt1,replayfalse',
        outputs=['summary.json','design.json','inputs.json','certificates.json','RUN_COMPLETE'],
        recovery='Cheap replay from originalprotectedcheckpoint/Git; no newcheckpoint/COS/inputcopies',
        scope='Ideal syntheticfeatureport theorem and numericalreplay; fixedrouting acrossvalueinterventions only. No naturaltext/full-hrouter exclusion/taskutility/probabilitygap/L5')

def validate_review():
    v=p.read(ROOT/REVIEW)
    if (v.get('verdict')!='PASS' or v.get('review_scope')!='both' or v.get('must_fix')!=[]
        or v.get('hashes')!={n:p.digest(ROOT/n) for n in FILES}
        or v.get('design_sha256')!=p.identity(design()) or v.get('core_sha256')!=p.identity(p.sources())):
        raise ValueError('Exact finite-port mechanism/lifecycle review required')

def guard(out):
    validate_review()
    if sys.flags.optimize or p.git('status','--porcelain') or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Clean CPU-only source required')
    out=out.resolve();old=CHECKPOINT.parent.parent
    if any(out.is_relative_to(r) or r.is_relative_to(out) for r in (ROOT,old)):
        raise ValueError('Independent output required')
    m.qa.fresh_output(out);return out

def event(out,phase,current):
    p.atomic_json(out/'progress.json',dict(schema='research.progress.v1',attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,
        phase=phase,stage=phase,current=current,completed=current,total=4,unit='audit_stage',observed_at=time.time(),batch_id='rmea-finite-probes-20261121'))

def rank_mod(a,prime=2147483647):
    rows=[[(n%prime)*pow(d%prime,-1,prime)%prime for n,d in (float(v).as_integer_ratio() for v in row)] for row in a]
    rank=0;pivots=[]
    for col in range(len(rows[0])):
        pivot=next((r for r in range(rank,len(rows)) if rows[r][col]),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank];inv=pow(rows[rank][col],-1,prime)
        rows[rank]=[(v*inv)%prime for v in rows[rank]]
        for r in range(rank+1,len(rows)):
            c=rows[r][col]
            if c:rows[r]=[(x-c*y)%prime for x,y in zip(rows[r],rows[rank])]
        pivots.append(col);rank+=1
        if rank==len(rows):break
    return rank,pivots

def iv_float(x,iv):
    n,d=float(x).as_integer_ratio();return iv.mpf(n)/iv.mpf(d)

def interval_circuit(angles,iv):
    state=[iv.mpc(0,0) for _ in range(8)];state[0]=iv.mpc(1,0)
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer][wire];c,s=iv.cos(y/2),iv.sin(y/2)
            minus,plus=iv.mpc(iv.cos(z/2),-iv.sin(z/2)),iv.mpc(iv.cos(z/2),iv.sin(z/2))
            bit=1<<(2-wire);old=state[:]
            for i in range(8):
                if not i&bit:
                    a,b=old[i],old[i|bit];state[i],state[i|bit]=(c*a-s*b)*minus,(s*a+c*b)*plus
        for control,target in ((0,1),(1,2),(2,0)):
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=[state[i] for i in perm]
    return state

def interval_scores(effect,frames,rows,columns,epsilon):
    from mpmath import iv
    iv.dps=40
    def theta(x):
        ex=iv.exp(2*iv_float(x,iv));return iv.pi*(ex-1)/(ex+1)
    effects=[interval_circuit([[[theta(x) for x in w] for w in l] for l in effect[r]],iv) for r in rows]
    choices=((0,0),(1,0),(1,1),(-1,0));states=[]
    for t in columns:
        first=[[iv.pi*choices[k][0]/2,iv.pi*choices[k][1]/2] for k in frames[t]]
        states.append(interval_circuit([first,[[iv.mpf(0),iv.mpf(0)] for _ in range(3)]],iv))
    result=[]
    for phi in effects:
        row=[]
        for psi in states:
            a=sum((iv.mpc(f.real,-f.imag)*s for f,s in zip(phi,psi)),iv.mpc(0,0))
            row.append(a.real*a.real+a.imag*a.imag+iv_float(epsilon,iv))
        result.append(row)
    return result

def interval_inverse_bound(c,matrix):
    from mpmath import iv
    bounds=[]
    for i in range(len(c)):
        total=F(0)
        for j in range(len(c)):
            e=iv.mpf(int(i==j))-sum((iv_float(c[i,k],iv)*matrix[k][j] for k in range(len(c))),iv.mpf(0))
            upper=math.nextafter(max(abs(float(e.a)),abs(float(e.b))),math.inf);total+=F.from_float(upper)
        bounds.append(total)
    return math.nextafter(float(max(bounds)),math.inf)

def audit(head,out):
    import numpy as np
    import torch
    from scipy.linalg import qr
    from baselines.retacred.rmea import EPS,circuit
    from baselines.retacred.rmea_token_transport import route
    matrix=np.concatenate((head.prepare.weight.detach().numpy()[:,:1024],head.value.weight.detach().numpy()))
    rank,pivots=rank_mod(matrix);singular=np.linalg.svd(matrix,compute_uv=False)
    right=np.linalg.lstsq(matrix,np.eye(76),rcond=None)[0];residual=float(np.max(np.abs(matrix@right-np.eye(76))))
    decoder=head.decoder.detach().numpy()[:,:2]
    det=F.from_float(float(decoder[0,0]))*F.from_float(float(decoder[1,1]))-F.from_float(float(decoder[0,1]))*F.from_float(float(decoder[1,0]))
    dmin=float(np.linalg.svd(decoder,compute_uv=False)[-1])
    cert=dict(port=dict(modular_rank=rank,pivot_columns=pivots,prime=2147483647,singular_min=float(singular[-1]),singular_max=float(singular[0]),inverse_residual=residual),
        decoder=dict(nonzero_rows=int(np.count_nonzero(np.any(decoder!=0,axis=1))),singular_min=dmin,rows01_determinant=dict(numerator=str(det.numerator),denominator=str(det.denominator))))
    event(out,'port_and_decoder',1)
    if not(rank==76 and residual<=TOL and cert['decoder']['nonzero_rows']==40 and det!=0 and dmin>1e-8):
        return dict(gate=False,stopped_at='port_or_decoder'),cert
    frames=list(itertools.product(range(4),repeat=3));choices=((0,0),(1,0),(1,1),(-1,0))
    angles=torch.zeros((64,2,3,2),dtype=torch.float64)
    for t,frame in enumerate(frames):
        for w,k in enumerate(frame):angles[t,0,w]=torch.tensor(choices[k],dtype=torch.float64)*math.pi/2
    psi=circuit(angles);phi=circuit(math.pi*torch.tanh(head.effect))
    score=((phi.conj()@psi.T).abs().square()+EPS).detach().numpy()
    columns=list(map(int,qr(score,mode='economic',pivoting=True)[2][:9]))
    rows=list(map(int,qr(score[:,columns].T,mode='economic',pivoting=True)[2][:9]))
    cert['score']=dict(columns=columns,rows=rows,epsilon=EPS)
    minor=score[np.ix_(rows,columns)]
    try:inverse=np.linalg.inv(minor)
    except np.linalg.LinAlgError:
        cert['score']['failure']='selected_minor_singular';event(out,'score_certificate',2)
        return dict(gate=False,stopped_at='score_interval_certificate'),cert
    bound=interval_inverse_bound(inverse,interval_scores(head.effect.detach().numpy(),frames,rows,columns,EPS))
    alpha=route(psi[columns][None],phi,torch.ones((1,9),dtype=torch.bool),kind='native')['alpha'][0]
    sv=np.linalg.svd(alpha.detach().numpy(),compute_uv=False)
    cert['score']=dict(columns=columns,rows=rows,interval_dps=40,neumann_upper=bound,native_singular_min=float(sv[-1]),minor_singular_min=float(np.linalg.svd(minor,compute_uv=False)[-1]),
        epsilon=EPS,inverse=inverse.tolist(),scope='Nonsingularity of idealrealcircuit9minor; not classifier probabilityerrorbound')
    event(out,'score_certificate',2)
    if not bound<1:return dict(gate=False,stopped_at='score_interval_certificate'),cert
    pre=np.arctanh(angles[columns].numpy().reshape(9,12)/math.pi);targets=[]
    offset=np.r_[head.prepare.bias.detach().numpy(),head.value.bias.detach().numpy()]
    for t in range(17):
        values=np.zeros((9,64))
        if t:
            token,direction=divmod(t-1,2);values[token,direction]=1
        targets.append((np.concatenate((pre,values),axis=1)-offset)@right.T)
    h=torch.from_numpy(np.stack(targets))
    d=head(h,torch.zeros((17,9,4),dtype=torch.float64),torch.ones((17,9),dtype=torch.bool),details=True)
    values=torch.zeros((17,9,64),dtype=torch.float64)
    for t in range(1,17):
        token,direction=divmod(t-1,2);values[t,token,direction]=1
    expected=torch.einsum('rt,btd,rd->br',alpha,values,head.decoder)+head.bias
    lp=torch.log_softmax(d['logits'],dim=-1);el=torch.log_softmax(expected,dim=-1)
    errors=dict(preparation_angles=float((d['angles']-angles[columns][None]).abs().max()),state=float((d['psi']-psi[columns][None]).abs().max()),
        routing=float((d['alpha']-alpha[None]).abs().max()),values=float((d['values']-values).abs().max()),logits=float((d['logits']-expected).abs().max()),
        log_odds=float(((lp-lp[:,0:1])-(el-el[:,0:1])).abs().max()))
    cert['probes']=dict(count=17,nonbaseline=16,tokens=9,errors=errors,feature_max_abs=float(h.abs().max()),
        interpretation='Binary64 replay residuals. Idealexact h existence from exactrational rank; not textencoder reachability')
    event(out,'finite_value_probes',3)
    return dict(gate=all(math.isfinite(x) and x<=TOL for x in errors.values()),stopped_at=None),cert

def run(args):
    out=guard(args.output);started=time.time();data=CHECKPOINT.read_bytes()
    if hashlib.sha256(data).hexdigest()!=SHA or p.read(ROOT/PROPOSAL)['checkpoint_sha256']!=SHA:raise ValueError('Protected checkpoint differs')
    import torch
    from baselines.retacred.rmea_token_transport import TrainableTokenTransportHead
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    head=TrainableTokenTransportHead('native').double().eval();head.load_state_dict(torch.load(io.BytesIO(data),map_location='cpu',weights_only=True))
    p.atomic_json(out/'design.json',design());p.atomic_json(out/'inputs.json',dict(checkpoint_sha256=SHA,original_source=m.qa.ORIGINAL,core_sha256=p.identity(p.sources())))
    event(out,'start',0)
    with torch.no_grad():result,cert=audit(head,out)
    result.update(schema='q-rmea.finite-probes.result.v1',source=p.git('rev-parse','HEAD'),checkpoint_sha256=SHA,seconds=time.time()-started,scope=design()['scope'])
    p.atomic_json(out/'summary.json',result);p.atomic_json(out/'certificates.json',cert)
    event(out,'complete',4);print(json.dumps(result),flush=True)
    p.atomic_json(out/'RUN_COMPLETE',dict(artifacts={n:p.digest(out/n) for n in design()['outputs'] if n!='RUN_COMPLETE'}))

def unit(args):
    import numpy as np
    import torch
    from mpmath import iv
    from baselines.retacred.rmea import circuit
    out=guard(args.output)
    assert rank_mod([[1.,2.],[2.,4.]])[0]==1
    assert rank_mod([[1.,1.],[1.,1.+2**-40]])[0]==2
    iv.dps=40
    assert interval_inverse_bound(np.eye(2),[[iv.mpf(1),iv.mpf(0)],[iv.mpf(0),iv.mpf(1)]])<1e-30
    assert interval_inverse_bound(np.eye(2),[[iv.mpf(1),iv.mpf(0)],[iv.mpf(0),iv.mpf(0)]])>=1
    angles=[[[.2,-.3],[.4,.1],[-.2,.6]],[[.5,-.1],[.1,.8],[-.7,.9]]]
    exact=interval_circuit([[[iv_float(v,iv) for v in w] for w in l] for l in angles],iv)
    actual=circuit(torch.tensor(angles,dtype=torch.float64)).numpy()
    frames=list(itertools.product(range(4),repeat=3))
    scores=interval_scores(np.zeros((40,2,3,2)),frames,[0],[0,16],0.)
    assert float(scores[0][0].a)<=1<=float(scores[0][0].b)
    assert float(scores[0][1].a)<=.5<=float(scores[0][1].b)
    assert max(abs(complex((float(v.real.a)+float(v.real.b))/2,(float(v.imag.a)+float(v.imag.b))/2)-a) for v,a in zip(exact,actual))<1e-14
    event(out,'unit',4)
    p.atomic_json(out/'unit-validation.json',dict(status='PASS',modular_rank=True,interval_positive_and_singular_rejection=True,interval_actual_circuit_parity=True,interval_born_score_regression=True,scope='Focused certifier/circuit checks; actualcheckpoint untouched'))
    p.atomic_json(out/'RUN_COMPLETE',dict(sha256=p.digest(out/'unit-validation.json')))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('unit','run'));ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();dict(unit=unit,run=run)[args.action](args)
if __name__=='__main__':main()
