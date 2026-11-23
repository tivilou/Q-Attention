"""Certified shared-weight matching after eliminating a common logit offset."""
import argparse
from fractions import Fraction as F
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import q_rmea_output_matching as m
p=m.p
PROPOSAL=Path('reports/q_rmea_output_matching/20261121_valid96/next-design.json')
REVIEW=Path('docs/evidence/rmea/20261121-task-factorization/centered-implementation-review.json')
FILES=('experiments/q_rmea_centered_output.py','experiments/q_rmea_output_matching.py',
       'experiments/q_rmea_query_alignment.py')


def design():
    return dict(schema='q-rmea.centered-output.design.v1',proposal=p.read(ROOT/PROPOSAL),
        proposal_sha256=p.digest(ROOT/PROPOSAL),threshold='1/10000000 anchored logit difference infinity error',
        arithmetic='Exact differences of original stored binary64 U,z; exact rational primal and dual verification',
        solver=dict(method='highs',options=m.LP_OPTIONS,primal='one unscaled minimax LP',
                    dual='one max-min separator with nonnegative positive/negative parts and L1 budget1'),
        launch='Three CPU workers within one scheduler allocation, each32cases; no GPU, models or cache reread',
        recovery='Cheap replay from sealed parent private arrays; no new checkpoints or COS',
        scope='Conditional development K1 anchored-error contrast against existing44rawK8 witnesses, not probability-error lowerbound or L5')


def validate_review():
    row=p.read(ROOT/REVIEW)
    if (row.get('verdict')!='PASS' or row.get('review_scope')!='both' or row.get('must_fix')
        or row.get('hashes')!={n:p.digest(ROOT/n) for n in FILES}
        or row.get('design_sha256')!=p.identity(design())
        or row.get('core_sha256')!=p.identity(p.sources())):
        raise ValueError('Exact centered mechanism/lifecycle review required')


def matrix_exact(u,z):
    import numpy as np
    if (u.ndim!=2 or z.shape!=(len(u),) or len(u)<2 or u.shape[1]<1
        or not np.isfinite(u).all() or not np.isfinite(z).all()):
        raise ValueError('Invalid centered inputs')
    # Subtract as rationals, never round the differences before certification.
    uu=[[m.fraction(v) for v in row] for row in u]
    zz=[m.fraction(v) for v in z]
    return [[uu[r][t]-uu[0][t]-zz[r]+zz[0] for t in range(u.shape[1])]
            for r in range(1,len(u))]


def primal(n,x):
    xx=[m.fraction(v) for v in x]
    if len(xx)!=len(n[0]) or min(xx)<0 or sum(xx)<=0:
        raise ValueError('Nonnegative nonzero primal required')
    bound=max(abs(sum(v*w for v,w in zip(row,xx))) for row in n)/sum(xx)
    return m.ratio_record(bound,True)


def dual(n,y):
    yy=[m.fraction(v) for v in y]
    if len(yy)!=len(n):raise ValueError('Dual shape differs')
    norm=sum(abs(v) for v in yy)
    delta=min(sum(yy[r]*n[r][t] for r in range(len(n))) for t in range(len(n[0])))
    return m.ratio_record(delta/norm,False) if norm>0 and delta>0 else None


def solve(u,z):
    import numpy as np
    from scipy.optimize import linprog
    exact=matrix_exact(u,z);n=np.asarray(exact,dtype=np.float64)
    if not np.isfinite(n).all():raise ValueError('Nonfinite search matrix')
    rows,t=n.shape
    fit=linprog(np.r_[np.zeros(t),1.],
        A_ub=np.vstack((np.c_[n,-np.ones(rows)],np.c_[-n,-np.ones(rows)])),
        b_ub=np.zeros(2*rows),A_eq=np.r_[np.ones(t),0.][None],b_eq=[1.],
        bounds=[(0.,None)]*(t+1),method='highs',options=m.LP_OPTIONS)
    out=dict(status='unresolved',upper=None,lower=None,primal_solver_status=int(fit.status),dual_solver_status=None)
    witness={}
    if fit.x is not None and np.isfinite(fit.x).all():
        x=np.maximum(fit.x[:-1],0.);witness['primal']=x
        if x.sum()>0:
            out['upper']=primal(exact,x)
            if m.record_fraction(out['upper'])<=m.ETA:
                out['status']='approximate_match';return out,witness
    # y = yplus-yminus; the LP's L1 constraint proposes a bounded separator.
    fit=linprog(np.r_[np.zeros(2*rows),-1.],
        A_ub=np.vstack((np.c_[-n.T,n.T,np.ones(t)],np.r_[np.ones(2*rows),0.][None])),
        b_ub=np.r_[np.zeros(t),1.],bounds=[(0.,None)]*(2*rows)+[(None,None)],
        method='highs',options=m.LP_OPTIONS)
    out['dual_solver_status']=int(fit.status)
    if fit.x is not None and np.isfinite(fit.x).all():
        y=fit.x[:rows]-fit.x[rows:2*rows];witness['dual']=y
        out['lower']=dual(exact,y)
        if out['lower'] and m.record_fraction(out['lower'])>m.ETA:
            out['status']='separated_above_tolerance'
    return out,witness


def event(out,phase,current,total=96,**extra):
    p.atomic_json(out/'progress.json',dict(schema='research.progress.v1',
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,phase=phase,stage=phase,
        unit='sample',current=current,completed=current,total=total,observed_at=time.time(),
        batch_id='rmea-centered-output-20261121',**extra))


def report(out,phase,states):
    event(out,phase,sum(x['current'] for x in states),jobs=states)


def verify_parent_seal(root,seal):
    required={'design.json','inputs.json','case-status.json','summary.json'}
    if set(seal.get('artifacts',{}))!=required:
        raise ValueError('Complete parent artifact seals required')
    for n,h in seal['artifacts'].items():
        if p.digest(root/n)!=h:raise ValueError('Parent artifact seal differs')


def parent_cases(root):
    proposal=design()['proposal']
    if root.name!='output' or root.parent.name!=proposal['parent_attempt']:
        raise ValueError('Frozen parent attempt differs')
    data=(root/'case-status.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=proposal['case_status_sha256']:
        raise ValueError('Sealed parent case identity differs')
    cases=__import__('json').loads(data)
    seal=p.read(root/'RUN_COMPLETE')
    verify_parent_seal(root,seal)
    summary=p.read(root/'summary.json')
    if (summary['source']!=proposal['parent_executed_source'] or summary['count']!=96
        or sorted(c['index'] for c in cases)!=sorted(proposal['indices'])):
        raise ValueError('Frozen parent source/coverage differs')
    return {c['index']:c for c in cases}


def command(args,shard):
    return [sys.executable,'-B',str(Path(__file__).resolve()),'worker',
        '--parent-output',str(args.parent_output),'--output',str(args.output),
        '--parent-pid',str(os.getpid()),'--shard',str(shard)]


def worker(args):
    import ctypes,signal
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Parent-death fencing failed')
    if os.getppid()!=args.parent_pid:raise RuntimeError('Parent identity changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('CPU allocation required')
    import numpy as np
    cases=parent_cases(args.parent_output)
    out=args.output/('shard'+str(args.shard));out.mkdir()
    values=out/'witnesses';values.mkdir()
    rows=[]
    for i in design()['proposal']['indices'][args.shard::3]:
        original=cases[i]
        # Original shard ownership is the frozen round-robin proposal.
        pos=design()['proposal']['indices'].index(i)
        f=args.parent_output/('shard'+str(pos%3))/'values'/(str(i)+'.npz')
        if f.is_symlink():raise ValueError('Symlinked parent witness')
        raw=f.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=original['private_values_sha256']:
            raise ValueError('Parent numeric input changed')
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
            u,z=arrays['utilities'],arrays['target_logits']
        if u.shape!=(40,original['tokens']) or z.shape!=(40,) or u.dtype!=np.float64 or z.dtype!=np.float64:
            raise ValueError('Original numeric shape/dtype differs')
        result,witness=solve(u,z)
        target=values/(str(i)+'.npz');np.savez_compressed(target,**witness)
        rows.append(dict(index=i,tokens=original['tokens'],centered_K1=result,
            existing_raw_K8=original['controls']['K8']['status'],
            parent_values_sha256=original['private_values_sha256'],private_witness_sha256=p.digest(target)))
        event(out,'centered_matching',len(rows),32,shard=args.shard)
    p.atomic_json(out/'cases.json',rows)
    p.atomic_json(out/'WORKER_COMPLETE',dict(count=len(rows),cases_sha256=p.digest(out/'cases.json')))


def run(args):
    validate_review()
    if p.git('status','--porcelain') or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Clean CPU-only exact source required')
    args.output,args.parent_output=args.output.resolve(),args.parent_output.resolve()
    if any(args.output.is_relative_to(d) or d.is_relative_to(args.output) for d in (ROOT,args.parent_output)):
        raise ValueError('Independent output required')
    m.qa.fresh_output(args.output)
    original=parent_cases(args.parent_output);started=time.time()
    p.atomic_json(args.output/'design.json',design())
    p.atomic_json(args.output/'inputs.json',dict(parent_source=design()['proposal']['parent_executed_source'],
        parent_attempt=design()['proposal']['parent_attempt'],case_status_sha256=p.digest(args.parent_output/'case-status.json')))
    m.qa.execute([('shard'+str(i),command(args,i),'') for i in range(3)],args.output,'centered_matching',report=report)
    rows=[]
    for i in range(3):
        out=args.output/('shard'+str(i));part=p.read(out/'cases.json');seal=p.read(out/'WORKER_COMPLETE')
        if p.digest(out/'cases.json')!=seal['cases_sha256'] or [x['index'] for x in part]!=design()['proposal']['indices'][i::3]:
            raise ValueError('Centered worker coverage/seal differs')
        for x in part:
            if p.digest(out/'witnesses'/(str(x['index'])+'.npz'))!=x['private_witness_sha256']:
                raise ValueError('Centered witness changed')
        rows.extend(part)
    if len(rows)!=96 or len({x['index'] for x in rows})!=96:raise ValueError('Coverage differs')
    counts={s:sum(x['centered_K1']['status']==s for x in rows) for s in m.STATUSES}
    paired=sum(x['centered_K1']['status']=='separated_above_tolerance' and x['existing_raw_K8']=='approximate_match' for x in rows)
    summary=dict(schema='q-rmea.centered-output.result.v1',source=p.git('rev-parse','HEAD'),count=96,
        counts=counts,existing_raw_K8_matches=sum(x['existing_raw_K8']=='approximate_match' for x in rows),
        paired_distinguishing_cases=paired,gate=paired>0,seconds=time.time()-started,
        scope='39anchoredlogitdifference K1 certificates and inherited rawK8 sufficiency; conditional development, no probability-error lowerbound/utility/L5')
    p.atomic_json(args.output/'case-status.json',rows);p.atomic_json(args.output/'summary.json',summary)
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={n:p.digest(args.output/n) for n in ('design.json','inputs.json','case-status.json','summary.json')}))
    event(args.output,'complete',96);print(__import__('json').dumps(summary),flush=True)


def unit(args):
    import numpy as np
    import tempfile
    validate_review()
    if p.git('status','--porcelain') or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Clean CPU-only exact source required')
    m.qa.fresh_output(args.output)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for n in ('design.json','inputs.json','case-status.json','summary.json'):
            (root/n).write_text(n)
        valid=dict(artifacts={n:p.digest(root/n) for n in ('design.json','inputs.json','case-status.json','summary.json')})
        verify_parent_seal(root,valid)
        for bad in ({'artifacts':{}},{'artifacts':{'summary.json':valid['artifacts']['summary.json']}}):
            try:verify_parent_seal(root,bad)
            except ValueError:pass
            else:raise AssertionError('Incomplete parent seal accepted')
        (root/'summary.json').write_text('changed')
        try:verify_parent_seal(root,valid)
        except ValueError:pass
        else:raise AssertionError('Changed parent artifact accepted')
    u=np.array([[0.,1.],[1.,0.]]);z=np.array([.9,.9]);n=matrix_exact(u,z)
    assert m.dual(np.ones((2,1)),u,z,[-1.,-1.])['bound']>.39
    assert primal(n,[1.,1.])['bound']==0 and solve(u,z)[0]['status']=='approximate_match'
    u=np.array([[0.,0.],[0.,1.]]);z=np.array([0.,2.]);n=matrix_exact(u,z)
    assert m.record_fraction(dual(n,[-1.]))==1 and dual(n,[1.]) is None
    result,_=solve(u,z);assert result['status']=='separated_above_tolerance'
    assert m.record_fraction(result['lower'])<=1<=m.record_fraction(result['upper'])
    # The original binary64 differences contain1 that a float-first subtraction loses.
    n=matrix_exact(np.array([[1e16],[1.]]),np.array([1e16,0.]))
    assert n==[[F(1)]]
    for x in ([0.,0.],[-1.,2.]):
        try:primal([[F(1),F(2)]],x)
        except ValueError:pass
        else:raise AssertionError('Invalid primal accepted')
    dummy=argparse.Namespace(parent_output=Path('/unused/output'),output=Path('/unused/new'))
    for i in range(3):
        v=parser().parse_args(command(dummy,i)[3:]);assert v.action=='worker' and v.shard==i and v.parent_pid==os.getpid()
    check=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--help'],capture_output=True,timeout=20)
    assert check.returncode==0
    report(args.output,'centered_matching',[dict(current=1),dict(current=2),dict(current=3)])
    v=p.read(args.output/'progress.json');assert v['current']==6 and v['total']==96
    p.atomic_json(args.output/'unit-validation.json',dict(status='PASS',raw_gap_centered_match=True,
        exact_dual_minimax_fixture=True,exact_original_differences=True,invalid_primals_rejected=True,
        incomplete_and_changed_parent_seals_rejected=True,
        real_CLI_and_callback=True,scope='Focused arithmetic/CLI contract, no GPU or new data'))
    p.atomic_json(args.output/'RUN_COMPLETE',dict(sha256=p.digest(args.output/'unit-validation.json')))


def parser():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('unit','worker','run'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--parent-output',type=Path)
    ap.add_argument('--parent-pid',type=int);ap.add_argument('--shard',type=int,choices=range(3));return ap


def main():
    ap=parser();args=ap.parse_args()
    if args.action!='unit' and args.parent_output is None:ap.error('--parent-output required')
    if args.action=='worker' and (args.parent_pid is None or args.shard is None):ap.error('Exact parent/shard required')
    dict(unit=unit,worker=worker,run=run)[args.action](args)


if __name__=='__main__':main()
