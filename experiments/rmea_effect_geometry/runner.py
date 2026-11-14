"""Read only two immutable checkpoints; emit aggregate geometry, no datasets."""
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import numpy as np
    import torch
    from legacy.rmea import RMEAHead,circuit
    from geometry import describe,dense_states
    from unit_checks import run_checks
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.5.1+cpu' and np.__version__=='1.26.4'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    src=Path(__file__).resolve().parent;repo=src.parents[1]
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
    assert not git('status','--porcelain=v1')
    out=args.output.resolve();assert not out.is_relative_to(repo) and not out.is_symlink()
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    contract=json.loads((src/'contract.json').read_text());started=time.monotonic()
    unit=run_checks();save(out/'unit.json',unit)
    torch.manual_seed(419);initial=RMEAHead('quantum')
    initial_sha=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in initial.parameters())).hexdigest()
    assert initial_sha==contract['small_initial_parameters_sha256']
    models={'small_seed419_initial':initial};input_identity={}
    for name,x in contract['checkpoints'].items():
        path=Path(x['path']);assert path.is_file() and not path.is_symlink() and path.stat().st_size<1048576
        assert sha(path)==x['sha256']
        model=RMEAHead('quantum');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        models[name]=model;input_identity[name]=dict(sha256=x['sha256'],bytes=path.stat().st_size,scope=x['scope'])
    reports={};maximum_parity=0.
    with torch.inference_mode():
        for name,model in models.items():
            assert sum(t.numel() for t in model.parameters())==81028
            # Descriptive float64 analysis of exactly stored fp32 parameters.
            angles=math.pi*torch.tanh(model.effect.detach().double())
            a=circuit(angles).numpy();b=dense_states(angles.numpy())
            np.testing.assert_allclose(a,b,atol=2e-14,rtol=0)
            maximum_parity=max(maximum_parity,float(np.abs(a-b).max()))
            reports[name]=describe(a)
    for name,x in contract['checkpoints'].items():assert sha(Path(x['path']))==x['sha256']
    summary=dict(schema='qattn.effect-geometry.descriptive.v1',status='PASS',geometry=reports,input_identity=input_identity,
        initial_parameters_sha256=initial_sha,max_dense_circuit_error=maximum_parity,
        source_revision=git('rev-parse','HEAD'),source_sha256=sha(Path(__file__)),contract_sha256=sha(src/'contract.json'),
        elapsed_seconds=time.monotonic()-started,training_runs=0,test_loaded=False,task_examples_read=0,cos_bytes=0,
        interpretation='Descriptive geometry only. No utility, causality, promotion threshold, optimal task geometry, or L3 claim. Tight isotropy still permits duplicate states. Independent binary measurements need not sum to identity.',
        numeric_scope='float64 of frozen fp32 parameters, not deployment float32 forward parity')
    save(out/'summary.json',summary)
    files={f.name:sha(f) for f in out.iterdir()};assert set(files)=={'summary.json','unit.json'}
    save(out/'RUN_COMPLETE',dict(schema='qattn.effect-geometry.closed.v1',files=files))
    print(json.dumps(dict(status='COMPLETE',checkpoints_unchanged=True,outputs=sorted(files))))

if __name__=='__main__':main()
